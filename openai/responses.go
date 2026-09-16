package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/matiasinsaurralde/rimeno"
)

// The Responses API (POST /v1/responses) is OpenAI's newer surface. Unlike Chat
// Completions it represents a turn as typed input/output *items* and keeps a
// reasoning model's thinking as first-class reasoning items — which is why Codex
// requires it (reasoning is preserved across tool calls instead of re-derived each
// turn). OpenRouter also serves /v1/responses. WithResponsesAPI flips this Client
// to speak it; the rimeno.Model contract is unchanged.
//
// This implements the non-streaming Generate path end-to-end (message history →
// input items incl. tool calls/results, and output items → text / reasoning /
// tool calls / usage). Streaming over Responses is a documented follow-on.

// WithResponsesAPI makes the client speak the Responses API (/responses) instead
// of Chat Completions (/chat/completions).
func WithResponsesAPI() Option { return func(c *Client) { c.useResponses = true } }

// ThreadsResponses implements rimeno.ResponseThreader: this client can continue a
// turn server-side (via previous_response_id) exactly when it speaks the Responses
// API. Chat Completions cannot, so it reports false and rimeno sends full history.
func (c *Client) ThreadsResponses() bool { return c.useResponses }

// ThreadingUnsupported reports whether err means the provider rejected a
// previous_response_id (so rimeno should stop threading this session and retry with full
// history). ThreadsResponses is a client capability — the client CAN send the field —
// but not every Responses backend honors it: OpenRouter, for one, 400s a non-null
// previous_response_id for models without server-side state ("expected null, received
// string"). Catching that here lets rimeno degrade gracefully instead of failing the run.
func (c *Client) ThreadingUnsupported(err error) bool {
	var ae *APIError
	if !errors.As(err, &ae) {
		return false
	}
	return ae.StatusCode == http.StatusBadRequest && strings.Contains(ae.Raw, "previous_response_id")
}

// --- wire types -------------------------------------------------------------

type respRequest struct {
	Model              string          `json:"model"`
	Input              []respItem      `json:"input"`
	Tools              []respTool      `json:"tools,omitempty"`
	ToolChoice         string          `json:"tool_choice,omitempty"`
	MaxOutputTokens    int             `json:"max_output_tokens,omitempty"`
	Temperature        *float64        `json:"temperature,omitempty"`
	Reasoning          *wireReasoning  `json:"reasoning,omitempty"`
	Text               *respTextFormat `json:"text,omitempty"`
	PreviousResponseID string          `json:"previous_response_id,omitempty"`
	Stream             bool            `json:"stream,omitempty"`
}

type respTextFormat struct {
	Format wireJSONSchemaResp `json:"format"`
}

type wireJSONSchemaResp struct {
	Type   string          `json:"type"` // "json_schema"
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

// respItem is a Responses input OR output item. Only the fields we produce/consume
// are modeled; unknown item types decode into Type and are skipped.
type respItem struct {
	Type      string        `json:"type"`              // message | function_call | function_call_output | reasoning
	Role      string        `json:"role,omitempty"`    // for message
	Content   []respContent `json:"content,omitempty"` // for message
	CallID    string        `json:"call_id,omitempty"` // for function_call / function_call_output
	Name      string        `json:"name,omitempty"`    // for function_call
	Arguments string        `json:"arguments,omitempty"`
	Output    string        `json:"output,omitempty"`  // for function_call_output
	Summary   []respContent `json:"summary,omitempty"` // for reasoning (some servers)
}

type respContent struct {
	Type string `json:"type"` // input_text | output_text | reasoning_text | summary_text
	Text string `json:"text"`
}

type respTool struct {
	Type        string          `json:"type"` // "function"
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type respResponse struct {
	ID     string     `json:"id"`
	Output []respItem `json:"output"`
	Usage  *respUsage `json:"usage"`
	Model  string     `json:"model"`
}

type respUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	TotalTokens        int `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details,omitempty"`
}

// --- generate ---------------------------------------------------------------

func (c *Client) generateResponses(ctx context.Context, req *rimeno.Request) (*rimeno.Response, error) {
	body, err := json.Marshal(c.buildRespRequest(req))
	if err != nil {
		return nil, fmt.Errorf("openai: encode responses request: %w", err)
	}
	raw, err := c.do(ctx, body)
	if err != nil {
		return nil, err
	}
	var rr respResponse
	if err := json.Unmarshal(raw, &rr); err != nil {
		return nil, fmt.Errorf("openai: decode responses: %w", err)
	}
	msg := parseRespMessage(rr)
	return &rimeno.Response{
		Message:    msg,
		Usage:      c.respUsage(rr.Usage),
		StopReason: respStop(msg),
		Model:      firstNonEmpty(rr.Model, c.model),
		ID:         rr.ID,
		Raw:        raw,
	}, nil
}

// parseRespMessage assembles an assistant message from a Responses output array
// (text, reasoning summary, and function-call items).
func parseRespMessage(rr respResponse) rimeno.Message {
	msg := rimeno.Message{Role: rimeno.RoleAssistant}
	for _, it := range rr.Output {
		switch it.Type {
		case "message":
			for _, ct := range it.Content {
				if ct.Type == "output_text" {
					msg.Text += ct.Text
				}
			}
		case "reasoning":
			for _, ct := range it.Summary {
				msg.Reasoning += ct.Text
			}
		case "function_call":
			msg.ToolCalls = append(msg.ToolCalls, rimeno.ToolCall{
				ID:        it.CallID,
				Name:      it.Name,
				Arguments: json.RawMessage(it.Arguments),
			})
		}
	}
	return msg
}

func (c *Client) buildRespRequest(req *rimeno.Request) *respRequest {
	model := req.Model
	if model == "" {
		model = c.model
	}
	out := &respRequest{
		Model:              model,
		Input:              toRespInput(req.Messages),
		Temperature:        req.Temperature,
		MaxOutputTokens:    req.MaxTokens,
		PreviousResponseID: req.PreviousResponseID,
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, respTool{
			Type: "function", Name: t.Name, Description: t.Description, Parameters: t.Parameters,
		})
	}
	if req.ToolChoice != "" && len(req.Tools) > 0 {
		out.ToolChoice = string(req.ToolChoice)
	}
	if req.ReasoningEffort != "" || req.ReasoningMaxTokens > 0 {
		out.Reasoning = &wireReasoning{Effort: req.ReasoningEffort, MaxTokens: req.ReasoningMaxTokens}
	}
	if req.ResponseFormat != nil {
		out.Text = &respTextFormat{Format: wireJSONSchemaResp{
			Type: "json_schema", Name: sanitizeName(req.ResponseFormat.Name),
			Schema: req.ResponseFormat.Schema, Strict: req.ResponseFormat.Strict,
		}}
	}
	return out
}

// toRespInput maps rimeno's chat-style history to Responses input items, including
// assistant tool calls (function_call) and tool results (function_call_output) so
// the agent loop's multi-turn tool use works.
func toRespInput(msgs []rimeno.Message) []respItem {
	var items []respItem
	for _, m := range msgs {
		switch m.Role {
		case rimeno.RoleTool:
			items = append(items, respItem{Type: "function_call_output", CallID: m.ToolCallID, Output: m.Text})
		case rimeno.RoleAssistant:
			if m.Text != "" {
				items = append(items, respItem{Type: "message", Role: "assistant",
					Content: []respContent{{Type: "output_text", Text: m.Text}}})
			}
			for _, tc := range m.ToolCalls {
				items = append(items, respItem{Type: "function_call", CallID: tc.ID, Name: tc.Name,
					Arguments: string(tc.Arguments)})
			}
		default: // system / user
			ct := "input_text"
			items = append(items, respItem{Type: "message", Role: string(m.Role),
				Content: []respContent{{Type: ct, Text: m.Text}}})
		}
	}
	return items
}

func (c *Client) respUsage(u *respUsage) rimeno.Usage {
	if u == nil {
		return rimeno.Usage{}
	}
	out := rimeno.Usage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
		CostUSD:      c.cost(wireUsage{PromptTokens: u.InputTokens, CompletionTokens: u.OutputTokens}),
	}
	if u.InputTokensDetails != nil {
		out.CachedInputTokens = u.InputTokensDetails.CachedTokens
	}
	if u.OutputTokensDetails != nil {
		out.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens
	}
	return out
}

func respStop(msg rimeno.Message) rimeno.StopReason {
	if len(msg.ToolCalls) > 0 {
		return rimeno.StopReasonToolCalls
	}
	return rimeno.StopReasonStop
}
