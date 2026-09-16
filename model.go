package rimeno

import (
	"context"
	"encoding/json"
)

// StopReason explains why the model stopped generating.
type StopReason string

const (
	StopReasonStop          StopReason = "stop"           // natural completion
	StopReasonToolCalls     StopReason = "tool_calls"     // model requested tools
	StopReasonLength        StopReason = "length"         // hit max output tokens
	StopReasonContentFilter StopReason = "content_filter" // provider filtered output
	StopReasonBudget        StopReason = "budget"         // rimeno budget exceeded
	StopReasonMaxSteps      StopReason = "max_steps"      // loop step cap reached
	StopReasonCanceled      StopReason = "canceled"       // context canceled
	StopReasonError         StopReason = "error"          // provider/tool error
)

// Usage accounts tokens and cost for one or more model calls.
type Usage struct {
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	TotalTokens  int     `json:"total_tokens"`
	CostUSD      float64 `json:"cost_usd"`
	// CachedInputTokens is the portion of InputTokens served from the provider's
	// prompt cache (when reported). It is a subset of InputTokens, surfaced so cost
	// estimates can reflect cache discounts.
	CachedInputTokens int `json:"cached_input_tokens,omitempty"`
	// ReasoningTokens is the portion of OutputTokens spent on a reasoning model's
	// internal chain-of-thought (when reported). It is a subset of OutputTokens.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// Add accumulates another Usage into u.
func (u *Usage) Add(o Usage) {
	u.InputTokens += o.InputTokens
	u.OutputTokens += o.OutputTokens
	u.TotalTokens += o.TotalTokens
	u.CostUSD += o.CostUSD
	u.CachedInputTokens += o.CachedInputTokens
	u.ReasoningTokens += o.ReasoningTokens
}

// ToolChoice controls whether/which tools the model may call.
type ToolChoice string

const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceRequired ToolChoice = "required"
)

// ResponseFormat requests a structured (JSON Schema) response.
type ResponseFormat struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

// ToolDef is the provider-facing declaration of a tool (name, description, and
// JSON Schema for its parameters).
type ToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// Request is a provider-agnostic model call.
type Request struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Tools          []ToolDef       `json:"tools,omitempty"`
	ToolChoice     ToolChoice      `json:"tool_choice,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
	// ReasoningEffort, for reasoning models, hints how much to think
	// ("minimal"|"low"|"medium"|"high"). Providers map it to their reasoning control
	// (OpenRouter `reasoning.effort`); ignored by non-reasoning models.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
	// ReasoningMaxTokens caps a reasoning model's thinking budget (OpenRouter
	// `reasoning.max_tokens`). Prevents `max_tokens` starvation where the model
	// spends the whole output budget thinking and returns an empty answer.
	ReasoningMaxTokens int `json:"reasoning_max_tokens,omitempty"`
	// PreviousResponseID threads a prior turn on the Responses API: the provider
	// continues from that response (reusing its server-side reasoning) so only the
	// new input items need be sent. Ignored by the Chat Completions path.
	PreviousResponseID string `json:"previous_response_id,omitempty"`
}

// Response is a provider-agnostic model reply.
type Response struct {
	Message    Message    `json:"message"`
	Usage      Usage      `json:"usage"`
	StopReason StopReason `json:"stop_reason"`
	Model      string     `json:"model"`
	// ID is the provider's response id, when it returns one (e.g. the Responses
	// API). It can be fed back as Request.PreviousResponseID to thread the next turn.
	ID  string          `json:"id,omitempty"`
	Raw json.RawMessage `json:"-"` // optional provider payload for debugging
}

// Model is the minimal provider abstraction: one synchronous completion.
type Model interface {
	// ID reports the model identifier (e.g. "gpt-4o-mini"), used in requests and
	// traces and to allow different models per agent/subagent.
	ID() string
	// Generate performs a single completion for the request.
	Generate(ctx context.Context, req *Request) (*Response, error)
}

// StreamingModel is an optional extension for providers that support streaming.
type StreamingModel interface {
	Model
	Stream(ctx context.Context, req *Request) (Stream, error)
}

// ResponseThreader is an optional Model extension: a provider that can continue a
// prior turn server-side (via Request.PreviousResponseID, e.g. the OpenAI Responses
// API) reports true. When Config.ResponsesThreading is set and the model reports it
// can thread, a Session sends only the new input each turn (with the prior response
// id) instead of resending the whole history, reusing the model's server-side
// reasoning across tool loops. A model that returns false is always sent full
// history, so enabling the option can never corrupt a non-threading provider.
type ResponseThreader interface {
	Model
	ThreadsResponses() bool
}

// supportsResponseThreading reports whether m can continue a turn server-side.
func supportsResponseThreading(m Model) bool {
	t, ok := m.(ResponseThreader)
	return ok && t.ThreadsResponses()
}

// threadingFallbacker is an optional Model capability: given a generate error, report
// whether it means the provider rejected the previous_response_id (as opposed to any
// other failure). A ResponseThreader can advertise threading yet sit in front of a
// backend that does not honor it (e.g. OpenRouter 400s a non-null previous_response_id
// for models without server-side state). When the model classifies such an error, a
// Session drops threading and retries the turn with full history instead of failing.
type threadingFallbacker interface {
	ThreadingUnsupported(err error) bool
}

// threadingUnsupported reports whether err is a provider rejection of
// previous_response_id that rimeno should recover from by resending full history.
func threadingUnsupported(m Model, err error) bool {
	f, ok := m.(threadingFallbacker)
	return ok && f.ThreadingUnsupported(err)
}

// Stream yields incremental output from a streaming completion. Recv returns
// io.EOF when the stream is exhausted; the final chunk carries a non-nil
// [StreamChunk.Final].
type Stream interface {
	Recv() (StreamChunk, error)
	Close() error
}

// StreamChunk is one increment of a streaming completion.
type StreamChunk struct {
	TextDelta      string    // incremental assistant text
	ReasoningDelta string    // incremental reasoning (chain-of-thought) text
	ToolCall       *ToolCall // a (possibly partial) tool call, when present
	Usage          *Usage    // usage, typically on the final chunk
	Final          *Response // set on the terminal chunk
}
