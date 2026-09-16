package openai

import "encoding/json"

// These types mirror the OpenAI Chat Completions wire format. They are internal
// to the package; rimeno callers use the provider-agnostic rimeno.Request/Response.

type chatRequest struct {
	Model          string              `json:"model"`
	Messages       []wireMessage       `json:"messages"`
	Tools          []wireTool          `json:"tools,omitempty"`
	ToolChoice     string              `json:"tool_choice,omitempty"`
	ResponseFormat *wireResponseFormat `json:"response_format,omitempty"`
	Temperature    *float64            `json:"temperature,omitempty"`
	MaxTokens      int                 `json:"max_tokens,omitempty"`
	Stop           []string            `json:"stop,omitempty"`
	Stream         bool                `json:"stream,omitempty"`
	StreamOptions  *streamOptions      `json:"stream_options,omitempty"`
	// Reasoning is OpenRouter's unified reasoning control (effort / max_tokens /
	// exclude). Non-reasoning models and providers that don't recognize it ignore it.
	Reasoning *wireReasoning `json:"reasoning,omitempty"`
}

type wireReasoning struct {
	Effort    string `json:"effort,omitempty"`
	MaxTokens int    `json:"max_tokens,omitempty"`
	Exclude   bool   `json:"exclude,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content,omitempty"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
	// Reasoning holds a reasoning model's output when the provider returns it in a
	// separate channel. Some reasoning models (and OpenRouter's normalization) put
	// the final answer here with content:null; "reasoning" is OpenRouter's spelling,
	// "reasoning_content" is used by other OpenAI-compatible providers. Both are
	// decode-only here (omitempty keeps them out of request bodies).
	Reasoning        string `json:"reasoning,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type wireToolCall struct {
	Index    int              `json:"index,omitempty"` // present in streaming deltas
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function wireFunctionCall `json:"function"`
}

type wireFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"` // JSON encoded as a string
}

type wireTool struct {
	Type     string          `json:"type"`
	Function wireFunctionDef `json:"function"`
}

type wireFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type wireResponseFormat struct {
	Type       string         `json:"type"`
	JSONSchema wireJSONSchema `json:"json_schema"`
}

type wireJSONSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Model   string       `json:"model"`
	Choices []wireChoice `json:"choices"`
	Usage   *wireUsage   `json:"usage"`
}

type wireChoice struct {
	Index        int         `json:"index"`
	Message      wireMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type wireUsage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *promptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *completionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type completionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type streamResponse struct {
	Choices []struct {
		Index        int         `json:"index"`
		Delta        wireMessage `json:"delta"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
}
