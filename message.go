package rimeno

import "encoding/json"

// Role identifies the author of a [Message].
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message is a single entry in a conversation. It intentionally models only what
// the harness needs across OpenAI-compatible providers: a role, text, optional
// assistant tool-call requests, and the linkage a tool result needs back to its
// call.
type Message struct {
	Role Role `json:"role"`
	// Text is the natural-language content (assistant/user/system) or the tool
	// result payload (tool).
	Text string `json:"text,omitempty"`
	// Reasoning is a reasoning model's chain-of-thought, when the provider surfaces
	// it in a separate channel (OpenRouter `reasoning`, DeepSeek `reasoning_content`).
	// Populated on assistant messages for observability only — it is decode-only and
	// is NEVER sent back to the provider as input (see openai.toWireMessages).
	Reasoning string `json:"reasoning,omitempty"`
	// ToolCalls is set on assistant messages that request tool invocations.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID links a RoleTool message back to the ToolCall it answers.
	ToolCallID string `json:"tool_call_id,omitempty"`
	// Name is the tool name on RoleTool messages (optional, aids some providers).
	Name string `json:"name,omitempty"`
}

// ToolCall is a request emitted by the model to invoke a tool.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// SystemMessage builds a system-role message.
func SystemMessage(text string) Message { return Message{Role: RoleSystem, Text: text} }

// UserMessage builds a user-role message.
func UserMessage(text string) Message { return Message{Role: RoleUser, Text: text} }

// AssistantMessage builds an assistant-role message.
func AssistantMessage(text string) Message { return Message{Role: RoleAssistant, Text: text} }

func toolResultMessage(callID, name, content string) Message {
	return Message{Role: RoleTool, ToolCallID: callID, Name: name, Text: content}
}
