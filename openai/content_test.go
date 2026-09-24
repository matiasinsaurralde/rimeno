package openai_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

// An assistant tool-call turn has empty text; the serialized message must still
// carry a "content" key (empty string), not omit it. Omitting it makes some
// OpenRouter upstreams normalize the missing field into a content part with
// text=undefined and reject the whole request with an http 400.
func TestToolCallTurn_KeepsContentKey(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = decodeReq(t, r)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	_, err := c.Generate(context.Background(), &rimeno.Request{
		Messages: []rimeno.Message{
			rimeno.UserMessage("hi"),
			{
				Role: rimeno.RoleAssistant,
				ToolCalls: []rimeno.ToolCall{
					{ID: "call_1", Name: "search", Arguments: json.RawMessage(`{"q":"x"}`)},
				},
			},
			{Role: rimeno.RoleTool, ToolCallID: "call_1", Name: "search", Text: "result"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	msgs, ok := got["messages"].([]any)
	if !ok {
		t.Fatalf("request had no messages array: %v", got)
	}
	// The assistant tool-call turn is the message carrying tool_calls.
	var assistant map[string]any
	for _, m := range msgs {
		mm, _ := m.(map[string]any)
		if _, hasTools := mm["tool_calls"]; hasTools {
			assistant = mm
			break
		}
	}
	if assistant == nil {
		t.Fatalf("no assistant tool-call message found in %v", msgs)
	}
	content, present := assistant["content"]
	if !present {
		t.Fatalf("assistant tool-call message omitted the content key: %v", assistant)
	}
	if content != "" {
		t.Errorf("content = %v, want empty string", content)
	}
}
