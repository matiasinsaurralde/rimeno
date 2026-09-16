package rpc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

// TestIntegration_RPCThroughOpenAIProvider drives the whole stack end-to-end:
// a JSON-RPC session/prompt → rimeno run loop → openai provider → HTTP, with a
// mock OpenAI-compatible server. It exercises a full tool-calling round trip
// (the model asks for a tool on the first HTTP call, answers on the second) that
// no single-package test covers.
func TestIntegration_RPCThroughOpenAIProvider(t *testing.T) {
	var httpCalls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpCalls++
		w.Header().Set("Content-Type", "application/json")
		if httpCalls == 1 {
			// First turn: request the "add" tool.
			_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"add","arguments":"{\"a\":21,\"b\":21}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
			return
		}
		// Second turn: the tool result is now in the request; answer.
		body := decodeReq(t, r)
		if !hasToolResult(body) {
			t.Errorf("second request missing tool result message: %v", body["messages"])
		}
		_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"The sum is 42."},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":6,"total_tokens":26}}`)
	}))
	defer srv.Close()

	factory := func(_ context.Context, _ NewSessionParams) (rimeno.Config, error) {
		model := openai.New(
			openai.WithBaseURL(srv.URL),
			openai.WithModel("test-model"),
			openai.WithAPIKey("sk-test"),
		)
		add := rimeno.NewTool("add", "add two ints",
			func(_ context.Context, in struct {
				A int `json:"a"`
				B int `json:"b"`
			}) (int, error) {
				return in.A + in.B, nil
			})
		return rimeno.Config{Model: model, Tools: []rimeno.Tool{add}}, nil
	}
	tc := serve(t, factory)

	resp, _ := tc.call("session/new", nil)
	var ns newSessionResult
	if err := json.Unmarshal(resp.Result, &ns); err != nil {
		t.Fatalf("session/new: %v", err)
	}

	resp, notes := tc.call("session/prompt", PromptParams{SessionID: ns.SessionID, Input: "what is 21+21?"})
	if resp.Error != nil {
		t.Fatalf("session/prompt error: %+v", resp.Error)
	}
	var pr promptResult
	if err := json.Unmarshal(resp.Result, &pr); err != nil {
		t.Fatalf("decode prompt: %v", err)
	}
	if pr.Text != "The sum is 42." {
		t.Errorf("text = %q, want %q", pr.Text, "The sum is 42.")
	}
	if pr.Summary.ToolCalls != 1 {
		t.Errorf("tool_calls = %d, want 1", pr.Summary.ToolCalls)
	}
	if pr.Usage.TotalTokens != 41 { // 15 + 26
		t.Errorf("total tokens = %d, want 41", pr.Usage.TotalTokens)
	}
	if httpCalls != 2 {
		t.Errorf("HTTP calls = %d, want 2", httpCalls)
	}

	// The tool round-trip should have streamed a tool_call_finished notification.
	var sawToolFinish bool
	for _, n := range notes {
		if n.Event["kind"] == "tool_call_finished" && n.Event["tool"] == "add" {
			sawToolFinish = true
		}
	}
	if !sawToolFinish {
		t.Errorf("missing tool_call_finished notification for add")
	}
}

func decodeReq(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	return m
}

// hasToolResult reports whether the messages array contains a tool-role message.
func hasToolResult(body map[string]any) bool {
	msgs, ok := body["messages"].([]any)
	if !ok {
		return false
	}
	for _, m := range msgs {
		if mm, ok := m.(map[string]any); ok {
			if role, _ := mm["role"].(string); strings.EqualFold(role, "tool") {
				return true
			}
		}
	}
	return false
}
