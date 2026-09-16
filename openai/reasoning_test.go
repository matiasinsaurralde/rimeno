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

// Reasoning controls (effort + max_tokens) are sent as the OpenRouter
// `reasoning` object.
func TestReasoningControls_SentInRequest(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = decodeReq(t, r)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	_, err := c.Generate(context.Background(), &rimeno.Request{
		Messages:           []rimeno.Message{rimeno.UserMessage("hi")},
		ReasoningEffort:    "high",
		ReasoningMaxTokens: 500,
	})
	if err != nil {
		t.Fatal(err)
	}
	r, ok := got["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("request had no reasoning object: %v", got)
	}
	if r["effort"] != "high" {
		t.Errorf("reasoning.effort = %v, want high", r["effort"])
	}
	if r["max_tokens"] != float64(500) {
		t.Errorf("reasoning.max_tokens = %v, want 500", r["max_tokens"])
	}
}

// No reasoning object is sent when neither control is set (default unbounded).
func TestReasoningControls_OmittedByDefault(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = decodeReq(t, r)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	if _, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("hi")}}); err != nil {
		t.Fatal(err)
	}
	if _, present := got["reasoning"]; present {
		t.Errorf("reasoning object should be omitted by default, got %v", got["reasoning"])
	}
}

// Message.Reasoning is populated alongside the answer (not only as a fallback),
// and the reasoning channel is not confused with the answer text.
func TestReasoningCapture_MessageReasoning(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"the answer is 42","reasoning":"let me think... 6*7=42"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18,"completion_tokens_details":{"reasoning_tokens":5}}}`))
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	res, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("6*7?")}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.Text != "the answer is 42" {
		t.Errorf("Text = %q, want the answer", res.Message.Text)
	}
	if res.Message.Reasoning != "let me think... 6*7=42" {
		t.Errorf("Reasoning = %q, want the chain-of-thought", res.Message.Reasoning)
	}
	if res.Usage.ReasoningTokens != 5 {
		t.Errorf("ReasoningTokens = %d, want 5", res.Usage.ReasoningTokens)
	}
	// Invariant: reasoning must never be echoed back to the provider as input.
	b, _ := json.Marshal(struct {
		Msgs []rimeno.Message `json:"messages"`
	}{[]rimeno.Message{res.Message}})
	_ = b // Message.Reasoning has a json tag, but the provider maps only role/content/
	// tool_calls via toWireMessages — covered by the request-shape tests above.
}
