package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

// The Responses-API path posts to /responses, maps chat history to input items,
// and parses output items (text + reasoning + tool calls + usage).
func TestResponses_RoundTrip(t *testing.T) {
	var path string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		_, _ = w.Write([]byte(`{
		  "model":"m",
		  "output":[
		    {"type":"reasoning","summary":[{"type":"summary_text","text":"thinking hard"}]},
		    {"type":"message","role":"assistant","content":[{"type":"output_text","text":"the answer is 42"}]}
		  ],
		  "usage":{"input_tokens":100,"output_tokens":20,"total_tokens":120,
		           "input_tokens_details":{"cached_tokens":80},
		           "output_tokens_details":{"reasoning_tokens":12}}
		}`))
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"),
		openai.WithResponsesAPI(),
		openai.WithPricing("m", openai.Price{InputPerMillion: 1, OutputPerMillion: 10}))
	res, err := c.Generate(context.Background(), &rimeno.Request{
		Messages:        []rimeno.Message{rimeno.SystemMessage("be terse"), rimeno.UserMessage("6*7?")},
		ReasoningEffort: "high",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(path, "/responses") {
		t.Errorf("posted to %q, want /responses", path)
	}
	// request used Responses input items + reasoning control
	if _, ok := body["input"].([]any); !ok {
		t.Errorf("request had no input array: %v", body)
	}
	if r, ok := body["reasoning"].(map[string]any); !ok || r["effort"] != "high" {
		t.Errorf("reasoning control not sent: %v", body["reasoning"])
	}
	// response parsing
	if res.Message.Text != "the answer is 42" {
		t.Errorf("Text = %q", res.Message.Text)
	}
	if res.Message.Reasoning != "thinking hard" {
		t.Errorf("Reasoning = %q", res.Message.Reasoning)
	}
	if res.Usage.InputTokens != 100 || res.Usage.OutputTokens != 20 {
		t.Errorf("tokens = %+v", res.Usage)
	}
	if res.Usage.CachedInputTokens != 80 || res.Usage.ReasoningTokens != 12 {
		t.Errorf("cached/reasoning = %d/%d, want 80/12", res.Usage.CachedInputTokens, res.Usage.ReasoningTokens)
	}
	if want := 100.0/1e6*1 + 20.0/1e6*10; res.Usage.CostUSD < want-1e-9 || res.Usage.CostUSD > want+1e-9 {
		t.Errorf("cost = %v, want ~%v", res.Usage.CostUSD, want)
	}
}

// A function_call output item becomes a rimeno tool call; the history's tool call +
// result map to Responses input items so the agent loop works over Responses.
func TestResponses_ToolCallsBothWays(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		_, _ = w.Write([]byte(`{"model":"m","output":[
		  {"type":"function_call","call_id":"call_1","name":"add","arguments":"{\"a\":1,\"b\":2}"}
		]}`))
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"), openai.WithResponsesAPI())
	res, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{
		rimeno.UserMessage("add 1 and 2"),
		{Role: rimeno.RoleAssistant, ToolCalls: []rimeno.ToolCall{{ID: "call_0", Name: "add", Arguments: json.RawMessage(`{}`)}}},
		{Role: rimeno.RoleTool, ToolCallID: "call_0", Text: "3"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	// output parsed into a tool call
	if len(res.Message.ToolCalls) != 1 || res.Message.ToolCalls[0].Name != "add" || res.Message.ToolCalls[0].ID != "call_1" {
		t.Fatalf("tool calls = %+v", res.Message.ToolCalls)
	}
	if res.StopReason != rimeno.StopReasonToolCalls {
		t.Errorf("stop = %v, want tool_calls", res.StopReason)
	}
	// input mapped the prior tool call + result into Responses items
	in, _ := body["input"].([]any)
	types := map[string]int{}
	for _, it := range in {
		if m, ok := it.(map[string]any); ok {
			types[m["type"].(string)]++
		}
	}
	if types["function_call"] != 1 || types["function_call_output"] != 1 {
		t.Errorf("input item types = %v, want one function_call and one function_call_output", types)
	}
}
