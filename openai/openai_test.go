package openai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

func decodeReq(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	b, _ := io.ReadAll(r.Body)
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	return m
}

func TestGenerate_TextAndUsageAndMapping(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeReq(t, r)
		// request mapping assertions
		if body["model"] != "test-model" {
			t.Errorf("model = %v", body["model"])
		}
		if r.Header.Get("Authorization") != "Bearer sk-test" {
			t.Errorf("auth header = %q", r.Header.Get("Authorization"))
		}
		if _, ok := body["tools"]; !ok {
			t.Errorf("expected tools in request")
		}
		if _, ok := body["response_format"]; !ok {
			t.Errorf("expected response_format in request")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000000,"completion_tokens":1000000,"total_tokens":2000000}}`)
	}))
	defer srv.Close()

	c := openai.New(
		openai.WithBaseURL(srv.URL),
		openai.WithAPIKey("sk-test"),
		openai.WithModel("test-model"),
		openai.WithPricing("test-model", openai.Price{InputPerMillion: 1, OutputPerMillion: 2}),
	)
	resp, err := c.Generate(context.Background(), &rimeno.Request{
		Model:          "test-model",
		Messages:       []rimeno.Message{rimeno.SystemMessage("sys"), rimeno.UserMessage("hi")},
		Tools:          []rimeno.ToolDef{{Name: "t", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
		ResponseFormat: &rimeno.ResponseFormat{Name: "out", Schema: json.RawMessage(`{"type":"object"}`), Strict: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Text != "hi there" {
		t.Errorf("text = %q", resp.Message.Text)
	}
	if resp.StopReason != rimeno.StopReasonStop {
		t.Errorf("stop = %q", resp.StopReason)
	}
	if resp.Usage.TotalTokens != 2000000 {
		t.Errorf("total tokens = %d", resp.Usage.TotalTokens)
	}
	if resp.Usage.CostUSD != 3.0 { // 1M in * $1 + 1M out * $2
		t.Errorf("cost = %v, want 3.0", resp.Usage.CostUSD)
	}
}

func TestGenerate_MaxTokensAndTemperature(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeReq(t, r)
		if mt, ok := body["max_tokens"]; !ok || mt.(float64) != 256 {
			t.Errorf("max_tokens = %v (ok=%v), want 256", body["max_tokens"], ok)
		}
		if tmp, ok := body["temperature"]; !ok || tmp.(float64) != 0.2 {
			t.Errorf("temperature = %v (ok=%v), want 0.2", body["temperature"], ok)
		}
		_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	temp := 0.2
	resp, err := c.Generate(context.Background(), &rimeno.Request{
		Messages:    []rimeno.Message{rimeno.UserMessage("hi")},
		MaxTokens:   256,
		Temperature: &temp,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != rimeno.StopReasonLength {
		t.Errorf("stop = %q, want length", resp.StopReason)
	}
}

func TestGenerate_ToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"add","arguments":"{\"a\":1,\"b\":2}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}`)
	}))
	defer srv.Close()
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	resp, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("go")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.StopReason != rimeno.StopReasonToolCalls {
		t.Fatalf("stop = %q", resp.StopReason)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.Name != "add" || string(tc.Arguments) != `{"a":1,"b":2}` {
		t.Errorf("tool call = %+v", tc)
	}
}

func TestGenerate_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"error":{"message":"bad model","type":"invalid_request_error"}}`)
	}))
	defer srv.Close()
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"), openai.WithMaxRetries(0))
	_, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}})
	var apiErr *openai.APIError
	if err == nil || !strings.Contains(err.Error(), "bad model") {
		t.Fatalf("expected API error, got %v", err)
	}
	if !asAPIError(err, &apiErr) || apiErr.StatusCode != 400 {
		t.Fatalf("expected 400 APIError, got %v", err)
	}
}

func TestStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		frames := []string{
			`{"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}`,
			`{"choices":[{"index":0,"delta":{"content":"lo"}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		}
		for _, f := range frames {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", f)
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	st, err := c.Stream(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = st.Close() }()

	var text strings.Builder
	var final *rimeno.Response
	for {
		chunk, err := st.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		text.WriteString(chunk.TextDelta)
		if chunk.Final != nil {
			final = chunk.Final
		}
	}
	if text.String() != "Hello" {
		t.Errorf("streamed text = %q, want Hello", text.String())
	}
	if final == nil || final.Message.Text != "Hello" {
		t.Fatalf("final = %+v", final)
	}
	if final.Usage.TotalTokens != 5 {
		t.Errorf("final usage = %+v", final.Usage)
	}
}

// TestEndToEnd drives the whole rimeno loop through the real client against a fake
// server that first requests a tool then answers.
func TestEndToEnd_LoopThroughClient(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		_ = decodeReq(t, r)
		if n == 1 {
			_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"add","arguments":"{\"a\":20,\"b\":22}"}}]},"finish_reason":"tool_calls"}],"usage":{"total_tokens":5}}`)
			return
		}
		_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"the sum is 42"},"finish_reason":"stop"}],"usage":{"total_tokens":7}}`)
	}))
	defer srv.Close()

	type addArgs struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	add := rimeno.NewTool("add", "add", func(_ context.Context, in addArgs) (int, error) { return in.A + in.B, nil })
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	agent, err := rimeno.New(rimeno.Config{Model: c, Tools: []rimeno.Tool{add}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "what is 20+22?")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "the sum is 42" {
		t.Fatalf("text = %q", res.Text)
	}
	if s := res.Trace.Summary(); s.ModelCalls != 2 || s.ToolCalls != 1 {
		t.Errorf("summary = %+v", s)
	}
}

// TestGenerate_ReasoningFallback covers reasoning models that return the answer in
// the reasoning channel with content:null (under json_schema). The
// answer must not be lost, but content must win when present and reasoning must be
// ignored on tool-call turns (where it is chain-of-thought, not the answer).
func TestGenerate_ReasoningFallback(t *testing.T) {
	cases := []struct {
		name     string
		message  string // the JSON for choices[0].message
		wantText string
	}{
		{
			name:     "content_null_reasoning_present",
			message:  `{"role":"assistant","content":null,"reasoning":"{\"name\":\"Ada\"}"}`,
			wantText: `{"name":"Ada"}`,
		},
		{
			name:     "reasoning_content_variant",
			message:  `{"role":"assistant","content":"","reasoning_content":"the answer"}`,
			wantText: "the answer",
		},
		{
			name:     "content_wins_when_present",
			message:  `{"role":"assistant","content":"real answer","reasoning":"thinking..."}`,
			wantText: "real answer",
		},
		{
			name:     "reasoning_ignored_on_toolcall_turn",
			message:  `{"role":"assistant","content":null,"reasoning":"cot","tool_calls":[{"id":"c1","type":"function","function":{"name":"add","arguments":"{}"}}]}`,
			wantText: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprintf(w, `{"choices":[{"index":0,"message":%s,"finish_reason":"stop"}],"usage":{"total_tokens":1}}`, tc.message)
			}))
			defer srv.Close()
			c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
			resp, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}})
			if err != nil {
				t.Fatal(err)
			}
			if resp.Message.Text != tc.wantText {
				t.Errorf("text = %q, want %q", resp.Message.Text, tc.wantText)
			}
		})
	}
}

// TestStream_ReasoningFallback covers the streaming path: when only reasoning
// deltas arrive (no content) and there are no tool calls, the assembled final
// message falls back to the reasoning text.
func TestStream_ReasoningFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		frames := []string{
			`{"choices":[{"index":0,"delta":{"role":"assistant","reasoning":"{\"na"}}]}`,
			`{"choices":[{"index":0,"delta":{"reasoning":"me\":\"Ada\"}"}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		}
		for _, f := range frames {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", f)
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	st, err := c.Stream(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = st.Close() }()
	var visibleText strings.Builder
	var final *rimeno.Response
	for {
		chunk, err := st.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		visibleText.WriteString(chunk.TextDelta) // reasoning must NOT surface as visible deltas
		if chunk.Final != nil {
			final = chunk.Final
		}
	}
	if visibleText.String() != "" {
		t.Errorf("reasoning leaked as text deltas: %q", visibleText.String())
	}
	if final == nil || final.Message.Text != `{"name":"Ada"}` {
		t.Fatalf("final text = %+v, want reasoning fallback", final)
	}
}

// TestDo_RetriesTruncatedBody covers the robustness gap the n=10 benchmark exposed:
// a 2xx with an empty/truncated body (provider hiccup under load) must be retried,
// not surfaced as "unexpected end of JSON input".
func TestDo_RetriesTruncatedBody(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.WriteHeader(http.StatusOK) // first attempt: empty 200 body
			return
		}
		_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"recovered"},"finish_reason":"stop"}],"usage":{"total_tokens":3}}`)
	}))
	defer srv.Close()
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m")) // default 2 retries
	resp, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}})
	if err != nil {
		t.Fatalf("expected retry to recover, got %v", err)
	}
	if resp.Message.Text != "recovered" {
		t.Errorf("text = %q, want recovered", resp.Message.Text)
	}
	if atomic.LoadInt32(&n) < 2 {
		t.Errorf("expected a retry (>=2 attempts), got %d", n)
	}
}

// TestDo_EmptyBodyExhaustsRetries verifies a persistently-empty 2xx eventually errors
// (rather than hanging or returning a bad response) after retries.
func TestDo_EmptyBodyExhaustsRetries(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK) // always empty
	}))
	defer srv.Close()
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"), openai.WithMaxRetries(1))
	_, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}})
	if err == nil {
		t.Fatal("expected an error after retries on a persistently empty body")
	}
}

// TestGenerate_CachedAndReasoningTokens verifies token-detail capture.
func TestGenerate_CachedAndReasoningTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1000,"completion_tokens":200,"total_tokens":1200,`+
			`"prompt_tokens_details":{"cached_tokens":880},"completion_tokens_details":{"reasoning_tokens":150}}}`)
	}))
	defer srv.Close()
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"))
	resp, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.CachedInputTokens != 880 {
		t.Errorf("cached input = %d, want 880", resp.Usage.CachedInputTokens)
	}
	if resp.Usage.ReasoningTokens != 150 {
		t.Errorf("reasoning tokens = %d, want 150", resp.Usage.ReasoningTokens)
	}
}

func asAPIError(err error, target **openai.APIError) bool {
	for err != nil {
		if e, ok := err.(*openai.APIError); ok {
			*target = e
			return true
		}
		type unwrapper interface{ Unwrap() error }
		u, ok := err.(unwrapper)
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}
