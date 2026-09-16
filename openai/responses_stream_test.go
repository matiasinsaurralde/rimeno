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

// The Responses SSE path yields text + reasoning deltas and a final assembled
// response with usage.
func TestResponses_Stream(t *testing.T) {
	sse := strings.Join([]string{
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","delta":"Hello "}`,
		``,
		`data: {"type":"response.output_text.delta","delta":"world"}`,
		``,
		`data: {"type":"response.reasoning_summary_text.delta","delta":"thinking"}`,
		``,
		`data: {"type":"response.completed","response":{"id":"resp_1","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world"}]}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		if body["stream"] != true {
			t.Errorf("stream flag not set: %v", body["stream"])
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"), openai.WithResponsesAPI())
	st, err := c.Stream(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = st.Close() }()
	var text, reasoning string
	var final *rimeno.Response
	for {
		ch, err := st.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		text += ch.TextDelta
		reasoning += ch.ReasoningDelta
		if ch.Final != nil {
			final = ch.Final
		}
	}
	if text != "Hello world" {
		t.Errorf("streamed text = %q", text)
	}
	if reasoning != "thinking" {
		t.Errorf("streamed reasoning = %q", reasoning)
	}
	if final == nil || final.Message.Text != "Hello world" || final.ID != "resp_1" {
		t.Fatalf("final = %+v", final)
	}
	if final.Usage.TotalTokens != 7 {
		t.Errorf("usage total = %d, want 7", final.Usage.TotalTokens)
	}
}

// previous_response_id is sent and the response id is captured, so a caller can
// thread turns over the Responses API.
func TestResponses_PreviousResponseID(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		_, _ = w.Write([]byte(`{"id":"resp_2","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`))
	}))
	defer srv.Close()

	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"), openai.WithResponsesAPI())
	res, err := c.Generate(context.Background(), &rimeno.Request{
		Messages:           []rimeno.Message{rimeno.UserMessage("continue")},
		PreviousResponseID: "resp_1",
	})
	if err != nil {
		t.Fatal(err)
	}
	if body["previous_response_id"] != "resp_1" {
		t.Errorf("previous_response_id = %v, want resp_1", body["previous_response_id"])
	}
	if res.ID != "resp_2" {
		t.Errorf("captured response id = %q, want resp_2", res.ID)
	}
}
