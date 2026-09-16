package rimeno_test

import (
	"context"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

func TestStreaming_EmitsDeltas(t *testing.T) {
	m := rimenotest.NewModel(rimenotest.Turn{Text: "Hello, streaming world"})
	var deltas int
	var full strings.Builder
	agent, err := rimeno.New(rimeno.Config{
		Model:  m,
		Stream: true,
		OnEvent: func(e rimeno.Event) {
			if d, ok := e.(rimeno.TextDeltaEvent); ok {
				deltas++
				full.WriteString(d.Delta)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "Hello, streaming world" {
		t.Fatalf("Text = %q", res.Text)
	}
	if deltas < 2 {
		t.Fatalf("expected multiple deltas, got %d", deltas)
	}
	if full.String() != "Hello, streaming world" {
		t.Fatalf("reassembled deltas = %q", full.String())
	}
}

func TestStreaming_WithToolCalls(t *testing.T) {
	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 2, B: 3})}},
		rimenotest.Turn{Text: "result is 5"},
	)
	add := rimeno.NewTool("add", "add", func(_ context.Context, in addArgs) (int, error) { return in.A + in.B, nil })
	agent, err := rimeno.New(rimeno.Config{Model: m, Stream: true, Tools: []rimeno.Tool{add}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "2+3")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "result is 5" {
		t.Fatalf("Text = %q", res.Text)
	}
	if sum := res.Trace.Summary(); sum.ToolCalls != 1 {
		t.Errorf("tool calls = %d, want 1", sum.ToolCalls)
	}
}

// Non-streaming model with Stream=true must fall back gracefully.
func TestStreaming_FallbackNonStreamingModel(t *testing.T) {
	m := &generateOnlyModel{text: "ok"}
	agent, err := rimeno.New(rimeno.Config{Model: m, Stream: true})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "ok" {
		t.Fatalf("Text = %q", res.Text)
	}
}

// generateOnlyModel implements rimeno.Model but NOT rimeno.StreamingModel.
type generateOnlyModel struct{ text string }

func (g *generateOnlyModel) ID() string { return "generate-only" }
func (g *generateOnlyModel) Generate(_ context.Context, _ *rimeno.Request) (*rimeno.Response, error) {
	return &rimeno.Response{
		Message:    rimeno.Message{Role: rimeno.RoleAssistant, Text: g.text},
		Usage:      rimeno.Usage{TotalTokens: 1},
		StopReason: rimeno.StopReasonStop,
	}, nil
}
