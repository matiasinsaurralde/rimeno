package rimeno_test

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

func TestLoadTrace_RoundTrip(t *testing.T) {
	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 1, B: 1})}},
		rimenotest.Turn{Text: "2"},
	)
	add := rimeno.NewTool("add", "add", func(_ context.Context, in addArgs) (int, error) { return in.A + in.B, nil })
	agent, _ := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{add}})
	res, err := agent.Run(context.Background(), "1+1")
	if err != nil {
		t.Fatal(err)
	}
	b, err := res.Trace.JSON()
	if err != nil {
		t.Fatal(err)
	}
	tr2, err := rimeno.LoadTrace(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	s1, s2 := res.Trace.Summary(), tr2.Summary()
	if s1.ModelCalls != s2.ModelCalls || s1.ToolCalls != s2.ToolCalls || s1.Usage.TotalTokens != s2.Usage.TotalTokens {
		t.Fatalf("round-trip mismatch:\n%+v\n%+v", s1, s2)
	}
}

func TestTrace_WriteTo(t *testing.T) {
	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 1, B: 1})}},
		rimenotest.Turn{Text: "2"},
	)
	add := rimeno.NewTool("add", "add", func(_ context.Context, in addArgs) (int, error) { return in.A + in.B, nil })
	agent, _ := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{add}})
	res, err := agent.Run(context.Background(), "1+1")
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	n, err := res.Trace.WriteTo(&buf)
	if err != nil || n == 0 {
		t.Fatalf("WriteTo: n=%d err=%v", n, err)
	}
	// run + 2 steps + 2 model + 1 tool = 6 span lines.
	lines := strings.Count(strings.TrimSpace(buf.String()), "\n") + 1
	if lines < 6 {
		t.Errorf("expected >=6 JSONL span lines, got %d:\n%s", lines, buf.String())
	}
	if !strings.Contains(buf.String(), `"kind":"tool"`) {
		t.Errorf("expected a tool span in JSONL output")
	}
}
