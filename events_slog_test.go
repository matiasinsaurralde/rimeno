package rimeno_test

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

func TestSlogSink(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))

	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 1, B: 2})}},
		rimenotest.Turn{Text: "3"},
	)
	add := rimeno.NewTool("add", "add", func(_ context.Context, in addArgs) (int, error) { return in.A + in.B, nil })
	agent, err := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{add}, OnEvent: rimeno.SlogSink(logger)})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(context.Background(), "1+2"); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{"rimeno.model_response", "rimeno.tool_call", "rimeno.run_finished"} {
		if !strings.Contains(out, want) {
			t.Errorf("slog output missing %q\n%s", want, out)
		}
	}
}
