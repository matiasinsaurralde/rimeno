package otel_test

import (
	"context"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/otel"
	"github.com/matiasinsaurralde/rimeno/rimenotest"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type addArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

// buildTrace runs a fake agent that calls a tool, producing a real rimeno trace.
func buildTrace(t *testing.T) *rimeno.Trace {
	t.Helper()
	model := rimenotest.NewModel(
		rimenotest.Turn{
			ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 2, B: 3})},
			Usage:     &rimeno.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15, CostUSD: 0.001},
		},
		rimenotest.Turn{Text: "5", Usage: &rimeno.Usage{InputTokens: 8, OutputTokens: 2, TotalTokens: 10, CostUSD: 0.0005}},
	).WithID("fake-model")
	add := rimeno.NewTool("add", "add", func(_ context.Context, in addArgs) (int, error) { return in.A + in.B, nil })
	agent, err := rimeno.New(rimeno.Config{Model: model, Tools: []rimeno.Tool{add}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "2+3")
	if err != nil {
		t.Fatal(err)
	}
	return res.Trace
}

func TestExport(t *testing.T) {
	tr := buildTrace(t)

	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	tracer := tp.Tracer("rimeno-test")

	n := otel.Export(context.Background(), tracer, tr)
	if n == 0 {
		t.Fatal("exported 0 spans")
	}

	ended := rec.Ended()
	if len(ended) != n {
		t.Fatalf("recorder saw %d spans, Export reported %d", len(ended), n)
	}

	// Index spans by name and collect the kinds present.
	kinds := map[string]int{}
	var modelSpan, rootSpan, toolSpan sdktrace.ReadOnlySpan
	for _, s := range ended {
		for _, kv := range s.Attributes() {
			if kv.Key == "rimeno.kind" {
				k := kv.Value.AsString()
				kinds[k]++
				switch k {
				case "model":
					modelSpan = s
				case "run":
					rootSpan = s
				case "tool":
					toolSpan = s
				}
			}
		}
	}

	for _, want := range []string{"run", "step", "model", "tool"} {
		if kinds[want] == 0 {
			t.Errorf("no span with rimeno.kind=%q (got %v)", want, kinds)
		}
	}
	if kinds["model"] != 2 {
		t.Errorf("model spans = %d, want 2", kinds["model"])
	}

	// A model span should carry usage attributes.
	if modelSpan == nil {
		t.Fatal("no model span")
	}
	if !hasAttr(modelSpan, "rimeno.usage.total_tokens") {
		t.Errorf("model span missing usage attributes: %v", modelSpan.Attributes())
	}

	// The tool span should be named for the tool and nested under the run tree
	// (i.e. it has a valid parent span id).
	if toolSpan == nil {
		t.Fatal("no tool span")
	}
	if !toolSpan.Parent().SpanID().IsValid() {
		t.Errorf("tool span has no parent — nesting was lost")
	}

	// The root run span should have no parent.
	if rootSpan == nil {
		t.Fatal("no run span")
	}
	if rootSpan.Parent().SpanID().IsValid() {
		t.Errorf("root span unexpectedly has a parent")
	}

	// Timestamps should be preserved (end >= start).
	if modelSpan.EndTime().Before(modelSpan.StartTime()) {
		t.Errorf("model span end %v before start %v", modelSpan.EndTime(), modelSpan.StartTime())
	}
}

func ExampleExport() {
	// After a run, export its trace to any configured OTel TracerProvider.
	// (Here a fake trace + in-memory recorder keep the example self-contained.)
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))

	model := rimenotest.NewModel(rimenotest.Turn{Text: "done"})
	agent, _ := rimeno.New(rimeno.Config{Model: model})
	res, _ := agent.Run(context.Background(), "hi")

	otel.Export(context.Background(), tp.Tracer("rimeno"), res.Trace)
	// The run now appears as OTel spans in your backend.
}

func TestExportNil(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	if n := otel.Export(context.Background(), tp.Tracer("x"), nil); n != 0 {
		t.Errorf("Export(nil) = %d, want 0", n)
	}
}

func hasAttr(s sdktrace.ReadOnlySpan, key attribute.Key) bool {
	for _, kv := range s.Attributes() {
		if kv.Key == key {
			return true
		}
	}
	return false
}
