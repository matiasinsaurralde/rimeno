// Package otel exports a completed rimeno [rimeno.Trace] to OpenTelemetry, so agent
// runs show up in your tracing backend (Jaeger, Tempo, Honeycomb, …) alongside
// the rest of a service's spans — with per-step timing, token usage, and cost.
//
// It lives in a separate module so the OpenTelemetry dependency stays out of
// rimeno's stdlib-only core; import it only if you want OTel export.
//
//	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter))
//	res, _ := agent.Run(ctx, "task")
//	otel.Export(ctx, tp.Tracer("rimeno"), res.Trace)
//
// Export maps each rimeno span to an OTel span, preserving the parent/child tree
// and start/end timestamps, and attaches rimeno.* attributes (kind, model, tool,
// stop_reason, usage.*). A span carrying an error is marked with an error status.
package otel

import (
	"context"

	"github.com/matiasinsaurralde/rimeno"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	oteltrace "go.opentelemetry.io/otel/trace"
)

// Export writes a completed rimeno Trace to tracer as a span tree nested under the
// active span in ctx (if any). It returns the number of spans emitted. Call it
// after a run finishes; it reads the trace without mutating it.
func Export(ctx context.Context, tracer oteltrace.Tracer, tr *rimeno.Trace) int {
	if tr == nil || tr.Root == nil {
		return 0
	}
	return exportSpan(ctx, tracer, tr.Root)
}

func exportSpan(ctx context.Context, tracer oteltrace.Tracer, sp *rimeno.Span) int {
	start := sp.Start
	child, os := tracer.Start(ctx, spanName(sp),
		oteltrace.WithTimestamp(start),
		oteltrace.WithAttributes(attrs(sp)...),
	)

	count := 1
	for _, c := range sp.Children {
		count += exportSpan(child, tracer, c)
	}

	if sp.Error != "" {
		os.SetStatus(codes.Error, sp.Error)
	}
	end := sp.End
	if end.IsZero() {
		end = start
	}
	os.End(oteltrace.WithTimestamp(end))
	return count
}

func spanName(sp *rimeno.Span) string {
	if sp.Name != "" {
		return sp.Name
	}
	return string(sp.Kind)
}

func attrs(sp *rimeno.Span) []attribute.KeyValue {
	a := []attribute.KeyValue{attribute.String("rimeno.kind", string(sp.Kind))}
	if sp.Model != "" {
		a = append(a, attribute.String("rimeno.model", sp.Model))
	}
	if sp.Tool != "" {
		a = append(a, attribute.String("rimeno.tool", sp.Tool))
	}
	if sp.StopReason != "" {
		a = append(a, attribute.String("rimeno.stop_reason", string(sp.StopReason)))
	}
	u := sp.Usage
	if u.InputTokens > 0 || u.OutputTokens > 0 || u.TotalTokens > 0 {
		a = append(a,
			attribute.Int("rimeno.usage.input_tokens", u.InputTokens),
			attribute.Int("rimeno.usage.output_tokens", u.OutputTokens),
			attribute.Int("rimeno.usage.total_tokens", u.TotalTokens),
		)
	}
	if u.CostUSD > 0 {
		a = append(a, attribute.Float64("rimeno.usage.cost_usd", u.CostUSD))
	}
	return a
}
