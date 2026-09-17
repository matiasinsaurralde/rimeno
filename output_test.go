package rimeno_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// decisionSchema is a hand-written schema (as OutputFromSchema's audience would
// have) rather than one derived from a Go type.
var decisionSchema = json.RawMessage(`{
	"type": "object",
	"properties": {
		"verdict": {"type": "string", "enum": ["accept", "reject"]},
		"confidence": {"type": "integer", "minimum": 0, "maximum": 100}
	},
	"required": ["verdict", "confidence"],
	"additionalProperties": false
}`)

func TestOutputFromSchema_Valid(t *testing.T) {
	m := rimenotest.NewModel(rimenotest.Turn{Text: `{"verdict":"accept","confidence":87}`})
	agent, err := rimeno.New(rimeno.Config{
		Model:  m,
		Output: rimeno.OutputFromSchema("decision", decisionSchema),
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "decide")
	if err != nil {
		t.Fatal(err)
	}
	// Default decoder is a raw-JSON passthrough; the caller decodes it.
	raw, ok := res.Output.(json.RawMessage)
	if !ok {
		t.Fatalf("Output type = %T, want json.RawMessage", res.Output)
	}
	var got struct {
		Verdict    string `json:"verdict"`
		Confidence int    `json:"confidence"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode raw output: %v", err)
	}
	if got.Verdict != "accept" || got.Confidence != 87 {
		t.Fatalf("decoded = %+v", got)
	}
	// The hand-written schema must reach the provider as response_format, strict
	// by default (matching OutputOf).
	rf := m.Requests[0].ResponseFormat
	if rf == nil {
		t.Fatal("expected ResponseFormat to be set")
	}
	if rf.Name != "decision" || !rf.Strict {
		t.Fatalf("ResponseFormat = %+v, want name=decision strict=true", rf)
	}
}

func TestOutputFromSchema_InvalidRejected(t *testing.T) {
	// confidence above maximum → local validation must fail.
	m := rimenotest.NewModel(rimenotest.Turn{Text: `{"verdict":"accept","confidence":150}`})
	agent, _ := rimeno.New(rimeno.Config{
		Model:  m,
		Output: rimeno.OutputFromSchema("decision", decisionSchema),
	})
	_, err := agent.Run(context.Background(), "decide")
	var ve *rimeno.OutputValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected OutputValidationError, got %v", err)
	}
}

func TestOutputFromSchema_RepairRetry(t *testing.T) {
	// Composes with OutputRepairAttempts: one invalid then one valid.
	m := rimenotest.NewModel(
		rimenotest.Turn{Text: `{"verdict":"maybe","confidence":50}`}, // invalid: enum
		rimenotest.Turn{Text: `{"verdict":"reject","confidence":40}`}, // valid after repair
	)
	agent, _ := rimeno.New(rimeno.Config{
		Model:                m,
		Output:               rimeno.OutputFromSchema("decision", decisionSchema),
		OutputRepairAttempts: 2,
	})
	res, err := agent.Run(context.Background(), "decide")
	if err != nil {
		t.Fatalf("expected repair to succeed, got %v", err)
	}
	if len(m.Requests) != 2 {
		t.Fatalf("expected exactly one repair round (2 model calls), got %d", len(m.Requests))
	}
	raw := res.Output.(json.RawMessage)
	if !strings.Contains(string(raw), "reject") {
		t.Fatalf("output = %s, want the repaired value", raw)
	}
}

func TestOutputFromSchema_Lax(t *testing.T) {
	m := rimenotest.NewModel(rimenotest.Turn{Text: `{"verdict":"accept","confidence":10}`})
	agent, _ := rimeno.New(rimeno.Config{
		Model:  m,
		Output: rimeno.OutputFromSchema("decision", decisionSchema, rimeno.Lax()),
	})
	if _, err := agent.Run(context.Background(), "decide"); err != nil {
		t.Fatal(err)
	}
	if rf := m.Requests[0].ResponseFormat; rf == nil || rf.Strict {
		t.Fatalf("Lax() should set Strict=false on response_format, got %+v", rf)
	}
}

func TestOutputFromSchema_DecodeInto(t *testing.T) {
	type decision struct {
		Verdict    string `json:"verdict"`
		Confidence int    `json:"confidence"`
	}
	decode := func(b []byte) (any, error) {
		var d decision
		if err := json.Unmarshal(b, &d); err != nil {
			return nil, err
		}
		return &d, nil
	}
	m := rimenotest.NewModel(rimenotest.Turn{Text: `{"verdict":"reject","confidence":3}`})
	agent, _ := rimeno.New(rimeno.Config{
		Model:  m,
		Output: rimeno.OutputFromSchema("decision", decisionSchema, rimeno.DecodeInto(decode)),
	})
	res, err := agent.Run(context.Background(), "decide")
	if err != nil {
		t.Fatal(err)
	}
	d, ok := res.Output.(*decision)
	if !ok {
		t.Fatalf("Output type = %T, want *decision", res.Output)
	}
	if d.Verdict != "reject" || d.Confidence != 3 {
		t.Fatalf("decoded = %+v", d)
	}
}

func TestOutputFromSchema_EmptySchema(t *testing.T) {
	// A nil/empty schema must produce a clear validation error, not a panic.
	m := rimenotest.NewModel(rimenotest.Turn{Text: `{"anything":true}`})
	agent, _ := rimeno.New(rimeno.Config{
		Model:  m,
		Output: rimeno.OutputFromSchema("empty", nil),
	})
	_, err := agent.Run(context.Background(), "go")
	var ve *rimeno.OutputValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected OutputValidationError for empty schema, got %v", err)
	}
	if !strings.Contains(err.Error(), "empty schema") {
		t.Fatalf("error = %v, want it to mention 'empty schema'", err)
	}
}

// TestOutputSpec_NilHooksNoPanic guards the break-free hardening: a hand-built
// OutputSpec that left validate/decode nil surfaces the raw text instead of
// panicking in the run loop.
func TestOutputSpec_NilHooksNoPanic(t *testing.T) {
	m := rimenotest.NewModel(rimenotest.Turn{Text: `{"x":1}`})
	agent, err := rimeno.New(rimeno.Config{
		Model:  m,
		Output: &rimeno.OutputSpec{Name: "bare", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("bare OutputSpec should not error, got %v", err)
	}
	raw, ok := res.Output.(json.RawMessage)
	if !ok {
		t.Fatalf("Output type = %T, want json.RawMessage passthrough", res.Output)
	}
	if strings.TrimSpace(string(raw)) != `{"x":1}` {
		t.Fatalf("output = %s, want raw passthrough", raw)
	}
}

type review struct {
	Score   int      `json:"score" jsonschema:"minimum=0,maximum=10"`
	Reasons []string `json:"reasons"`
}

func TestStructuredOutput_Valid(t *testing.T) {
	m := rimenotest.NewModel(rimenotest.Turn{Text: `{"score":8,"reasons":["clean","tested"]}`})
	agent, err := rimeno.New(rimeno.Config{Model: m, Output: rimeno.OutputOf[review]()})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "review")
	if err != nil {
		t.Fatal(err)
	}
	out, ok := res.Output.(*review)
	if !ok {
		t.Fatalf("Output type = %T, want *review", res.Output)
	}
	if out.Score != 8 || len(out.Reasons) != 2 {
		t.Fatalf("decoded = %+v", out)
	}
	// the request must carry a response_format built from the schema.
	if m.Requests[0].ResponseFormat == nil {
		t.Fatal("expected ResponseFormat to be set")
	}
}

func TestStructuredOutput_InvalidRejected(t *testing.T) {
	// score above maximum → validation must fail.
	m := rimenotest.NewModel(rimenotest.Turn{Text: `{"score":99,"reasons":[]}`})
	agent, _ := rimeno.New(rimeno.Config{Model: m, Output: rimeno.OutputOf[review]()})
	_, err := agent.Run(context.Background(), "review")
	var ve *rimeno.OutputValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected OutputValidationError, got %v", err)
	}
}

func TestStructuredOutput_RepairRetry(t *testing.T) {
	m := rimenotest.NewModel(
		rimenotest.Turn{Text: `{"score":99,"reasons":[]}`},    // invalid: score > 10
		rimenotest.Turn{Text: `{"score":7,"reasons":["ok"]}`}, // valid after repair
	)
	agent, _ := rimeno.New(rimeno.Config{Model: m, Output: rimeno.OutputOf[review]()})
	res, err := agent.Run(context.Background(), "review")
	if err != nil {
		t.Fatalf("expected repair to succeed, got %v", err)
	}
	r := res.Output.(*review)
	if r.Score != 7 {
		t.Fatalf("score = %d, want 7", r.Score)
	}
	if len(m.Requests) < 2 {
		t.Fatalf("expected a repair round (>=2 model calls), got %d", len(m.Requests))
	}
	// the repair round must include a corrective user message.
	repairFound := false
	for _, msg := range m.Requests[1].Messages {
		if msg.Role == rimeno.RoleUser && strings.Contains(msg.Text, "did not satisfy") {
			repairFound = true
		}
	}
	if !repairFound {
		t.Error("repair prompt not sent to model")
	}
}

func TestStructuredOutput_RepairExhausted(t *testing.T) {
	m := rimenotest.NewModel(
		rimenotest.Turn{Text: `{"score":99,"reasons":[]}`},
		rimenotest.Turn{Text: `{"score":98,"reasons":[]}`}, // still invalid
	)
	agent, _ := rimeno.New(rimeno.Config{Model: m, Output: rimeno.OutputOf[review](), OutputRepairAttempts: 1})
	_, err := agent.Run(context.Background(), "review")
	var ve *rimeno.OutputValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected validation error after repair exhausted, got %v", err)
	}
}

func TestStructuredOutput_MalformedRejected(t *testing.T) {
	m := rimenotest.NewModel(rimenotest.Turn{Text: `not json`})
	agent, _ := rimeno.New(rimeno.Config{Model: m, Output: rimeno.OutputOf[review]()})
	_, err := agent.Run(context.Background(), "review")
	if err == nil {
		t.Fatal("expected error for malformed JSON output")
	}
}
