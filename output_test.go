package rimeno_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

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
