package rimeno_test

import (
	"context"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// fixedTokenizer returns a constant count per text, so we can assert the agent
// uses the configured tokenizer for compaction estimates.
type fixedTokenizer struct{ per int }

func (f fixedTokenizer) CountTokens(string) int { return f.per }

// recordingCompactor captures the estimate it was given and never compacts.
type recordingCompactor struct{ gotEstimate int }

func (r *recordingCompactor) ShouldCompact(_ []rimeno.Message, est int) bool {
	r.gotEstimate = est
	return false
}
func (r *recordingCompactor) Compact(_ context.Context, m []rimeno.Message) ([]rimeno.Message, error) {
	return m, nil
}

func TestTokenizer_UsedForEstimate(t *testing.T) {
	rc := &recordingCompactor{}
	m := rimenotest.NewModel(rimenotest.Turn{Text: "hi"})
	agent, err := rimeno.New(rimeno.Config{
		Model:     m,
		Compactor: rc,
		Tokenizer: fixedTokenizer{per: 100},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(context.Background(), "one user message"); err != nil {
		t.Fatal(err)
	}
	// history before the first step = [user] → 100 (text) + 3 overhead = 103.
	if rc.gotEstimate != 103 {
		t.Fatalf("estimate = %d, want 103 (tokenizer not used?)", rc.gotEstimate)
	}
}

func TestTokenizer_DefaultHeuristic(t *testing.T) {
	rc := &recordingCompactor{}
	m := rimenotest.NewModel(rimenotest.Turn{Text: "hi"})
	agent, _ := rimeno.New(rimeno.Config{Model: m, Compactor: rc}) // no Tokenizer
	if _, err := agent.Run(context.Background(), "hello"); err != nil {
		t.Fatal(err)
	}
	// default heuristic is non-zero for a non-empty message.
	if rc.gotEstimate <= 0 {
		t.Fatalf("default estimate = %d, want > 0", rc.gotEstimate)
	}
}
