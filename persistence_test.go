package rimeno_test

import (
	"bytes"
	"context"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// TestSessionSnapshotResume verifies a conversation can be snapshotted, restored
// into a new session, and continued — with history and usage carried across.
func TestSessionSnapshotResume(t *testing.T) {
	model := rimenotest.NewModel(
		rimenotest.Turn{Text: "first answer", Usage: &rimeno.Usage{TotalTokens: 10}},
		rimenotest.Turn{Text: "second answer", Usage: &rimeno.Usage{TotalTokens: 20}},
	)
	agent, err := rimeno.New(rimeno.Config{Model: model, Instructions: "be helpful"})
	if err != nil {
		t.Fatal(err)
	}

	// Turn 1 on the original session.
	sess := agent.NewSession()
	if _, err := sess.Send(context.Background(), "hello"); err != nil {
		t.Fatal(err)
	}
	snap := sess.Snapshot()
	if snap.Version != 1 {
		t.Errorf("version = %d, want 1", snap.Version)
	}
	if snap.Usage.TotalTokens != 10 {
		t.Errorf("snapshot usage = %d, want 10", snap.Usage.TotalTokens)
	}
	// history: system + user + assistant = 3 messages.
	if len(snap.Messages) != 3 {
		t.Fatalf("snapshot messages = %d, want 3: %+v", len(snap.Messages), snap.Messages)
	}
	if snap.Messages[0].Role != rimeno.RoleSystem {
		t.Errorf("first message role = %q, want system", snap.Messages[0].Role)
	}

	// Round-trip through JSON.
	var buf bytes.Buffer
	if _, err := snap.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := rimeno.LoadSessionState(&buf)
	if err != nil {
		t.Fatal(err)
	}

	// Resume in a "new process": fresh agent, restored state.
	agent2, _ := rimeno.New(rimeno.Config{Model: model, Instructions: "be helpful"})
	resumed := agent2.ResumeSession(loaded)

	res, err := resumed.Send(context.Background(), "again")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "second answer" {
		t.Errorf("resumed answer = %q, want %q", res.Text, "second answer")
	}
	// Usage accumulated across the resume boundary (10 + 20).
	if res.Usage.TotalTokens != 30 {
		t.Errorf("cumulative usage = %d, want 30", res.Usage.TotalTokens)
	}
	// The resumed history should not duplicate the system message.
	msgs := resumed.Messages()
	systemCount := 0
	for _, m := range msgs {
		if m.Role == rimeno.RoleSystem {
			systemCount++
		}
	}
	if systemCount != 1 {
		t.Errorf("system messages after resume = %d, want 1", systemCount)
	}
	// The original turn-1 exchange is still present (context preserved).
	var sawFirst bool
	for _, m := range msgs {
		if m.Role == rimeno.RoleAssistant && m.Text == "first answer" {
			sawFirst = true
		}
	}
	if !sawFirst {
		t.Error("resumed session lost turn-1 context")
	}
}

// TestResumeEmptyState verifies resuming from a zero-value state is safe.
func TestResumeEmptyState(t *testing.T) {
	model := rimenotest.NewModel(rimenotest.Turn{Text: "ok"})
	agent, _ := rimeno.New(rimeno.Config{Model: model})
	sess := agent.ResumeSession(rimeno.SessionState{})
	res, err := sess.Send(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "ok" {
		t.Errorf("text = %q", res.Text)
	}
}
