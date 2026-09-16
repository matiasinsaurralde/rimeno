package rimeno_test

import (
	"context"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

func TestSummarizingCompactor_Compact(t *testing.T) {
	summarizer := rimenotest.NewModel(rimenotest.Turn{Text: "<summary>condensed</summary>"})
	c := &rimeno.SummarizingCompactor{Model: summarizer, MaxContextTokens: 100, KeepRecent: 2}

	msgs := []rimeno.Message{
		rimeno.SystemMessage("system prompt"),
		rimeno.UserMessage("turn 1"),
		rimeno.AssistantMessage("reply 1"),
		rimeno.UserMessage("turn 2"),
		rimeno.AssistantMessage("reply 2"),
		rimeno.UserMessage("turn 3"),
	}
	out, err := c.Compact(context.Background(), msgs)
	if err != nil {
		t.Fatal(err)
	}
	// expect: system prompt + summary + last 2 verbatim = 4 messages
	if len(out) != 4 {
		t.Fatalf("compacted len = %d, want 4: %+v", len(out), out)
	}
	if out[0].Text != "system prompt" {
		t.Errorf("system prompt not preserved: %q", out[0].Text)
	}
	if !strings.Contains(out[1].Text, "condensed") {
		t.Errorf("summary missing: %q", out[1].Text)
	}
	if out[len(out)-1].Text != "turn 3" {
		t.Errorf("recent tail not preserved: %q", out[len(out)-1].Text)
	}
}

func TestSummarizingCompactor_TriggersInLoop(t *testing.T) {
	// A compactor with a tiny window should fire; assert a compaction event.
	summarizer := rimenotest.NewModel(rimenotest.Turn{Text: "SUMMARY"})
	c := &rimeno.SummarizingCompactor{Model: summarizer, MaxContextTokens: 1, KeepRecent: 1}

	var compacted bool
	m := rimenotest.NewModel(rimenotest.Turn{Text: "answer"})
	agent, _ := rimeno.New(rimeno.Config{
		Model:     m,
		Compactor: c,
		OnEvent: func(e rimeno.Event) {
			if _, ok := e.(rimeno.CompactionStartedEvent); ok {
				compacted = true
			}
		},
	})
	if _, err := agent.Run(context.Background(), "hello with enough text to exceed the tiny window"); err != nil {
		t.Fatal(err)
	}
	if !compacted {
		t.Error("expected compaction to trigger")
	}
}

func TestEstimateTokens(t *testing.T) {
	got := rimeno.EstimateTokens([]rimeno.Message{rimeno.UserMessage(strings.Repeat("x", 40))})
	if got < 10 {
		t.Errorf("EstimateTokens = %d, want >= 10", got)
	}
}
