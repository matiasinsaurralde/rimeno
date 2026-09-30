package memory

import (
	"context"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// stubInner is a minimal inner Compactor that always compacts and replaces the
// body with a fixed summary, keeping the leading system messages. It stands in
// for a SummarizingCompactor without needing a model, so the test isolates the
// memory re-injection behavior.
type stubInner struct{ summary string }

func (s stubInner) ShouldCompact([]rimeno.Message, int) bool { return true }

func (s stubInner) Compact(_ context.Context, msgs []rimeno.Message) ([]rimeno.Message, error) {
	var head []rimeno.Message
	i := 0
	for i < len(msgs) && msgs[i].Role == rimeno.RoleSystem {
		head = append(head, msgs[i])
		i++
	}
	// Drop the body entirely, replacing it with a summary — this is what would
	// otherwise lose the facts the agent stored mid-conversation.
	out := append(head, rimeno.SystemMessage("SUMMARY: "+s.summary))
	return out, nil
}

func TestCompactor_ReInjectsMemory(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "task-1")
	// A fact the agent stored earlier, e.g. via the remember tool.
	if _, err := mem.Remember(ctx, Record{Kind: "fact", Text: "the API base url is https://x", Salience: 0.9}); err != nil {
		t.Fatal(err)
	}

	c := Compactor(mem, stubInner{summary: "did some work"})

	msgs := []rimeno.Message{
		rimeno.SystemMessage("You are an assistant."),
		rimeno.UserMessage("hello"),
		rimeno.AssistantMessage("hi"),
		rimeno.UserMessage("more"),
		rimeno.AssistantMessage("ok"),
	}

	out, err := c.Compact(ctx, msgs)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}

	joined := renderMessages(out)
	if !strings.Contains(joined, "the API base url is https://x") {
		t.Fatalf("fact not re-injected after compaction:\n%s", joined)
	}
	if !strings.Contains(joined, "SUMMARY: did some work") {
		t.Fatalf("inner summary missing:\n%s", joined)
	}

	// The original instructions must remain first; the recalled memory sits right
	// after the leading system messages, before the summary.
	if out[0].Text != "You are an assistant." {
		t.Fatalf("instructions no longer first: %q", out[0].Text)
	}
	memIdx := indexOfContains(out, preambleHeader)
	sumIdx := indexOfContains(out, "SUMMARY:")
	if memIdx < 1 || memIdx > sumIdx {
		t.Fatalf("recalled memory misplaced: memIdx=%d sumIdx=%d", memIdx, sumIdx)
	}
}

func TestCompactor_ShouldCompactDelegates(t *testing.T) {
	mem := New(NewInMemory(), "ns")
	// With a nil inner, ShouldCompact is always false.
	if Compactor(mem, nil).ShouldCompact(nil, 1<<30) {
		t.Fatal("nil inner should never trigger compaction")
	}
	// With an inner that always compacts, it delegates true.
	if !Compactor(mem, stubInner{}).ShouldCompact(nil, 0) {
		t.Fatal("should delegate ShouldCompact to inner")
	}
}

func TestCompactor_NoRecordsIsNoOp(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "empty")
	c := Compactor(mem, stubInner{summary: "s"})
	in := []rimeno.Message{rimeno.SystemMessage("sys"), rimeno.UserMessage("u")}
	out, err := c.Compact(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	// No memory -> no injected memory block, only the inner's output.
	if indexOfContains(out, preambleHeader) != -1 {
		t.Fatalf("injected an empty memory block:\n%s", renderMessages(out))
	}
}

// TestCompactor_FactSurvivesForcedCompactionInSession is the end-to-end check: a
// multi-turn session with the memory Compactor keeps a stored fact available to
// the model after compaction runs between turns.
func TestCompactor_FactSurvivesForcedCompactionInSession(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "task-1")
	if _, err := mem.Remember(ctx, Record{Kind: "fact", Text: "secret token = abc123", Salience: 1}); err != nil {
		t.Fatal(err)
	}

	model := rimenotest.NewModel(
		rimenotest.Turn{Text: "turn one"},
		rimenotest.Turn{Text: "turn two"},
	)
	agent, err := rimeno.New(rimeno.Config{
		Model:        model,
		Instructions: "You are an assistant.",
		Compactor:    Compactor(mem, stubInner{summary: "prior work"}),
	})
	if err != nil {
		t.Fatal(err)
	}

	sess := agent.NewSession()
	if _, err := sess.Send(ctx, "first"); err != nil {
		t.Fatalf("send 1: %v", err)
	}
	// Compaction runs at the start of the next turn; the second request the model
	// sees should carry the recalled fact.
	if _, err := sess.Send(ctx, "second"); err != nil {
		t.Fatalf("send 2: %v", err)
	}

	lastReq := model.Requests[len(model.Requests)-1]
	if !transcriptContains(lastReq.Messages, "secret token = abc123") {
		t.Fatalf("stored fact did not survive compaction into the next turn:\n%s", renderMessages(lastReq.Messages))
	}
}

func indexOfContains(msgs []rimeno.Message, sub string) int {
	for i, m := range msgs {
		if strings.Contains(m.Text, sub) {
			return i
		}
	}
	return -1
}
