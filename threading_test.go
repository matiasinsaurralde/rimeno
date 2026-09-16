package rimeno_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// threadingModel is a fake provider that speaks response-id threading: it returns a
// fresh response id each turn and captures every request, so a test can assert what
// the session actually sent (full history vs. delta + previous_response_id). When
// threads is false it still returns ids but reports it cannot thread, exercising the
// safe fallback where rimeno must send full history regardless of the opt-in.
type threadingModel struct {
	threads bool
	calls   int
	reqs    []*rimeno.Request
	turns   []rimeno.Message // assistant message per call; beyond it, a terminal empty reply
}

func (m *threadingModel) ID() string             { return "thread-fake" }
func (m *threadingModel) ThreadsResponses() bool { return m.threads }

func (m *threadingModel) Generate(_ context.Context, req *rimeno.Request) (*rimeno.Response, error) {
	// Snapshot the request (messages slice is reused/appended by the caller).
	snap := *req
	snap.Messages = append([]rimeno.Message(nil), req.Messages...)
	m.reqs = append(m.reqs, &snap)

	i := m.calls
	m.calls++
	var msg rimeno.Message
	if i < len(m.turns) {
		msg = m.turns[i]
	} else {
		msg = rimeno.Message{Role: rimeno.RoleAssistant}
	}
	stop := rimeno.StopReasonStop
	if len(msg.ToolCalls) > 0 {
		stop = rimeno.StopReasonToolCalls
	}
	return &rimeno.Response{
		Message:    msg,
		Usage:      rimeno.Usage{InputTokens: 5, OutputTokens: 3, TotalTokens: 8},
		StopReason: stop,
		Model:      m.ID(),
		ID:         fmt.Sprintf("resp_%d", i+1),
	}, nil
}

// TestResponsesThreading_SendsDeltaWithPreviousID verifies the happy path: turn 1
// sends the full history and no previous id; turn 2 (after a tool call) sends only
// the new tool-result message, threaded onto turn 1's response id.
func TestResponsesThreading_SendsDeltaWithPreviousID(t *testing.T) {
	var toolCalls int
	tool := newAddTool(&toolCalls)
	m := &threadingModel{
		threads: true,
		turns: []rimeno.Message{
			{Role: rimeno.RoleAssistant, ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 2, B: 3})}},
			{Role: rimeno.RoleAssistant, Text: "the answer is 5"},
		},
	}
	ag, err := rimeno.New(rimeno.Config{
		Model:              m,
		Instructions:       "sys",
		Tools:              []rimeno.Tool{tool},
		ResponsesThreading: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := ag.Run(context.Background(), "add 2 and 3")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "the answer is 5" {
		t.Fatalf("Text = %q", res.Text)
	}
	if len(m.reqs) != 2 {
		t.Fatalf("model calls = %d, want 2", len(m.reqs))
	}

	// Turn 1: full history (system + user), no previous_response_id.
	r1 := m.reqs[0]
	if r1.PreviousResponseID != "" {
		t.Errorf("turn 1 PreviousResponseID = %q, want empty", r1.PreviousResponseID)
	}
	if len(r1.Messages) != 2 {
		t.Fatalf("turn 1 messages = %d, want 2 (system+user)", len(r1.Messages))
	}
	if r1.Messages[0].Role != rimeno.RoleSystem || r1.Messages[1].Role != rimeno.RoleUser {
		t.Errorf("turn 1 roles = %v, %v; want system, user", r1.Messages[0].Role, r1.Messages[1].Role)
	}

	// Turn 2: only the delta the server does not yet have — the tool result — plus
	// previous_response_id pointing at turn 1's response.
	r2 := m.reqs[1]
	if r2.PreviousResponseID != "resp_1" {
		t.Errorf("turn 2 PreviousResponseID = %q, want resp_1", r2.PreviousResponseID)
	}
	if len(r2.Messages) != 1 {
		t.Fatalf("turn 2 messages = %d, want 1 (tool result only)", len(r2.Messages))
	}
	if r2.Messages[0].Role != rimeno.RoleTool {
		t.Errorf("turn 2 message role = %v, want tool", r2.Messages[0].Role)
	}
}

// TestResponsesThreading_NonThreadingModelGetsFullHistory verifies the safety
// guarantee: enabling the opt-in against a model that reports it cannot thread must
// never send a previous_response_id and must always resend full history.
func TestResponsesThreading_NonThreadingModelGetsFullHistory(t *testing.T) {
	var toolCalls int
	tool := newAddTool(&toolCalls)
	m := &threadingModel{
		threads: false, // model cannot thread
		turns: []rimeno.Message{
			{Role: rimeno.RoleAssistant, ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 2, B: 3})}},
			{Role: rimeno.RoleAssistant, Text: "5"},
		},
	}
	ag, err := rimeno.New(rimeno.Config{
		Model:              m,
		Instructions:       "sys",
		Tools:              []rimeno.Tool{tool},
		ResponsesThreading: true, // opt-in set, but ignored because the model can't thread
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ag.Run(context.Background(), "add"); err != nil {
		t.Fatal(err)
	}
	if len(m.reqs) != 2 {
		t.Fatalf("model calls = %d, want 2", len(m.reqs))
	}
	r2 := m.reqs[1]
	if r2.PreviousResponseID != "" {
		t.Errorf("non-threading model got PreviousResponseID = %q, want empty", r2.PreviousResponseID)
	}
	// Full history on turn 2: system, user, assistant(tool call), tool result.
	if len(r2.Messages) != 4 {
		t.Errorf("turn 2 messages = %d, want 4 (full history)", len(r2.Messages))
	}
}

// rejectingThreadModel advertises threading but errors on any request carrying a
// previous_response_id, and classifies that error as a threading rejection — modeling
// a provider (like OpenRouter for kimi-k3) that speaks the Responses API but will not
// honor previous_response_id. It exercises rimeno's transparent fallback.
type rejectingThreadModel struct {
	calls  int
	reqs   []*rimeno.Request
	turns  []rimeno.Message
	rejErr error
}

func (m *rejectingThreadModel) ID() string                          { return "reject-thread" }
func (m *rejectingThreadModel) ThreadsResponses() bool              { return true }
func (m *rejectingThreadModel) ThreadingUnsupported(err error) bool { return errors.Is(err, m.rejErr) }

func (m *rejectingThreadModel) Generate(_ context.Context, req *rimeno.Request) (*rimeno.Response, error) {
	snap := *req
	snap.Messages = append([]rimeno.Message(nil), req.Messages...)
	m.reqs = append(m.reqs, &snap)
	if req.PreviousResponseID != "" {
		return nil, m.rejErr // provider rejects threading
	}
	i := m.calls
	m.calls++
	var msg rimeno.Message
	if i < len(m.turns) {
		msg = m.turns[i]
	} else {
		msg = rimeno.Message{Role: rimeno.RoleAssistant}
	}
	stop := rimeno.StopReasonStop
	if len(msg.ToolCalls) > 0 {
		stop = rimeno.StopReasonToolCalls
	}
	return &rimeno.Response{Message: msg, Usage: rimeno.Usage{InputTokens: 5, OutputTokens: 3, TotalTokens: 8},
		StopReason: stop, Model: m.ID(), ID: fmt.Sprintf("resp_%d", i+1)}, nil
}

// TestResponsesThreading_FallsBackWhenProviderRejects verifies the hardening the live
// test found necessary: when a threaded request is rejected because the provider won't
// honor previous_response_id, the session drops threading, resends full history, and
// completes — instead of failing the run.
func TestResponsesThreading_FallsBackWhenProviderRejects(t *testing.T) {
	var toolCalls int
	tool := newAddTool(&toolCalls)
	m := &rejectingThreadModel{
		rejErr: errors.New("http 400: previous_response_id not supported"),
		turns: []rimeno.Message{
			{Role: rimeno.RoleAssistant, ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 2, B: 3})}},
			{Role: rimeno.RoleAssistant, Text: "sum is 5"},
		},
	}
	ag, err := rimeno.New(rimeno.Config{
		Model:              m,
		Instructions:       "sys",
		Tools:              []rimeno.Tool{tool},
		ResponsesThreading: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := ag.Run(context.Background(), "add 2 and 3")
	if err != nil {
		t.Fatalf("run failed instead of falling back: %v", err)
	}
	if res.Text != "sum is 5" {
		t.Fatalf("Text = %q, want the completed answer", res.Text)
	}
	// Expect: turn 1 (full history), turn 2 threaded (rejected), turn 2 retried with
	// full history and no previous_response_id.
	if len(m.reqs) != 3 {
		t.Fatalf("requests = %d, want 3 (turn1, rejected threaded turn2, full-history retry)", len(m.reqs))
	}
	if m.reqs[1].PreviousResponseID == "" {
		t.Errorf("second request should have attempted threading (previous_response_id set)")
	}
	if m.reqs[2].PreviousResponseID != "" {
		t.Errorf("retry should drop threading, got PreviousResponseID=%q", m.reqs[2].PreviousResponseID)
	}
	if len(m.reqs[2].Messages) != 4 {
		t.Errorf("retry messages = %d, want 4 (full history: system,user,assistant,tool)", len(m.reqs[2].Messages))
	}
}

// onceCompactor rewrites history to a fixed 3-message summary the first time the
// history grows past a threshold, then never again. It lets a test force a single
// compaction mid-loop.
type onceCompactor struct{ fired bool }

func (c *onceCompactor) ShouldCompact(msgs []rimeno.Message, _ int) bool {
	return !c.fired && len(msgs) >= 4
}

func (c *onceCompactor) Compact(_ context.Context, msgs []rimeno.Message) ([]rimeno.Message, error) {
	c.fired = true
	// Keep the system message and the last two messages (still >= threadSentIdx, so
	// the reset is what actually prevents an out-of-range/empty threaded request).
	return []rimeno.Message{msgs[0], msgs[len(msgs)-2], msgs[len(msgs)-1]}, nil
}

// TestResponsesThreading_CompactionResetsThread verifies that a mid-loop compaction
// drops the server-side thread: the turn after compaction must resend the full
// (compacted) history with no previous_response_id, since the rewritten history no
// longer matches what the server holds.
func TestResponsesThreading_CompactionResetsThread(t *testing.T) {
	var toolCalls int
	tool := newAddTool(&toolCalls)
	m := &threadingModel{
		threads: true,
		turns: []rimeno.Message{
			{Role: rimeno.RoleAssistant, ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 2, B: 3})}},
			{Role: rimeno.RoleAssistant, Text: "done"},
		},
	}
	ag, err := rimeno.New(rimeno.Config{
		Model:              m,
		Instructions:       "sys",
		Tools:              []rimeno.Tool{tool},
		ResponsesThreading: true,
		Compactor:          &onceCompactor{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ag.Run(context.Background(), "add"); err != nil {
		t.Fatal(err)
	}
	if len(m.reqs) != 2 {
		t.Fatalf("model calls = %d, want 2", len(m.reqs))
	}
	// Turn 1 established the thread (resp_1). Compaction fires before turn 2 and
	// rewrites history to 3 messages, so turn 2 must be full history, unthreaded.
	r2 := m.reqs[1]
	if r2.PreviousResponseID != "" {
		t.Errorf("post-compaction PreviousResponseID = %q, want empty (thread reset)", r2.PreviousResponseID)
	}
	if len(r2.Messages) != 3 {
		t.Errorf("post-compaction messages = %d, want 3 (full compacted history)", len(r2.Messages))
	}
}
