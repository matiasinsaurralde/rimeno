// Package rimenotest provides deterministic test doubles for rimeno, so agents and
// workflows can be tested without a live model provider.
package rimenotest

import (
	"context"
	"encoding/json"
	"io"
	"sync"
	"time"

	"github.com/matiasinsaurralde/rimeno"
)

// Turn scripts one model response returned by [Model] on successive Generate
// calls.
type Turn struct {
	// Text is the assistant's message text.
	Text string
	// ToolCalls, if non-empty, makes this a tool-requesting turn.
	ToolCalls []rimeno.ToolCall
	// Usage overrides the default usage for this turn.
	Usage *rimeno.Usage
	// StopReason overrides the inferred stop reason.
	StopReason rimeno.StopReason
	// Delay, if set, makes Generate block for this long before returning the turn,
	// simulating provider latency without any real model call. The wait is
	// context-aware: a canceled or deadline-exceeded context returns that error
	// instead. Used by tests that need wall-clock to parallelize. Applies to
	// scripted turns only.
	Delay time.Duration
}

// Model is a scripted, deterministic rimeno.Model for tests. It returns the
// configured turns in order, captures every request for assertions, and is safe
// for concurrent use.
type Model struct {
	id    string
	mu    sync.Mutex
	turns []Turn
	idx   int
	// Requests captures every request passed to Generate, in order.
	Requests []*rimeno.Request
	// GenerateFn, if set, overrides scripted turns with custom logic.
	GenerateFn func(ctx context.Context, req *rimeno.Request, call int) (*rimeno.Response, error)
}

// NewModel returns a scripted model that replays the given turns.
func NewModel(turns ...Turn) *Model { return &Model{id: "fake", turns: turns} }

// WithID sets the model identifier reported by ID.
func (m *Model) WithID(id string) *Model { m.id = id; return m }

// ID implements rimeno.Model.
func (m *Model) ID() string {
	if m.id == "" {
		return "fake"
	}
	return m.id
}

// Generate implements rimeno.Model.
func (m *Model) Generate(ctx context.Context, req *rimeno.Request) (*rimeno.Response, error) {
	m.mu.Lock()
	call := m.idx
	m.idx++
	m.Requests = append(m.Requests, req)
	fn := m.GenerateFn
	var turn Turn
	scripted := false
	if call < len(m.turns) {
		turn = m.turns[call]
		scripted = true
	}
	m.mu.Unlock()

	if fn != nil {
		return fn(ctx, req, call)
	}
	if !scripted {
		// Default terminal response once the script is exhausted.
		return &rimeno.Response{
			Message:    rimeno.Message{Role: rimeno.RoleAssistant, Text: ""},
			Usage:      rimeno.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
			StopReason: rimeno.StopReasonStop,
			Model:      m.ID(),
		}, nil
	}

	// Simulate provider latency for this turn (context-aware), so fan-out and
	// concurrency benchmarks have real wall-clock to parallelize. The mutex is
	// already released above, so concurrent Generate calls wait in parallel.
	if turn.Delay > 0 {
		t := time.NewTimer(turn.Delay)
		defer t.Stop()
		select {
		case <-t.C:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	stop := turn.StopReason
	if stop == "" {
		if len(turn.ToolCalls) > 0 {
			stop = rimeno.StopReasonToolCalls
		} else {
			stop = rimeno.StopReasonStop
		}
	}
	usage := rimeno.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15}
	if turn.Usage != nil {
		usage = *turn.Usage
	}
	return &rimeno.Response{
		Message:    rimeno.Message{Role: rimeno.RoleAssistant, Text: turn.Text, ToolCalls: turn.ToolCalls},
		Usage:      usage,
		StopReason: stop,
		Model:      m.ID(),
	}, nil
}

// ToolCall is a helper to build a rimeno.ToolCall with JSON-encoded arguments.
func ToolCall(id, name string, args any) rimeno.ToolCall {
	b, _ := json.Marshal(args)
	return rimeno.ToolCall{ID: id, Name: name, Arguments: b}
}

// Stream implements rimeno.StreamingModel: it produces the next scripted turn as a
// series of small text deltas followed by a final response. This lets streaming
// code paths be tested deterministically.
func (m *Model) Stream(ctx context.Context, req *rimeno.Request) (rimeno.Stream, error) {
	resp, err := m.Generate(ctx, req)
	if err != nil {
		return nil, err
	}
	return &fakeStream{resp: resp, deltas: chunkString(resp.Message.Text, 4)}, nil
}

type fakeStream struct {
	resp      *rimeno.Response
	deltas    []string
	i         int
	sentFinal bool
}

func (s *fakeStream) Recv() (rimeno.StreamChunk, error) {
	if s.i < len(s.deltas) {
		d := s.deltas[s.i]
		s.i++
		return rimeno.StreamChunk{TextDelta: d}, nil
	}
	if !s.sentFinal {
		s.sentFinal = true
		u := s.resp.Usage
		return rimeno.StreamChunk{Final: s.resp, Usage: &u}, nil
	}
	return rimeno.StreamChunk{}, io.EOF
}

func (s *fakeStream) Close() error { return nil }

// chunkString splits s into pieces of at most n bytes so that concatenating the
// pieces reproduces s exactly.
func chunkString(s string, n int) []string {
	if s == "" {
		return nil
	}
	var out []string
	for i := 0; i < len(s); i += n {
		end := i + n
		if end > len(s) {
			end = len(s)
		}
		out = append(out, s[i:end])
	}
	return out
}
