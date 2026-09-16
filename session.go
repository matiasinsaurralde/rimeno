package rimeno

import "sync"

// Session is a stateful, multi-turn conversation with an [Agent]. It retains
// history across [Session.Send] calls and accumulates usage. A Session is not
// safe for concurrent Send calls; use one Session per conversation.
type Session struct {
	agent   *Agent
	mu      sync.Mutex
	history []Message
	usage   Usage
	traces  []*Trace
	// Responses-threading state (used only when Config.ResponsesThreading is set and
	// the model can thread): lastResponseID is the prior turn's response id and
	// threadSentIdx is how many history messages the server already has, so the next
	// request sends only history[threadSentIdx:]. Reset on compaction.
	lastResponseID string
	threadSentIdx  int
	// threadingDisabled latches on when a threaded request is rejected by the provider
	// (see threadingFallbacker): the session then reverts to sending full history for
	// the rest of its life, so a backend that advertises Responses but won't honor
	// previous_response_id degrades gracefully instead of failing every turn.
	threadingDisabled bool
}

// NewSession starts a fresh multi-turn conversation seeded with the agent's
// system instructions.
func (a *Agent) NewSession() *Session {
	s := &Session{agent: a}
	if a.instructions != "" {
		s.history = append(s.history, SystemMessage(a.instructions))
	}
	return s
}

// Send is implemented in run.go.

// Messages returns a copy of the current conversation history.
func (s *Session) Messages() []Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Message(nil), s.history...)
}

// Usage returns cumulative usage across all turns in the session.
func (s *Session) Usage() Usage {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.usage
}

// Traces returns the trace of each turn, in order.
func (s *Session) Traces() []*Trace {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*Trace(nil), s.traces...)
}
