package rimeno

import (
	"encoding/json"
	"io"
)

// SessionState is a serializable snapshot of a [Session]'s conversation: the full
// message history and cumulative usage. It is everything needed to resume a
// multi-turn conversation in a later process.
//
// It is plain JSON — persist it however you like (file, database, KV store) with
// [SessionState.WriteTo]/[LoadSessionState] or your own encoder — rimeno does not
// dictate storage. This keeps the harness embeddable: the host owns durability.
type SessionState struct {
	// Version identifies the snapshot format for forward compatibility.
	Version int `json:"version"`
	// Messages is the conversation history, including the system message.
	Messages []Message `json:"messages"`
	// Usage is cumulative usage across the session so far.
	Usage Usage `json:"usage"`
}

// currentStateVersion is the SessionState format version written by Snapshot.
const currentStateVersion = 1

// Snapshot returns a serializable copy of the session's current state.
func (s *Session) Snapshot() SessionState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return SessionState{
		Version:  currentStateVersion,
		Messages: append([]Message(nil), s.history...),
		Usage:    s.usage,
	}
}

// ResumeSession creates a session seeded from a prior [SessionState], so a
// conversation can continue across process restarts. The snapshot already
// contains the system message, so the agent's Instructions are not re-added;
// the snapshot is the source of truth for history. Usage continues to accumulate
// from the snapshot's total.
func (a *Agent) ResumeSession(state SessionState) *Session {
	return &Session{
		agent:   a,
		history: append([]Message(nil), state.Messages...),
		usage:   state.Usage,
	}
}

// WriteTo writes the state as indented JSON, implementing [io.WriterTo].
func (s SessionState) WriteTo(w io.Writer) (int64, error) {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return 0, err
	}
	n, err := w.Write(b)
	return int64(n), err
}

// LoadSessionState reads a [SessionState] previously written by
// [SessionState.WriteTo] (or any equivalent JSON).
func LoadSessionState(r io.Reader) (SessionState, error) {
	var st SessionState
	if err := json.NewDecoder(r).Decode(&st); err != nil {
		return SessionState{}, err
	}
	return st, nil
}
