package workflow

import "sync"

// State is the shared blackboard passed to every step. It holds the workflow
// input and each completed step's output (keyed by step name). It is safe for
// concurrent use so parallel steps can read/write it.
type State struct {
	mu     sync.Mutex
	values map[string]string
	// Input is the top-level workflow input.
	Input string
}

func newState(input string) *State {
	return &State{values: map[string]string{}, Input: input}
}

// Get returns the value stored under key (typically a completed step's output),
// or "" if absent.
func (s *State) Get(key string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.values[key]
}

// Set stores a value under key.
func (s *State) Set(key, value string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.values[key] = value
}
