package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// fileStore is a durable [Store] that keeps one JSON file per namespace under a
// directory. Each write rewrites that namespace's file atomically (temp file +
// rename). It is safe for concurrent use within one process; it is not a
// multi-process database (use a SQLite backend for that).
type fileStore struct {
	dir string
	now func() time.Time

	mu    sync.Mutex
	cache map[string]map[string]Record // namespace -> id -> record (lazy-loaded)
}

// NewFileStore returns a durable [Store] rooted at dir (created if absent),
// persisting each namespace as a JSON file. It makes a run resumable across
// process restarts and is the substrate for a cross-run knowledge store.
func NewFileStore(dir string) (Store, error) {
	if dir == "" {
		return nil, fmt.Errorf("memory: NewFileStore requires a directory")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("memory: create store dir: %w", err)
	}
	return &fileStore{dir: dir, now: time.Now, cache: make(map[string]map[string]Record)}, nil
}

func (s *fileStore) Put(_ context.Context, r Record) error {
	r = normalize(r, s.now())
	s.mu.Lock()
	defer s.mu.Unlock()
	ns, err := s.load(r.Namespace)
	if err != nil {
		return err
	}
	if prev, ok := ns[r.ID]; ok {
		r.CreatedAt = prev.CreatedAt
	}
	ns[r.ID] = r
	return s.flush(r.Namespace, ns)
}

func (s *fileStore) Get(_ context.Context, nsName, id string) (Record, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ns, err := s.load(nsName)
	if err != nil {
		return Record{}, false, err
	}
	r, ok := ns[id]
	return r, ok, nil
}

func (s *fileStore) Query(_ context.Context, q Query) ([]Record, error) {
	s.mu.Lock()
	ns, err := s.load(q.Namespace)
	if err != nil {
		s.mu.Unlock()
		return nil, err
	}
	var out []Record
	for _, r := range ns {
		if matches(r, q) {
			out = append(out, r)
		}
	}
	s.mu.Unlock()
	rank(out, q)
	return applyLimit(out, q), nil
}

func (s *fileStore) Delete(_ context.Context, nsName, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ns, err := s.load(nsName)
	if err != nil {
		return err
	}
	if _, ok := ns[id]; !ok {
		return nil
	}
	delete(ns, id)
	return s.flush(nsName, ns)
}

// load returns the (cached) record map for a namespace, reading its file on first
// access. Caller holds s.mu.
func (s *fileStore) load(nsName string) (map[string]Record, error) {
	if ns, ok := s.cache[nsName]; ok {
		return ns, nil
	}
	ns := make(map[string]Record)
	b, err := os.ReadFile(s.path(nsName))
	switch {
	case os.IsNotExist(err):
		// New namespace; empty map.
	case err != nil:
		return nil, fmt.Errorf("memory: read namespace %q: %w", nsName, err)
	default:
		var recs []Record
		if err := json.Unmarshal(b, &recs); err != nil {
			return nil, fmt.Errorf("memory: decode namespace %q: %w", nsName, err)
		}
		for _, r := range recs {
			ns[r.ID] = r
		}
	}
	s.cache[nsName] = ns
	return ns, nil
}

// flush writes a namespace's records to disk atomically. Caller holds s.mu.
func (s *fileStore) flush(nsName string, ns map[string]Record) error {
	recs := make([]Record, 0, len(ns))
	for _, r := range ns {
		recs = append(recs, r)
	}
	// Stable on-disk order (by ID) keeps files diff-friendly and deterministic.
	sort.Slice(recs, func(i, j int) bool { return recs[i].ID < recs[j].ID })
	b, err := json.MarshalIndent(recs, "", "  ")
	if err != nil {
		return fmt.Errorf("memory: encode namespace %q: %w", nsName, err)
	}
	path := s.path(nsName)
	tmp, err := os.CreateTemp(s.dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("memory: temp file: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("memory: write namespace %q: %w", nsName, err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("memory: close namespace %q: %w", nsName, err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("memory: commit namespace %q: %w", nsName, err)
	}
	return nil
}

// path maps a namespace to its on-disk JSON file, encoding characters that are
// unsafe in filenames so keys like "repo@commit" or "a/b" map to a flat file.
func (s *fileStore) path(nsName string) string {
	return filepath.Join(s.dir, safeFilename(nsName)+".json")
}

// safeFilename replaces path separators and other awkward characters so any
// namespace string maps to a single, safe filename.
func safeFilename(ns string) string {
	if ns == "" {
		return "_default"
	}
	r := strings.NewReplacer(
		"/", "_", "\\", "_", ":", "_", "*", "_",
		"?", "_", "\"", "_", "<", "_", ">", "_", "|", "_", " ", "_",
	)
	return r.Replace(ns)
}
