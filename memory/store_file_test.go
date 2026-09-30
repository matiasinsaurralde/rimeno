package memory

import (
	"context"
	"testing"
)

// TestFileStore_DurableAcrossRestart writes with one store instance and reads
// with a fresh instance over the same directory, simulating a process restart.
func TestFileStore_DurableAcrossRestart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	s1, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("NewFileStore: %v", err)
	}
	if err := s1.Put(ctx, Record{ID: "a", Namespace: "task-1", Kind: "fact", Text: "durable"}); err != nil {
		t.Fatalf("Put: %v", err)
	}

	s2, err := NewFileStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	got, ok, err := s2.Get(ctx, "task-1", "a")
	if err != nil || !ok {
		t.Fatalf("Get after reopen: ok=%v err=%v", ok, err)
	}
	if got.Text != "durable" {
		t.Fatalf("Text = %q, want durable", got.Text)
	}
}

// TestFileStore_NamespaceFilenameSafety ensures awkward namespace strings map to
// a usable file and round-trip.
func TestFileStore_NamespaceFilenameSafety(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	ns := "github.com/acme/repo@abc123"
	if err := s.Put(ctx, Record{ID: "x", Namespace: ns, Text: "ok"}); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, ok, err := s.Get(ctx, ns, "x")
	if err != nil || !ok || got.Text != "ok" {
		t.Fatalf("round-trip of awkward namespace failed: ok=%v err=%v got=%+v", ok, err, got)
	}
}
