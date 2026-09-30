package memory

import (
	"context"
	"sync"
	"testing"
)

// TestInMemory_ConcurrentSharedBlackboard exercises the claim that the default
// store is safe for many agents sharing one namespace: goroutines write and read
// the same scope concurrently. Run with -race to catch data races.
func TestInMemory_ConcurrentSharedBlackboard(t *testing.T) {
	ctx := context.Background()
	store := NewInMemory()

	const agents = 8
	const perAgent = 50
	var wg sync.WaitGroup
	for a := 0; a < agents; a++ {
		wg.Add(1)
		go func(a int) {
			defer wg.Done()
			mem := New(store, "shared", WithSource("agent-"+itoa(a)))
			for i := 0; i < perAgent; i++ {
				if _, err := mem.Remember(ctx, Record{Kind: "fact", Text: "a" + itoa(a) + "-" + itoa(i)}); err != nil {
					t.Errorf("Remember: %v", err)
					return
				}
				if _, err := mem.Recall(ctx, Query{Limit: 5}); err != nil {
					t.Errorf("Recall: %v", err)
					return
				}
			}
		}(a)
	}
	wg.Wait()

	all, err := New(store, "shared").Recall(ctx, Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != agents*perAgent {
		t.Fatalf("got %d records, want %d (all writes visible on shared blackboard)", len(all), agents*perAgent)
	}
}
