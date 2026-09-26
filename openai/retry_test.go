package openai_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

const okBody = `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"total_tokens":1}}`

// fastBackoff keeps retry waits negligible so tests don't sleep for real.
func fastBackoff() openai.Option {
	return openai.WithBackoff(openai.Backoff{Base: time.Millisecond, Cap: 5 * time.Millisecond, MaxRetries: 5})
}

// TestDo_RetriesStatus covers 429 and 5xx recovery: a retriable status on the
// first attempt followed by a 200 must succeed.
func TestDo_RetriesStatus(t *testing.T) {
	for _, status := range []int{http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusBadGateway} {
		t.Run(fmt.Sprintf("status_%d", status), func(t *testing.T) {
			var n int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if atomic.AddInt32(&n, 1) == 1 {
					w.WriteHeader(status)
					_, _ = fmt.Fprint(w, `{"error":{"message":"try again"}}`)
					return
				}
				_, _ = fmt.Fprint(w, okBody)
			}))
			defer srv.Close()
			c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"), fastBackoff())
			resp, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}})
			if err != nil {
				t.Fatalf("expected recovery, got %v", err)
			}
			if resp.Message.Text != "ok" {
				t.Errorf("text = %q", resp.Message.Text)
			}
			if got := atomic.LoadInt32(&n); got != 2 {
				t.Errorf("attempts = %d, want 2", got)
			}
		})
	}
}

// TestDo_StopsAtMaxRetries verifies the total attempt count is MaxRetries+1 when
// the endpoint never recovers.
func TestDo_StopsAtMaxRetries(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&n, 1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = fmt.Fprint(w, `{"error":{"message":"down"}}`)
	}))
	defer srv.Close()
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"),
		openai.WithBackoff(openai.Backoff{Base: time.Millisecond, Cap: 5 * time.Millisecond, MaxRetries: 3}))
	_, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}})
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	if got := atomic.LoadInt32(&n); got != 4 { // MaxRetries(3) + 1
		t.Errorf("attempts = %d, want 4", got)
	}
}

// TestDo_HonorsRetryAfterSeconds verifies a 429 Retry-After (delta-seconds) delays
// the next attempt by at least the hinted duration.
func TestDo_HonorsRetryAfterSeconds(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = fmt.Fprint(w, okBody)
	}))
	defer srv.Close()
	// Tiny jitter schedule; Retry-After must override it and dominate the wait.
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"),
		openai.WithBackoff(openai.Backoff{Base: time.Millisecond, Cap: 30 * time.Second, MaxRetries: 3}))
	start := time.Now()
	if _, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < time.Second {
		t.Errorf("elapsed = %v, want >= 1s (Retry-After honored)", elapsed)
	}
}

// TestDo_RetryAfterClampedToCap verifies a large Retry-After hint is clamped to
// Cap rather than stalling the caller for the full hinted time.
func TestDo_RetryAfterClampedToCap(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.Header().Set("Retry-After", "3600") // 1h hint
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = fmt.Fprint(w, okBody)
	}))
	defer srv.Close()
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"),
		openai.WithBackoff(openai.Backoff{Base: time.Millisecond, Cap: 20 * time.Millisecond, MaxRetries: 3}))
	start := time.Now()
	if _, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("elapsed = %v, want clamped to Cap (<1s)", elapsed)
	}
}

// TestDo_ContextCanceledDuringBackoff verifies a canceled context aborts the wait
// between attempts promptly with ctx.Err().
func TestDo_ContextCanceledDuringBackoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = fmt.Fprint(w, `{"error":{"message":"down"}}`)
	}))
	defer srv.Close()
	// Long backoff so the wait is still in progress when the deadline fires.
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"),
		openai.WithBackoff(openai.Backoff{Base: 10 * time.Second, Cap: 10 * time.Second, MaxRetries: 5}))
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := c.Generate(ctx, &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}})
	if err == nil {
		t.Fatal("expected context error")
	}
	if !strings.Contains(err.Error(), "context") {
		t.Errorf("err = %v, want context error", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("elapsed = %v, want prompt cancellation (<1s)", elapsed)
	}
}

// TestWithMaxConcurrency_LimitsInFlight verifies the client never holds more than
// k requests in flight to the endpoint, even with many concurrent callers.
func TestWithMaxConcurrency_LimitsInFlight(t *testing.T) {
	const k = 2
	var inFlight, maxSeen int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		cur := atomic.AddInt32(&inFlight, 1)
		for {
			old := atomic.LoadInt32(&maxSeen)
			if cur <= old || atomic.CompareAndSwapInt32(&maxSeen, old, cur) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond) // hold the slot so overlap is observable
		atomic.AddInt32(&inFlight, -1)
		_, _ = fmt.Fprint(w, okBody)
	}))
	defer srv.Close()
	c := openai.New(openai.WithBaseURL(srv.URL), openai.WithModel("m"), openai.WithMaxConcurrency(k))

	const callers = 10
	done := make(chan error, callers)
	for i := 0; i < callers; i++ {
		go func() {
			_, err := c.Generate(context.Background(), &rimeno.Request{Messages: []rimeno.Message{rimeno.UserMessage("x")}})
			done <- err
		}()
	}
	for i := 0; i < callers; i++ {
		if err := <-done; err != nil {
			t.Fatalf("caller %d: %v", i, err)
		}
	}
	if got := atomic.LoadInt32(&maxSeen); got > k {
		t.Errorf("max in-flight = %d, want <= %d", got, k)
	}
}
