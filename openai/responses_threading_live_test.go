package openai_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

// capModel wraps the Responses client to record whether rimeno ever sent a
// previous_response_id (i.e. threading actually engaged) and how many model calls
// happened. It re-exposes ThreadsResponses so rimeno still recognizes the capability
// through the wrapper.
type capModel struct {
	inner     *openai.Client
	mu        sync.Mutex
	sawPrevID bool
	calls     int
}

func (c *capModel) ID() string                        { return c.inner.ID() }
func (c *capModel) ThreadsResponses() bool            { return c.inner.ThreadsResponses() }
func (c *capModel) ThreadingUnsupported(e error) bool { return c.inner.ThreadingUnsupported(e) }

func (c *capModel) Generate(ctx context.Context, req *rimeno.Request) (*rimeno.Response, error) {
	c.mu.Lock()
	c.calls++
	if req.PreviousResponseID != "" {
		c.sawPrevID = true
	}
	c.mu.Unlock()
	return c.inner.Generate(ctx, req)
}

// TestResponsesThreading_Live validates the Responses session-loop threading against a
// real endpoint end-to-end, and measures its token/cost effect. It runs the SAME
// multi-turn tool loop twice — threading on and off — over /v1/responses.
//
// The decisive correctness check is that the THREADED run still reaches the right
// answer: with threading, turns after the first send only the new tool result plus
// previous_response_id, so a correct answer proves the server reconstructed the full
// context from the prior response id (had it ignored previous_response_id, the later
// turns would lack the task and the tool words, and the answer would be wrong).
//
//	RIMENO_LIVE=1 OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
//	  go test -run TestResponsesThreading_Live -v ./openai/
func TestResponsesThreading_Live(t *testing.T) {
	if os.Getenv("RIMENO_LIVE") != "1" || os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set RIMENO_LIVE=1 and OPENAI_API_KEY to run the live threading test")
	}
	base := os.Getenv("OPENAI_BASE_URL")
	if base == "" {
		base = "https://openrouter.ai/api/v1"
	}
	const modelID = "moonshotai/kimi-k3"

	// A tool loop the model cannot short-circuit: the three code words are only
	// obtainable by calling the tool, so producing the final answer forces multiple
	// tool-result turns — exactly where threading avoids resending history.
	words := map[int]string{1: "alpha", 2: "bravo", 3: "charlie"}
	codeword := rimeno.NewTool("codeword",
		"Return the secret code word for step n (n is 1, 2, or 3). Call it once per step.",
		func(_ context.Context, in struct {
			N int `json:"n"`
		}) (string, error) {
			w, ok := words[in.N]
			if !ok {
				return "", fmt.Errorf("no word for n=%d", in.N)
			}
			return w, nil
		})
	const task = "Get three code words by calling `codeword` for n=1, then n=2, then n=3 " +
		"(one call at a time, wait for each result). Then reply with exactly: RESULT: <w1>-<w2>-<w3>."
	const wantAnswer = "alpha-bravo-charlie"

	run := func(threading bool) (correct bool, in, cached, out int, cost float64, calls int, sawPrev bool) {
		cm := &capModel{inner: openai.New(
			openai.WithBaseURL(base),
			openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
			openai.WithModel(modelID),
			openai.WithResponsesAPI(),
			openai.WithMaxRetries(3),
			openai.WithPricing(modelID, openai.Price{InputPerMillion: 2.375, OutputPerMillion: 13.30}),
		)}
		ag, err := rimeno.New(rimeno.Config{
			Model:              cm,
			Instructions:       "You are a careful assistant that uses tools exactly as instructed.",
			Tools:              []rimeno.Tool{codeword},
			ResponsesThreading: threading,
			MaxSteps:           8,
			Budget:             rimeno.Budget{MaxTotalTokens: 40000, MaxToolCalls: 8, Timeout: 120 * time.Second},
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
		defer cancel()
		res, err := ag.Run(ctx, task)
		if err != nil {
			t.Fatalf("threading=%v run failed: %v", threading, err)
		}
		u := res.Usage
		return strings.Contains(strings.ToLower(res.Text), wantAnswer),
			u.InputTokens, u.CachedInputTokens, u.OutputTokens, u.CostUSD, cm.calls, cm.sawPrevID
	}

	onCorrect, onIn, onCached, onOut, onCost, onCalls, onPrev := run(true)
	offCorrect, offIn, offCached, offOut, offCost, offCalls, _ := run(false)

	t.Logf("threading ON : correct=%v in=%d cached=%d out=%d cost=$%.5f model_calls=%d sawPrevID=%v",
		onCorrect, onIn, onCached, onOut, onCost, onCalls, onPrev)
	t.Logf("threading OFF: correct=%v in=%d cached=%d out=%d cost=$%.5f model_calls=%d",
		offCorrect, offIn, offCached, offOut, offCost, offCalls)
	if onIn > 0 {
		t.Logf("payoff: threaded sent %+d input tokens (%.1f%%) vs unthreaded; cost %+.5f",
			onIn-offIn, 100*float64(onIn-offIn)/float64(offIn), onCost-offCost)
	}

	// Hard gates (non-flaky):
	// 1. Threading must actually have engaged (previous_response_id sent).
	if onCalls >= 2 && !onPrev {
		t.Errorf("threading ON but no previous_response_id was ever sent across %d calls", onCalls)
	}
	// 2. The threaded run must still be correct — proof the server reconstructed
	//    context from the response id rather than the (omitted) history.
	if !onCorrect {
		t.Errorf("threaded run wrong answer — server may not honor previous_response_id for %s", modelID)
	}
	// 3. Control: the unthreaded run (full history) must also be correct.
	if !offCorrect {
		t.Errorf("unthreaded control run wrong answer (%q expected)", wantAnswer)
	}
}
