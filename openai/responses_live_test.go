package openai_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

// TestResponses_Live exercises the Responses provider against a real endpoint
// (OpenRouter's /v1/responses). It is skipped unless RIMENO_LIVE=1 and OPENAI_API_KEY
// are set, so normal `go test` stays offline and free.
//
//	RIMENO_LIVE=1 OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
//	  go test -run TestResponses_Live -v ./openai/
func TestResponses_Live(t *testing.T) {
	if os.Getenv("RIMENO_LIVE") != "1" || os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set RIMENO_LIVE=1 and OPENAI_API_KEY to run the live Responses test")
	}
	base := os.Getenv("OPENAI_BASE_URL")
	if base == "" {
		base = "https://openrouter.ai/api/v1"
	}
	const model = "moonshotai/kimi-k3"
	c := openai.New(
		openai.WithBaseURL(base),
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithModel(model),
		openai.WithResponsesAPI(),
		openai.WithMaxRetries(3),
		openai.WithPricing(model, openai.Price{InputPerMillion: 2.375, OutputPerMillion: 13.30}),
	)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	res, err := c.Generate(ctx, &rimeno.Request{
		Messages:  []rimeno.Message{rimeno.UserMessage("Reply with exactly: PONG")},
		MaxTokens: 2000,
	})
	if err != nil {
		t.Fatalf("live Responses Generate failed: %v", err)
	}
	t.Logf("text=%q reasoning_len=%d usage in/out/total=%d/%d/%d cost=$%.5f",
		res.Message.Text, len(res.Message.Reasoning),
		res.Usage.InputTokens, res.Usage.OutputTokens, res.Usage.TotalTokens, res.Usage.CostUSD)
	if res.Message.Text == "" && res.Message.Reasoning == "" {
		t.Fatalf("empty answer AND empty reasoning — Responses parsing likely wrong; Raw=%s", string(res.Raw))
	}
	if res.Usage.TotalTokens == 0 {
		t.Errorf("usage not parsed (total=0)")
	}
}
