package rimeno_test

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

// TestOutputFromSchema_Live drives OutputFromSchema end-to-end against a real
// endpoint: it hands a hand-written JSON Schema straight to the provider as
// response_format and asserts the model's answer validates and decodes. Skipped
// unless RIMENO_LIVE=1 and OPENAI_API_KEY are set, so a plain `go test` stays
// offline and free even when an OPENAI_API_KEY happens to be in the environment.
//
//	RIMENO_LIVE=1 OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
//	  go test -run TestOutputFromSchema_Live -v .
func TestOutputFromSchema_Live(t *testing.T) {
	if os.Getenv("RIMENO_LIVE") != "1" || os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set RIMENO_LIVE=1 and OPENAI_API_KEY to run the live OutputFromSchema test")
	}
	base := os.Getenv("OPENAI_BASE_URL")
	if base == "" {
		base = "https://openrouter.ai/api/v1"
	}
	const modelID = "openai/gpt-4o-mini"

	model := openai.New(
		openai.WithBaseURL(base),
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithModel(modelID),
		openai.WithMaxRetries(3),
	)

	// A hand-written schema — the case OutputFromSchema exists for. Lax() keeps
	// non-strict providers from rejecting it.
	sentiment := json.RawMessage(`{
		"type": "object",
		"properties": {
			"label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
			"score": {"type": "integer", "minimum": 0, "maximum": 100}
		},
		"required": ["label", "score"],
		"additionalProperties": false
	}`)

	agent, err := rimeno.New(rimeno.Config{
		Model:                model,
		Instructions:         "Classify the sentiment of the user's text. Reply with ONLY JSON matching the schema.",
		Output:               rimeno.OutputFromSchema("sentiment", sentiment, rimeno.Lax()),
		OutputRepairAttempts: 2,
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	res, err := agent.Run(ctx, "I absolutely loved this movie, best I've seen all year!")
	if err != nil {
		t.Fatalf("live run failed: %v", err)
	}

	raw, ok := res.Output.(json.RawMessage)
	if !ok {
		t.Fatalf("Output type = %T, want json.RawMessage", res.Output)
	}
	var got struct {
		Label string `json:"label"`
		Score int    `json:"score"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode live output %s: %v", raw, err)
	}
	t.Logf("label=%q score=%d text=%q tokens=%d cost=$%.5f",
		got.Label, got.Score, res.Text, res.Usage.TotalTokens, res.Usage.CostUSD)

	if got.Label != "positive" {
		t.Errorf("label = %q, want positive for clearly-positive text", got.Label)
	}
	if got.Score < 0 || got.Score > 100 {
		t.Errorf("score = %d, out of the schema's [0,100] range", got.Score)
	}
}
