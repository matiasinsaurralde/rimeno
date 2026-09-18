// Command quickstart is a minimal rimeno example: one agent with one Go tool,
// against any OpenAI-compatible endpoint.
//
// Run (reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment or a .env file):
//
//	OPENAI_API_KEY=sk-... go run ./examples/quickstart
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/dotenv"
	"github.com/matiasinsaurralde/rimeno/openai"
)

// model is hardcoded for the example; swap it for whatever your endpoint serves.
const model = "gpt-4o-mini"

type addArgs struct {
	A int `json:"a" jsonschema:"description=first addend"`
	B int `json:"b" jsonschema:"description=second addend"`
}

func main() {
	// Load .env for local dev (never overrides variables already set).
	_, _ = dotenv.Load()

	m := openai.New(
		openai.WithBaseURL(env("OPENAI_BASE_URL", openai.DefaultBaseURL)),
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithModel(model),
	)

	add := rimeno.NewTool("add", "Add two integers",
		func(ctx context.Context, in addArgs) (int, error) { return in.A + in.B, nil })

	agent, err := rimeno.New(rimeno.Config{
		Model:        m,
		Instructions: "You are precise. Use tools for arithmetic.",
		Tools:        []rimeno.Tool{add},
	})
	if err != nil {
		log.Fatal(err)
	}

	res, err := agent.Run(context.Background(), "what is 21 + 21?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Text)

	s := res.Trace.Summary()
	fmt.Printf("model_calls=%d tool_calls=%d tokens=%d cost=$%.4f\n",
		s.ModelCalls, s.ToolCalls, s.Usage.TotalTokens, s.Usage.CostUSD)
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
