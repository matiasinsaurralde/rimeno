// Command quickstart is a minimal rimeno example: one agent with one Go tool,
// against any OpenAI-compatible endpoint.
//
// Run:
//
//	RIMENO_MODEL=gpt-4o-mini RIMENO_API_KEY=sk-... go run ./examples/quickstart
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

type addArgs struct {
	A int `json:"a" jsonschema:"description=first addend"`
	B int `json:"b" jsonschema:"description=second addend"`
}

func main() {
	model := openai.New(
		openai.WithBaseURL(env("RIMENO_BASE_URL", openai.DefaultBaseURL)),
		openai.WithAPIKey(os.Getenv("RIMENO_API_KEY")),
		openai.WithModel(env("RIMENO_MODEL", "gpt-4o-mini")),
	)

	add := rimeno.NewTool("add", "Add two integers",
		func(ctx context.Context, in addArgs) (int, error) { return in.A + in.B, nil })

	agent, err := rimeno.New(rimeno.Config{
		Model:        model,
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
