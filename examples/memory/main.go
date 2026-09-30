// Command memory demonstrates the rimeno/memory layer: an agent that keeps
// durable, structured memory across turns and across process restarts, with the
// memory tools wired in and a memory-aware compactor so facts survive
// compaction.
//
// Run (reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment or a .env file):
//
//	OPENAI_API_KEY=sk-... go run ./examples/memory
//
// It uses a durable file store under a temp dir, so a second run of the same
// process (or a real restart pointed at the same dir) recalls what the first run
// remembered.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/dotenv"
	"github.com/matiasinsaurralde/rimeno/memory"
	"github.com/matiasinsaurralde/rimeno/openai"
)

// model is hardcoded for the example; swap it for whatever your endpoint serves.
const model = "gpt-4o-mini"

func main() {
	// Load .env for local dev (never overrides variables already set).
	_, _ = dotenv.Load()

	ctx := context.Background()

	// A durable, JSON-per-namespace store. Point it anywhere you want memory to
	// persist; here we use a stable temp dir so re-running the example recalls
	// prior facts.
	dir := filepath.Join(os.TempDir(), "rimeno-memory-example")
	store, err := memory.NewFileStore(dir)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("memory dir:", dir)

	// Scope the handle to a task. Everything the agent remembers lands here and is
	// isolated from other namespaces.
	mem := memory.New(store, "trip-planning", memory.WithSource("assistant"))

	// Show what (if anything) we already know from a previous run.
	if prior, _ := mem.Recall(ctx, memory.Query{}); len(prior) > 0 {
		fmt.Println("\nrecalled from a previous run:")
		for _, r := range prior {
			fmt.Printf("  - [%s] %s\n", r.Kind, r.Text)
		}
	}

	m := openai.New(
		openai.WithBaseURL(env("OPENAI_BASE_URL", openai.DefaultBaseURL)),
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithModel(model),
	)

	// Give the model the memory tools (remember / recall / note / update_todo /
	// forget) and a memory-aware compactor so stored facts are re-injected if the
	// conversation is ever compacted.
	inner := &rimeno.SummarizingCompactor{Model: m, MaxContextTokens: 128_000}
	agent, err := rimeno.New(rimeno.Config{
		Model: m,
		Instructions: "You are a travel-planning assistant. When the user tells you a " +
			"durable preference or fact, call `remember` to store it. Before answering, " +
			"call `recall` to check what you already know about the user.",
		Tools:     mem.Tools(),
		Compactor: memory.Compactor(mem, inner),
	})
	if err != nil {
		log.Fatal(err)
	}

	// Turn 1: tell it something worth remembering.
	res, err := agent.Run(ctx, "I'm vegetarian and I hate red-eye flights. Remember that. Then suggest a destination.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("\nassistant:", res.Text)

	// Show the store contents after the run.
	recs, _ := mem.Recall(ctx, memory.Query{})
	fmt.Printf("\nmemory now holds %d record(s):\n", len(recs))
	for _, r := range recs {
		fmt.Printf("  - [%s] %s\n", r.Kind, r.Text)
	}

	// A budgeted preamble is what you'd inject at the top of a fresh session to
	// ground the model in prior knowledge without dumping the whole store.
	preamble, _ := mem.Preamble(ctx, memory.Query{Limit: 5}, 512)
	if preamble != "" {
		fmt.Println("\ninjectable preamble for next session:")
		fmt.Println(preamble)
	}

	fmt.Println("\nrun again and it will recall these facts from disk.")
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
