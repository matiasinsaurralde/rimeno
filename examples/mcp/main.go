// Command mcp is a rimeno example that connects to an external Model Context
// Protocol (MCP) server, exposes its tools to a rimeno agent, and runs a prompt.
//
// It needs two things: an MCP server command and an OpenAI-compatible model.
//
//	# Example: the reference filesystem server over npx, and an OpenAI model.
//	# Credentials come from OPENAI_API_KEY / OPENAI_BASE_URL (env or a .env file).
//	OPENAI_API_KEY=sk-... \
//	  go run ./examples/mcp -- npx -y @modelcontextprotocol/server-filesystem /tmp
//
// Everything after `--` is the MCP server command and its arguments.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/dotenv"
	"github.com/matiasinsaurralde/rimeno/mcp"
	"github.com/matiasinsaurralde/rimeno/openai"
)

// modelID is hardcoded for the example; swap it for whatever your endpoint serves.
const modelID = "gpt-4o-mini"

func main() {
	// Load .env for local dev (never overrides variables already set).
	_, _ = dotenv.Load()

	cmd, args := serverCommand()
	if cmd == "" {
		log.Fatal("usage: go run ./examples/mcp -- <mcp-server-command> [args...]")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	// 1. Start the MCP server and complete the handshake.
	client, err := mcp.NewStdioClient(cmd, args, mcp.WithToolNamePrefix("mcp_"))
	if err != nil {
		log.Fatalf("start MCP server %q: %v", cmd, err)
	}
	defer client.Close()

	info, err := client.Connect(ctx)
	if err != nil {
		log.Fatalf("MCP handshake: %v", err)
	}
	fmt.Printf("connected to %s v%s\n", info.ServerInfo.Name, info.ServerInfo.Version)

	// 2. Pull the server's tools as rimeno tools.
	tools, err := client.Tools(ctx)
	if err != nil {
		log.Fatalf("list tools: %v", err)
	}
	fmt.Printf("discovered %d tool(s):\n", len(tools))
	for _, t := range tools {
		fmt.Printf("  - %s: %s\n", t.Name(), t.Description())
	}

	// 3. Hand them to an agent.
	model := openai.New(
		openai.WithBaseURL(env("OPENAI_BASE_URL", openai.DefaultBaseURL)),
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithModel(modelID),
	)
	agent, err := rimeno.New(rimeno.Config{
		Model:        model,
		Instructions: "You are a helpful assistant. Use the available tools when relevant.",
		Tools:        tools,
	})
	if err != nil {
		log.Fatal(err)
	}

	const prompt = "What tools do you have, and what can they do?"
	res, err := agent.Run(ctx, prompt)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("\n" + res.Text)

	s := res.Trace.Summary()
	fmt.Printf("\nmodel_calls=%d tool_calls=%d tokens=%d\n", s.ModelCalls, s.ToolCalls, s.Usage.TotalTokens)
}

// serverCommand returns the MCP server command and args given after "--".
func serverCommand() (string, []string) {
	for i, a := range os.Args {
		if a == "--" && i+1 < len(os.Args) {
			rest := os.Args[i+1:]
			return rest[0], rest[1:]
		}
	}
	return "", nil
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
