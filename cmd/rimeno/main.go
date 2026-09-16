// Command rimeno is a small CLI for exercising the rimeno harness: a zero-config
// `demo` (fake model, shows tool-calling + tracing), a `run` against any
// OpenAI-compatible endpoint, and a `workflow` demo.
//
// Usage:
//
//	rimeno demo
//	RIMENO_API_KEY=... RIMENO_MODEL=gpt-4o-mini rimeno run -p "what is 21+21?" -demo-tools -trace
//	rimeno workflow -p "the Go programming language"
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/dotenv"
	"github.com/matiasinsaurralde/rimeno/openai"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
	"github.com/matiasinsaurralde/rimeno/rpc"
	"github.com/matiasinsaurralde/rimeno/workflow"

	"flag"
)

func main() {
	// Load .env for local dev (never overrides variables already set).
	if p, _ := dotenv.Load(); p != "" {
		fmt.Fprintf(os.Stderr, "rimeno: loaded env from %s\n", p)
	}
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "demo":
		err = cmdDemo(os.Args[2:])
	case "run":
		err = cmdRun(os.Args[2:])
	case "workflow":
		err = cmdWorkflow(os.Args[2:])
	case "serve":
		err = cmdServe(os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "rimeno: unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "rimeno:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `rimeno — embeddable Go agent harness (CLI for testing)

Commands:
  demo                Run a scripted demo with a fake model (no API key needed).
  run       -p TEXT   Run a prompt against an OpenAI-compatible endpoint.
  workflow  -p TEXT   Run a demo multi-agent workflow.
  serve               Serve rimeno over JSON-RPC 2.0 on stdio (embed from any host).

run/workflow read credentials from flags or env:
  -model / RIMENO_MODEL / OPENAI_MODEL
  -base-url / RIMENO_BASE_URL / OPENAI_BASE_URL   (default `+openai.DefaultBaseURL+`)
  -api-key via RIMENO_API_KEY / OPENAI_API_KEY

Examples:
  rimeno demo
  RIMENO_MODEL=gpt-4o-mini rimeno run -p "what is 21+21?" -demo-tools -trace
`)
}

// --- demo -------------------------------------------------------------------

func cmdDemo(args []string) error {
	fs := flag.NewFlagSet("demo", flag.ExitOnError)
	verbose := fs.Bool("v", true, "print the event stream")
	_ = fs.Parse(args)

	// A scripted model: first turn calls the `add` tool, second turn answers.
	model := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", map[string]int{"a": 20, "b": 22})}},
		rimenotest.Turn{Text: "The sum of 20 and 22 is 42."},
	).WithID("demo-model")

	agent, err := rimeno.New(rimeno.Config{
		Model:        model,
		Name:         "demo",
		Instructions: "You are a calculator. Use the add tool.",
		Tools:        []rimeno.Tool{demoAddTool()},
		OnEvent:      eventPrinter(*verbose),
	})
	if err != nil {
		return err
	}
	res, err := agent.Run(context.Background(), "what is 20 + 22?")
	if err != nil {
		return err
	}
	fmt.Println("\nAnswer:", res.Text)
	printTrace(res.Trace)
	return nil
}

// --- run --------------------------------------------------------------------

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	prompt := fs.String("p", "", "prompt text")
	file := fs.String("f", "", "read prompt from file")
	model := fs.String("model", "", "model id (or RIMENO_MODEL/OPENAI_MODEL)")
	baseURL := fs.String("base-url", "", "API base URL (or RIMENO_BASE_URL/OPENAI_BASE_URL)")
	system := fs.String("system", "", "system instructions")
	demoTools := fs.Bool("demo-tools", false, "register built-in demo tools (add, now)")
	trace := fs.Bool("trace", false, "print trace summary")
	verbose := fs.Bool("v", false, "print the event stream")
	stream := fs.Bool("stream", false, "stream assistant text as it arrives")
	outputJSON := fs.Bool("json", false, "print result as JSON")
	maxSteps := fs.Int("max-steps", 0, "max loop steps (0 = default)")
	timeout := fs.Duration("timeout", 2*time.Minute, "overall timeout")
	saveTrace := fs.String("save-trace", "", "write the run trace JSON to a file")
	_ = fs.Parse(args)

	text, err := promptText(*prompt, *file)
	if err != nil {
		return err
	}

	m, err := modelFromEnv(*model, *baseURL)
	if err != nil {
		return err
	}
	cfg := rimeno.Config{
		Model:        m,
		Instructions: *system,
		MaxSteps:     *maxSteps,
		Stream:       *stream,
		OnEvent:      runEventHandler(*verbose, *stream),
	}
	if *demoTools {
		cfg.Tools = []rimeno.Tool{demoAddTool(), demoNowTool()}
	}
	agent, err := rimeno.New(cfg)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	res, err := agent.Run(ctx, text)
	if err != nil {
		return err
	}
	if *saveTrace != "" {
		b, jerr := res.Trace.JSON()
		if jerr != nil {
			return jerr
		}
		if werr := os.WriteFile(*saveTrace, b, 0o600); werr != nil {
			return werr
		}
		fmt.Fprintf(os.Stderr, "trace written to %s\n", *saveTrace)
	}
	if *outputJSON {
		return printJSON(res)
	}
	if *stream {
		fmt.Println() // deltas already printed the answer
	} else {
		fmt.Println(res.Text)
	}
	if *trace {
		printTrace(res.Trace)
	}
	return nil
}

// runEventHandler prints streamed text deltas to stdout (when stream) and tool/
// subagent activity to stderr (when verbose).
func runEventHandler(verbose, stream bool) func(rimeno.Event) {
	if !verbose && !stream {
		return nil
	}
	inner := eventPrinter(verbose)
	return func(e rimeno.Event) {
		if stream {
			if d, ok := e.(rimeno.TextDeltaEvent); ok {
				fmt.Print(d.Delta)
			}
		}
		if inner != nil {
			inner(e)
		}
	}
}

// --- workflow ---------------------------------------------------------------

func cmdWorkflow(args []string) error {
	fs := flag.NewFlagSet("workflow", flag.ExitOnError)
	prompt := fs.String("p", "the Go programming language", "workflow input")
	model := fs.String("model", "", "model id (or RIMENO_MODEL/OPENAI_MODEL)")
	baseURL := fs.String("base-url", "", "API base URL")
	_ = fs.Parse(args)

	// Use a real model if credentials are present; otherwise a fake for offline demo.
	var mk func(name string) *rimeno.Agent
	if hasCredentials() {
		m, err := modelFromEnv(*model, *baseURL)
		if err != nil {
			return err
		}
		mk = func(name string) *rimeno.Agent {
			a, _ := rimeno.New(rimeno.Config{Model: m, Name: name})
			return a
		}
	} else {
		fmt.Fprintln(os.Stderr, "(no credentials found — using a fake model for an offline demo)")
		mk = func(name string) *rimeno.Agent {
			a, _ := rimeno.New(rimeno.Config{Model: rimenotest.NewModel(rimenotest.Turn{Text: name + " output"}).WithID(name), Name: name})
			return a
		}
	}

	planner := mk("planner")
	research := mk("researcher")
	writer := mk("writer")

	wf := workflow.New("research").
		Step(workflow.AgentStep("plan", planner, func(s *workflow.State) string {
			return "Make a short research plan for: " + s.Input
		})).
		Step(workflow.AgentStep("research", research, func(s *workflow.State) string {
			return "Research according to this plan:\n" + s.Get("plan")
		}, "plan")).
		Step(workflow.AgentStep("write", writer, func(s *workflow.State) string {
			return "Write a concise summary using this research:\n" + s.Get("research")
		}, "research"))

	res, err := wf.Run(context.Background(), *prompt)
	if err != nil {
		return err
	}
	for name, out := range res.Outputs {
		fmt.Printf("── %s ──\n%s\n\n", name, out)
	}
	fmt.Printf("aggregated usage: %+v\n", res.Usage)
	return nil
}

// --- serve ------------------------------------------------------------------

// cmdServe runs the JSON-RPC server on stdio so a non-Go host can embed rimeno.
// With -demo it uses a fake echo model (no API key); otherwise it builds a real
// OpenAI-compatible model per session from flags/env.
func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	model := fs.String("model", "", "model id (or RIMENO_MODEL/OPENAI_MODEL)")
	baseURL := fs.String("base-url", "", "API base URL")
	system := fs.String("system", "", "default system instructions")
	demoTools := fs.Bool("demo-tools", false, "register built-in demo tools (add, now)")
	demo := fs.Bool("demo", false, "use a fake echo model (no API key needed)")
	_ = fs.Parse(args)

	factory := func(_ context.Context, p rpc.NewSessionParams) (rimeno.Config, error) {
		var m rimeno.Model
		if *demo {
			m = echoModel()
		} else {
			rm, err := modelFromEnv(*model, *baseURL)
			if err != nil {
				return rimeno.Config{}, err
			}
			m = rm
		}
		instructions := *system
		if p.Instructions != "" { // let the client override per session
			instructions = p.Instructions
		}
		cfg := rimeno.Config{Model: m, Instructions: instructions}
		if *demoTools {
			cfg.Tools = []rimeno.Tool{demoAddTool(), demoNowTool()}
		}
		return cfg, nil
	}

	srv := rpc.NewServer(factory, rpc.WithServerInfo("rimeno", "0.1.0"))
	fmt.Fprintln(os.Stderr, "rimeno rpc: serving JSON-RPC 2.0 on stdio "+
		"(methods: initialize, ping, session/new, session/prompt, session/close)")
	return srv.ServeStdio(context.Background())
}

// echoModel is a fake model that echoes the last user message, for exercising the
// RPC protocol without a provider.
func echoModel() rimeno.Model {
	m := rimenotest.NewModel()
	m.GenerateFn = func(_ context.Context, req *rimeno.Request, _ int) (*rimeno.Response, error) {
		var last string
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == rimeno.RoleUser {
				last = req.Messages[i].Text
				break
			}
		}
		return &rimeno.Response{
			Message:    rimeno.Message{Role: rimeno.RoleAssistant, Text: "echo: " + last},
			Usage:      rimeno.Usage{InputTokens: len(last) / 4, OutputTokens: 1, TotalTokens: len(last)/4 + 1},
			StopReason: rimeno.StopReasonStop,
		}, nil
	}
	return m
}

// --- helpers ----------------------------------------------------------------

func modelFromEnv(model, baseURL string) (rimeno.Model, error) {
	if model == "" {
		model = firstEnv("RIMENO_MODEL", "OPENAI_MODEL")
	}
	if model == "" {
		return nil, fmt.Errorf("no model set (use -model or RIMENO_MODEL)")
	}
	if baseURL == "" {
		baseURL = firstEnv("RIMENO_BASE_URL", "OPENAI_BASE_URL")
	}
	if baseURL == "" {
		baseURL = openai.DefaultBaseURL
	}
	apiKey := firstEnv("RIMENO_API_KEY", "OPENAI_API_KEY")
	return openai.New(
		openai.WithModel(model),
		openai.WithBaseURL(baseURL),
		openai.WithAPIKey(apiKey),
	), nil
}

func hasCredentials() bool {
	return firstEnv("RIMENO_MODEL", "OPENAI_MODEL") != "" &&
		firstEnv("RIMENO_API_KEY", "OPENAI_API_KEY") != ""
}

func firstEnv(keys ...string) string {
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}

func promptText(prompt, file string) (string, error) {
	if file != "" {
		b, err := os.ReadFile(file) // #nosec G304 -- file is the prompt path the user passed on the CLI
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
	if prompt == "" {
		return "", fmt.Errorf("provide a prompt with -p or -f")
	}
	return prompt, nil
}

func eventPrinter(verbose bool) func(rimeno.Event) {
	if !verbose {
		return nil
	}
	return func(e rimeno.Event) {
		switch ev := e.(type) {
		case rimeno.ToolCallStartedEvent:
			fmt.Fprintf(os.Stderr, "  → tool %s(%s)\n", ev.Call.Name, string(ev.Call.Arguments))
		case rimeno.ToolCallFinishedEvent:
			fmt.Fprintf(os.Stderr, "  ← tool %s => %s\n", ev.Call.Name, truncate(ev.Result, 80))
		case rimeno.SubAgentStartedEvent:
			fmt.Fprintf(os.Stderr, "  ⇢ subagent %s\n", ev.Name)
		case rimeno.CompactionStartedEvent:
			fmt.Fprintf(os.Stderr, "  ⟳ compacting (%d msgs, ~%d tokens)\n", ev.Messages, ev.EstimatedTokens)
		}
	}
}

func printTrace(tr *rimeno.Trace) {
	if tr == nil {
		return
	}
	s := tr.Summary()
	fmt.Printf("\n── trace summary ──\n")
	fmt.Printf("wallclock=%s steps=%d model_calls=%d tool_calls=%d subagents=%d\n",
		s.Wallclock.Round(time.Millisecond), s.Steps, s.ModelCalls, s.ToolCalls, s.Subagents)
	fmt.Printf("tokens: in=%d out=%d total=%d cost=$%.4f\n",
		s.Usage.InputTokens, s.Usage.OutputTokens, s.Usage.TotalTokens, s.Usage.CostUSD)
	for tool, st := range s.ToolBreakdown {
		fmt.Printf("  tool %s: calls=%d errors=%d\n", tool, st.Calls, st.Errors)
	}
}

func printJSON(res *rimeno.Result) error {
	s := res.Trace.Summary()
	out := map[string]any{
		"text":        res.Text,
		"steps":       res.Steps,
		"stop_reason": res.StopReason,
		"usage":       res.Usage,
		"summary":     s,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// --- demo tools -------------------------------------------------------------

type addArgs struct {
	A float64 `json:"a" jsonschema:"description=first addend"`
	B float64 `json:"b" jsonschema:"description=second addend"`
}

func demoAddTool() rimeno.Tool {
	return rimeno.NewTool("add", "Add two numbers", func(_ context.Context, in addArgs) (float64, error) {
		return in.A + in.B, nil
	})
}

func demoNowTool() rimeno.Tool {
	return rimeno.NewTool("now", "Return the current time (RFC3339)", func(_ context.Context, _ struct{}) (string, error) {
		return time.Now().Format(time.RFC3339), nil
	})
}
