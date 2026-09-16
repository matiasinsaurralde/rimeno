package rimeno

import (
	"context"
	"fmt"
	"time"
)

// DefaultMaxSteps caps the number of model/tool loop iterations per run when
// Config.MaxSteps is zero.
const DefaultMaxSteps = 32

// ApprovalDecision is returned by an [ApprovalFunc].
type ApprovalDecision int

const (
	// ApprovalAllow permits the tool call.
	ApprovalAllow ApprovalDecision = iota
	// ApprovalDeny blocks the tool call; the reason is fed back to the model.
	ApprovalDeny
)

// ApprovalRequest is presented to an [ApprovalFunc] before a tool runs.
type ApprovalRequest struct {
	AgentName string
	Call      ToolCall
	Tool      Tool
}

// ApprovalFunc is an optional hook consulted before each tool invocation. It is
// the Go analog of an app-server "requestApproval": returning ApprovalDeny (with
// an optional reason) blocks the call. The hook may block to wait for a human.
type ApprovalFunc func(ctx context.Context, req ApprovalRequest) (ApprovalDecision, string)

// Config configures an [Agent]. The zero value is not valid (Model is required);
// every other field has a sensible default.
type Config struct {
	// Model is the provider used for completions. Required.
	Model Model
	// Instructions is the system prompt prepended to every session.
	Instructions string
	// Tools the agent may call.
	Tools []Tool
	// Output, if set, requires and validates a structured result (see OutputOf).
	Output *OutputSpec
	// OutputRepairAttempts is how many times to ask the model to fix output that
	// fails schema validation before giving up. Applies only when Output is set;
	// defaults to 1.
	OutputRepairAttempts int
	// Budget bounds resource use for each run.
	Budget Budget
	// MaxSteps caps loop iterations per run (default DefaultMaxSteps).
	MaxSteps int
	// Compactor manages context-window compaction (nil disables it).
	Compactor Compactor
	// Tokenizer, if set, is used to estimate token counts for compaction and
	// budgeting (default: a ~4-chars/token heuristic).
	Tokenizer Tokenizer
	// Approve, if set, gates every tool call.
	Approve ApprovalFunc
	// OnEvent receives run events synchronously (keep handlers cheap).
	OnEvent func(Event)
	// Temperature overrides the provider default when non-nil.
	Temperature *float64
	// MaxTokens caps output tokens per model call (0 = provider default).
	MaxTokens int
	// ReasoningEffort, for reasoning models, hints how hard to think
	// ("minimal"|"low"|"medium"|"high"); empty leaves the provider default.
	ReasoningEffort string
	// ReasoningMaxTokens caps a reasoning model's thinking budget per call, so a
	// small MaxTokens can't be entirely consumed by reasoning (0 = unbounded).
	ReasoningMaxTokens int
	// ResponsesThreading, when the model supports it (see ResponseThreader — the
	// openai provider does under WithResponsesAPI), makes a Session continue each turn
	// server-side: only the new input is sent, with the prior response id, reusing the
	// model's reasoning across tool loops. Ignored by models that can't thread.
	ResponsesThreading bool
	// ToolChoice controls tool selection (default provider "auto").
	ToolChoice ToolChoice
	// ParallelTools runs multiple tool calls requested in a single turn
	// concurrently (each as a goroutine). Order of results is preserved. This is
	// how a parent fans out to multiple subagents in parallel.
	ParallelTools bool
	// MaxParallelTools caps concurrency when ParallelTools is set (default 8).
	MaxParallelTools int
	// MaxToolErrors, if > 0, is a per-tool circuit breaker: once a tool has failed
	// this many times in a single run, further calls to it are refused (with a
	// message telling the model to stop) instead of invoked. This bounds a model
	// that thrashes retrying a broken tool — most usefully a dead subagent in a
	// fan-out — before it burns the run's budget. Default 0 (disabled).
	MaxToolErrors int
	// Stream uses the provider's streaming API when it implements StreamingModel,
	// emitting TextDeltaEvent as assistant text arrives. Falls back to a single
	// Generate call for non-streaming models. Tool calls are handled from the
	// assembled final response either way.
	Stream bool
	// Name labels the agent in events and traces (default "agent").
	Name string
}

// Agent is a configured, reusable harness. It is safe for concurrent use: each
// call to Run or NewSession is independent. Create sessions for multi-turn
// conversations; use Run for one-shot tasks.
type Agent struct {
	model              Model
	instructions       string
	tools              map[string]Tool
	toolDefs           []ToolDef
	output             *OutputSpec
	outputRepairs      int
	budget             Budget
	maxSteps           int
	compactor          Compactor
	tokenizer          Tokenizer
	approve            ApprovalFunc
	onEvent            func(Event)
	temperature        *float64
	maxTokens          int
	reasoningEffort    string
	reasoningMaxTokens int
	responsesThreading bool
	toolChoice         ToolChoice
	parallelTools      bool
	maxParallelTools   int
	maxToolErrors      int
	stream             bool
	name               string
}

// New validates cfg and returns an Agent.
func New(cfg Config) (*Agent, error) {
	if cfg.Model == nil {
		return nil, ErrNoModel
	}
	a := &Agent{
		model:              cfg.Model,
		instructions:       cfg.Instructions,
		tools:              make(map[string]Tool, len(cfg.Tools)),
		output:             cfg.Output,
		outputRepairs:      cfg.OutputRepairAttempts,
		budget:             cfg.Budget,
		maxSteps:           cfg.MaxSteps,
		compactor:          cfg.Compactor,
		tokenizer:          cfg.Tokenizer,
		approve:            cfg.Approve,
		onEvent:            cfg.OnEvent,
		temperature:        cfg.Temperature,
		maxTokens:          cfg.MaxTokens,
		reasoningEffort:    cfg.ReasoningEffort,
		reasoningMaxTokens: cfg.ReasoningMaxTokens,
		responsesThreading: cfg.ResponsesThreading,
		toolChoice:         cfg.ToolChoice,
		parallelTools:      cfg.ParallelTools,
		maxParallelTools:   cfg.MaxParallelTools,
		maxToolErrors:      cfg.MaxToolErrors,
		stream:             cfg.Stream,
		name:               cfg.Name,
	}
	for _, t := range cfg.Tools {
		if t == nil {
			return nil, fmt.Errorf("rimeno: nil tool in Config.Tools")
		}
		if _, dup := a.tools[t.Name()]; dup {
			return nil, fmt.Errorf("rimeno: duplicate tool name %q", t.Name())
		}
		a.tools[t.Name()] = t
		a.toolDefs = append(a.toolDefs, ToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.ParametersSchema(),
		})
	}
	if a.maxSteps == 0 {
		a.maxSteps = DefaultMaxSteps
	}
	if a.output != nil && a.outputRepairs == 0 {
		a.outputRepairs = 1
	}
	if a.name == "" {
		a.name = "agent"
	}
	return a, nil
}

// Run executes a one-shot task in a fresh session and returns the result.
func (a *Agent) Run(ctx context.Context, input string) (*Result, error) {
	return a.NewSession().Send(ctx, input)
}

func (a *Agent) emit(e Event) {
	if a.onEvent != nil {
		a.onEvent(e)
	}
}

func (a *Agent) base() EventBase { return EventBase{Agent: a.name, Time: time.Now()} }

func (a *Agent) maxParallel() int {
	if a.maxParallelTools > 0 {
		return a.maxParallelTools
	}
	return 8
}
