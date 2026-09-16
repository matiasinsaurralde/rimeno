package rimeno

import (
	"context"
	"encoding/json"
	"time"

	"github.com/matiasinsaurralde/rimeno/schema"
)

type agentToolArgs struct {
	Task string `json:"task" jsonschema:"description=The task or question to delegate. Give complete, standalone instructions; the subagent cannot see this conversation."`
}

// AgentTool exposes a subagent as a [Tool] the parent can call to delegate a
// bounded task. The subagent runs in its own [Session] (its own context window,
// its own model, its own tools), under a budget derived from the parent's
// remaining allowance. Its trace nests into the parent's trace and its usage
// rolls up into the parent's budget and the parent trace's [Summary].
//
// This is the core multi-agent primitive: an orchestrator agent (model A) can be
// given AgentTools wrapping worker agents (model B), and deep graphs fall out
// because a worker's own Tools may include further AgentTools. Combine with
// Config.ParallelTools to fan out to several subagents concurrently (goroutines).
//
//	worker, _ := rimeno.New(rimeno.Config{Model: modelB, Instructions: "..."} )
//	boss, _ := rimeno.New(rimeno.Config{
//	    Model: modelA,
//	    Tools: []rimeno.Tool{rimeno.AgentTool(worker, "research", "Delegate research")},
//	    ParallelTools: true,
//	})
func AgentTool(sub *Agent, name, description string) Tool {
	return &agentTool{
		sub:         sub,
		name:        name,
		description: description,
		schema:      schema.MustOf[agentToolArgs](),
	}
}

type agentTool struct {
	sub         *Agent
	name        string
	description string
	schema      json.RawMessage
}

func (t *agentTool) Name() string                      { return t.name }
func (t *agentTool) Description() string               { return t.description }
func (t *agentTool) ParametersSchema() json.RawMessage { return t.schema }

// Invoke is the standalone fallback used when the tool runs outside a parent loop
// (e.g. directly in tests). Inside a run, the loop calls invokeSub instead.
func (t *agentTool) Invoke(ctx context.Context, args json.RawMessage) (any, error) {
	task, err := parseTask(args)
	if err != nil {
		return nil, &ToolError{Tool: t.name, Err: err}
	}
	res, err := t.sub.Run(ctx, task)
	if err != nil {
		return "subagent error: " + err.Error(), nil
	}
	return res.Text, nil
}

// invokeSub integrates the subagent with the parent run: hierarchical (reserved)
// budget, trace grafting, and lifecycle events. The returned bool reports whether
// the subagent failed, so the parent loop can count it toward the tool circuit
// breaker.
func (t *agentTool) invokeSub(ctx context.Context, args json.RawMessage, rc *runContext) (string, bool) {
	task, err := parseTask(args)
	if err != nil {
		return "error: " + err.Error(), true
	}
	rc.emit(SubAgentStartedEvent{EventBase{Agent: t.sub.name, Time: time.Now()}, t.sub.name})

	childBudget, grant := rc.budget.childBudget(t.sub.budget)
	sess := t.sub.NewSession()
	res, runErr := sess.runWith(ctx, UserMessage(task), childBudget)

	var usage Usage
	if res != nil {
		if res.Trace != nil && res.Trace.Root != nil {
			res.Trace.Root.Name = t.sub.name + ":subagent"
			rc.trace.graft(rc.currentSpan, res.Trace.Root)
		}
		usage = res.Usage
		rc.emit(SubAgentFinishedEvent{EventBase{Agent: t.sub.name, Time: time.Now()}, t.sub.name, usage})
	}
	// Always settle: record actual usage and release the reservation, even on error.
	rc.budget.settleChild(grant, usage)
	if runErr != nil {
		return "subagent error: " + runErr.Error(), true
	}
	return res.Text, false
}

func parseTask(args json.RawMessage) (string, error) {
	var in agentToolArgs
	if len(args) > 0 && string(args) != "null" {
		if err := json.Unmarshal(args, &in); err != nil {
			return "", err
		}
	}
	return in.Task, nil
}
