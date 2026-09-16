package sandbox

import (
	"context"

	"github.com/matiasinsaurralde/rimeno"
)

type commandArgs struct {
	Command string `json:"command" jsonschema:"description=the shell command to run"`
	Stdin   string `json:"stdin,omitempty" jsonschema:"description=optional data for the command's standard input"`
}

// CommandOutput is the structured result the model receives from [CommandTool].
type CommandOutput struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	ExitCode int    `json:"exit_code"`
}

// CommandTool exposes a [Sandbox] to the model as a "run_command" tool: the model
// supplies a shell command, which runs as `sh -c <command>` inside the sandbox,
// and receives the stdout/stderr/exit code. The host owns the sandbox's lifecycle
// (create it, build the tool, run the agent, close it) and should gate the tool
// with rimeno's approval hook when the sandbox is not fully isolated.
//
//	sb, _ := sandbox.LocalProvider{}.Create(ctx, sandbox.Spec{WorkDir: "/work"})
//	defer sb.Close()
//	agent, _ := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{sandbox.CommandTool(sb)}})
func CommandTool(sb Sandbox) rimeno.Tool {
	return rimeno.NewTool("run_command",
		"Run a shell command in the sandbox and return its stdout, stderr, and exit code.",
		func(ctx context.Context, in commandArgs) (CommandOutput, error) {
			res, err := sb.Exec(ctx, ExecRequest{
				Argv:  []string{"sh", "-c", in.Command},
				Stdin: in.Stdin,
			})
			if err != nil {
				// A run failure (timeout, spawn error, policy denial) is fed back to
				// the model as a recoverable tool error.
				return CommandOutput{}, err
			}
			return CommandOutput(res), nil
		})
}
