package sandbox_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
	"github.com/matiasinsaurralde/rimeno/sandbox"
)

func newLocal(t *testing.T, p sandbox.LocalProvider) sandbox.Sandbox {
	t.Helper()
	sb, err := p.Create(context.Background(), sandbox.Spec{})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	t.Cleanup(func() { _ = sb.Close() })
	return sb
}

func TestLocal_Exec(t *testing.T) {
	sb := newLocal(t, sandbox.LocalProvider{})
	res, err := sb.Exec(context.Background(), sandbox.ExecRequest{Argv: []string{"echo", "hello"}})
	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if strings.TrimSpace(res.Stdout) != "hello" {
		t.Errorf("stdout = %q, want hello", res.Stdout)
	}
	if res.ExitCode != 0 {
		t.Errorf("exit = %d, want 0", res.ExitCode)
	}
}

func TestLocal_NonZeroExit(t *testing.T) {
	sb := newLocal(t, sandbox.LocalProvider{})
	res, err := sb.Exec(context.Background(), sandbox.ExecRequest{Argv: []string{"sh", "-c", "exit 3"}})
	if err != nil {
		t.Fatalf("exec returned Go error for non-zero exit: %v", err)
	}
	if res.ExitCode != 3 {
		t.Errorf("exit = %d, want 3", res.ExitCode)
	}
}

func TestLocal_Stdin(t *testing.T) {
	sb := newLocal(t, sandbox.LocalProvider{})
	res, err := sb.Exec(context.Background(), sandbox.ExecRequest{Argv: []string{"cat"}, Stdin: "piped in"})
	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if res.Stdout != "piped in" {
		t.Errorf("stdout = %q, want %q", res.Stdout, "piped in")
	}
}

func TestLocal_Allowlist(t *testing.T) {
	sb := newLocal(t, sandbox.LocalProvider{AllowedCommands: []string{"echo"}})
	if _, err := sb.Exec(context.Background(), sandbox.ExecRequest{Argv: []string{"echo", "ok"}}); err != nil {
		t.Errorf("allowed command errored: %v", err)
	}
	if _, err := sb.Exec(context.Background(), sandbox.ExecRequest{Argv: []string{"ls"}}); err == nil {
		t.Error("expected disallowed command to error")
	}
}

func TestLocal_Timeout(t *testing.T) {
	sb := newLocal(t, sandbox.LocalProvider{})
	_, err := sb.Exec(context.Background(), sandbox.ExecRequest{
		Argv:    []string{"sh", "-c", "sleep 5"},
		Timeout: 50 * time.Millisecond,
	})
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("error = %v, want a timeout", err)
	}
}

func TestLocal_OutputCap(t *testing.T) {
	sb := newLocal(t, sandbox.LocalProvider{MaxOutputBytes: 16})
	res, err := sb.Exec(context.Background(), sandbox.ExecRequest{
		Argv: []string{"sh", "-c", "printf 'a%.0s' $(seq 1 1000)"},
	})
	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if !strings.Contains(res.Stdout, "truncated") {
		t.Errorf("expected truncation marker, got %d bytes", len(res.Stdout))
	}
	if len(res.Stdout) > 100 {
		t.Errorf("output not capped: %d bytes", len(res.Stdout))
	}
}

func TestLocal_EmptyArgv(t *testing.T) {
	sb := newLocal(t, sandbox.LocalProvider{})
	if _, err := sb.Exec(context.Background(), sandbox.ExecRequest{}); err == nil {
		t.Error("expected error for empty argv")
	}
}

// TestCommandTool_InAgent proves the sandbox tool works inside a real agent run:
// the model calls run_command, rimeno executes it in the sandbox, and the output
// flows back.
func TestCommandTool_InAgent(t *testing.T) {
	sb := newLocal(t, sandbox.LocalProvider{})
	model := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{
			rimenotest.ToolCall("c1", "run_command", map[string]string{"command": "echo sandboxed"}),
		}},
		rimenotest.Turn{Text: "the command printed sandboxed"},
	)
	agent, err := rimeno.New(rimeno.Config{Model: model, Tools: []rimeno.Tool{sandbox.CommandTool(sb)}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "run echo")
	if err != nil {
		t.Fatal(err)
	}
	var toolResult string
	for _, m := range res.Messages {
		if m.Role == rimeno.RoleTool {
			toolResult = m.Text
		}
	}
	if !strings.Contains(toolResult, "sandboxed") {
		t.Errorf("tool result = %q, want it to contain 'sandboxed'", toolResult)
	}
	if !strings.Contains(toolResult, "exit_code") {
		t.Errorf("tool result should be structured JSON with exit_code: %q", toolResult)
	}
}

func TestCommandTool_ErrorFeedsBack(t *testing.T) {
	// A denied command should surface to the model as a recoverable tool error.
	sb := newLocal(t, sandbox.LocalProvider{AllowedCommands: []string{"echo"}})
	model := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{
			rimenotest.ToolCall("c1", "run_command", map[string]string{"command": "whatever"}),
		}},
		rimenotest.Turn{Text: "handled"},
	)
	agent, _ := rimeno.New(rimeno.Config{Model: model, Tools: []rimeno.Tool{sandbox.CommandTool(sb)}})
	res, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("run should not fail on a recoverable tool error: %v", err)
	}
	if res.Text != "handled" {
		t.Errorf("text = %q", res.Text)
	}
}
