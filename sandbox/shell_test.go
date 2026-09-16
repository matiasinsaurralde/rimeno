package sandbox

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func newShell(t *testing.T, p PersistentShellProvider) Sandbox {
	t.Helper()
	sb, err := p.Create(context.Background(), Spec{WorkDir: t.TempDir()})
	if err != nil {
		t.Fatalf("create shell: %v", err)
	}
	t.Cleanup(func() { _ = sb.Close() })
	return sb
}

// run executes a shell command the way CommandTool does (Argv = sh -c ...).
func run(t *testing.T, sb Sandbox, cmd string) ExecResult {
	t.Helper()
	r, err := sb.Exec(context.Background(), ExecRequest{Argv: []string{"sh", "-c", cmd}})
	if err != nil {
		t.Fatalf("exec %q: %v", cmd, err)
	}
	return r
}

// TestPersistentShell_StatePersists is the core guarantee: cwd, exported vars, and
// shell vars set in one command are visible in the next.
func TestPersistentShell_StatePersists(t *testing.T) {
	sb := newShell(t, PersistentShellProvider{})

	if r := run(t, sb, "mkdir -p sub && cd sub"); r.ExitCode != 0 {
		t.Fatalf("cd setup exit=%d stderr=%q", r.ExitCode, r.Stderr)
	}
	if r := run(t, sb, "pwd"); !strings.HasSuffix(strings.TrimSpace(r.Stdout), "/sub") {
		t.Errorf("cwd not persisted: pwd=%q", strings.TrimSpace(r.Stdout))
	}
	run(t, sb, "export FOO=bar123")
	if r := run(t, sb, "echo $FOO"); strings.TrimSpace(r.Stdout) != "bar123" {
		t.Errorf("exported var not persisted: %q", strings.TrimSpace(r.Stdout))
	}
	run(t, sb, "X=42")
	if r := run(t, sb, `echo "val=$X"`); strings.TrimSpace(r.Stdout) != "val=42" {
		t.Errorf("shell var not persisted: %q", strings.TrimSpace(r.Stdout))
	}
}

// TestPersistentShell_StreamsAndExit checks stdout/stderr separation and exit code.
func TestPersistentShell_StreamsAndExit(t *testing.T) {
	sb := newShell(t, PersistentShellProvider{})
	// A real command's non-zero exit is captured via $? and the persistent shell
	// survives. (A literal `exit` builtin would end the shell — see the doc comment
	// and the self-heal path; here we use a subshell to get a specific code, as an
	// external program would.)
	r := run(t, sb, "echo out; echo err 1>&2; (exit 3)")
	if strings.TrimSpace(r.Stdout) != "out" {
		t.Errorf("stdout=%q, want out", r.Stdout)
	}
	if strings.TrimSpace(r.Stderr) != "err" {
		t.Errorf("stderr=%q, want err", r.Stderr)
	}
	if r.ExitCode != 3 {
		t.Errorf("exit=%d, want 3", r.ExitCode)
	}
	// The shell survives a failing command.
	if r := run(t, sb, "false; echo still-alive"); strings.TrimSpace(r.Stdout) != "still-alive" {
		t.Errorf("shell did not survive a failing command: %q", r.Stdout)
	}
}

// TestPersistentShell_Multiline handles a command spanning multiple lines / pipes.
func TestPersistentShell_Multiline(t *testing.T) {
	sb := newShell(t, PersistentShellProvider{})
	r := run(t, sb, "printf 'a\\nb\\nc\\n' | grep -c b")
	if strings.TrimSpace(r.Stdout) != "1" {
		t.Errorf("pipeline stdout=%q, want 1", r.Stdout)
	}
}

// TestPersistentShell_Stdin feeds stdin to a command.
func TestPersistentShell_Stdin(t *testing.T) {
	sb := newShell(t, PersistentShellProvider{})
	r, err := sb.Exec(context.Background(), ExecRequest{Argv: []string{"sh", "-c", "cat"}, Stdin: "hello-stdin"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(r.Stdout) != "hello-stdin" {
		t.Errorf("stdin not delivered: %q", r.Stdout)
	}
}

// TestPersistentShell_Timeout: a hung command times out, and the sandbox self-heals.
func TestPersistentShell_Timeout(t *testing.T) {
	sb := newShell(t, PersistentShellProvider{})
	_, err := sb.Exec(context.Background(), ExecRequest{
		Argv:    []string{"sh", "-c", "sleep 10"},
		Timeout: 300 * time.Millisecond,
	})
	if err == nil {
		t.Fatal("expected a timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("err=%v, want timeout", err)
	}
	// Self-heal: the next command runs in a fresh shell.
	if r := run(t, sb, "echo recovered"); strings.TrimSpace(r.Stdout) != "recovered" {
		t.Errorf("did not self-heal after timeout: %q", r.Stdout)
	}
}

// TestPersistentShell_OutputCap truncates oversized output.
func TestPersistentShell_OutputCap(t *testing.T) {
	sb := newShell(t, PersistentShellProvider{MaxOutputBytes: 200})
	r := run(t, sb, `awk 'BEGIN{for(i=0;i<1000;i++)print "LINE"i}'`)
	if !strings.Contains(r.Stdout, "truncated") {
		t.Errorf("expected truncation marker, got %d bytes", len(r.Stdout))
	}
	if len(r.Stdout) > 400 { // ~limit + marker, not the full ~8KB
		t.Errorf("output not capped: %d bytes", len(r.Stdout))
	}
}

// TestPersistentShell_CommandTool: cwd persists across two CommandTool invocations,
// proving the drop-in swap works end-to-end through the tool the model calls.
func TestPersistentShell_CommandTool(t *testing.T) {
	sb := newShell(t, PersistentShellProvider{})
	tool := CommandTool(sb)

	if _, err := tool.Invoke(context.Background(), json.RawMessage(`{"command":"mkdir -p deep/dir && cd deep/dir"}`)); err != nil {
		t.Fatalf("invoke 1: %v", err)
	}
	out, err := tool.Invoke(context.Background(), json.RawMessage(`{"command":"pwd"}`))
	if err != nil {
		t.Fatalf("invoke 2: %v", err)
	}
	co, ok := out.(CommandOutput)
	if !ok {
		t.Fatalf("unexpected tool output type %T", out)
	}
	if !strings.HasSuffix(strings.TrimSpace(co.Stdout), "/deep/dir") {
		t.Errorf("cwd not persisted across tool calls: %q", strings.TrimSpace(co.Stdout))
	}
}
