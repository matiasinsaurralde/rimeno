package sandbox

import (
	"context"
	"time"
)

// ExecRequest describes one command execution inside a [Sandbox].
type ExecRequest struct {
	// Argv is the command and its arguments (argv[0] is the program). Required.
	Argv []string
	// Stdin is fed to the command's standard input.
	Stdin string
	// Env are extra environment variables ("KEY=value"), appended to the
	// sandbox's base environment.
	Env []string
	// Dir overrides the working directory for this command.
	Dir string
	// Timeout bounds this command (0 = the provider default).
	Timeout time.Duration
}

// ExecResult is the outcome of an [ExecRequest]. A non-zero ExitCode is a normal
// result (the command ran and failed), not a Go error; Go errors are reserved for
// commands that could not run (spawn failure, timeout, policy denial).
type ExecResult struct {
	Stdout   string
	Stderr   string
	ExitCode int
}

// Sandbox is an isolated environment in which commands run. Implementations range
// from the unsandboxed [LocalProvider] (host exec, for dev) to container/microVM
// backends (Docker/Firecracker/e2b — see the package roadmap). A Sandbox may be
// reused for many Exec calls; Close releases it.
type Sandbox interface {
	Exec(ctx context.Context, req ExecRequest) (ExecResult, error)
	Close() error
}

// Spec requests a sandbox with a given image/workdir/environment. Fields that a
// particular provider cannot honor (e.g. Image on the local provider) are ignored.
type Spec struct {
	// Image is the container/microVM image (ignored by the local provider).
	Image string
	// WorkDir is the default working directory for commands.
	WorkDir string
	// Env is the base environment ("KEY=value") for every command.
	Env []string
}

// Provider creates sandboxes. Swapping the provider is how a program moves from
// unsandboxed local execution to real isolation without touching agent/tool code.
type Provider interface {
	Create(ctx context.Context, spec Spec) (Sandbox, error)
}
