package sandbox

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// LocalProvider creates sandboxes that run commands directly on the host with NO
// isolation. It is intended for development and trusted use; for untrusted code
// or production, plug in a container/microVM provider (Docker/Firecracker/e2b)
// implementing the same [Provider] interface. Pair any command execution with
// rimeno's approval hook for human-in-the-loop control.
type LocalProvider struct {
	// AllowedCommands, if non-empty, restricts argv[0] (by full path or basename)
	// to this set. Note: a shell tool that runs "sh -c ..." only needs "sh"
	// allowed, so the allowlist mainly constrains direct Exec callers — gate shell
	// tools with the approval hook instead.
	AllowedCommands []string
	// MaxOutputBytes caps captured stdout and stderr per command (default 1 MiB).
	MaxOutputBytes int
	// DefaultTimeout applies when ExecRequest.Timeout is zero (default 30s).
	DefaultTimeout time.Duration
}

// Create returns a local sandbox bound to the spec's workdir and environment.
func (p LocalProvider) Create(_ context.Context, spec Spec) (Sandbox, error) {
	return &localSandbox{provider: p, dir: spec.WorkDir, env: spec.Env}, nil
}

func (p LocalProvider) defaultTimeout() time.Duration {
	if p.DefaultTimeout > 0 {
		return p.DefaultTimeout
	}
	return 30 * time.Second
}

func (p LocalProvider) maxOutput() int {
	if p.MaxOutputBytes > 0 {
		return p.MaxOutputBytes
	}
	return 1 << 20
}

func (p LocalProvider) checkAllowed(cmd string) error {
	if len(p.AllowedCommands) == 0 {
		return nil
	}
	base := filepath.Base(cmd)
	for _, a := range p.AllowedCommands {
		if a == cmd || a == base {
			return nil
		}
	}
	return fmt.Errorf("sandbox: command %q not allowed", cmd)
}

type localSandbox struct {
	provider LocalProvider
	dir      string
	env      []string
}

func (s *localSandbox) Exec(ctx context.Context, req ExecRequest) (ExecResult, error) {
	if len(req.Argv) == 0 {
		return ExecResult{}, errors.New("sandbox: empty argv")
	}
	if err := s.provider.checkAllowed(req.Argv[0]); err != nil {
		return ExecResult{}, err
	}

	timeout := req.Timeout
	if timeout <= 0 {
		timeout = s.provider.defaultTimeout()
	}
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Running the caller-supplied argv is exactly what a sandbox does; callers
	// are responsible for the isolation policy around it.
	cmd := exec.CommandContext(cctx, req.Argv[0], req.Argv[1:]...) // #nosec G204
	if req.Dir != "" {
		cmd.Dir = req.Dir
	} else {
		cmd.Dir = s.dir
	}
	cmd.Env = append(append([]string(nil), s.env...), req.Env...)
	if req.Stdin != "" {
		cmd.Stdin = strings.NewReader(req.Stdin)
	}
	stdout := &cappedBuffer{limit: s.provider.maxOutput()}
	stderr := &cappedBuffer{limit: s.provider.maxOutput()}
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	err := cmd.Run()
	res := ExecResult{Stdout: stdout.String(), Stderr: stderr.String()}
	if cctx.Err() == context.DeadlineExceeded {
		return res, fmt.Errorf("sandbox: command timed out after %s", timeout)
	}
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			res.ExitCode = ee.ExitCode() // non-zero exit is a normal result
			return res, nil
		}
		return res, fmt.Errorf("sandbox: exec %q: %w", req.Argv[0], err)
	}
	return res, nil
}

func (s *localSandbox) Close() error { return nil }

// cappedBuffer accumulates output up to limit bytes, then discards the rest and
// notes the truncation. Writes never fail, so they don't abort the command's I/O.
type cappedBuffer struct {
	buf       bytes.Buffer
	limit     int
	truncated bool
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	if b.limit > 0 {
		remaining := b.limit - b.buf.Len()
		if remaining <= 0 {
			b.truncated = true
			return len(p), nil
		}
		if len(p) > remaining {
			b.buf.Write(p[:remaining])
			b.truncated = true
			return len(p), nil
		}
	}
	return b.buf.Write(p)
}

func (b *cappedBuffer) String() string {
	if b.truncated {
		return b.buf.String() + "\n…(truncated)"
	}
	return b.buf.String()
}
