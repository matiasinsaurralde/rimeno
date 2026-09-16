package sandbox

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// DockerProvider creates sandboxes backed by Docker containers: Create starts a
// long-lived container, Exec runs `docker exec` inside it, and Close removes it.
// It shells out to the `docker` CLI (no Go Docker SDK dependency), so it requires
// Docker installed with a running daemon.
//
// This gives real process/filesystem isolation through the same
// [Provider]/[Sandbox] interface as [LocalProvider] — swapping one for the other
// needs no changes to agent or tool code. The command-construction logic is
// unit-tested; end-to-end behavior requires a Docker daemon (the integration test
// skips when none is available), so validate it in your environment.
type DockerProvider struct {
	// Image is the default container image when Spec.Image is empty.
	Image string
	// Docker is the docker binary to invoke (default "docker").
	Docker string
	// DefaultTimeout applies when ExecRequest.Timeout is zero (default 60s).
	DefaultTimeout time.Duration
	// MaxOutputBytes caps captured output per command (default 1 MiB).
	MaxOutputBytes int

	// host runs commands on the host; nil uses a default local sandbox. Injectable
	// for tests so the docker CLI need not actually run.
	host Sandbox
}

func (p DockerProvider) docker() string {
	if p.Docker != "" {
		return p.Docker
	}
	return "docker"
}

func (p DockerProvider) timeout() time.Duration {
	if p.DefaultTimeout > 0 {
		return p.DefaultTimeout
	}
	return 60 * time.Second
}

func (p DockerProvider) hostSandbox() Sandbox {
	if p.host != nil {
		return p.host
	}
	return &localSandbox{provider: LocalProvider{MaxOutputBytes: p.MaxOutputBytes, DefaultTimeout: p.timeout()}}
}

// Create starts a container (image from Spec.Image, else DockerProvider.Image)
// running `sleep infinity`, and returns a sandbox bound to it.
func (p DockerProvider) Create(ctx context.Context, spec Spec) (Sandbox, error) {
	image := spec.Image
	if image == "" {
		image = p.Image
	}
	if image == "" {
		return nil, errors.New("sandbox: DockerProvider requires an image (Spec.Image or DockerProvider.Image)")
	}
	host := p.hostSandbox()
	res, err := host.Exec(ctx, ExecRequest{Argv: dockerRunArgv(p.docker(), image, spec)})
	if err != nil {
		return nil, fmt.Errorf("sandbox: docker run: %w", err)
	}
	if res.ExitCode != 0 {
		return nil, fmt.Errorf("sandbox: docker run failed (exit %d): %s", res.ExitCode, strings.TrimSpace(res.Stderr))
	}
	id := strings.TrimSpace(res.Stdout)
	if id == "" {
		return nil, errors.New("sandbox: docker run returned no container id")
	}
	return &dockerSandbox{provider: p, host: host, container: id, workdir: spec.WorkDir}, nil
}

// dockerRunArgv builds the `docker run -d --rm [...] image sleep infinity` argv.
func dockerRunArgv(docker, image string, spec Spec) []string {
	argv := []string{docker, "run", "-d", "--rm"}
	if spec.WorkDir != "" {
		argv = append(argv, "-w", spec.WorkDir)
	}
	for _, e := range spec.Env {
		argv = append(argv, "-e", e)
	}
	argv = append(argv, image, "sleep", "infinity")
	return argv
}

type dockerSandbox struct {
	provider  DockerProvider
	host      Sandbox
	container string
	workdir   string
}

// execArgv builds the `docker exec [-i] [-w dir] [-e ...] container <argv...>`.
func (s *dockerSandbox) execArgv(req ExecRequest) []string {
	argv := []string{s.provider.docker(), "exec"}
	if req.Stdin != "" {
		argv = append(argv, "-i")
	}
	dir := req.Dir
	if dir == "" {
		dir = s.workdir
	}
	if dir != "" {
		argv = append(argv, "-w", dir)
	}
	for _, e := range req.Env {
		argv = append(argv, "-e", e)
	}
	argv = append(argv, s.container)
	argv = append(argv, req.Argv...)
	return argv
}

func (s *dockerSandbox) Exec(ctx context.Context, req ExecRequest) (ExecResult, error) {
	if len(req.Argv) == 0 {
		return ExecResult{}, errors.New("sandbox: empty argv")
	}
	// Run `docker exec ...` on the host; the inner command's exit code and output
	// pass straight through.
	return s.host.Exec(ctx, ExecRequest{Argv: s.execArgv(req), Stdin: req.Stdin, Timeout: req.Timeout})
}

// Close removes the container (force, ignoring whether it is still running).
func (s *dockerSandbox) Close() error {
	_, err := s.host.Exec(context.Background(), ExecRequest{
		Argv:    []string{s.provider.docker(), "rm", "-f", s.container},
		Timeout: 20 * time.Second,
	})
	return err
}
