package sandbox

import (
	"context"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fakeHost is an injected host Sandbox that records the argv it is asked to run
// and replays scripted results, so DockerProvider's CLI wiring can be tested
// without a Docker daemon.
type fakeHost struct {
	calls   [][]string
	results []ExecResult
	i       int
}

func (f *fakeHost) Exec(_ context.Context, req ExecRequest) (ExecResult, error) {
	f.calls = append(f.calls, req.Argv)
	if f.i < len(f.results) {
		r := f.results[f.i]
		f.i++
		return r, nil
	}
	return ExecResult{}, nil
}
func (f *fakeHost) Close() error { return nil }

func TestDockerRunArgv(t *testing.T) {
	got := dockerRunArgv("docker", "alpine:3", Spec{WorkDir: "/work", Env: []string{"A=1", "B=2"}})
	want := []string{"docker", "run", "-d", "--rm", "-w", "/work", "-e", "A=1", "-e", "B=2", "alpine:3", "sleep", "infinity"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("run argv =\n  %v\nwant\n  %v", got, want)
	}
}

func TestDockerExecArgv(t *testing.T) {
	s := &dockerSandbox{provider: DockerProvider{}, container: "abc123", workdir: "/w"}
	got := s.execArgv(ExecRequest{Argv: []string{"sh", "-c", "echo hi"}, Stdin: "x", Env: []string{"K=V"}})
	want := []string{"docker", "exec", "-i", "-w", "/w", "-e", "K=V", "abc123", "sh", "-c", "echo hi"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("exec argv =\n  %v\nwant\n  %v", got, want)
	}
	// Dir on the request overrides the sandbox default; no stdin drops -i.
	got = s.execArgv(ExecRequest{Argv: []string{"ls"}, Dir: "/other"})
	want = []string{"docker", "exec", "-w", "/other", "abc123", "ls"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("exec argv (override) =\n  %v\nwant\n  %v", got, want)
	}
}

func TestDockerProvider_CreateExecCloseFlow(t *testing.T) {
	host := &fakeHost{results: []ExecResult{
		{Stdout: "container-xyz\n"},   // docker run → container id
		{Stdout: "hello from inside"}, // docker exec → command output
		{},                            // docker rm -f
	}}
	p := DockerProvider{Image: "alpine", host: host}

	sb, err := p.Create(context.Background(), Spec{WorkDir: "/app"})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	// The run used the configured image and workdir.
	if run := host.calls[0]; run[len(run)-3] != "alpine" || !contains(run, "/app") {
		t.Errorf("docker run argv = %v", run)
	}

	res, err := sb.Exec(context.Background(), ExecRequest{Argv: []string{"echo", "hi"}})
	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if res.Stdout != "hello from inside" {
		t.Errorf("stdout = %q", res.Stdout)
	}
	// The exec was wrapped as `docker exec ... container-xyz echo hi`.
	execCall := host.calls[1]
	if !contains(execCall, "container-xyz") || execCall[len(execCall)-2] != "echo" {
		t.Errorf("docker exec argv = %v", execCall)
	}

	if err := sb.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	rm := host.calls[2]
	if !contains(rm, "rm") || !contains(rm, "container-xyz") {
		t.Errorf("docker rm argv = %v", rm)
	}
}

func TestDockerProvider_NoImage(t *testing.T) {
	p := DockerProvider{host: &fakeHost{}}
	if _, err := p.Create(context.Background(), Spec{}); err == nil {
		t.Error("expected error when no image is set")
	}
}

func TestDockerProvider_RunFailure(t *testing.T) {
	host := &fakeHost{results: []ExecResult{{ExitCode: 125, Stderr: "no such image"}}}
	p := DockerProvider{Image: "nope", host: host}
	if _, err := p.Create(context.Background(), Spec{}); err == nil {
		t.Error("expected error when docker run fails")
	}
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

// TestDockerProvider_Integration runs a real container if a Docker daemon is
// available; it skips otherwise so the suite stays green without Docker.
func TestDockerProvider_Integration(t *testing.T) {
	if !dockerAvailable(t) {
		t.Skip("docker daemon not available")
	}
	p := DockerProvider{Image: "alpine:3"}
	sb, err := p.Create(context.Background(), Spec{})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer func() { _ = sb.Close() }()
	res, err := sb.Exec(context.Background(), ExecRequest{Argv: []string{"sh", "-c", "echo containerized"}})
	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if strings.TrimSpace(res.Stdout) != "containerized" {
		t.Errorf("stdout = %q", res.Stdout)
	}
}

func dockerAvailable(t *testing.T) bool {
	t.Helper()
	host := &localSandbox{provider: LocalProvider{}}
	res, err := host.Exec(context.Background(), ExecRequest{
		Argv:    []string{"docker", "info"},
		Timeout: 3 * time.Second,
	})
	return err == nil && res.ExitCode == 0
}
