package sandbox

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// PersistentShellProvider creates sandboxes backed by a single long-lived shell
// process per session — the model used by Claude Code's Bash tool. Unlike
// [LocalProvider], which runs each command as a fresh `sh -c`, here every command
// runs in one persistent shell, so working directory (`cd`), exported variables,
// and other shell state PERSIST across [Sandbox.Exec] calls.
//
// It runs on the host with NO isolation (dev/trusted use), like LocalProvider; pair
// command execution with rimeno's approval hook for control. It implements [Provider],
// so it is a drop-in swap — build [CommandTool] on the returned Sandbox and the
// agent/tool code is unchanged:
//
//	sb, _ := sandbox.PersistentShellProvider{}.Create(ctx, sandbox.Spec{WorkDir: dir, Env: os.Environ()})
//	defer sb.Close()
//	tool := sandbox.CommandTool(sb)
//
// Because there is one shell, Exec calls are serialized (a mutex); a shell-backed
// tool therefore does not benefit from rimeno's ParallelTools. A command that exceeds
// its timeout — or one that exits the shell (`exit`, or `set -e` then a failure) —
// terminates the shell (and its process group); the next Exec transparently starts
// a fresh shell, with shell state reset (the expected cost of that exceptional path).
type PersistentShellProvider struct {
	// Shell is the shell program (default: "bash" if on PATH, else "sh").
	Shell string
	// MaxOutputBytes caps captured stdout and stderr per command (default 1 MiB).
	MaxOutputBytes int
	// DefaultTimeout applies when ExecRequest.Timeout is zero (default 30s).
	DefaultTimeout time.Duration
}

func (p PersistentShellProvider) shell() string {
	if p.Shell != "" {
		return p.Shell
	}
	if path, err := exec.LookPath("bash"); err == nil {
		return path
	}
	return "sh"
}

func (p PersistentShellProvider) maxOutput() int {
	if p.MaxOutputBytes > 0 {
		return p.MaxOutputBytes
	}
	return 1 << 20
}

func (p PersistentShellProvider) defaultTimeout() time.Duration {
	if p.DefaultTimeout > 0 {
		return p.DefaultTimeout
	}
	return 30 * time.Second
}

// Create starts the shell process and returns a session-scoped sandbox.
func (p PersistentShellProvider) Create(_ context.Context, spec Spec) (Sandbox, error) {
	s := &persistentShell{provider: p, dir: spec.WorkDir, env: spec.Env}
	if err := s.start(); err != nil {
		return nil, err
	}
	return s, nil
}

type persistentShell struct {
	provider PersistentShellProvider
	dir      string
	env      []string

	mu      sync.Mutex // serializes Exec (a single shell process)
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	lines   chan string // reader goroutine feeds marker lines here
	tmpDir  string
	token   string
	counter int
	alive   bool
}

func (s *persistentShell) start() error {
	if s.tmpDir != "" { // self-heal path: clean the dead shell's temp dir first
		_ = os.RemoveAll(s.tmpDir)
		s.tmpDir = ""
	}
	tmp, err := os.MkdirTemp("", "rimeno-shell-")
	if err != nil {
		return fmt.Errorf("sandbox: shell tmpdir: %w", err)
	}
	var tok [8]byte
	_, _ = rand.Read(tok[:])

	// Launching the configured shell is the sandbox's job; the command it
	// interprets arrives over stdin, not as an argument here.
	cmd := exec.Command(s.provider.shell()) // #nosec G204
	if s.dir != "" {
		cmd.Dir = s.dir
	}
	cmd.Env = append([]string(nil), s.env...)
	setPgid(cmd) // own process group, so a timeout can kill the whole tree

	stdin, err := cmd.StdinPipe()
	if err != nil {
		_ = os.RemoveAll(tmp)
		return fmt.Errorf("sandbox: shell stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = os.RemoveAll(tmp)
		return fmt.Errorf("sandbox: shell stdout: %w", err)
	}
	cmd.Stderr = io.Discard // per-command stderr is redirected to files; the shell's own stderr is unused
	if err := cmd.Start(); err != nil {
		_ = os.RemoveAll(tmp)
		return fmt.Errorf("sandbox: start shell: %w", err)
	}

	s.cmd = cmd
	s.stdin = stdin
	s.tmpDir = tmp
	s.token = hex.EncodeToString(tok[:])
	s.lines = make(chan string, 8)
	s.alive = true

	// One reader goroutine feeds marker lines to s.lines. Command stdout/stderr go
	// to temp files, so the shell's own stdout only ever carries our marker lines.
	go func(r io.Reader, out chan<- string) {
		br := bufio.NewReaderSize(r, 64*1024)
		for {
			line, err := br.ReadString('\n')
			if line != "" {
				out <- line
			}
			if err != nil {
				close(out)
				return
			}
		}
	}(stdout, s.lines)
	return nil
}

// Exec runs one command in the persistent shell and returns its output/exit code.
func (s *persistentShell) Exec(ctx context.Context, req ExecRequest) (ExecResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	command := commandLine(req)
	if strings.TrimSpace(command) == "" {
		return ExecResult{}, errors.New("sandbox: empty command")
	}
	if !s.alive {
		if err := s.start(); err != nil {
			return ExecResult{}, err
		}
	}

	timeout := req.Timeout
	if timeout <= 0 {
		timeout = s.provider.defaultTimeout()
	}

	s.counter++
	id := s.counter
	outFile := filepath.Join(s.tmpDir, "o"+strconv.Itoa(id))
	errFile := filepath.Join(s.tmpDir, "e"+strconv.Itoa(id))
	marker := fmt.Sprintf("__RIMENO_%s_%d__", s.token, id)
	defer func() { _ = os.Remove(outFile) }()
	defer func() { _ = os.Remove(errFile) }()

	// Redirect the command's stdin from a file (if provided) or /dev/null — never
	// from the shell's own stdin, which is our command channel.
	stdinRedir := " < /dev/null"
	if req.Stdin != "" {
		inFile := filepath.Join(s.tmpDir, "i"+strconv.Itoa(id))
		if err := os.WriteFile(inFile, []byte(req.Stdin), 0o600); err == nil {
			stdinRedir = " < " + shquote(inFile)
			defer func() { _ = os.Remove(inFile) }()
		}
	}

	// Per-command env exports (rare; CommandTool passes none).
	var exports strings.Builder
	for _, kv := range req.Env {
		if i := strings.IndexByte(kv, '='); i > 0 {
			fmt.Fprintf(&exports, "export %s=%s\n", kv[:i], shquote(kv[i+1:]))
		}
	}

	// Run the command as a GROUP (not a subshell), so cd/exports persist in the
	// session; redirect its streams to files; then print the marker + exit code on
	// the shell's real stdout (our read channel).
	script := fmt.Sprintf("%s{ %s\n} > %s 2> %s%s\nprintf '%s %%d\\n' \"$?\"\n",
		exports.String(), command, shquote(outFile), shquote(errFile), stdinRedir, marker)

	if _, err := io.WriteString(s.stdin, script); err != nil {
		s.killLocked()
		return s.partial(outFile, errFile), fmt.Errorf("sandbox: write to shell: %w", err)
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			s.killLocked()
			return s.partial(outFile, errFile), ctx.Err()
		case <-timer.C:
			s.killLocked()
			return s.partial(outFile, errFile), fmt.Errorf("sandbox: command timed out after %s", timeout)
		case line, ok := <-s.lines:
			if !ok {
				s.alive = false
				return s.partial(outFile, errFile), errors.New("sandbox: shell exited unexpectedly")
			}
			idx := strings.Index(line, marker)
			if idx < 0 {
				continue // stray line (should not happen); ignore
			}
			return ExecResult{
				Stdout:   readCapped(outFile, s.provider.maxOutput()),
				Stderr:   readCapped(errFile, s.provider.maxOutput()),
				ExitCode: parseExit(line[idx+len(marker):]),
			}, nil
		}
	}
}

func (s *persistentShell) partial(outFile, errFile string) ExecResult {
	return ExecResult{
		Stdout: readCapped(outFile, s.provider.maxOutput()),
		Stderr: readCapped(errFile, s.provider.maxOutput()),
	}
}

// killLocked terminates the shell and its process group (caller holds s.mu). It
// drains the line channel so the reader goroutine can finish, and reaps the process.
func (s *persistentShell) killLocked() {
	if !s.alive && s.cmd == nil {
		return
	}
	if s.cmd != nil {
		killProc(s.cmd)
		go func(c *exec.Cmd) { _ = c.Wait() }(s.cmd)
	}
	if s.stdin != nil {
		_ = s.stdin.Close()
	}
	if s.lines != nil { // unblock/close the reader goroutine
		go func(ch chan string) {
			for range ch {
			}
		}(s.lines)
	}
	s.alive = false
}

// Close terminates the shell and removes its temp directory.
func (s *persistentShell) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.killLocked()
	if s.tmpDir != "" {
		_ = os.RemoveAll(s.tmpDir)
		s.tmpDir = ""
	}
	return nil
}

// commandLine derives the shell command string from an ExecRequest. CommandTool
// sends Argv=["sh","-c",cmd], which becomes cmd; any other argv is shell-quoted and
// joined so a direct Exec caller still works.
func commandLine(req ExecRequest) string {
	a := req.Argv
	if len(a) >= 3 && a[1] == "-c" {
		if b := filepath.Base(a[0]); b == "sh" || b == "bash" {
			return a[2]
		}
	}
	if len(a) == 0 {
		return ""
	}
	parts := make([]string, len(a))
	for i, x := range a {
		parts[i] = shquote(x)
	}
	return strings.Join(parts, " ")
}

func parseExit(s string) int {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return -1
	}
	return n
}

// shquote single-quotes s for safe embedding in a shell command.
func shquote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// readCapped reads up to limit bytes from path, appending a truncation marker when
// the file is larger (it reads at most limit+1 bytes, so a huge file is not slurped).
func readCapped(path string, limit int) string {
	f, err := os.Open(path) // #nosec G304 -- path is an internal temp file created by the sandbox
	if err != nil {
		return ""
	}
	defer func() { _ = f.Close() }()
	b := &cappedBuffer{limit: limit}
	_, _ = io.Copy(b, io.LimitReader(f, int64(limit)+1))
	return b.String()
}
