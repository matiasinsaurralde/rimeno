//go:build !unix

package sandbox

import "os/exec"

// setPgid is a no-op on platforms without POSIX process groups.
func setPgid(*exec.Cmd) {}

// killProc kills just the shell process (no process-group semantics available).
func killProc(cmd *exec.Cmd) {
	if cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}
