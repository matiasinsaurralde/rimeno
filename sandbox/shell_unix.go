//go:build unix

package sandbox

import (
	"os/exec"
	"syscall"
)

// setPgid puts the shell in its own process group so the whole tree (the shell and
// any children it spawned, e.g. a hung `git`) can be killed together.
func setPgid(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProc kills the shell's entire process group.
func killProc(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	// Negative pid targets the process group (the shell is the group leader).
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	_ = cmd.Process.Kill()
}
