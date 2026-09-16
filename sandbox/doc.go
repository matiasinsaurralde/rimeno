// Package sandbox is rimeno's execution-isolation seam: an interface for running
// commands in a controlled environment, plus a tool that exposes it to a model.
//
// The abstraction is a [Provider] that creates [Sandbox]es, so a program can move
// from unsandboxed local execution to real isolation by swapping one value — no
// changes to agent or tool code. [CommandTool] turns any Sandbox into a rimeno
// "run_command" tool.
//
// # Providers
//
//   - [LocalProvider] (included): runs commands directly on the host with NO
//     isolation, plus optional guards (command allowlist, output cap, timeout).
//     For development and trusted use.
//   - [DockerProvider] (included): each sandbox is a Docker container; real
//     process/filesystem isolation via the `docker` CLI (no Go Docker SDK
//     dependency). Requires a running Docker daemon.
//
// # Roadmap
//
// Further isolating providers implement the same [Provider]/[Sandbox] interface:
//
//   - Firecracker — each sandbox is a microVM; strongest isolation, for running
//     untrusted code with near-native speed.
//   - e2b — hosted sandboxes over the e2b API; no local daemon needed.
//
// Because the interface is stable, agents and tools written against it today work
// unchanged as more providers land. Gate the local provider behind rimeno's
// approval hook and prefer a strict command allowlist; prefer DockerProvider (or
// a stricter backend) for untrusted code.
package sandbox
