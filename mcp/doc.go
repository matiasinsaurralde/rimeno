// Package mcp is a Model Context Protocol (MCP) client: it connects rimeno to
// external tool servers so a rimeno agent can call their tools as if they were
// local Go tools.
//
// MCP is JSON-RPC 2.0, most commonly spoken over a subprocess's stdio. A Client
// performs the initialize handshake, lists the server's tools, and wraps each as
// a [github.com/matiasinsaurralde/rimeno.Tool]. A model's tool call then turns
// into an MCP tools/call request transparently.
//
// # Usage
//
//	c, err := mcp.NewStdioClient("mcp-server-filesystem", []string{"/tmp"})
//	if err != nil { log.Fatal(err) }
//	defer c.Close()
//
//	if _, err := c.Connect(ctx); err != nil { log.Fatal(err) }
//	tools, err := c.Tools(ctx) // []rimeno.Tool
//	if err != nil { log.Fatal(err) }
//
//	agent, _ := rimeno.New(rimeno.Config{Model: model, Tools: tools})
//	res, _ := agent.Run(ctx, "list the files in /tmp")
//
// Tools from several servers can be combined; use [WithToolNamePrefix] to avoid
// name collisions. This package is the mirror image of rimeno/rpc: rpc lets other
// programs drive rimeno, while mcp lets rimeno drive other programs' tools.
package mcp
