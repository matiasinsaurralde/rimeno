package rimeno

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/matiasinsaurralde/rimeno/schema"
)

// Tool is something the agent can call. Most users never implement this directly:
// [NewTool] builds one from a typed Go function.
type Tool interface {
	// Name is the identifier the model uses to call the tool.
	Name() string
	// Description tells the model what the tool does and when to use it.
	Description() string
	// ParametersSchema is the JSON Schema for the tool's arguments.
	ParametersSchema() json.RawMessage
	// Invoke runs the tool with JSON arguments and returns a result to be
	// serialized back to the model. Returning an ordinary error feeds the error
	// to the model as a recoverable tool result; wrap with [Fatal] to abort.
	Invoke(ctx context.Context, args json.RawMessage) (any, error)
}

// NewTool builds a [Tool] from a typed function. The parameter schema is derived
// from In by reflection (json tags for names/optionality, `jsonschema` tags for
// descriptions/enums/bounds); the model's JSON arguments are unmarshaled into In,
// fn is called, and its Out result is JSON-encoded back to the model.
//
// This is the primary way to expose your own Go code to an agent:
//
//	type addArgs struct {
//	    A int `json:"a" jsonschema:"description=first addend"`
//	    B int `json:"b" jsonschema:"description=second addend"`
//	}
//	add := rimeno.NewTool("add", "Add two integers",
//	    func(ctx context.Context, in addArgs) (int, error) { return in.A + in.B, nil })
//
// NewTool panics if In cannot be reflected into a schema, which is a programming
// error surfaced at construction (mirrors regexp.MustCompile).
func NewTool[In, Out any](name, description string, fn func(context.Context, In) (Out, error)) Tool {
	sch, err := schema.Of[In]()
	if err != nil {
		panic(fmt.Sprintf("rimeno.NewTool(%q): cannot derive schema: %v", name, err))
	}
	return &typedTool[In, Out]{name: name, description: description, schema: sch, fn: fn}
}

type typedTool[In, Out any] struct {
	name        string
	description string
	schema      json.RawMessage
	fn          func(context.Context, In) (Out, error)
}

func (t *typedTool[In, Out]) Name() string                      { return t.name }
func (t *typedTool[In, Out]) Description() string               { return t.description }
func (t *typedTool[In, Out]) ParametersSchema() json.RawMessage { return t.schema }

func (t *typedTool[In, Out]) Invoke(ctx context.Context, args json.RawMessage) (any, error) {
	var in In
	if len(args) > 0 && string(args) != "null" {
		if err := json.Unmarshal(args, &in); err != nil {
			return nil, &ToolError{Tool: t.name, Err: fmt.Errorf("invalid arguments: %w", err)}
		}
	}
	return t.fn(ctx, in)
}

// RawTool builds a [Tool] from a hand-written parameter schema and a raw handler.
// Use it when reflection-derived schemas are insufficient.
func RawTool(name, description string, params json.RawMessage, fn func(ctx context.Context, args json.RawMessage) (any, error)) Tool {
	return &rawTool{name: name, description: description, schema: params, fn: fn}
}

type rawTool struct {
	name        string
	description string
	schema      json.RawMessage
	fn          func(ctx context.Context, args json.RawMessage) (any, error)
}

func (t *rawTool) Name() string                      { return t.name }
func (t *rawTool) Description() string               { return t.description }
func (t *rawTool) ParametersSchema() json.RawMessage { return t.schema }
func (t *rawTool) Invoke(ctx context.Context, args json.RawMessage) (any, error) {
	return t.fn(ctx, args)
}

// marshalToolResult renders a tool's return value into the string fed back to the
// model. Strings and JSON bytes pass through; everything else is JSON-encoded.
func marshalToolResult(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	case []byte:
		return string(x)
	case json.RawMessage:
		return string(x)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(b)
	}
}
