package rimeno

import (
	"encoding/json"
	"reflect"

	"github.com/matiasinsaurralde/rimeno/schema"
)

// OutputSpec describes a required structured output for an agent: a JSON Schema
// sent to the provider as response_format, plus local validation and decoding.
// Build one with [OutputOf].
type OutputSpec struct {
	Name   string
	Schema json.RawMessage
	Strict bool

	validate func([]byte) error
	decode   func([]byte) (any, error)
}

// OutputOf configures an agent to return a value of type T. The schema is derived
// from T by reflection and sent as the provider's structured-output format; the
// model's final message is validated against the schema and decoded into *T,
// surfaced as Result.Output.
//
//	type review struct {
//	    Score   int      `json:"score" jsonschema:"minimum=0,maximum=10"`
//	    Reasons []string `json:"reasons"`
//	}
//	agent, _ := rimeno.New(rimeno.Config{Model: m, Output: rimeno.OutputOf[review]()})
//	res, _ := agent.Run(ctx, "review this PR")
//	r := res.Output.(*review)
func OutputOf[T any]() *OutputSpec {
	sch := schema.MustOf[T]()
	return &OutputSpec{
		Name:   outputName[T](),
		Schema: sch,
		Strict: true,
		validate: func(b []byte) error {
			return schema.Validate(sch, b)
		},
		decode: func(b []byte) (any, error) {
			var v T
			if err := json.Unmarshal(b, &v); err != nil {
				return nil, err
			}
			return &v, nil
		},
	}
}

// OutputFromSchema configures an agent to return output conforming to a
// hand-written JSON Schema, for callers who already have a schema (generated
// from protobuf/OpenAPI, or hand-tuned with $ref/anyOf/enum descriptions) and do
// not want to mirror it as a Go type. It is the [OutputSpec] analogue of
// [RawTool].
//
// The schema is sent to the provider as response_format and validated locally
// with [schema.Validate]; by default the model's final message is surfaced as
// Result.Output as a [json.RawMessage] for the caller to decode. Pair with
// Config.OutputRepairAttempts to have the model repair invalid output.
//
//	spec := rimeno.OutputFromSchema("decision", decisionSchema, rimeno.Lax())
//	agent, _ := rimeno.New(rimeno.Config{Model: m, Output: spec, OutputRepairAttempts: 2})
//	res, _ := agent.Run(ctx, task)
//	raw := res.Output.(json.RawMessage) // decode with your own parser
func OutputFromSchema(name string, sch json.RawMessage, opts ...OutputOption) *OutputSpec {
	spec := &OutputSpec{
		Name:   name,
		Schema: sch,
		Strict: true,
		validate: func(b []byte) error {
			return schema.Validate(sch, b)
		},
		// Default decode is a passthrough copy: the caller owns the concrete type.
		decode: func(b []byte) (any, error) {
			return json.RawMessage(append([]byte(nil), b...)), nil
		},
	}
	for _, o := range opts {
		o(spec)
	}
	return spec
}

// OutputOption configures an [OutputSpec] built by [OutputFromSchema].
type OutputOption func(*OutputSpec)

// Lax disables provider "strict" structured-output mode. Strict mode requires a
// closed schema (every property required, additionalProperties:false); many real
// schemas are not closed, and some providers reject them under strict — Lax is
// the escape hatch.
func Lax() OutputOption { return func(s *OutputSpec) { s.Strict = false } }

// DecodeInto replaces the default json.RawMessage passthrough decoder, so
// Result.Output can be a typed value while the schema stays raw. The bytes passed
// to decode have already been validated against the schema.
func DecodeInto(decode func([]byte) (any, error)) OutputOption {
	return func(s *OutputSpec) { s.decode = decode }
}

func outputName[T any]() string {
	t := reflect.TypeFor[T]()
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Name() != "" {
		return t.Name()
	}
	return "output"
}
