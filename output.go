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
