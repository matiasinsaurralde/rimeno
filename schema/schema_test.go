package schema_test

import (
	"encoding/json"
	"testing"

	"github.com/matiasinsaurralde/rimeno/schema"
)

type address struct {
	City string `json:"city"`
	Zip  string `json:"zip,omitempty"`
}

type person struct {
	Name    string   `json:"name" jsonschema:"description=full legal name"`
	Age     int      `json:"age" jsonschema:"minimum=0,maximum=130"`
	Role    string   `json:"role" jsonschema:"enum=admin|user|guest"`
	Email   *string  `json:"email,omitempty"`
	Tags    []string `json:"tags,omitempty"`
	Address address  `json:"address"`
	Ignored string   `json:"-"`
	//lint:ignore U1000 present so Of can be shown to skip unexported fields
	private string
}

func TestOf_Person(t *testing.T) {
	raw, err := schema.Of[person]()
	if err != nil {
		t.Fatalf("Of: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["type"] != "object" {
		t.Fatalf("type = %v, want object", m["type"])
	}
	props := m["properties"].(map[string]any)
	for _, want := range []string{"name", "age", "role", "email", "tags", "address"} {
		if _, ok := props[want]; !ok {
			t.Errorf("missing property %q", want)
		}
	}
	if _, ok := props["Ignored"]; ok {
		t.Error("json:\"-\" field should be omitted")
	}
	if _, ok := props["private"]; ok {
		t.Error("unexported field should be omitted")
	}

	req := toStringSet(m["required"])
	for _, want := range []string{"name", "age", "role", "address"} {
		if !req[want] {
			t.Errorf("expected %q to be required", want)
		}
	}
	for _, notWant := range []string{"email", "tags"} {
		if req[notWant] {
			t.Errorf("expected %q to be optional", notWant)
		}
	}

	name := props["name"].(map[string]any)
	if name["description"] != "full legal name" {
		t.Errorf("name description = %v", name["description"])
	}
	age := props["age"].(map[string]any)
	if age["type"] != "integer" || age["minimum"].(float64) != 0 || age["maximum"].(float64) != 130 {
		t.Errorf("age schema wrong: %v", age)
	}
	role := props["role"].(map[string]any)
	if got := toStringSet(role["enum"]); !got["admin"] || !got["user"] || !got["guest"] {
		t.Errorf("role enum wrong: %v", role["enum"])
	}
}

func TestValidate(t *testing.T) {
	raw := schema.MustOf[person]()

	good := `{"name":"Ada","age":36,"role":"admin","address":{"city":"London"}}`
	if err := schema.Validate(raw, []byte(good)); err != nil {
		t.Fatalf("valid doc rejected: %v", err)
	}

	cases := map[string]string{
		"missing required name": `{"age":36,"role":"admin","address":{"city":"x"}}`,
		"bad enum":              `{"name":"A","age":1,"role":"root","address":{"city":"x"}}`,
		"age above maximum":     `{"name":"A","age":999,"role":"user","address":{"city":"x"}}`,
		"wrong type for age":    `{"name":"A","age":"old","role":"user","address":{"city":"x"}}`,
		"nested missing city":   `{"name":"A","age":1,"role":"user","address":{}}`,
		"malformed json":        `{`,
	}
	for name, doc := range cases {
		if err := schema.Validate(raw, []byte(doc)); err == nil {
			t.Errorf("%s: expected validation error, got nil", name)
		}
	}
}

func TestOf_Scalars(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  json.RawMessage
		typ  string
	}{
		{"string", schema.MustOf[string](), "string"},
		{"int", schema.MustOf[int](), "integer"},
		{"float", schema.MustOf[float64](), "number"},
		{"bool", schema.MustOf[bool](), "boolean"},
	} {
		var m map[string]any
		_ = json.Unmarshal(tc.raw, &m)
		if m["type"] != tc.typ {
			t.Errorf("%s: type = %v, want %v", tc.name, m["type"], tc.typ)
		}
	}
}

func toStringSet(v any) map[string]bool {
	out := map[string]bool{}
	if arr, ok := v.([]any); ok {
		for _, e := range arr {
			if s, ok := e.(string); ok {
				out[s] = true
			}
		}
	}
	return out
}

// FuzzValidate ensures Validate never panics regardless of the schema/data bytes
// it is handed — it must always return normally (nil or an error). Runs its seed
// corpus under plain `go test`; explore more with `go test -fuzz=FuzzValidate`.
func FuzzValidate(f *testing.F) {
	f.Add(`{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}`, `{"a":1}`)
	f.Add(`{"type":"string","enum":["x","y"]}`, `"z"`)
	f.Add(`{"type":"number","minimum":0,"maximum":10}`, `42`)
	f.Add(`{"type":"array","items":{"type":"boolean"}}`, `[true,false]`)
	f.Add(`not json`, `also not json`)
	f.Add(``, ``)
	f.Fuzz(func(t *testing.T, schemaJSON, data string) {
		// Must not panic; the return value is irrelevant for the robustness check.
		_ = schema.Validate(json.RawMessage(schemaJSON), []byte(data))
	})
}
