package schema

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
)

// ValidationError describes why a JSON value failed schema validation.
type ValidationError struct {
	Path string // JSON path to the offending value, e.g. "$.items[2].name"
	Msg  string
}

func (e *ValidationError) Error() string {
	if e.Path == "" {
		return "schema: " + e.Msg
	}
	return fmt.Sprintf("schema: %s: %s", e.Path, e.Msg)
}

// Validate checks data against the JSON Schema, returning a *ValidationError on
// the first violation or nil if the value conforms.
//
// It supports the subset of JSON Schema that [For] emits: object properties and
// required, arrays and items, string/number/integer/boolean types, enums, and
// numeric minimum/maximum. Unknown keywords are ignored.
func Validate(schemaJSON json.RawMessage, data []byte) error {
	var sch map[string]any
	if err := json.Unmarshal(schemaJSON, &sch); err != nil {
		return &ValidationError{Msg: "invalid schema: " + err.Error()}
	}
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		return &ValidationError{Msg: "invalid JSON: " + err.Error()}
	}
	return validateValue(sch, v, "$")
}

func validateValue(sch map[string]any, v any, path string) error {
	if enum, ok := sch["enum"].([]any); ok {
		matched := false
		for _, e := range enum {
			if equalJSON(e, v) {
				matched = true
				break
			}
		}
		if !matched {
			return &ValidationError{path, "value not permitted by enum"}
		}
	}

	typ, _ := sch["type"].(string)
	switch typ {
	case "object":
		obj, ok := v.(map[string]any)
		if !ok {
			return &ValidationError{path, "expected object"}
		}
		if req, ok := sch["required"].([]any); ok {
			for _, r := range req {
				name, _ := r.(string)
				if _, present := obj[name]; !present {
					return &ValidationError{path, "missing required property " + strconv.Quote(name)}
				}
			}
		}
		props, _ := sch["properties"].(map[string]any)
		allowExtra := true
		if ap, ok := sch["additionalProperties"].(bool); ok {
			allowExtra = ap
		}
		for name, val := range obj {
			ps, ok := props[name].(map[string]any)
			if !ok {
				if !allowExtra {
					return &ValidationError{childPath(path, name), "unexpected property"}
				}
				continue
			}
			if err := validateValue(ps, val, childPath(path, name)); err != nil {
				return err
			}
		}
	case "array":
		arr, ok := v.([]any)
		if !ok {
			return &ValidationError{path, "expected array"}
		}
		if items, ok := sch["items"].(map[string]any); ok {
			for i, el := range arr {
				if err := validateValue(items, el, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
	case "string":
		if _, ok := v.(string); !ok {
			return &ValidationError{path, "expected string"}
		}
	case "integer":
		f, ok := v.(float64)
		if !ok {
			return &ValidationError{path, "expected integer"}
		}
		if math.Trunc(f) != f {
			return &ValidationError{path, "expected integer, got fractional number"}
		}
		if err := checkBounds(sch, f, path); err != nil {
			return err
		}
	case "number":
		f, ok := v.(float64)
		if !ok {
			return &ValidationError{path, "expected number"}
		}
		if err := checkBounds(sch, f, path); err != nil {
			return err
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			return &ValidationError{path, "expected boolean"}
		}
	case "", "null":
		// Unconstrained or explicitly nullable: accept.
	}
	return nil
}

func checkBounds(sch map[string]any, f float64, path string) error {
	if mn, ok := sch["minimum"].(float64); ok && f < mn {
		return &ValidationError{path, fmt.Sprintf("value %v below minimum %v", f, mn)}
	}
	if mx, ok := sch["maximum"].(float64); ok && f > mx {
		return &ValidationError{path, fmt.Sprintf("value %v above maximum %v", f, mx)}
	}
	return nil
}

func childPath(path, name string) string { return path + "." + name }

func equalJSON(a, b any) bool {
	ab, _ := json.Marshal(a)
	bb, _ := json.Marshal(b)
	return string(ab) == string(bb)
}
