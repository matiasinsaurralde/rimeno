package schema

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// maxDepth guards against pathologically deep or recursive types.
const maxDepth = 32

// Of returns the JSON Schema for the type parameter T.
func Of[T any]() (json.RawMessage, error) { return For(reflect.TypeFor[T]()) }

// MustOf is like Of but panics on error. Use it where the type is fixed at
// compile time and a malformed type is a programming error (mirrors
// regexp.MustCompile ergonomics).
func MustOf[T any]() json.RawMessage {
	s, err := Of[T]()
	if err != nil {
		panic("rimeno/schema: " + err.Error())
	}
	return s
}

// For returns the JSON Schema for a reflect.Type.
func For(t reflect.Type) (json.RawMessage, error) {
	node, err := build(t, fieldTag{}, 0)
	if err != nil {
		return nil, err
	}
	return json.Marshal(node)
}

// fieldTag holds schema hints parsed from a struct field's `jsonschema` tag.
type fieldTag struct {
	description string
	enum        []string
	minimum     *float64
	maximum     *float64
	format      string
	required    *bool
}

func build(t reflect.Type, tag fieldTag, depth int) (map[string]any, error) {
	if depth > maxDepth {
		return nil, fmt.Errorf("schema: max nesting depth exceeded at %s", t)
	}
	t = deref(t)

	if t == reflect.TypeOf(time.Time{}) {
		m := map[string]any{"type": "string", "format": "date-time"}
		applyTag(m, tag)
		return m, nil
	}

	var m map[string]any
	switch t.Kind() {
	case reflect.String:
		m = map[string]any{"type": "string"}
	case reflect.Bool:
		m = map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		m = map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		m = map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		if t.Elem().Kind() == reflect.Uint8 { // []byte encodes as a base64 string
			m = map[string]any{"type": "string", "contentEncoding": "base64"}
			break
		}
		items, err := build(t.Elem(), fieldTag{}, depth+1)
		if err != nil {
			return nil, err
		}
		m = map[string]any{"type": "array", "items": items}
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("schema: map key must be string, got %s", t.Key())
		}
		vals, err := build(t.Elem(), fieldTag{}, depth+1)
		if err != nil {
			return nil, err
		}
		m = map[string]any{"type": "object", "additionalProperties": vals}
	case reflect.Struct:
		return buildStruct(t, tag, depth)
	case reflect.Interface:
		m = map[string]any{} // unconstrained ("any")
	default:
		return nil, fmt.Errorf("schema: unsupported kind %s (%s)", t.Kind(), t)
	}
	applyTag(m, tag)
	return m, nil
}

func buildStruct(t reflect.Type, tag fieldTag, depth int) (map[string]any, error) {
	props := map[string]any{}
	var required []string
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		name, opts := jsonName(f)
		if name == "-" {
			continue
		}
		// Anonymous struct fields without an explicit json name are flattened.
		if f.Anonymous && f.Tag.Get("json") == "" && deref(f.Type).Kind() == reflect.Struct {
			sub, err := buildStruct(deref(f.Type), fieldTag{}, depth+1)
			if err != nil {
				return nil, err
			}
			if sp, ok := sub["properties"].(map[string]any); ok {
				for k, v := range sp {
					props[k] = v
				}
			}
			if rr, ok := sub["required"].([]string); ok {
				required = append(required, rr...)
			}
			continue
		}
		ft := parseFieldTag(f)
		child, err := build(f.Type, ft, depth+1)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", f.Name, err)
		}
		props[name] = child
		if isRequired(f, opts, ft) {
			required = append(required, name)
		}
	}
	m := map[string]any{"type": "object", "properties": props, "additionalProperties": false}
	if len(required) > 0 {
		m["required"] = required
	}
	applyTag(m, tag)
	return m, nil
}

func deref(t reflect.Type) reflect.Type {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t
}

func jsonName(f reflect.StructField) (string, map[string]bool) {
	opts := map[string]bool{}
	tag := f.Tag.Get("json")
	if tag == "" {
		return f.Name, opts
	}
	parts := strings.Split(tag, ",")
	for _, o := range parts[1:] {
		opts[o] = true
	}
	if parts[0] == "" {
		return f.Name, opts
	}
	return parts[0], opts
}

func isRequired(f reflect.StructField, opts map[string]bool, tag fieldTag) bool {
	if tag.required != nil {
		return *tag.required
	}
	if f.Type.Kind() == reflect.Pointer {
		return false
	}
	return !opts["omitempty"]
}

func parseFieldTag(f reflect.StructField) fieldTag {
	var ft fieldTag
	raw := f.Tag.Get("jsonschema")
	if raw == "" {
		if d := f.Tag.Get("desc"); d != "" {
			ft.description = d
		}
		return ft
	}
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		kv := strings.SplitN(part, "=", 2)
		key := strings.TrimSpace(kv[0])
		if len(kv) == 1 {
			switch key {
			case "required":
				b := true
				ft.required = &b
			case "optional":
				b := false
				ft.required = &b
			default:
				if ft.description == "" {
					ft.description = key
				}
			}
			continue
		}
		val := strings.TrimSpace(kv[1])
		switch key {
		case "description", "desc":
			ft.description = val
		case "enum":
			ft.enum = strings.Split(val, "|")
		case "format":
			ft.format = val
		case "minimum", "min":
			if n, err := strconv.ParseFloat(val, 64); err == nil {
				ft.minimum = &n
			}
		case "maximum", "max":
			if n, err := strconv.ParseFloat(val, 64); err == nil {
				ft.maximum = &n
			}
		case "required":
			b := val == "true"
			ft.required = &b
		}
	}
	return ft
}

func applyTag(m map[string]any, tag fieldTag) {
	if tag.description != "" {
		m["description"] = tag.description
	}
	if len(tag.enum) > 0 {
		e := make([]any, len(tag.enum))
		for i, s := range tag.enum {
			e[i] = s
		}
		m["enum"] = e
	}
	if tag.minimum != nil {
		m["minimum"] = *tag.minimum
	}
	if tag.maximum != nil {
		m["maximum"] = *tag.maximum
	}
	if tag.format != "" {
		m["format"] = tag.format
	}
}
