// Package schema derives JSON Schema documents from Go types via reflection and
// validates JSON values against them.
//
// It is the shared foundation for two rimeno features:
//
//   - Tool parameters: rimeno.NewTool[In, Out] reflects In into a parameter schema
//     that is advertised to the model.
//   - Structured output: rimeno.OutputOf[T] reflects T into a response schema that
//     is sent as the provider's response_format and validated on return.
//
// The generated schemas target the subset of JSON Schema understood by
// OpenAI-compatible "structured outputs": objects with typed properties, a
// required list, enums, numeric bounds, arrays, and nested objects. The goal is
// pragmatic fidelity for LLM tool/argument use, not full Draft 2020-12 coverage.
package schema
