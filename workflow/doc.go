// Package workflow orchestrates multiple rimeno agents as an explicit DAG, for when
// you want deterministic control over routing instead of letting an LLM decide.
//
// A workflow is a set of named [Step]s with declared dependencies. Steps share a
// [State] blackboard; a step's output is stored under its name and is available
// to dependents. Independent steps run concurrently (each as a goroutine), so
// parallelism is implicit in the dependency graph — no explicit fan-out needed.
// Each agent step may use a different model, so an orchestrator (model A) and
// workers (model B) is just configuration.
//
//	wf := workflow.New("research").
//	    Step(workflow.AgentStep("plan", planner, func(s *workflow.State) string { return s.Input })).
//	    Step(workflow.AgentStep("web", webWorker, func(s *workflow.State) string { return s.Get("plan") }, "plan")).
//	    Step(workflow.AgentStep("docs", docsWorker, func(s *workflow.State) string { return s.Get("plan") }, "plan")).
//	    Step(workflow.AgentStep("write", writer, func(s *workflow.State) string {
//	        return "Synthesize:\n" + s.Get("web") + "\n" + s.Get("docs")
//	    }, "web", "docs"))
//	res, err := wf.Run(ctx, "state of Go generics")
//	fmt.Println(res.Output)              // terminal step output
//	fmt.Printf("%+v\n", res.Usage)       // aggregated cost/tokens across steps
//
// Here "web" and "docs" both depend only on "plan", so they run in parallel.
package workflow
