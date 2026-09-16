package workflow

import (
	"context"
	"fmt"
	"sync"

	"github.com/matiasinsaurralde/rimeno"
)

// StepOutput is what a step produces: its text plus optional usage/trace (set by
// agent steps) for aggregation and comparison.
type StepOutput struct {
	Text  string
	Usage rimeno.Usage
	Trace *rimeno.Trace
}

// Step is a named unit of work with declared dependencies. It runs once all
// dependencies have completed; its Text output is stored in State under Name.
type Step struct {
	Name      string
	DependsOn []string
	Run       func(ctx context.Context, s *State) (StepOutput, error)
}

// AgentStep runs a rimeno.Agent. The prompt is built from State (e.g. the workflow
// input or upstream step outputs). Usage and trace are captured for aggregation.
func AgentStep(name string, agent *rimeno.Agent, prompt func(s *State) string, deps ...string) *Step {
	return &Step{
		Name:      name,
		DependsOn: deps,
		Run: func(ctx context.Context, s *State) (StepOutput, error) {
			res, err := agent.Run(ctx, prompt(s))
			if err != nil {
				return StepOutput{}, err
			}
			return StepOutput{Text: res.Text, Usage: res.Usage, Trace: res.Trace}, nil
		},
	}
}

// FuncStep runs arbitrary Go for glue/transform logic between agents.
func FuncStep(name string, fn func(ctx context.Context, s *State) (string, error), deps ...string) *Step {
	return &Step{
		Name:      name,
		DependsOn: deps,
		Run: func(ctx context.Context, s *State) (StepOutput, error) {
			out, err := fn(ctx, s)
			return StepOutput{Text: out}, err
		},
	}
}

// Workflow is a DAG of steps.
type Workflow struct {
	name  string
	steps []*Step
	index map[string]*Step
}

// New creates an empty workflow.
func New(name string) *Workflow {
	return &Workflow{name: name, index: map[string]*Step{}}
}

// Step adds a step. It panics on a duplicate step name (a construction error).
func (w *Workflow) Step(s *Step) *Workflow {
	if _, dup := w.index[s.Name]; dup {
		panic(fmt.Sprintf("workflow: duplicate step %q", s.Name))
	}
	w.index[s.Name] = s
	w.steps = append(w.steps, s)
	return w
}

// Result is the outcome of a workflow run.
type Result struct {
	// Output is the terminal step's output (the last step added).
	Output string
	// Outputs holds every step's output keyed by name.
	Outputs map[string]string
	// State is the final blackboard.
	State *State
	// Usage is the aggregated usage across all agent steps.
	Usage rimeno.Usage
	// StepTraces holds each agent step's trace, keyed by step name.
	StepTraces map[string]*rimeno.Trace
}

// Run executes the workflow. Steps run as soon as their dependencies complete;
// independent steps run concurrently. The first step error cancels the rest and
// is returned (fail-fast).
func (w *Workflow) Run(ctx context.Context, input string) (*Result, error) {
	if err := w.validate(); err != nil {
		return nil, err
	}
	state := newState(input)
	res := &Result{
		Outputs:    map[string]string{},
		State:      state,
		StepTraces: map[string]*rimeno.Trace{},
	}

	var mu sync.Mutex // guards done/res
	done := map[string]bool{}
	cctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var firstErr error

	for len(done) < len(w.steps) {
		// Collect steps whose dependencies are all satisfied.
		var ready []*Step
		mu.Lock()
		for _, st := range w.steps {
			if done[st.Name] {
				continue
			}
			if depsMet(st, done) {
				ready = append(ready, st)
			}
		}
		mu.Unlock()

		if len(ready) == 0 {
			return res, fmt.Errorf("workflow %q: no runnable steps (cycle or missing dependency)", w.name)
		}

		var wg sync.WaitGroup
		for _, st := range ready {
			wg.Add(1)
			go func(st *Step) {
				defer wg.Done()
				out, err := safeRunStep(cctx, st, state)
				mu.Lock()
				defer mu.Unlock()
				if err != nil {
					if firstErr == nil {
						firstErr = fmt.Errorf("workflow %q: step %q: %w", w.name, st.Name, err)
						cancel()
					}
					return
				}
				state.Set(st.Name, out.Text)
				res.Outputs[st.Name] = out.Text
				res.Usage.Add(out.Usage)
				if out.Trace != nil {
					res.StepTraces[st.Name] = out.Trace
				}
				done[st.Name] = true
			}(st)
		}
		wg.Wait()

		if firstErr != nil {
			return res, firstErr
		}
	}

	if n := len(w.steps); n > 0 {
		res.Output = res.Outputs[w.steps[n-1].Name]
	}
	return res, nil
}

// safeRunStep runs a step, converting a panic into a step error rather than
// crashing the process (steps run in goroutines, where a panic is fatal). The
// workflow then fails fast on it like any other step error.
func safeRunStep(ctx context.Context, st *Step, state *State) (out StepOutput, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()
	return st.Run(ctx, state)
}

func depsMet(s *Step, done map[string]bool) bool {
	for _, d := range s.DependsOn {
		if !done[d] {
			return false
		}
	}
	return true
}

// validate checks that dependencies exist and the graph is acyclic.
func (w *Workflow) validate() error {
	for _, st := range w.steps {
		for _, d := range st.DependsOn {
			if _, ok := w.index[d]; !ok {
				return fmt.Errorf("workflow %q: step %q depends on unknown step %q", w.name, st.Name, d)
			}
		}
	}
	// Kahn's algorithm cycle check.
	indeg := map[string]int{}
	for _, st := range w.steps {
		indeg[st.Name] = len(st.DependsOn)
	}
	queue := []string{}
	for name, d := range indeg {
		if d == 0 {
			queue = append(queue, name)
		}
	}
	dependents := map[string][]string{}
	for _, st := range w.steps {
		for _, d := range st.DependsOn {
			dependents[d] = append(dependents[d], st.Name)
		}
	}
	visited := 0
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		visited++
		for _, m := range dependents[n] {
			indeg[m]--
			if indeg[m] == 0 {
				queue = append(queue, m)
			}
		}
	}
	if visited != len(w.steps) {
		return fmt.Errorf("workflow %q: dependency cycle detected", w.name)
	}
	return nil
}
