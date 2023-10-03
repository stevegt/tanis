package node

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"testing"

	. "github.com/stevegt/goadapt"

	"net/http"
	_ "net/http/pprof"
)

func add(args ...float64) float64 {
	var res float64
	for _, arg := range args {
		res += arg
	}
	return res
}

func sub(args ...float64) float64 {
	var res float64
	for _, arg := range args {
		res -= arg
	}
	return res
}

func mul(args ...float64) float64 {
	var res float64
	for _, arg := range args {
		res *= arg
	}
	return res
}

func div(args ...float64) float64 {
	var res float64
	for _, arg := range args {
		res /= arg
	}
	return res
}

// RandomKey returns a random key from a map. In this code, `K` refers
// to the key data type, and `V` refers to the value data type. The
// keyword `comparable` indicates that `K` must be a comparable type,
// while V can be any type.
func RandomKey[K comparable, V any](m map[K]V) K {
	Assert(len(m) > 0, "expected map to be non-empty")
	var keys []K
	for k := range m {
		keys = append(keys, k)
	}
	key := keys[rand.Intn(len(keys))]
	return key
}

// saveDot saves a graphviz dot file to /tmp
func saveDot(g *Graph) {
	name := g.Name
	dot := g.Draw()
	err := ioutil.WriteFile(fmt.Sprintf("/tmp/%s.dot", name), []byte(dot), 0644)
	Ck(err)
	Pf("saved /tmp/%s.dot\n", name)
}

func TestWrap(t *testing.T) {
	// create a function
	fn := func(args ...float64) float64 {
		var sum float64
		for _, arg := range args {
			sum += arg
		}
		return sum
	}

	// create a wrapped function
	inputNames := []string{"a", "b", "c"}
	outputName := "y"
	node := Wrap(fn, inputNames, outputName)

	// make an input map
	inputMap := map[string]float64{
		"a": 1.0,
		"b": 2.0,
		"c": 3.0,
	}

	// call the wrapped function via the node
	resultMap := node.F(inputMap)
	Assert(resultMap[outputName] == 6.0, "expected result %f, got %f", 6.0, resultMap[outputName])
}

func TestEdge(t *testing.T) {
	// create an edge
	edge := NewEdge("a", nil, 0)

	// subscribe to the edge
	rc1 := edge.Subscribe(nil, 0)
	rc2 := edge.Subscribe(nil, 0)

	// publish a value to the edge
	edge.Send(1.0)
	close(edge.Publish)

	// read the result from the result channels
	var result float64
	result = <-rc1
	Tassert(t, result == 1.0, "expected result %f, got %f", 1.0, result)
	result = <-rc2
	Tassert(t, result == 1.0, "expected result %f, got %f", 1.0, result)
	for result = range rc1 {
		Tassert(t, false, "got extra result %f", result)
	}
	for result = range rc2 {
		Tassert(t, false, "got extra result %f", result)
	}

}

func TestNode(t *testing.T) {
	// create a zero node
	node := &Node{}
	// try calling it
	result := node.F(nil)
	Tassert(t, result == nil, "expected result %v, got %v", nil, result)

	// create a function with multiple outputs
	fn := func(args ...float64) []float64 {
		var sum float64
		var product float64 = 1.0
		for _, arg := range args {
			sum += arg
			product *= arg
		}
		return []float64{sum, product}
	}

	// create a node with multiple outputs
	inputNames := []string{"a", "b", "c", "d"}
	outputNames := []string{"x", "y"}
	node = WrapMulti(fn, inputNames, outputNames)
	// try calling it
	inputMap := map[string]float64{
		"a": 1.0,
		"b": 2.0,
		"c": 3.0,
		"d": 4.0,
	}
	resultMap := node.F(inputMap)
	Pf("resultMap: %#v\n", resultMap)
	Tassert(t, resultMap["x"] == 10.0, "expected result %f, got %f", 10.0, resultMap["x"])
	Tassert(t, resultMap["y"] == 24.0, "expected result %f, got %f", 24.0, resultMap["y"])

	// ensure we're implementing Function
	var _ Function = node

}

func TestJoin(t *testing.T) {
	inputNames := []string{"a", "b", "c"}
	width := len(inputNames)

	// create an empty graph -- this should create an edge for each
	// input name
	g := NewGraph("joiner", 0, inputNames)

	// join the inputs into a single channel
	outChan := g.join(nil, inputNames)

	// send a value to each input
	inputs := map[string]float64{
		"a": 1.0,
		"b": 2.0,
		"c": 3.0,
	}
	for name, value := range inputs {
		g.edges[name].Send(value)
	}

	// read the result from the output channel
	result := <-outChan

	// check the result
	Assert(len(result) == width, "expected %d results, got %d", width, len(result))
	for name, value := range inputs {
		Tassert(t, result[name] == value, "expected result %f, got %f", value, result[name])
	}

	// close the input channels
	for _, edge := range g.edges {
		close(edge.Publish)
	}

	// make sure output channel is closed
	for result = range outChan {
		Tassert(t, false, "got extra result %f", result)
	}

}

func TestGraph(t *testing.T) {

	/*
		// see http://localhost:6060/debug/pprof/goroutine?debug=2 for deadlocks
		go func() {
			Pl(http.ListenAndServe("localhost:6060", nil))
		}()
	*/

	// Build a function table
	functions := map[string]func(args ...float64) float64{
		"add": add,
		"sub": sub,
		"mul": mul,
		"div": div,
		"one": func(args ...float64) float64 { return 1 },
		"two": func(args ...float64) float64 { return 2 },
	}

	// Build a simple network using a Graph
	inputNames := []string{"a", "b"}
	g := NewGraph("fib", 0, inputNames)
	// - this particular graph is a fibonacci sequence
	g.AddNode(Wrap(functions["add"], []string{"a", "b"}, "c"))
	g.AddNode(Wrap(functions["add"], []string{"b", "c"}, "d"))
	g.AddNode(Wrap(functions["add"], []string{"c", "d"}, "e"))
	g.AddNode(Wrap(functions["add"], []string{"d", "e"}, "f"))
	g.AddNode(Wrap(functions["add"], []string{"e", "f"}, "g"))
	g.Start()

	// make an input map
	inputMap := map[string]float64{
		"a": 0.0,
		"b": 1.0,
	}
	res := g.F(inputMap)
	Pl(res)

	// make sure the result is correct
	Assert(len(res) == 1, "expected %d result, got %d", 1, len(res))
	Assert(res["g"] == 8.0, "expected result %f, got %f", 8.0, res["g"])

	// save the graph
	saveDot(g)
}

func TestRandomGraph(t *testing.T) {

	// see http://localhost:6060/debug/pprof/goroutine?debug=2 for deadlocks
	go func() {
		Pl(http.ListenAndServe("localhost:6060", nil))
	}()

	rand.Seed(1)
	// Build a function table
	functions := map[string]func(args ...float64) float64{
		"add": add,
		"sub": sub,
		"mul": mul,
		"div": div,
		"one": func(args ...float64) float64 { return 1 },
		"two": func(args ...float64) float64 { return 2 },
	}

	// Build a small random network using a Graph
	inputNames := []string{"a", "b", "c"}
	g := NewGraph("randomGraph", 0, inputNames)
	// create a bunch of nodes
	for i := 0; i < 100; i++ {
		// pick a random function
		fn := functions[RandomKey(functions)]

		// pick one or more random edges from the existing edges to use
		// as inputs
		upstreams := make(map[string]bool)
		for {
			if rand.Float64() < 0.3 {
				name := RandomKey(g.edges)
				Tassert(t, name != "", "expected name to be non-empty")
				// prevent duplicate upstreams
				_, ok := upstreams[name]
				if ok {
					continue
				}
				upstreams[name] = true
			} else {
				// ensure we have at least one upstream
				if len(upstreams) > 1 {
					break
				}
			}
		}

		// add the node
		var upstreamNames []string
		for name, _ := range upstreams {
			upstreamNames = append(upstreamNames, name)
		}
		g.AddNode(Wrap(fn, upstreamNames, uname()))
	}

	// start the graph
	g.Start()

	// make an input map
	inputMap := map[string]float64{
		"a": 0.0,
		"b": 1.0,
		"c": 2.0,
	}

	saveDot(g)

	// run the graph
	res := g.F(inputMap)
	Pl(res)

	// shut down
	g.Stop()

	// try reading from the output joiner
	for result := range g.graphOutputChan {
		Tassert(t, false, "got extra result %f", result)
	}

}
