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
	for i := 0; i < 2; i++ {
		// pick a random function
		fn := functions[RandomKey(functions)]

		// pick one or more random edges from the existing edges to use
		// as inputs
		upstreams := make(map[string]bool)
		for {
			if rand.Float64() < 0.7 {
				name := RandomKey(g.edges)
				Assert(name != "", "expected name to be non-empty")
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
		upstreamNames := make([]string, len(upstreams))
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

	// dot := g.DrawDot()
	// ioutil.WriteFile("/tmp/node_test.dot", []byte(dot), 0644)

}

/*

func TestSimple(t *testing.T) {
	logger = NewLog()

	// see localhost:6060/debug/pprof
	go func() {
		Pl(http.ListenAndServe("localhost:6060", nil))
	}()

	nodeCount := 3

	// create functions
	addFn := Function{
		Fn: func(args ...float64) float64 {
			var sum float64
			for _, arg := range args {
				sum += arg
			}
			return sum
		},
	}

	// an acyclic graph should not need a buffer size > 0
	size := 0

	// create input topics
	topic0 := NewTopic("0", size)
	topic1 := NewTopic("1", size)

	// create nodes
	// - subscriptions are the graph edges
	// - this particular graph is a fibonacci sequence
	nodes := make([]*Graph, nodeCount)
	for i := 0; i < nodeCount; i++ {
		name := fmt.Sprintf("%d", i)
		switch i {
		case 0:
			nodes[i] = NewGraph(name, addFn, size, topic0, topic1)
		case 1:
			nodes[i] = NewGraph(name, addFn, size, topic1, nodes[i-1])
		default:
			nodes[i] = NewGraph(name, addFn, size, nodes[i-2], nodes[i-1])
		}
	}

	// subscribe to the last node's output
	resultChan := nodes[nodeCount-1].Subscribe(size)

	// inspect the node graph
	for i := 0; i < nodeCount; i++ {
		iname := fmt.Sprintf("%d", i)
		Assert(nodes[i] != nil, "expected node %d to be non-nil", i)
		Assert(nodes[i].Name == iname, "expected node %d to have id %d, got %d", i, i, nodes[i].Name)
		Pf("node %d: ", i)
		ag := nodes[i].Input
		Pf("input: %#v ", ag)
		Assert(ag != nil, "expected node %d to have non-nil input joiner", i)
		Pf("joiner inputs: ")
		for _, inputChan := range ag.InputChans {
			Pf("%#v ", inputChan)
		}
		Pf("node output: %#v ", nodes[i].Output)
		Pl()
	}

	// publish values to the first two topics
	topic0.Publish <- 1.0
	topic1.Publish <- 2.0

	// read results from the result channel
	var result float64
	for result = range resultChan {
		Pl(result)
		// close the input topics -- this will cause
		// everything to shut down and the result channel to close
		close(topic0.Publish)
		close(topic1.Publish)
	}

	// simulate the node results
	expecteds := make([]float64, nodeCount+2)
	for i := 0; i < nodeCount+2; i++ {
		switch i {
		case 0:
			expecteds[i] = 1.0
		case 1:
			expecteds[i] = 2.0
		default:
			expecteds[i] = expecteds[i-1] + expecteds[i-2]
		}
	}
	expected := expecteds[len(expecteds)-1]
	Pf("expected result: %f got %f\n", expected, result)
	Assert(result == expected, "expected result %f, got %f", expected, result)
}

func TestSplitter(t *testing.T) {
	// create an input channel
	inputChan := make(chan []float64)
	// create a splitter
	inputNames := []string{"a", "b", "c", "d", "e", "f", "g"}
	width := len(inputNames)
	splitter := NewSplitter("1", inputNames, 0, inputChan)
	// subscribe to the splitter's output
	var outputChans []chan float64
	for i := 0; i < width; i++ {
		outputChans = append(outputChans, splitter.Subscribe(i, 0))
	}
	// publish values to the input channel
	inputs := []float64{0, 1.0, 2.0, 3.0, 4.0, 5.0, 6}
	inputChan <- inputs
	close(inputChan)
	// read results from the output channels
	results := make([]float64, width)
	// start a goroutine for each output channel
	received := 0
	wg := sync.WaitGroup{}
	for i, outputChan := range outputChans {
		wg.Add(1)
		go func(i int, outputChan chan float64) {
			defer wg.Done()
			for result := range outputChan {
				received++
				results[i] = result
			}
		}(i, outputChan)
	}
	wg.Wait()
	Assert(received == width, "expected %d results, got %d", width, received)
	Assert(len(results) == width, "expected %d results, got %d", width, len(results))
	for i := 0; i < width; i++ {
		Assert(results[i] == inputs[i], "expected result %f, got %f", inputs[i], results[i])
	}
}

func TestGraph(t *testing.T) {

	// create functions
	addFn := Function{
		Fn: func(args ...float64) float64 {
			var sum float64
			for _, arg := range args {
				sum += arg
			}
			return sum
		},
	}

	nodeCount := 3

	// an acyclic graph should not need a buffer size > 0
	size := 0

	//	    g.AddNode("x", ...)
	//      g.AddNode("y", ...)
	//      g.AddNode("z", ..., "x", "y")
	//      ok := g.Verify()
	// 		outputmap := g.F(inputmap) // both are map[string]float64

	// create a graph with a buffer size of 0
	// - AddNode calls include the graph edges
	// - this particular graph is a fibonacci sequence
	inputNames := []string{"a", "b"}
	outputNames := []string{"y"}
	g := NewGraph(0, inputNames, outputNames)
	g.AddNode("0", addFn, "a", "b")
	g.AddNode("1", addFn, "b", "0")
	for i := 2; i < nodeCount; i++ {
		names := []string{int2str(i - 2), int2str(i - 1)}
		node := g.AddNode(int2str(i), addFn, names...)
	}
	ok := g.Verify()
	Assert(ok, "expected graph to verify")

	inputMap := map[string]float64{
		"a": 1.0,
		"b": 2.0,
	}
	outputmap := g.F(inputmap) // both are map[string]float64

	// simulate the node results
	expecteds := make(map[string]float64)
	for i := 0; i < nodeCount+2; i++ {
		switch i {
		case 0:
			expecteds[int2str(i)] = 1.0
		case 1:
			expecteds[int2str(i)] = 2.0
		default:
			expecteds[int2str(i)] = expecteds[int2str(i-1)] + expecteds[int2str(i-2)]
		}
	}
	// check the results
	for k, expected := range expecteds {
		got := outputmap[k]
		Pf("expected result: %f got %f\n", expected, got)
		Assert(got == expected, "expected result %f, got %f", expected, got)
	}
}


*/
