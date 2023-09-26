package node

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"sync"
	"testing"

	. "github.com/stevegt/goadapt"

	_ "net/http/pprof"
)

// RandomKey returns a random key from a map. In this code, `K` refers
// to the key data type, and `V` refers to the value data type. The
// keyword `comparable` indicates that `K` must be a comparable type,
// while V can be any type.
func RandomKey[K comparable, V any](m map[K]V) K {
	var keys []K
	for k := range m {
		keys = append(keys, k)
	}

	return keys[rand.Intn(len(keys))]
}

func TestSimple(t *testing.T) {
	logger = NewLog()

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
	nodes := make([]*Node, nodeCount)
	for i := 0; i < nodeCount; i++ {
		name := fmt.Sprintf("%d", i)
		switch i {
		case 0:
			nodes[i] = NewNode(name, addFn, size, topic0, topic1)
		case 1:
			nodes[i] = NewNode(name, addFn, size, topic1, nodes[i-1])
		default:
			nodes[i] = NewNode(name, addFn, size, nodes[i-2], nodes[i-1])
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

func TestNet(t *testing.T) {
	rand.Seed(1)

	// Build a function table
	functions := map[string]Function{
		"add": Function{add},
		"sub": Function{sub},
		"mul": Function{mul},
		"div": Function{div},
		"one": Function{func(args ...float64) float64 { return 1 }},
		"two": Function{func(args ...float64) float64 { return 2 }},
	}

	// Build a small random network using a Graph
	g := NewGraph(0)
	inputTopics := g.NameInputs("a", "b", "c")
	for i := 0; i < 20; i++ {
		// pick a random function
		fn := functions[RandomKey(functions)]
		// pick random inputs for node
		inputs := make([]Publisher, 0)

		// maybe pick a random input topic
		if rand.Float64() < 0.1 {
			inputs = append(inputs, inputTopics[RandomKey(inputTopics)])
		}

		// pick zero or more random nodes from the existing nodes to use
		// as inputs
		for {
			if rand.Float64() < 0.7 && len(g.Nodes) > 0 {
				j := rand.Intn(len(g.Nodes))
				inputs = append(inputs, g.Nodes[j])
			} else {
				break
			}
		}

		// add the node
		g.AddNode(uint64(i), fn, inputs...)
	}
	Pf("open outputs: %d\n", len(g.OpenOutputs()))
	// assign names to all of the open outputs
	var names []string
	for i := 0; i < len(g.OpenOutputs()); i++ {
		name := fmt.Sprintf("y%d", i)
		names = append(names, name)
	}
	outputTopics := g.NameOutputs(names...)

	// subscribe to all of the output topics
	var resultChans []chan float64
	for _, outputTopic := range outputTopics {
		resultChans = append(resultChans, outputTopic.Subscribe(0))
	}
	Assert(len(g.OpenOutputs()) == 0, "expected 0 open outputs, got %d", g.OpenOutputs)

	// publish values to the input topics
	for _, inputTopic := range inputTopics {
		inputTopic.Publish <- rand.Float64()
	}

	// read results from the result channels
	var results []float64
	for _, resultChan := range resultChans {
		results = append(results, <-resultChan)
	}
	Pl(results)

	dot := g.DrawDot()
	ioutil.WriteFile("/tmp/node_test.dot", []byte(dot), 0644)

	/*

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
	*/
}
