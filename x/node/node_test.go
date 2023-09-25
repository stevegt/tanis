package node

import (
	"net/http"
	"testing"

	. "github.com/stevegt/goadapt"

	_ "net/http/pprof"
)

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
	topic0 := NewTopic(size)
	topic1 := NewTopic(size)

	// create nodes
	// - subscriptions are the graph edges
	// - this particular graph is a fibonacci sequence
	nodes := make([]*Node, nodeCount)
	for i := 0; i < nodeCount; i++ {
		switch i {
		case 0:
			nodes[i] = NewNode(uint64(i), addFn, size, topic0, topic1)
		case 1:
			nodes[i] = NewNode(uint64(i), addFn, size, topic1, nodes[i-1])
		default:
			nodes[i] = NewNode(uint64(i), addFn, size, nodes[i-2], nodes[i-1])
		}
	}

	// subscribe to the last node's output
	resultChan := nodes[nodeCount-1].Subscribe(size)

	// inspect the node graph
	for i := 0; i < nodeCount; i++ {
		Assert(nodes[i] != nil, "expected node %d to be non-nil", i)
		Assert(nodes[i].Id == uint64(i), "expected node %d to have id %d, got %d", i, i, nodes[i].Id)
		Pf("node %d: ", i)
		ag := nodes[i].Input
		Pf("input: %#v ", ag)
		Assert(ag != nil, "expected node %d to have non-nil input aggregator", i)
		Pf("aggregator inputs: ")
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

	// create a graph with a buffer size of 0
	// - AddNode calls are the graph edges
	// - this particular graph is a fibonacci sequence
	g := NewGraph(0)
	inputTopics := g.NameInputs("a", "b")
	node0 := g.AddNode(uint64(0), addFn, inputTopics["a"], inputTopics["b"])
	node1 := g.AddNode(uint64(1), addFn, inputTopics["b"], node0)
	for i := 2; i < nodeCount; i++ {
		node := g.AddNode(uint64(i), addFn, node0, node1)
		node0, node1 = node1, node
	}
	outputTopics := g.NameOutputs("y")

	// subscribe to the last node's output
	resultChan := outputTopics["y"].Subscribe(size)
	Assert(len(g.OpenOutputs()) == 0, "expected 0 open outputs, got %d", g.OpenOutputs)

	// publish values to the first two topics
	inputTopics["a"].Publish <- 1.0
	inputTopics["b"].Publish <- 2.0

	// read results from the result channel
	var result float64
	for result = range resultChan {
		Pl(result)
		// close the input topics -- this will cause
		// everything to shut down and the result channel to close
		for _, inputTopic := range inputTopics {
			close(inputTopic.Publish)
		}
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
