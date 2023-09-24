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

	nodeCount := 4

	// create topics, one for each node
	topics := make([]*Topic, nodeCount)
	for i := 0; i < nodeCount; i++ {
		topics[i] = NewTopic()
	}

	// create aggregators -- this is the graph edge structure
	aggregators := make([]*Aggregator, nodeCount)
	for i := 2; i < nodeCount; i++ {
		// connect each aggregator's inputs to the previous two topics
		ag := NewAggregator([]*Topic{topics[i-2], topics[i-1]})
		aggregators[i] = ag
	}

	// create nodes
	nodes := make([]*Node, nodeCount)

	/*
		// create input nodes
		fn0 := Function{Fn: func(args ...float64) float64 { return 1.0 }, ArgCount: 0}
		fn1 := Function{Fn: func(args ...float64) float64 { return 2.0 }, ArgCount: 0}
		node0 := NewNode(0, fn0, nil, topics[0])
		node1 := NewNode(1, fn1, nil, topics[1])
		nodes[0] = node0
		nodes[1] = node1
	*/

	// create a function
	fn := Function{
		Fn: func(args ...float64) float64 {
			return args[0] + args[1]
		},
		ArgCount: 2,
	}

	// create nodes
	for i := 2; i < nodeCount; i++ {
		node := NewNode(uint64(i), fn, aggregators[i], topics[i])
		nodes[i] = node
	}

	// subscribe to the last node's output
	resultChan := nodes[nodeCount-1].Output.Subscribe()

	// inspect the node graph
	for i := 2; i < nodeCount; i++ {
		Assert(nodes[i] != nil, "expected node %d to be non-nil", i)
		Assert(nodes[i].Id == uint64(i), "expected node %d to have id %d, got %d", i, i, nodes[i].Id)
		Pf("node %d: ", i)
		ag := nodes[i].Input
		Pf("input aggregator: %p ", ag)
		Assert(ag != nil, "expected node %d to have non-nil input aggregator", i)
		Pf("aggregator inputs: ")
		for j, input := range ag.Input {
			Pf("%p ", input)
			Assert(input == topics[i-2+j], "expected aggregator %d input %d to be topic %d, got %d", i, j, i-2+j, input)
		}
		Pf("node output: %p ", nodes[i].Output)
		Assert(nodes[i].Output == topics[i], "expected node %d output to be topic %d, got %d", i, i, nodes[i].Output)
		Pl()
	}

	// publish values to the first two topics
	topics[0].Publish <- 1.0
	topics[1].Publish <- 2.0

	// read results from the result channel
	var result float64
	for result = range resultChan {
		Pl(result)
		// close the first two topics -- this will cause
		// everything to shut down and the result channel to close
		close(topics[0].Publish)
		close(topics[1].Publish)
	}

	// simulate the node results
	expecteds := make([]float64, nodeCount)
	for i := 0; i < nodeCount; i++ {
		switch i {
		case 0:
			expecteds[i] = 1.0
		case 1:
			expecteds[i] = 2.0
		default:
			expecteds[i] = expecteds[i-1] + expecteds[i-2]
		}
	}
	expected := expecteds[nodeCount-1]
	Pf("expected result: %f got %f\n", expected, result)
	Assert(result == expected, "expected result %f, got %f", expected, result)
}
