package main

import (
	"sync"
	"time"

	. "github.com/stevegt/goadapt"
)

// WorkRequest is a struct that contains the work to be done
type WorkRequest struct {
	Work func()
}

// StartWorker starts a worker. The work channel is used to send work
// to the worker.  The worker will quit when the work channel is
// closed.
func StartWorker(work chan *WorkRequest) {
	go func() {
		for {
			workRequest, ok := <-work
			if !ok {
				// Channel closed
				return
			}
			workRequest.Work()
		}
	}()
}

// Function is a function which takes a slice of float64 arguments and
// returns a float64 result.  The number of arguments is specified by
// ArgCount.
type Function struct {
	ArgCount int
	Fn       func(...float64) float64
}

// Node is a node in a graph.  It contains a function and a slice
// of argument ids.  The result field is filled in by a worker.  The
// result is the result of the function applied to the arguments. The
// Id field is used to identify the node, and the ArgIds field is used
// to identify the nodes where the function arguments are generated.
type Node struct {
	Id       uint64
	Function string
	ArgIds   []uint64
}

// Xeq executes a node.  It looks up the function in a function table,
// looks up the arguments in a sync.Map, and posts a work request and
// the original message back to the work channel if there are any
// cache misses.  If there are no cache misses, the worker
// executes the function and posts the result in the sync.Map.
func Xeq(id uint64, graph Graph, functions map[string]Function, cache *sync.Map, work chan *WorkRequest) {
	Pf("Executing node %d... ", id)
	// Look up the function
	node, ok := graph[id]
	Assert(ok, "node not found: %d", id)
	function, ok := functions[node.Function]
	Assert(ok, "function not implemented: %s", node.Function)
	// Look up the arguments
	args := make([]float64, len(node.ArgIds))
	for i, argId := range node.ArgIds {
		arg, ok := cache.Load(argId)
		if !ok {
			// Post a work request and the original message back to the work channel
			Pf("posting: %d ", argId)
			work <- &WorkRequest{Work: func() {
				Xeq(argId, graph, functions, cache, work)
			}}
			Pf("reposting: %d ", id)
			work <- &WorkRequest{Work: func() {
				Xeq(id, graph, functions, cache, work)
			}}
			return
		}
		Pf("found arg: %d ", argId)
		args[i] = arg.(float64)
	}
	// Execute the function
	result := function.Fn(args...)
	// Post the result in the cache
	cache.Store(node.Id, result)
	Pl()
}

// Graph is a directed acyclic graph.  It is a map from node ids to
// nodes.  The nodes are messages.  The edges are the ArgIds in the
// messages.
type Graph map[uint64]Node

func main() {

	// Build a function table
	functions := map[string]Function{
		"add": Function{2, func(args ...float64) float64 { return args[0] + args[1] }},
		"sub": Function{2, func(args ...float64) float64 { return args[0] - args[1] }},
		"mul": Function{2, func(args ...float64) float64 { return args[0] * args[1] }},
		"div": Function{2, func(args ...float64) float64 { return args[0] / args[1] }},
		"one": Function{0, func(args ...float64) float64 { return 1 }},
		"two": Function{0, func(args ...float64) float64 { return 2 }},
	}

	// Build a graph
	graph := make(Graph, 10)
	graph[0] = Node{0, "add", []uint64{2, 3}}
	graph[1] = Node{1, "sub", []uint64{3, 4}}
	graph[2] = Node{2, "mul", []uint64{4, 5}}
	graph[3] = Node{3, "div", []uint64{5, 6}}
	graph[4] = Node{4, "one", []uint64{}}
	graph[5] = Node{5, "two", []uint64{}}
	graph[6] = Node{6, "two", []uint64{}}

	// Execute the graph
	//
	// The idea here is that a node is a message that is sent to a
	// worker.  The worker looks up the function in a function table,
	// looks up the arguments in a sync.Map, and posts a work request and
	// the original message back to the work channel if there are any
	// cache misses.  If there are no cache misses, the worker
	// executes the function and posts the result in the sync.Map.
	//
	// The main thread waits for the sync.Map to be filled in, then
	// prints the result.

	work := make(chan *WorkRequest, 100)
	for i := 0; i < 50; i++ {
		StartWorker(work)
	}

	var cache sync.Map

	for i, _ := range graph {
		work <- &WorkRequest{Work: func() {
			Xeq(i, graph, functions, &cache, work)
		}}
	}

	// Print the result from node 0
	// Wait for the result to be posted
	var result float64
	for {
		time.Sleep(1 * time.Second)
		// show the cache
		for i := 0; i < 10; i++ {
			res, ok := cache.Load(uint64(i))
			if ok {
				Pf("%d: %f\n", i, res.(float64))
			}
		}
		res, ok := cache.Load(uint64(0))
		if !ok {
			work <- &WorkRequest{Work: func() {
				Xeq(0, graph, functions, &cache, work)
			}}
			continue
		}
		result = res.(float64)
		Pf("Result = %v\n", result)
	}

}
