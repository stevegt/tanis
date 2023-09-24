package main

import (
	"sync"
	"time"

	. "github.com/stevegt/goadapt"

	_ "net/http/pprof"

	"net/http"
)

type Function struct {
	Fn       func(...float64) float64
	ArgCount int
}

type Node struct {
	Id          uint64
	Function    Function
	InputChans  []chan float64
	Subscribe   chan chan float64
	Subscribers []chan float64
}

// NewNode creates a new node with the given Function.  It returns a
// slice of input channels and a single Subscribe channel.  Reading
// from the Subscribe channel will return a channel which will receive
// each output of the Function.  An output is generated each time all
// input channels have received a value and the Function has
// calculated the result.  The number of channels in the input slice
// is equal to the number of arguments expected by the Function.
func NewNode(id uint64, fn Function) (n *Node) {
	n = &Node{
		Id:        id,
		Function:  fn,
		Subscribe: make(chan chan float64),
	}

	// generate output channel for each subscriber
	go func() {
		for {
			// create a new output channel
			output := make(chan float64, 10)
			// send to the next subscriber
			I("node %d waiting for subscriber... ", n.Id)
			n.Subscribe <- output
			I("node %d sent output channel to subscriber\n", n.Id)
			// add to the list of outputs
			n.Subscribers = append(n.Subscribers, output)
		}
	}()
	return
}

// Run causes the node to subscribe to the given nodes, then starts
// the calculation goroutine. Before starting the goroutine, each node
// in the given slice is connected to the corresponding input channel
// of this node.  The number of nodes in the slice must be equal to
// the number of arguments expected by the node's function.
func (n *Node) Run(topics []*Node) {
	Assert(len(topics) == n.Function.ArgCount, "wrong number of topics: %d, argcount %d", len(topics), n.Function.ArgCount)
	// for each upstream node...
	I("node %d subscribing to %d upstream nodes\n", n.Id, len(topics))
	for i := range topics {
		// subscribe to the node's output
		I("node %d subscribing to node %d... ", n.Id, topics[i].Id)
		outChan := <-topics[i].Subscribe
		I("node %d received output channel from node %d\n", n.Id, topics[i].Id)
		// add to our input channels
		n.InputChans = append(n.InputChans, outChan)
	}

	go func() {
		defer func() {
			// notify subscribers on exit
			for _, subscriber := range n.Subscribers {
				close(subscriber)
			}
		}()

		// read from all input channels until they are closed
		closedCount := 0
		for {
			// read from all input channels. we need to read from each
			// channel in a separate goroutine to avoid blocking and
			// handle the case where one or more channels are closed.
			inputs := make([]float64, len(n.InputChans))
			wg := &sync.WaitGroup{}
			for i := range n.InputChans {
				wg.Add(1)
				// start goroutine for channel i
				go func(i int) {
					defer wg.Done()
					// read one value from channel i
					I("node %d waiting for input on channel %d... ", n.Id, i)
					input, ok := <-n.InputChans[i]
					if ok {
						I("node %d received input on channel %d\n", n.Id, i)
						// channel is open
						inputs[i] = input
						Pl("node", n.Id, "received", input, "on input", i)
						return
					}
					// channel is closed
					closedCount++
					Pl("node", n.Id, "input", i, "closed")
				}(i)
			}
			// wait for all input channels to be read
			wg.Wait()
			if closedCount == len(n.InputChans) {
				Pl("node", n.Id, "all inputs closed")
				return
			}

			// calculate the result
			result := n.Function.Fn(inputs...)
			// send the result to all subscribers
			for _, subscriber := range n.Subscribers {
				I("node %d sending result to subscriber... ", n.Id)
				subscriber <- result
				I("node %d sent result to subscriber\n", n.Id)
			}
		}
	}()
}

func main() {
	logger = NewLog()

	go func() {
		Pl(http.ListenAndServe("localhost:6060", nil))
	}()

	// create a function
	fn := Function{
		Fn: func(args ...float64) float64 {
			return args[0] + args[1]
		},
		ArgCount: 2,
	}

	var nodes []*Node

	// create input nodes
	input1 := NewNode(0, Function{Fn: func(args ...float64) float64 { return 1.0 }, ArgCount: 0})
	input2 := NewNode(1, Function{Fn: func(args ...float64) float64 { return 2.0 }, ArgCount: 0})
	nodes = append(nodes, input1)
	nodes = append(nodes, input2)

	// create more nodes
	for i := 0; i < 2; i++ {
		nodes = append(nodes, NewNode(uint64(len(nodes)), fn))
	}

	// run the nodes
	input1.Run(nil)
	input2.Run(nil)
	for i := 2; i < len(nodes); i++ {
		// connect each node's inputs to the previous two nodes' outputs
		me := nodes[i]
		upstream1 := nodes[i-2]
		upstream2 := nodes[i-1]
		Pl("connecting node", me.Id, "to upstream nodes", upstream1.Id, upstream2.Id)
		nodes[i].Run([]*Node{upstream1, upstream2})
	}

	lastNode := nodes[len(nodes)-1]
	Pl("main subscribing to last node", lastNode.Id)
	resultChan := <-lastNode.Subscribe

	time.Sleep(1 * time.Second)

	for _, node := range nodes {
		println("node", node.Id, "has", len(node.Subscribers), "subscribers")
	}

	// read results from the last node
	Pl("main reading from result channel... ")
	for result := range resultChan {
		println(result)
	}

}

// Log collects messages in a channel and writes them to stdout.
type Log struct {
	MsgChan chan string
}

// NewLog creates a new Log.
func NewLog() (l *Log) {
	l = &Log{
		MsgChan: make(chan string, 100),
	}
	go func() {
		for msg := range l.MsgChan {
			Pl(msg)
		}
	}()
	return
}

// I logs a message.
func I(args ...interface{}) {
	msg := FormatArgs(args...)
	logger.MsgChan <- msg
}

var logger *Log
