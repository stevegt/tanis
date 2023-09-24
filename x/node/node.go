package main

import (
	"sync"

	. "github.com/stevegt/goadapt"

	_ "net/http/pprof"

	"net/http"
)

// Topic is a pub/sub topic.
type Topic struct {
	// Subscribers is a slice of channels which will receive values
	// published to the topic.
	Subscribers []chan []float64
	// Publish is a channel which will receive Values to be published
	// to the topic.
	Publish chan []float64
}

// NewTopic creates a new Topic with the given name.
func NewTopic() (t *Topic) {
	t = &Topic{
		Subscribers: make([]chan []float64, 0),
		Publish:     make(chan []float64, 10),
	}
	go func() {
		defer func() {
			// notify subscribers on exit
			for _, subscriber := range t.Subscribers {
				close(subscriber)
			}
		}()

		for {
			// wait for a value to be published
			value, ok := <-t.Publish
			if !ok {
				// channel is closed
				return
			}
			// send the value to all subscribers
			for _, subscriber := range t.Subscribers {
				subscriber <- value
			}
		}
	}()
	return
}

// Subscribe returns a channel which will receive values published to
// the topic.
func (t *Topic) Subscribe() (c chan []float64) {
	c = make(chan []float64, 10)
	t.Subscribers = append(t.Subscribers, c)
	return
}

// Aggregator takes several input topics and publishes a single output
// topic.  The output topic is published each time all input topics
// have received a value.
type Aggregator struct {
	InputTopics []*Topic
	OutputTopic *Topic
}

// NewAggregator creates a new Aggregator with the given input and
// output topics.
func NewAggregator(inputTopics []*Topic, outputTopic *Topic) (a *Aggregator) {
	a = &Aggregator{
		InputTopics: inputTopics,
		OutputTopic: outputTopic,
	}

	// subscribe to the input topics
	var inputChans []chan []float64
	for _, inputTopic := range a.InputTopics {
		inputChans = append(inputChans, inputTopic.Subscribe())
	}

	go func() {
		defer func() {
			// notify output topic on exit
			close(a.OutputTopic.Publish)
		}()

		// read from all input channels until they are closed
		closedCount := 0
		for {
			// read from all input channels. we need to read from each
			// in a separate goroutine to avoid blocking and handle
			// the case where one or more channels are closed.
			values := make([]float64, len(inputChans))
			wg := &sync.WaitGroup{}
			for i := range inputChans {
				wg.Add(1)
				// start goroutine for topic i
				go func(i int) {
					defer wg.Done()
					// read one value from topic i
					input, ok := <-inputChans[i]
					if ok {
						// channel is open
						// XXX only using the first value in the input
						// slice
						values[i] = input[0]
						return
					}
					// channel is closed
					closedCount++
				}(i)
			}
			// wait for all input channels to be read
			wg.Wait()
			if closedCount == len(inputChans) {
				return
			}

			// send the values slice to the output topic
			a.OutputTopic.Publish <- values

		}
	}()
	return
}

// Function is a function which takes a slice of float64 arguments and
// returns a float64 result.  The number of arguments is specified by
// ArgCount.
type Function struct {
	Fn       func(...float64) float64
	ArgCount int
}

// Node is a node in a dataflow graph.  It has a Function which
// calculates a result from a slice of input values.
type Node struct {
	Id          uint64
	Function    Function
	InputTopic  *Topic
	OutputTopic *Topic
}

// NewNode creates a new node with the given Function.
// An output is generated each time a value slice is received on the
// input topic.
func NewNode(id uint64, fn Function, inputTopic *Topic, outputTopic *Topic) (n *Node) {

	n = &Node{
		Id:          id,
		Function:    fn,
		InputTopic:  inputTopic,
		OutputTopic: outputTopic,
	}

	var inputChan chan []float64
	if n.InputTopic != nil {
		// subscribe to the input topic
		c := n.InputTopic.Subscribe()
		inputChan = c
	}

	outputChan := n.OutputTopic.Publish

	go func() {
		defer func() {
			// notify output topic on exit
			close(outputChan)
		}()

		for {
			var inputs []float64
			if inputChan != nil {
				// read a value slice from the input topic
				in, ok := <-inputChan
				inputs = in
				if !ok {
					// channel is closed
					return
				}
			}

			// calculate the result
			result := n.Function.Fn(inputs...)
			// send the result to the output topic
			outputChan <- []float64{result}
		}
	}()
	return
}

func main() {
	logger = NewLog()

	go func() {
		Pl(http.ListenAndServe("localhost:6060", nil))
	}()

	nodeCount := 4

	// create topics, one for each node
	var topics []*Topic
	for i := 0; i < nodeCount; i++ {
		topics = append(topics, NewTopic())
	}

	// create aggregators
	var aggregators []*Aggregator
	for i := 2; i < nodeCount; i++ {
		// connect each aggregator's inputs to the previous two
		// topics, and its output to the current topic
		ag := NewAggregator([]*Topic{topics[i-2], topics[i-1]}, topics[i])
		aggregators = append(aggregators, ag)
	}

	// subscribe to the last aggregator's output
	resultChan := aggregators[len(aggregators)-1].OutputTopic.Subscribe()

	// create nodes
	var nodes []*Node

	// create input nodes
	fn0 := Function{Fn: func(args ...float64) float64 { return 1.0 }, ArgCount: 0}
	fn1 := Function{Fn: func(args ...float64) float64 { return 2.0 }, ArgCount: 0}
	node0 := NewNode(0, fn0, nil, topics[0])
	node1 := NewNode(1, fn1, nil, topics[1])
	nodes = append(nodes, node0)
	nodes = append(nodes, node1)

	// create a function
	fn := Function{
		Fn: func(args ...float64) float64 {
			return args[0] + args[1]
		},
		ArgCount: 2,
	}

	// create more nodes
	for i := 0; i < 2; i++ {
		node := NewNode(uint64(len(nodes)), fn, aggregators[i].OutputTopic, topics[i])
		nodes = append(nodes, node)
	}

	Assert(len(nodes) == nodeCount, "expected %d nodes, got %d", nodeCount, len(nodes))

	// read results from the result channel
	for result := range resultChan {
		Pl(result)
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
