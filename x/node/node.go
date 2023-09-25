package node

import (
	"sync"

	. "github.com/stevegt/goadapt"

	_ "net/http/pprof"
)

// Log collects messages in a channel and writes them to stdout.
type Log struct {
	MsgChan chan string
}

// NewLog creates a new Log.
func NewLog() (l *Log) {
	l = &Log{
		MsgChan: make(chan string, 99999),
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

// Publisher is an interface that supports a Subscribe method.
type Publisher interface {
	Subscribe(int) chan float64
}

// Topic is a pub/sub topic that fans out single values to multiple
// subscribers.
type Topic struct {
	// Subscribers is a slice of channels which will receive values
	// published to the topic.
	Subscribers []chan float64
	// Publish is a channel which will receive Values to be published
	// to the topic.
	Publish chan float64
}

// NewTopic creates a new Topic. Size is the size of the Publish
// channel.  If size is 0, the channel is unbuffered.  An unbuffered
// channel will block the publisher until all subscribers have
// received the previously published value, which can be useful when
// needed for barrier problem synchronization.
func NewTopic(size int) (t *Topic) {
	t = &Topic{
		Subscribers: make([]chan float64, 0),
		Publish:     make(chan float64, size),
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
// the topic.  Size is the size of the channel.  If size is 0, the
// channel is unbuffered.  An unbuffered channel can be used to block
// other subscribers until this subscriber has received the value,
// which can be useful when needed for barrier problem synchronization.
func (t *Topic) Subscribe(size int) (c chan float64) {
	c = make(chan float64, size)
	t.Subscribers = append(t.Subscribers, c)
	return
}

// Aggregator is a fan-in -- it takes several input topics and publishes a single output
// slice.  The output slice is published each time all input topics
// have received a value.
type Aggregator struct {
	InputChans  []chan float64
	Subscribers []chan []float64
}

// NewAggregator creates a new Aggregator with the given input topics.
func NewAggregator(inputChans []chan float64) (a *Aggregator) {
	a = &Aggregator{
		InputChans:  inputChans,
		Subscribers: make([]chan []float64, 0),
	}

	go func() {
		defer func() {
			// notify subscribers on exit
			for _, subscriber := range a.Subscribers {
				close(subscriber)
			}
		}()

		// read from all input channels until they are closed
		closedCount := 0
		for {
			// read from all input channels. we need to read from each
			// in a separate goroutine to avoid blocking and to handle
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
						values[i] = input
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

			// send the values slice to all subscribers
			for _, subscriber := range a.Subscribers {
				subscriber <- values
			}
		}
	}()
	return
}

// Subscribe returns a channel which will receive each slice of values
// published by the aggregator.
func (a *Aggregator) Subscribe() (c chan []float64) {
	c = make(chan []float64, 10)
	a.Subscribers = append(a.Subscribers, c)
	return
}

// Function is a function which takes a slice of float64 arguments and
// returns a float64 result.
type Function struct {
	Fn func(...float64) float64
}

// Node is a node in a dataflow graph.  It has a Function which
// calculates a result from a slice of input values.
type Node struct {
	Id       uint64
	Function Function
	Input    *Aggregator
	Output   *Topic
}

// NewNode creates a new node with the given Function and input
// channels. An output is generated each time value are received from
// all input channels.  If the node has no input channels, the output
// is generated immediately by the node's Function on each read of an
// output channel.
func NewNode(id uint64, fn Function, chanSize int, publishers ...Publisher) (n *Node) {

	// subscribe to the input topics
	inputChans := make([]chan float64, len(publishers))
	for i, publisher := range publishers {
		inputChans[i] = publisher.Subscribe(chanSize)
	}

	var input *Aggregator
	if len(inputChans) > 0 {
		// create an aggregator for the input channels
		input = NewAggregator(inputChans)
	}

	output := NewTopic(10)

	n = &Node{
		Id:       id,
		Function: fn,
		Input:    input,
		Output:   output,
	}

	var inputChan chan []float64
	if n.Input != nil {
		// subscribe to the input topic
		c := n.Input.Subscribe()
		inputChan = c
	}

	outputChan := n.Output.Publish

	go func() {
		defer func() {
			// notify output topic on exit
			close(outputChan)
		}()

		for {
			var inputs []float64
			if inputChan != nil {
				// read a value slice from the input aggregator
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
			outputChan <- result
		}
	}()
	return
}

// Subscribe returns a channel which will receive each value
// calculated by the node's Function.  Size is the size of the
// channel.  If size is 0, the channel is unbuffered.  An unbuffered
// channel can be used to block other subscribers until this
// subscriber has received the value, which can be useful when needed
// for barrier problem synchronization.  A buffered channel might be
// used with cyclic graphs in neural networks to provide state
// feedback.
func (n *Node) Subscribe(size int) (c chan float64) {
	c = n.Output.Subscribe(size)
	return
}

// Graph is a dataflow graph.
type Graph struct {
	Nodes    []*Node
	ChanSize int
}

// NewGraph creates a new Graph with the given size for all channels.
func NewGraph(size int) (g *Graph) {
	g = &Graph{
		Nodes:    make([]*Node, 0),
		ChanSize: size,
	}
	return
}

// AddNode adds a node to the graph.
func (g *Graph) AddNode(id uint64, fn Function, publishers ...Publisher) (node *Node) {
	node = NewNode(id, fn, g.ChanSize, publishers...)
	g.Nodes = append(g.Nodes, node)
	return
}

// NameInputs creates a map of input topics indexed by name.
func (g *Graph) NameInputs(names ...string) (m map[string]*Topic) {
	m = make(map[string]*Topic)
	for _, name := range names {
		m[name] = NewTopic(g.ChanSize)
	}
	return
}

// OpenOutputs returns a slice of nodes which
// have no subscribers.
func (g *Graph) OpenOutputs() (nodes []*Node) {
	for _, node := range g.Nodes {
		if len(node.Output.Subscribers) == 0 {
			nodes = append(nodes, node)
		}
	}
	return
}

// NameOutputs creates topics for nodes with open outputs and returns a map of
// topics indexed by name.
func (g *Graph) NameOutputs(names ...string) (m map[string]Publisher) {
	m = make(map[string]Publisher)
	open := g.OpenOutputs()
	Assert(len(open) == len(names), "Number of names must match number of open outputs")
	for i, node := range g.OpenOutputs() {
		name := names[i]
		m[name] = node.Output
	}
	return
}
