package node

import (
	"sync"

	. "github.com/stevegt/goadapt"

	_ "net/http/pprof"

	"github.com/emicklei/dot"
)


// Publisher is an interface that represents a node in a directed
// dataflow graph.  It accepts a slice of float64 values and publishes
// a single float64 value.  It can be used as a node in a dataflow
// graph.
// XXX merge Node and Function into Function, rename Publisher to 
// XXX Node
type Publisher interface {
	GetName() string
	Subscribe(int) chan float64
	Subscribers() []Publisher
}

// Edge is a connection between two nodes.
type Edge struct {
	From Publisher
	To   Publisher
	c chan float64
}

// NewEdge creates a new edge between two nodes.
func NewEdge(size int, from, to Publisher) (e *Edge) {
	e = &Edge{
		From: from,
		To:   to,
		c:    from.Subscribe(size),
	}
	go func() {
		defer close(e.c)
		for {
			value, ok := <-e.From.Subscribe(size)
			if !ok {
				return
			}
			e.c <- value
		}


// Topic is a pub/sub topic that fans out single values to multiple
// subscribers.
type Topic struct {
	// Id is a unique identifier for the topic.
	Name string
	// subscribers is a slice of channels which will receive values
	// published to the topic.
	subscribers []chan float64
	// Publish is a channel which will receive Values to be published
	// to the topic.
	Publish chan float64
}

// GetName returns the topic's name.
func (t *Topic) GetName() string {
	return t.Name
}

// NewTopic creates a new Topic. Size is the size of the Publish
// channel.  If size is 0, the channel is unbuffered.  An unbuffered
// channel will block the publisher until all subscribers have
// received the previously published value, which can be useful when
// needed for barrier problem synchronization.
func NewTopic(name string, size int) (t *Topic) {
	t = &Topic{
		Name:        name,
		subscribers: make([]chan float64, 0),
		Publish:     make(chan float64, size),
	}
	go func() {
		defer func() {
			// notify subscribers on exit
			for _, subscriber := range t.subscribers {
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
			for _, subscriber := range t.subscribers {
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
	t.subscribers = append(t.subscribers, c)
	return
}

// Subscribers returns a slice of subscribers.
func (t *Topic) Subscribers() (names []string) {
	for _, subscriber := range t.subscribers {
		names = append(names, int2str(cap(subscriber)))


// Splitter takes a single input slice and publishes each value to a
// separate output topic.
type Splitter struct {
	Name      string
	Width     int
	InputChan chan []float64
	// Topics is a slice of output topics.  Each topic receives a single
	// value from the input slice.
	Topics []*Topic
}

// Subscribe given a slice position returns a channel which will
// receive values from that position.  Size is the size of the
// channel.  If size is 0, the channel is unbuffered.
func (s *Splitter) Subscribe(i, size int) (c chan float64) {
	c = s.Topics[i].Subscribe(size)
	return
}

// NewSplitter creates a new splitter and returns a slice of output
// topics.  Width is the number of values in each
// input slice.  Size is the size of the ouput topic channels.  If
// size is 0, the channels are unbuffered.
func NewSplitter(name string, inputNames []string, size int, inputChan chan []float64) (s *Splitter) {
	s = &Splitter{
		Name:      name,
		Width:     len(inputNames),
		InputChan: inputChan,
		Topics:    make([]*Topic, len(inputNames)),
	}

	// create a topic for each value in the input slice
	for i := range s.Topics {
		s.Topics[i] = NewTopic(uname(), size)
	}

	go func() {
		defer func() {
			// notify subscribers on exit
			for _, topic := range s.Topics {
				close(topic.Publish)
			}
		}()

		// read from input channel until it is closed
		for {
			// read one slice of values from input channel
			inputs, ok := <-inputChan
			if !ok {
				// channel is closed
				return
			}

			// send each value to all subscribers
			for i, topic := range s.Topics {
				topic.Publish <- inputs[i]
			}
		}
	}()
	return
}

// Joiner takes several input topics and publishes a single output
// slice.  The output slice is published each time all input topics
// have received a value.
type Joiner struct {
	Name        string
	InputChans  []chan float64
	Subscribers []chan []float64
}

// NewJoiner creates a new joiner with the given input topics.
func NewJoiner(name string, inputChans []chan float64) (a *Joiner) {
	a = &Joiner{
		Name:        name,
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
// published by the joiner.
func (a *Joiner) Subscribe() (c chan []float64) {
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
	Name     string
	Function Function
	Input    *Joiner
	Output   *Topic
}

// NewNode creates a new node with the given Function and input
// channels. An output is generated each time value are received from
// all input channels.  If the node has no input channels, the output
// is generated immediately by the node's Function on each read of an
// output channel.
func NewNode(name string, fn Function, chanSize int, publishers ...Publisher) (n *Node) {

	// subscribe to the input topics
	inputChans := make([]chan float64, len(publishers))
	for i, publisher := range publishers {
		inputChans[i] = publisher.Subscribe(chanSize)
	}

	var input *Joiner
	if len(inputChans) > 0 {
		// create a joiner for the input channels
		input = NewJoiner(name, inputChans)
	}

	output := NewTopic(name, 10)

	n = &Node{
		Name:     name,
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
				// read a value slice from the input joiner
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

// GetName returns the node's name.
func (n *Node) GetName() string {
	return n.Name
}

// Graph is a dataflow graph.
type Graph struct {
	// Nodes       []*Node
	ChanSize    int
	InputNames  []string
	OutputNames []string
	Publishers  map[string]Publisher
	// input splitter
	inputs *Splitter
	// output joiner
	outputs *Joiner
	// graphviz bits XXX this feels like a hack though
	gvnodes map[string]*dot.Node
	gv      *dot.Graph
}

// NewGraph creates a new Graph with the given size for all channels
// and the given input and output names.
//
// XXX caller should be able to do this:
//
//	     g := NewGraph(10, []string{"x", "y"}, []string{"z"})
//	     g.AddNode("x", ...)
//	     g.AddNode("y", ...)
//	     g.AddNode("z", ..., "x", "y")
//	     ok := g.Verify()
//		 outputmap := g.F(inputmap) // both are map[string]float64
func NewGraph(size int, inputNames, outputNames []string) (g *Graph) {
	g = &Graph{
		// Nodes:       make([]*Node, 0),
		ChanSize:    size,
		InputNames:  inputNames,
		OutputNames: outputNames,
		Publishers:  make(map[string]Publisher),
		gvnodes:     make(map[string]*dot.Node),
		gv:          dot.NewGraph(dot.Directed),
	}

	// create input splitter
	inputChan := make(chan []float64, len(inputNames))
	g.inputs = NewSplitter(uname(), inputNames, size, inputChan)
	// add to gv
	g.gv.Node(Spf("%d", g.inputs.Name))

	// joiner gets created after all nodes are added

	return
}

// AddNode adds a node to the graph.
func (g *Graph) AddNode(name string, fn Function, pubNames ...string) {
	// find publishers from names
	publishers := make([]Publisher, len(pubNames))
	for i, pubName := range pubNames {
		pub, ok := g.Publishers[pubName]
		Assert(ok, "Publisher %v not found", pubName)
		publishers[i] = pub
	}
	// node := NewNode(name, fn, g.ChanSize, publishers...)
	// g.Nodes = append(g.Nodes, node)

	// add node to graphviz graph
	gvnode := g.gv.Node(Spf("%d", name))
	g.gvnodes[name] = &gvnode
	for _, publisher := range publishers {
		pname := publisher.GetName()
		// add edge to graphviz graph
		Assert(pname != name, "Node %v cannot publish to itself", name)
		from := g.gvnodes[pname]
		if from == nil {
			// input node
			inputNode := g.gv.Node(Spf("%d", pname))
			from = &inputNode
		}
		to := g.gvnodes[name]
		Pf("g.gv %v from %v to %v\n", g.gv, from, to)
		g.gv.Edge(*from, *to)
	}
	return
}

// GetInputs returns a map of input publishers indexed by name.  These
// publishers are the outputs of the input splitter.  It's possible to
// provide inputs via either this map or the splitter, but not both,
// and it's important to close the splitter input channel to shut
// things down.
//
// XXX deprecate.
func (g *Graph) XXXGetInputs(names ...string) (m map[string]*Topic) {
	// get input topics from splitter
	m = make(map[string]*Topic)
	for i, name := range g.InputNames {
		m[name] = g.inputs.Topics[i]
	}
	return
}

// OpenOutputs returns a slice of publisher names which
// have no subscribers.
func (g *Graph) OpenOutputs() (names []string) {
	for name, pub := range g.Publishers {
		if len(pub.Subscribers()) == 0 {
			names = append(names, name)
		}
	}
	return
}

// DrawDot returns a graphviz rendering of the graph.
func (g *Graph) DrawDot() string {
	return g.gv.String()
}

var uniqueId uint64

// uname returns a unique name string
func uname() string {
	uniqueId++
	return Spf("u%d", uniqueId)
}

// int2str returns a string representation of an int
func int2str(i int) string {
	return Spf("%d", i)
}
