package node

import (
	. "github.com/stevegt/goadapt"

	_ "net/http/pprof"
)

/*
- graph and node both implement the Function interface
- nodes don't use channels
- nodes only have node.F(inputMap map[string]float64) (outputMap map[string]float64)
- we use a helper to convert between maps and slices of float64 for simple
  function definitions
- graph knows channels, uses Edges to connect internal nodes to each other
- the zero-value of a graph is a valid graph
- the zero-value of a node is a valid node
*/

// Function is an interface for anything that accepts a map of float64
// values and returns a map of float64 values.
type Function interface {
	GetName() string
	F(inputMap map[string]float64) (outputMap map[string]float64)
	GetInputNames() []string
	GetOutputNames() []string
	Draw() string
}

// Node is a node in a dataflow graph.  It implements the Function
// interface.  The zero-value of a Node is a valid Node with no inputs
// or outputs.
type Node struct {
	Name string
	Fn   func(inputMap map[string]float64) (outputMap map[string]float64)
	// InputNames is a slice of input names.
	InputNames []string
	// OutputNames is a slice of output names.
	OutputNames []string
}

// GetName returns the node's name.
func (n *Node) GetName() string {
	return n.Name
}

// GetInputNames returns the node's input names.
func (n *Node) GetInputNames() []string {
	return n.InputNames
}

// GetOutputNames returns the node's output names.
func (n *Node) GetOutputNames() []string {
	return n.OutputNames
}

func isNil(f Function) bool {
	if f == nil {
		return true
	}
	if f.(*Node) == nil {
		return true
	}
	return false
}

// Wrap creates a new Node, accepting a function that accepts a
// variadic list of float64 values along with a slice of input names
// and an output name.
// XXX this isn't great, because fmaps is going to be called
// every time the node is executed, and it's going to do a lot of
// work to convert between maps and slices.  it would be better
// to have functions always accept and return maps and not use
// Wrap() at all.
func Wrap(fslices func(...float64) float64, inputNames []string, outputName string) (n *Node) {
	fmaps := func(inputMap map[string]float64) (outputMap map[string]float64) {
		Assert(len(inputMap) == len(inputNames), "expected %v inputs, got %v", len(inputNames), len(inputMap))
		// convert input map to slice
		inputs := make([]float64, len(inputNames))
		for i, inputName := range inputNames {
			Assert(inputName != "", "Empty input name")
			var ok bool
			inputs[i], ok = inputMap[inputName]
			Assert(ok, "input %v missing from provided map", inputName)
		}
		// calculate result
		output := fslices(inputs...)
		// convert result to map
		outputMap = make(map[string]float64)
		outputMap[outputName] = output
		return
	}
	n = &Node{
		Fn:          fmaps,
		InputNames:  inputNames,
		OutputNames: []string{outputName},
	}
	return
}

// WrapMulti creates a new Node, accepting a function that accepts a
// variadic list of float64 values along with a slice of input names
// and a slice of output names.
func WrapMulti(fslices func(...float64) []float64, inputNames, outputNames []string) (n *Node) {
	fmaps := func(inputMap map[string]float64) (outputMap map[string]float64) {
		Assert(len(inputMap) == len(inputNames), "expected %v inputs, got %v", len(inputNames), len(inputMap))
		// convert input map to slice
		inputs := make([]float64, len(inputNames))
		for i, inputName := range inputNames {
			var ok bool
			inputs[i], ok = inputMap[inputName]
			Assert(ok, "input %v missing from provided map", inputName)
		}
		// calculate result
		outputs := fslices(inputs...)
		Assert(len(outputs) == len(outputNames), "expected %v outputs, got %v", len(outputNames), len(outputs))
		// convert result to map
		outputMap = make(map[string]float64)
		for i, outputName := range outputNames {
			outputMap[outputName] = outputs[i]
		}
		return
	}
	n = &Node{
		Fn:          fmaps,
		InputNames:  inputNames,
		OutputNames: outputNames,
	}
	return
}

// Draw returns a graphviz entry for the node.
func (n *Node) Draw() (dot string) {
	dot += Spf("  %s;\n", n.GetName())
	return
}

// F executes the node's function.
func (n *Node) F(inputMap map[string]float64) (outputMap map[string]float64) {
	if n.Fn == nil {
		// no function
		return
	}
	outputMap = n.Fn(inputMap)
	return
}

// Edge is a hyperedge of one or more float64 channels.  An edge has a
// Subscribe() method which returns a channel which will receive
// values sent to the edge.  To send values to the edge, send them to
// the edge's Publish channel or call the edge's Send() method.
type Edge struct {
	// Name is the edge's name.
	Name string
	// Publish is a channel which can be used to send values to the
	// edge.
	Publish chan float64
	// Subscribers is a slice of channels which will receive values
	// sent to the edge.
	Subscribers []chan float64
	// FromNode is the node that the edge gets its values from.
	FromNode Function
	// ToNodes is a slice of subscriber nodes.
	ToNodes []Function
}

// NewEdge creates a new edge with the given name and size.  Size is
// the size of the Publish channel.  If size is 0, the channel is
// unbuffered.  An unbuffered channel will block the publisher until
// all subscribers have received the previously published value,
// which can be useful when needed for barrier problem
// synchronization.
func NewEdge(name string, fromNode *Node, size int) (e *Edge) {
	if name == "" {
		name = uname()
	}
	e = &Edge{
		Name:        name,
		FromNode:    fromNode,
		Publish:     make(chan float64, size),
		Subscribers: make([]chan float64, 0),
	}
	Debug("NewEdge %s\n", name)
	go func() {
		defer func() {
			// notify subscribers on exit
			for _, subscriber := range e.Subscribers {
				close(subscriber)
			}
		}()

		for {
			// wait for a value to be published
			value, ok := <-e.Publish
			if !ok {
				// channel is closed
				return
			}
			// send the value to all subscribers
			for _, subscriber := range e.Subscribers {
				Debug("edge %s sending %v to subscriber addr %v\n", e.Name, value, &subscriber)
				subscriber <- value
				Debug("edge %s sent %v to subscriber\n", e.Name, value)
			}
		}
	}()
	return
}

// Subscribe returns a channel which will receive values published to
// the edge.  Size is the size of the channel.  If size is 0, the
// channel is unbuffered.  An unbuffered channel can be used to block
// other subscribers until this subscriber has received the value;
// this might be useful if needed for barrier problem synchronization.
func (e *Edge) Subscribe(toNode Function, size int) (c chan float64) {
	c = make(chan float64, size)
	// Debug("edge %s subscribe creating chan addr %v\n", e.Name, &c)
	e.Subscribers = append(e.Subscribers, c)
	e.ToNodes = append(e.ToNodes, toNode)
	return
}

// Draw returns a graphviz representation of the edge
func (e *Edge) Draw() (dot string) {
	var fromName string
	if isNil(e.FromNode) {
		fromName = "in"
	} else {
		fromName = e.FromNode.GetName()
	}
	for _, node := range e.ToNodes {
		var toName string
		if isNil(node) {
			toName = "out"
		} else {
			toName = node.GetName()
		}
		dot += Spf("  %s -> %s [label=%s];\n", fromName, toName, e.Name)
	}
	return
}

// Send sends a value to the edge.
func (e *Edge) Send(value float64) {
	Debug("edge sending %v to %v\n", value, e.Name)
	e.Publish <- value
	Debug("edge sent %v to %v\n", value, e.Name)
}

// Graph is a collection of Nodes and Graphs. Graph implements the
// Function interface. Graphs manage channels and use them to connect
// internal Functions to each other.
type Graph struct {
	Name        string
	Size        int
	InputNames  []string
	OutputNames []string
	// dot is a graphviz graph
	Dot string
	// nodes is a map of all node names to nodes
	nodes map[string]Function
	// edges is a map of all edge names to edges
	edges map[string]*Edge
	// graphOutputChan is a channel created by join(), and contains
	// the final output map created by the graph
	graphOutputChan chan map[string]float64
}

// NewGraph creates a new graph with the given channel buffer size and
// input and output edge names.
func NewGraph(name string, size int, inputNames []string) (g *Graph) {
	g = &Graph{
		Name:        name,
		Size:        size,
		InputNames:  inputNames,
		OutputNames: make([]string, 0),
		nodes:       make(map[string]Function),
		edges:       make(map[string]*Edge),
	}

	// create an edge for each graph input
	for _, name := range g.InputNames {
		g.edges[name] = NewEdge(name, nil, size)
	}

	return
}

// AddNode adds a node to the graph.  If node.Name is empty, a unique
// node name is generated.
func (g *Graph) AddNode(node *Node) {
	if node.Name == "" {
		node.Name = uname()
	}
	_, ok := g.nodes[node.Name]
	Assert(!ok, "Duplicate node name %v", node.Name)
	g.nodes[node.Name] = node

	// create an edge for each node output
	for _, name := range node.GetOutputNames() {
		Debug("creating node %v output %v\n", node.GetName(), name)
		edge, ok := g.edges[name]
		if ok {
			Debug("edges: %#v\n", g.edges)
			existingNode := edge.FromNode
			Assert(false, "Node %v output %v already used by node %v", node.GetName(), name, existingNode.GetName())
		}
		g.edges[name] = NewEdge(name, node, g.Size)
	}

	return
}

// Start starts the graph.  It creates the graph's internal channels
// and starts the graph's internal goroutines.
func (g *Graph) Start() {
	if g == nil {
		// no graph
		return
	}

	// start graph
	g.Dot = Spf("digraph %s {\n", g.Name)

	// start goroutine for each node
	for _, node := range g.nodes {
		g.Dot += node.Draw()
		g.start(node)
	}

	// find all unsubscribed edges -- these are the graph outputs
	for name, edge := range g.edges {
		if len(edge.Subscribers) == 0 {
			g.OutputNames = append(g.OutputNames, name)
		}
	}

	// join the graph outputs into a single channel
	g.graphOutputChan = g.join(nil, g.OutputNames)

	// render all edges
	for _, edge := range g.edges {
		g.Dot += edge.Draw()
	}

	// end graph
	g.Dot += "}\n"

}

// start starts a node's goroutine.
func (g *Graph) start(node Function) {

	// join the node inputs
	inputChan := g.join(node, node.GetInputNames())

	// read from joiner until it is closed
	go func() {
		defer func() {
			// notify output channels on exit
			for _, output := range node.GetOutputNames() {
				close(g.edges[output].Publish)
			}
		}()
		for inputMap := range inputChan {
			// calculate result
			Debug("node %v inputMap: %#v\n", node.GetName(), inputMap)
			outputMap := node.F(inputMap)
			// send result to output channels
			for name, value := range outputMap {
				g.edges[name].Send(value)
			}
		}
	}()
}

// F executes the graph's function.
func (g *Graph) F(inputMap map[string]float64) (outputMap map[string]float64) {
	if g == nil {
		// no graph
		return
	}
	// for each input, send the value to the edge of the same name
	for name, value := range inputMap {
		edge, ok := g.edges[name]
		Assert(ok, "Input %v not found", name)
		Debug("graph sending %v to %v\n", value, name)
		edge.Send(value)
	}
	Debug("sent inputs\n")

	// get the results from graph output joiner
	outputMap = <-g.graphOutputChan
	return
}

// join() given a list of edge names, subscribes to each edge and
// returns a channel which will contain a map of values published to
// the edges.
func (g *Graph) join(toNode Function, names []string) (outChan chan map[string]float64) {

	inputChans := make(map[string]chan float64)

	// create channels by subscribing to each edge
	for _, name := range names {
		Assert(name != "", "Empty edge name")
		var ok bool
		edge, ok := g.edges[name]
		Assert(ok, "Edge %v not found", name)
		// prevent duplicate subscriptions
		_, ok = inputChans[name]
		Assert(!ok, "Edge %v already subscribed", name)
		inputChans[name] = edge.Subscribe(toNode, g.Size)
	}

	// create outChan
	outChan = make(chan map[string]float64, g.Size)

	go func() {
		defer func() {
			// notify output channel on exit
			close(outChan)
		}()

		// read from all input channels. we need to read from each
		// in a separate goroutine to avoid blocking and to handle
		// the case where one or more channels are closed.  we start a
		// goroutine for each input channel, and then we read values
		// from each channel, waiting until we have one value from
		// each channel while filling the output map, then we publish
		// the output map on outchan.

		// create intermediate rx channels to store the most recent
		// value from each input channel.
		rxChans := make(map[string]chan float64)
		for name, _ := range inputChans {
			rxChans[name] = make(chan float64, 1)
		}

		// start a goroutine for each input channel.  each goroutine
		// reads one value from its channel and stores it in the
		// rx channel
		for name, c := range inputChans {
			go func(name string, c chan float64) {
				for {
					// read one value from channel
					Debug("joiner waiting for %v addr %v\n", name, &c)
					// wait until the parent goroutine says it's ok to read
					input, ok := <-c
					if !ok {
						// channel is closed
						Debug("joiner got closed %v\n", name)
						close(rxChans[name])
						return
					}
					// channel is open
					rxChans[name] <- input
					Debug("joiner got %v, input: %#v\n", name, input)
				}
			}(name, c)
		}

		// read from rx channels until they are closed
		closed := false
		for {
			outputMap := make(map[string]float64)
			// read one value from each input channel
			for name, c := range rxChans {
				// get one value from channel
				input, ok := <-c
				if !ok {
					// channel is closed
					closed = true
					break
				}
				// channel is open
				outputMap[name] = input
			}
			if closed {
				// one or more channels are closed
				break
			}
			outChan <- outputMap
		}

	}()
	return
}

// Draw analyzes the graph and returns a graphviz graph.
// XXX move all this stuff into NewGraph and AddNode, create actual
// edges during Start()
func (g *Graph) Draw() (dot string) {
	// start graph
	dot = Spf("digraph %s {\n", g.Name)
	// add nodes
	for _, node := range g.nodes {
		dot += Spf("  %s;\n", node.GetName())
	}
	// add edges
	for _, edge := range g.edges {
		var fromName string
		if isNil(edge.FromNode) {
			// edge is a graph input
			fromName = "in"
		} else {
			fromName = edge.FromNode.GetName()
		}
		for _, toNode := range edge.ToNodes {
			var toName string
			if isNil(toNode) {
				// edge is a graph output
				toName = "out"
			} else {
				toName = toNode.GetName()
			}
			dot += Spf("  %s -> %s [label=\"%s\"];\n", fromName, toName, edge.Name)
		}
	}
	// end graph
	dot += "}\n"
	return
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

// Stop stops the graph.
func (g *Graph) Stop() {
	if g == nil {
		// no graph
		return
	}
	// close all graph input channels
	for _, name := range g.InputNames {
		edge := g.edges[name]
		close(edge.Publish)
	}
}
