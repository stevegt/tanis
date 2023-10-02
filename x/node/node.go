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
			Pf("edge %#v fromNode %#v\n", edge, edge.FromNode)
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

/*
// Joiner takes several input channels and publishes a single slice.
// The output slice is published each time all input topics have
// received a value.
type Joiner struct {
	Name       string
	InputChans []chan float64
}

// NewJoiner creates a new joiner with the given input channels and
// returns a channel of slices.
func NewJoiner(name string, size int, inputChans []chan float64) (a *Joiner) {
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
*/

//	OpenInputs() (inputMap map[string]chan float64)
//	OpenOutputs() (outputMap map[string]chan float64)

/*

// Node is an interface for a node in a dataflow graph.
type Node interface {
	// GetName returns the node's name.
	GetName() string
	// F executes the node's function.
	F(inputMap map[string]float64) (outputMap map[string]float64)
	// internal use
	setInputChannels(map[string]*Edge)
	getOutputEdges() map[string]*Edge
}
*/

//

/*
// NewFunction creates a new function.
func NewFunction(name string, size int, inputNames, outputNames []string, fn func([]float64) []float64) (f Function) {
	f = Function{
		Name: name,
		InputNames: inputNames,
		OutputNames: outputNames,
		Fn:   fn,
	}

	// create input and output edges



	return
}

// GetName returns the function's name.
func (f *Function) GetName() string {
	return f.Name
}

// F executes the function.
func (f *Function) F(inputMap map[string]float64) (outputMap map[string]float64) {
	// convert input map to slice
	inputs := make([]float64, len(inputMap))
	i := 0




	// Graph is a dataflow graph.  A graph manages its own internal edges.
	//
	// XXX caller should be able to do this:
	//
	//	     g := NewNode("foo", []string{"x", "y"}, []string{"z"})
	//	     g.AddNode("", addFn, []string{"x", "y"}, []string{"a"})
	//	     g.AddNode("", addFn, []string{"a", "y"}, []string{"z"})
	//	     err := g.Start()
	//		 outputmap := g.F(inputmap) // both are map[string]float64
	type Graph struct {
		Name        string
		fn          Function
		InputNames  []string
		OutputNames []string
		// internal nodes created by this node
		// map[nodeName]*Node
		nodes map[string]*Graph
		// edges connecting internal nodes
		// map[edgeName]chan float64
		edges map[string]*Edge
		// internal channels for connecting things together
		chans map[string]chan float64

		// graphviz bits
		gvnodes map[string]*dot.Node
		gv      *dot.Graph
	}

	// NewGraph creates a new node with the given name and input and
	// output edge names.  The input and output edge names are used later
	// in either the AddNode() or SetFunction() methods.  Size is the
	// size of the internal edges.  If size is 0, the channels are
	// unbuffered.  An unbuffered channel will block the publisher until
	// all subscribers have received the previously published value, which
	// might be useful if needed for barrier problem synchronization.
	func NewGraph(name string, size int, inputNames, outputNames []string) (n *Graph) {

		// create input and output edges
		edges := make(map[string]*Edge)
		for _, inputName := range inputNames {
			edges[inputName] = NewEdge(inputName, size)
		}
		for _, outputName := range outputNames {
			edges[outputName] = NewEdge(outputName, size)
		}
		n = newNode(name, size, inputNames, outputNames, edges)
		return
	}

	// newNode creates a new node with the given name and input and
	// output edge names and edges.
	func newNode(name string, size int, inputNames, outputNames []string, edges map[string]*Edge) (n *Graph) {
		if name == "" {
			name = uname()
		}
		// ensure input and output names are unique
		allNames := make(map[string]bool)
		for _, inputName := range inputNames {
			Assert(!allNames[inputName], "Duplicate input name %v", inputName)
			allNames[inputName] = true
		}
		for _, outputName := range outputNames {
			Assert(!allNames[outputName], "Duplicate output name %v", outputName)
			allNames[outputName] = true
		}

		// create graphviz nodes for input and output edge names
		for _, inputName := range inputNames {
			gvnode := n.gv.Node(inputName)
			n.gvnodes[inputName] = &gvnode
			// set node shape
			n.gvnodes[inputName].Attr("shape", "box")
		}
		for _, outputName := range outputNames {
			gvnode := n.gv.Node(outputName)
			n.gvnodes[outputName] = &gvnode
			// set node shape
			n.gvnodes[outputName].Attr("shape", "box")
		}

		// create node
		n = &Graph{
			Name:        name,
			InputNames:  inputNames,
			OutputNames: outputNames,
			nodes:       make(map[string]*Graph),
			edges:       edges,
			gvnodes:     make(map[string]*dot.Node),
			gv:          dot.NewGraph(dot.Directed),
		}

		return
	}

	// SetFunction sets the node's function.  The input and output edge
	// names given in the NewNode() method are mapped to the function's
	// input and output slices in the order given. It is an error to call
	// this method if AddNode has been called.
	func (n *Graph) SetFunction(fn Function) (err error) {
		defer Return(&err)
		Assert(len(n.nodes) == 0, "Cannot set function after adding nodes")
		n.fn = fn
		return
	}

	// AddNode adds a node to the parent node's internal graph.  If name
	// is empty, a unique node name is generated.  The inputnames are a
	// list of edge names that have already been added to the graph.  If
	// any of the output edge names match a parent node's output edge
	// name, the new node's output will be used as the parent node's
	// output when F() is called.  If any output name is empty, a unique
	// name is generated and can be retrieved from the node's OutputNames
	// field. It is an error to call this method if SetFunction has been
	// called.
	func (n *Graph) AddNode(name string, size int, fn Function, inputNames, outputNames []string) (nn *Graph, err error) {
		defer Return(&err)
		Assert(n.fn == nil, "Cannot add nodes after setting function")
		if name == "" {
			name = uname()
		}

		edges := make(map[string]*Edge)

		// ensure input names are valid edge names in the parent node
		for _, inputName := range inputNames {
			edge, ok := n.edges[inputName]
			Assert(ok, "Edge %v not found", inputName)
			edges[inputName] = edge
		}
		// create output names if needed
		for i, outputName := range outputNames {
			if outputName == "" {
				outputName = uname()
				outputNames[i] = outputName
			}
		}
		// create output edges or use parent node's output edges
		for _, outputName := range outputNames {
			// create output edge if needed
			edge, ok := n.edges[outputName]
			if !ok {
				edges[outputName] = NewEdge(outputName, size)
			} else {
				edges[outputName] = edge
			}
		}
		// create new node
		nn = newNode(name, size, inputNames, outputNames, edges)
		return
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

		}
		return
	}

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

*/

/*
	// OldNode is a node in a dataflow graph.  It has a Function which
	// calculates a result from a slice of input values.
	type OldNode struct {
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
	func NewOldNode(name string, fn Function, chanSize int, publishers ...Publisher) (n *OldNode) {

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

		n = &OldNode{
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
	func (n *OldNode) Subscribe(size int) (c chan float64) {
		c = n.Output.Subscribe(size)
		return
	}

	// GetName returns the node's name.
	func (n *OldNode) GetName() string {
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
*/

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
