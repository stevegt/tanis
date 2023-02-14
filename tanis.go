package tanis

import (
	"encoding/json"
	"math"
	"math/rand"
	"sync"

	. "github.com/stevegt/goadapt"
)

// sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoid derivative
func sigmoidD1(x float64) float64 {
	// return math.Exp(-x) / math.Pow(1+math.Exp(-x), 2)
	return x * (1 - x)
}

// tanh activation function
func tanh(x float64) float64 {
	return math.Tanh(x)
}

// tanh derivative
func tanhD1(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}

/*
func main() {
	// Create a network with 2 inputs, 2 hidden nodes, and 2 outputs
	// XXX do this in JSON
	network := &Network{
		InputCount: 2,
		Layers: []*Layer{
			// hidden layer
			&Layer{
				Nodes: []*Node{
					NewNode(sigmoid, sigmoidD1),
					NewNode(sigmoid, sigmoidD1),
				},
			},
			// outputs
			&Layer{
				Nodes: []*Node{
					NewNode(sigmoid, sigmoidD1),
					NewNode(sigmoid, sigmoidD1),
				},
			},
		},
	}

	// Connect the layers
	network.Init()

	// randomize weights
	network.RandomizeWeights(-1, 1)

	// Create a training set
	// XXX do this in JSON
	trainingSet := &TrainingSet{
		Cases: []*TrainingCase{
			NewTrainingCase([]float64{0, 0}, []float64{0, 0}),
			NewTrainingCase([]float64{0, 1}, []float64{0, 1}),
			NewTrainingCase([]float64{1, 0}, []float64{0, 1}),
			NewTrainingCase([]float64{1, 1}, []float64{1, 0}),
		},
	}

	// Train the network
	for i := 0; i < 100000; i++ {
		cost := 0.0
		for _, trainingCase := range trainingSet.Cases {
			cost += network.Train(trainingCase, 0.1)
		}
		cost /= float64(len(trainingSet.Cases))
		if cost < 0.001 {
			break
		}
	}

	// Print the weights
	Pl("Weights:")
	for _, layer := range network.Layers {
		for _, node := range layer.Nodes {
			Pl(node.Weights)
		}
	}

	// Make some predictions
	Pl("Predictions:")
	for _, trainingCase := range trainingSet.Cases {
		Pf("%v -> ", trainingCase.Inputs)
		outputs := network.Predict(trainingCase.Inputs)
		Pl(outputs)
	}
}
*/

// Network represents a neural network
type Network struct {
	Name       string
	InputCount int
	Layers     []*Layer
}

// Init initializes a network by connecting the layers and starting
// the worker pool.
func (n *Network) Init() {
	Assert(n.InputCount > 0)
	layer0 := n.Layers[0]
	layer0.Init(n.InputCount, nil)
	for i := 1; i < len(n.Layers); i++ {
		layer := n.Layers[i]
		// The test cases always create an input slice even if it's
		// not going to be used. Since we're in a downstream layer
		// here, we don't need the input slice, so we can set it to
		// nil.
		layer.inputs = nil
		upstreamLayer := n.Layers[i-1]
		layer.Init(0, upstreamLayer)
	}
}

// Save serializes the network configuration, weights, and biases to a
// JSON string.
func (n *Network) Save() string {
	var buf []byte
	buf, err := json.MarshalIndent(n, "", "  ")
	Ck(err)
	return string(buf)
}

// Load deserializes a network configuration, weights, and biases from
// a JSON string.
func Load(txt string) (n *Network, err error) {
	Return(&err)
	n = &Network{}
	err = json.Unmarshal([]byte(txt), &n)
	Ck(err)
	n.Init()
	return
}

// Clone returns a deep copy of the network, giving it a new name.
func (n *Network) Clone(newName string) (clone *Network) {
	txt := n.Save()
	clone, err := Load(txt)
	Ck(err)
	clone.Name = newName
	return
}

// Predict executes the forward function of a network and returns its
// output values.
func (n *Network) Predict(inputs []float64) (outputs []float64) {
	// clear caches
	for _, layer := range n.Layers {
		for _, node := range layer.Nodes {
			node.cached = false
		}
	}
	// set input values
	inputLayer := n.Layers[0]
	inputLayer.SetInputs(inputs)
	// execute forward function
	outputLayer := n.Layers[len(n.Layers)-1]
	for _, outputNode := range outputLayer.Nodes {
		outputs = append(outputs, outputNode.Output())
	}
	return
}

// RandomizeWeights sets the weights of all nodes to random values
// between min and max.
func (n *Network) Randomize() {
	for _, layer := range n.Layers {
		layer.Randomize()
	}
}

// Layer represents a layer of nodes.
type Layer struct {
	Nodes    []*Node
	inputs   []float64
	upstream *Layer
	net      *Network
}

// Init initializes a layer by connecting it to the upstream layer.
func (l *Layer) Init(inputs int, upstream *Layer) {
	// allow for test cases to pre-set inputs by only creating the
	// inputs slice if it's not already set
	if inputs > 0 && l.inputs == nil {
		l.inputs = make([]float64, inputs)
	}
	l.upstream = upstream
	for _, node := range l.Nodes {
		node.Init(l)
	}
}

// Randomize sets the weights and biases to random values.
func (l *Layer) Randomize() {
	for _, node := range l.Nodes {
		node.Randomize()
	}
}

// SetWeights sets the weights of all nodes to the given values.
func (l *Layer) SetWeights(weights [][]float64) {
	Assert(len(l.Nodes) == len(weights))
	for i, node := range l.Nodes {
		node.SetWeights(weights[i])
		node.cached = false
	}
}

// SetBias accepts a slice of bias values and sets the bias of each
// node to the corresponding value.
func (l *Layer) SetBiases(bias []float64) {
	for i, node := range l.Nodes {
		node.Bias = bias[i]
		node.cached = false
	}
}

// SetInputs sets the input values of this layer to the given vector.
func (l *Layer) SetInputs(inputs []float64) {
	Assert(l.upstream == nil)
	l.inputs = inputs
	for _, node := range l.Nodes {
		node.cached = false
	}
}

// Node represents a node in a neural network.
type Node struct {
	Weights        []float64 // weights for the inputs of this node
	Bias           float64   // bias for this node
	ActivationName string    // name of the activation function
	layer          *Layer    // layer this node belongs to
	activation     func(float64) float64
	activationD1   func(float64) float64
	// XXX remove cached, replace output with a pointer to a float64, and
	// set the pointer to nil when the node is modified
	output float64
	cached bool
	lock   sync.Mutex
}

// NewNode creates a new node.  The arguments are the activation
// function and its derivative.
func NewNode(activationName string) (n *Node) {
	n = &Node{ActivationName: activationName}
	return
}

// activationFuncs returns the activation function and its derivative
// for the given name.
func activationFuncs(name string) (activation, activationD1 func(float64) float64) {
	switch name {
	case "sigmoid":
		activation = sigmoid
		activationD1 = sigmoidD1
	case "tanh":
		activation = tanh
		activationD1 = tanhD1
	}
	return
}

// Init initializes a node by connecting it to the upstream layer and
// creating a weight slot for each upstream node.
func (n *Node) Init(layer *Layer) {
	activation, activationD1 := activationFuncs(n.ActivationName)
	Assert(activation != nil)
	Assert(activationD1 != nil)
	n.activation = activation
	n.activationD1 = activationD1

	n.layer = layer
	// allow for test cases to pre-set weights by only creating the
	// weights slice if it's not already set
	if n.Weights == nil {
		n.Weights = make([]float64, len(layer.Inputs()))
	}
}

// SetWeights sets the weights of this node to the given values.
func (n *Node) SetWeights(weights []float64) {
	// Assert(len(n.Weights) == len(weights), Spf("n.Weights: %v, weights: %v", n.Weights, weights))
	Assert(len(n.Weights) == len(weights))
	copy(n.Weights, weights)
}

var pool *Pool

// Pool is a worker pool for executing arbitrary functions in parallel
// without filling RAM.
type Pool struct {
	// Number of workers in the pool.
	Workers int
	// Queue of functions to execute.
	Queue chan func()
}

// worker is a Pool worker.
func worker(pool *Pool) {
	for f := range pool.Queue {
		f()
	}
}

// StartPool starts a global worker pool with the given number of workers.
func StartPool(workers, queuesize int) {
	pool = &Pool{
		Workers: workers,
		Queue:   make(chan func(), queuesize),
	}
	for i := 0; i < workers; i++ {
		go worker(pool)
		Pl("started worker", i)
	}
}

// Inputs returns either the input values of the current layer
// or the output values of the upstream layer.  If we need to fetch
// values from the upstream layer, we spawn a goroutine for each
// upstream node to parallelize the computations.
func (l *Layer) Inputs() (inputs []float64) {
	if l.upstream == nil {
		Assert(len(l.inputs) > 0, "layer: %#v", l)
		inputs = l.inputs
	} else {
		Assert(len(l.inputs) == 0, Spf("layer: %#v", l))
		// singleThreaded := false
		if pool == nil {
			// single-threaded
			for _, upstreamNode := range l.upstream.Nodes {
				inputs = append(inputs, upstreamNode.Output())
			}
		} else {
			// parallelized using the worker pool
			// create slice of channels to receive the output values
			outputChans := make([]chan float64, len(l.upstream.Nodes))
			// iterate over upstream nodes and spawn a goroutine for each
			go func() {
				for i, upstreamNode := range l.upstream.Nodes {
					outputChan := make(chan float64)
					outputChans[i] = outputChan
					pool.Queue <- mkWorker(outputChan, upstreamNode)
				}
			}()
			// iterate over the channels and get the output values
			for _, outputChan := range outputChans {
				inputs = append(inputs, <-outputChan)
			}
		}
	}
	return
}

// mkWorker creates a worker function for the given output channel and
// node.
func mkWorker(outputChan chan float64, node *Node) func() {
	return func() {
		// send the output value to the channel
		Pf("running worker %p\n", node)
		outputChan <- node.Output()
		Pf("worker done %p\n", node)
	}
}

// Output executes the forward function of a node and returns its
// output value.
func (n *Node) Output() (output float64) {
	// lock the node so that only one goroutine can access it at a time
	n.lock.Lock()
	defer n.lock.Unlock()
	if !n.cached {
		inputs := n.layer.Inputs()
		weightedSum := 0.0
		// add weighted inputs
		for i, input := range inputs {
			weightedSum += input * n.Weights[i]
		}
		// add bias
		weightedSum += n.Bias
		// apply activation function
		n.output = n.activation(weightedSum)
		n.cached = true
	}
	return n.output
}

func (n *Node) Randomize() {
	for i := range n.Weights {
		n.Weights[i] = rand.Float64()*2 - 1
	}
	n.Bias = rand.Float64()*2 - 1
}

// TrainingSet represents a set of training cases.
type TrainingSet struct {
	Cases []*TrainingCase
}

// TrainingCase represents a single training case.
type TrainingCase struct {
	Inputs  []float64
	Targets []float64
}

// NewTrainingCase creates a new training case.
func NewTrainingCase(inputs, targets []float64) (c *TrainingCase) {
	c = &TrainingCase{}
	c.Inputs = inputs
	c.Targets = targets
	return
}

// TrainOne runs one backpropagation iteration through the network. It
// takes a training case as input and returns the total error cost of
// the output nodes.
func (n *Network) TrainOne(trainingCase *TrainingCase, learningRate float64) (cost float64) {

	// provide inputs, get outputs
	outputs := n.Predict(trainingCase.Inputs)

	// initialize the error vector with the output errors
	errors := make([]float64, len(outputs))
	for i, target := range trainingCase.Targets {
		// populate the vector of errors for the output layer -- this is the
		// derivative of the cost function, which is just the difference
		// between the expected and actual output values.
		//
		// cost = 0.5 * (y - x)^2
		// dcost/dx = y - x
		// XXX DcostDoutput
		errors[i] = target - outputs[i]
		// accumulate total cost
		cost += 0.5 * math.Pow(target-outputs[i], 2)
	}

	// Backpropagate the errors through the network and update the
	// weights, starting from the output layer and working backwards.
	// Since Backprop is recursive, we only need to call it on the
	// output layer.
	outputLayer := n.Layers[len(n.Layers)-1]
	outputLayer.Backprop(errors, learningRate)

	return
}

// InputErrors returns the errors for the inputs of the given layer
func (l *Layer) InputErrors(outputErrors []float64) (inputErrors []float64) {
	Assert(len(outputErrors) == len(l.Nodes))
	inputErrors = make([]float64, len(l.Inputs()))
	for i, node := range l.Nodes {
		node.AddInputErrors(outputErrors[i], inputErrors)
	}
	return
}

// AddInputErrors adds the errors for the inputs of this node to the
// given inputErrors slice, updating the slice in place.
func (n *Node) AddInputErrors(outputError float64, inputErrors []float64) {
	Assert(len(inputErrors) == len(n.Weights))
	for i, weight := range n.Weights {
		delta := outputError * n.activationD1(n.Output())
		inputErrors[i] += weight * delta
	}
}

// UpdateWeights updates the weights of this node
func (n *Node) UpdateWeights(outputError float64, inputs []float64, learningRate float64) {
	for j, input := range inputs {
		// update the weight for the j-th input to this node
		n.Weights[j] += learningRate * outputError * n.activationD1(n.Output()) * input
	}
	// update the bias
	n.Bias += learningRate * outputError * n.activationD1(n.Output())
	// Debug("Backprop: node %d errs %v weights %v, bias %v", i, outputErrs, node.Weights, node.Bias)

	// mark cache dirty last so we only use the old output value
	// in the above calculations
	n.cached = false
}

// Backprop performs backpropagation on a layer.  It takes a vector of
// errors as input, updates the weights of the nodes in the layer, and
// recurses to the upstream layer.
// XXX move errs into the node struct, have node calculate its own
// deltas and errors
func (l *Layer) Backprop(outputErrs []float64, learningRate float64) {

	// Pf("Backprop: outputErrs: %v layer: %#v\n", outputErrs, l)

	// update the errors for the inputs to this layer
	inputErrors := l.InputErrors(outputErrs)

	// update the weights for the inputs to this layer
	for i, node := range l.Nodes {
		node.UpdateWeights(outputErrs[i], l.Inputs(), learningRate)
	}

	if l.upstream != nil {
		// recurse to the upstream layer
		l.upstream.Backprop(inputErrors, learningRate)
	}
}

/*

XXX reconcile the following docs with the above

	// We use the chain rule to calculate the partial derivative of the cost
	// function with respect to the weight:
	//
	// XXX
	// dcost/dweight = dcost/doutput * doutput/dweight
	//
	// The cost function is the sum of the squares of the errors.  We
	// multiply by 0.5 to simplify the derivative.
	//
	// cost = 0.5 * (target - output)^2
	//
	// The derivative of the cost function with respect to the output
	// is simply (target - output).
	//
	// dcost/doutput = target - output

	// The derivative of the output with respect to the weighted input
	// is the derivative of the activation function. Why?  Because
	// the output is the weighted sum of the inputs, which is the
	// XXX



	// XXX
	// doutput/dweight = activationD1(output)

	// The derivative of the weighted sum with respect to a weight is
	// the output of the related upstream node.  Why?  Because the
	// weighted sum is the sum of the products of each weight and the
	// output of the correspending upstream node.  To simplify, if for
	// example we only had one upstream node, then the weighted sum
	// would be:
	//
	// weightedsum = weight * input
	//
	// We can't change the upstream node output, so we can only change
	// the weight.  So we can think of the above equation in the form
	// of:
	//
	// y = mx + b
	//
	// where y is the weighted sum, m is the weight, x is the upstream
	// node output, and b is the bias.
	//
	// The derivative of that equation with respect to m is x:
	//
	// dy/dm = x
	//
	// So the derivative of the weighted sum with respect to a weight
	// is simply the input to the node.
	//
	// dweightedsum/dweight = input
	//
	// So going back to the original equation using the partial
	// derivatives in the chain rule:
	//
	// dcost/dweight = dcost/doutput * doutput/dweightedsum * dweightedsum/dweight
	//               = (target - output) * activationD1(output) * input
	//
	// Let's do it in code:
	for i, upstreamNode := range n.Upstream {
		dweightedsum_dweight := upstreamNode.Output()
		// dcost/dweight = dcost/doutput * doutput/dweightedsum * dweightedsum/dweight
		dcost_dweight := dcost_doutput * doutput_dweightedsum * dweightedsum_dweight
		// adjust weight
		n.Weights[i] += dcost_dweight

		// Now we need to adjust the upstream node weights.  We do
		// this by calling the upstream node's backprop function. but
		// we need a target to pass to Backprop().  Let's ask the upstream
		// node what its target is, passing in what we already know:
		targetUpstream := upstreamNode.Target(dweightedsum_dweight)
		// Now we can call the upstream node's backprop function:
		upstreamNode.Backprop(targetUpstream)
	}
*/
