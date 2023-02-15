package tanis

import (
	"encoding/json"
	"fmt"
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
	lock       sync.Mutex
}

// init initializes a network by connecting the layers.
func (n *Network) init() {
	n.lock.Lock()
	defer n.lock.Unlock()

	Assert(n.InputCount > 0)
	layer0 := n.Layers[0]
	layer0.init(n.InputCount, nil)
	for i := 1; i < len(n.Layers); i++ {
		layer := n.Layers[i]
		// The test cases always create an input slice even if it's
		// not going to be used. Since we're in a downstream layer
		// here, we don't need the input slice, so we can set it to
		// nil.
		layer.inputs = nil
		upstreamLayer := n.Layers[i-1]
		layer.init(0, upstreamLayer)
	}
}

// Save serializes the network configuration, weights, and biases to a
// JSON string.
func (n *Network) Save() string {
	n.lock.Lock()
	defer n.lock.Unlock()

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
	n.init()
	return
}

// NewNetwork creates a new network with the given configuration.  The
// configuration is a slice of integers, where each integer is the
// number of nodes in a layer. The first integer is the number of
// inputs, and the last integer is the number of outputs.  The
// activation function defaults to sigmoid.
func NewNetwork(name string, conf ...int) (n *Network) {
	n = &Network{
		Name:       name,
		InputCount: conf[0],
		Layers:     make([]*Layer, len(conf)-1),
	}
	for i := 1; i < len(conf); i++ {
		inputCount := conf[i-1]
		nodeCount := conf[i]
		n.Layers[i-1] = newLayer(inputCount, nodeCount, "sigmoid")
	}
	n.init()
	n.Randomize()
	return
}

// newLayer creates a new layer with the given number of inputs and
// nodes.
func newLayer(inputCount, nodeCount int, activationName string) (l *Layer) {
	l = &Layer{
		Nodes: make([]*Node, nodeCount),
	}
	for i := 0; i < nodeCount; i++ {
		l.Nodes[i] = newNode(activationName)
	}
	l.init(inputCount, nil)
	return
}

// Clone returns a deep copy of the network, giving it a new name.
func (n *Network) Clone(newName string) (clone *Network) {
	// lock not needed here because Save() locks
	txt := n.Save()
	clone, err := Load(txt)
	Ck(err)
	clone.Name = newName
	return
}

// Predict executes the forward function of a network and returns its
// output values.
func (n *Network) Predict(inputs []float64) (outputs []float64) {
	n.lock.Lock()
	defer n.lock.Unlock()
	outputs = n.predict(inputs)
	return
}

func (n *Network) predict(inputs []float64) (outputs []float64) {
	// clear caches
	for _, layer := range n.Layers {
		for _, node := range layer.Nodes {
			node.cached = false
		}
	}
	// set input values
	inputLayer := n.Layers[0]
	inputLayer.setInputs(inputs)
	// execute forward function
	outputLayer := n.Layers[len(n.Layers)-1]
	for _, outputNode := range outputLayer.Nodes {
		outputs = append(outputs, outputNode.getOutput())
	}
	return
}

// RandomizeWeights sets the weights of all nodes to random values
func (n *Network) Randomize() {
	n.lock.Lock()
	defer n.lock.Unlock()
	for _, layer := range n.Layers {
		layer.randomize()
	}
}

// Layer represents a layer of nodes.
type Layer struct {
	Nodes    []*Node
	inputs   []float64
	upstream *Layer
}

// init initializes a layer by connecting it to the upstream layer.
func (l *Layer) init(inputs int, upstream *Layer) {
	// allow for test cases to pre-set inputs by only creating the
	// inputs slice if it's not already set
	if inputs > 0 && l.inputs == nil {
		l.inputs = make([]float64, inputs)
	}
	l.upstream = upstream
	for _, node := range l.Nodes {
		node.init(l)
	}
}

// randomize sets the weights and biases to random values.
func (l *Layer) randomize() {
	for _, node := range l.Nodes {
		node.randomize()
	}
}

// setWeights sets the weights of all nodes to the given values.
func (l *Layer) setWeights(weights [][]float64) {
	Assert(len(l.Nodes) == len(weights))
	for i, node := range l.Nodes {
		node.setWeights(weights[i])
		node.cached = false
	}
}

// SetBias accepts a slice of bias values and sets the bias of each
// node to the corresponding value.
func (l *Layer) setBiases(bias []float64) {
	for i, node := range l.Nodes {
		node.Bias = bias[i]
		node.cached = false
	}
}

// setInputs sets the input values of this layer to the given vector.
func (l *Layer) setInputs(inputs []float64) {
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

// newNode creates a new node.  The arguments are the activation
// function and its derivative.
func newNode(activationName string) (n *Node) {
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

// init initializes a node by connecting it to the upstream layer and
// creating a weight slot for each upstream node.
func (n *Node) init(layer *Layer) {
	activation, activationD1 := activationFuncs(n.ActivationName)
	Assert(activation != nil)
	Assert(activationD1 != nil)
	n.activation = activation
	n.activationD1 = activationD1

	n.layer = layer
	// allow for test cases to pre-set weights by only creating the
	// weights slice if it's not already set
	if n.Weights == nil {
		n.Weights = make([]float64, len(layer.getInputs()))
	}
}

// setWeights sets the weights of this node to the given values.
func (n *Node) setWeights(weights []float64) {
	// Assert(len(n.Weights) == len(weights), Spf("n.Weights: %v, weights: %v", n.Weights, weights))
	Assert(len(n.Weights) == len(weights))
	copy(n.Weights, weights)
}

// getInputs returns either the input values of the current layer
// or the output values of the upstream layer.
func (l *Layer) getInputs() (inputs []float64) {
	if l.upstream == nil {
		Assert(len(l.inputs) > 0, "layer: %#v", l)
		inputs = l.inputs
	} else {
		Assert(len(l.inputs) == 0, Spf("layer: %#v", l))
		for _, upstreamNode := range l.upstream.Nodes {
			inputs = append(inputs, upstreamNode.getOutput())
		}
	}
	return
}

// getOutput executes the forward function of a node and returns its
// output value.
func (n *Node) getOutput() (output float64) {
	// lock the node so that only one goroutine can access it at a time
	n.lock.Lock()
	defer n.lock.Unlock()
	if !n.cached {
		inputs := n.layer.getInputs()
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

func (n *Node) randomize() {
	for i := range n.Weights {
		n.Weights[i] = rand.Float64()*2 - 1
	}
	n.Bias = rand.Float64()*2 - 1
}

// TrainingSet represents a set of training cases.
type TrainingSet struct {
	Cases []*TrainingCase
}

// NewTrainingSet creates a new training set.
func NewTrainingSet() (ts *TrainingSet) {
	ts = &TrainingSet{}
	return
}

// Validate validates a network against a training set, ensuring that
// the network outputs are within maxCost of the expected outputs.
func (n *Network) Validate(ts *TrainingSet, maxCost float64) (err error) {
	for _, tc := range ts.Cases {
		outputs := n.Predict(tc.Inputs)
		cost := 0.0
		for i, output := range outputs {
			cost += math.Abs(output - tc.Targets[i])
		}
		if cost > maxCost {
			return fmt.Errorf("cost too high for inputs: %v, expected: %v, got: %v", tc.Inputs, tc.Targets, outputs)
		}
	}
	return
}

// Add adds a training case to the set.
func (ts *TrainingSet) Add(inputs, targets []float64) {
	ts.Cases = append(ts.Cases, &TrainingCase{Inputs: inputs, Targets: targets})
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

// Mimic trains the network to match the outputs of oldNet given
// trainingSet inputs.  Ignores the target values in trainingSet; instead
// asks oldNet to predict output values for trainingSet inputs, then
// uses those output values as targets for training the network.
func (n *Network) Mimic(oldNet *Network, trainingSet *TrainingSet, learningRate float64, iterations int, maxCost float64) (cost float64, err error) {
	// build a new training set by asking oldNet to predict the outputs
	newSet := oldNet.MkTrainingSet(trainingSet)
	// train the network to match the outputs of oldNet
	cost, err = n.Train(newSet, learningRate, iterations, maxCost)
	return
}

// MkTrainingSet creates a new training set by running the given inputs through
// the network.  The targets in the input training set are ignored.
func (n *Network) MkTrainingSet(trainingSet *TrainingSet) (newSet *TrainingSet) {
	newSet = NewTrainingSet()
	for _, tc := range trainingSet.Cases {
		outputs := n.Predict(tc.Inputs)
		newSet.Add(tc.Inputs, outputs)
	}
	return
}

// Append appends the given training set to the current training set,
// returning a new training set.
func (ts *TrainingSet) Append(other *TrainingSet) (newSet *TrainingSet) {
	newSet = NewTrainingSet()
	newSet.Cases = append(newSet.Cases, ts.Cases...)
	newSet.Cases = append(newSet.Cases, other.Cases...)
	return
}

// Train the network given a training set.
func (n *Network) Train(trainingSet *TrainingSet, learningRate float64, iterations int, maxCost float64) (cost float64, err error) {
	n.lock.Lock()
	defer n.lock.Unlock()

	for i := 0; i < iterations; i++ {
		cost = 0.0
		for _, trainingCase := range trainingSet.Cases {
			cost += n.trainOne(trainingCase, learningRate)
		}
		cost /= float64(len(trainingSet.Cases))
		if cost < maxCost {
			return cost, nil
		}
	}
	return cost, fmt.Errorf("max iterations reached")
}

// trainOne runs one backpropagation iteration through the network. It
// takes a training case as input and returns the total error cost of
// the output nodes.
func (n *Network) trainOne(trainingCase *TrainingCase, learningRate float64) (cost float64) {
	// provide inputs, get outputs
	outputs := n.predict(trainingCase.Inputs)

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
	outputLayer.backprop(errors, learningRate)

	return
}

// InputErrors returns the errors for the inputs of the given layer
func (l *Layer) InputErrors(outputErrors []float64) (inputErrors []float64) {
	Assert(len(outputErrors) == len(l.Nodes))
	inputErrors = make([]float64, len(l.getInputs()))
	for i, node := range l.Nodes {
		node.addInputErrors(outputErrors[i], inputErrors)
	}
	return
}

// addInputErrors adds the errors for the inputs of this node to the
// given inputErrors slice, updating the slice in place.
func (n *Node) addInputErrors(outputError float64, inputErrors []float64) {
	Assert(len(inputErrors) == len(n.Weights))
	for i, weight := range n.Weights {
		delta := outputError * n.activationD1(n.getOutput())
		inputErrors[i] += weight * delta
	}
}

// updateWeights updates the weights of this node
func (n *Node) updateWeights(outputError float64, inputs []float64, learningRate float64) {
	for j, input := range inputs {
		// update the weight for the j-th input to this node
		n.Weights[j] += learningRate * outputError * n.activationD1(n.getOutput()) * input
	}
	// update the bias
	n.Bias += learningRate * outputError * n.activationD1(n.getOutput())
	// Debug("Backprop: node %d errs %v weights %v, bias %v", i, outputErrs, node.Weights, node.Bias)

	// mark cache dirty last so we only use the old output value
	// in the above calculations
	n.cached = false
}

// backprop performs backpropagation on a layer.  It takes a vector of
// errors as input, updates the weights of the nodes in the layer, and
// recurses to the upstream layer.
// XXX move errs into the node struct, have node calculate its own
// deltas and errors
func (l *Layer) backprop(outputErrs []float64, learningRate float64) {

	// Pf("Backprop: outputErrs: %v layer: %#v\n", outputErrs, l)

	// update the errors for the inputs to this layer
	inputErrors := l.InputErrors(outputErrs)

	// update the weights for the inputs to this layer
	for i, node := range l.Nodes {
		node.updateWeights(outputErrs[i], l.getInputs(), learningRate)
	}

	if l.upstream != nil {
		// recurse to the upstream layer
		l.upstream.backprop(inputErrors, learningRate)
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
