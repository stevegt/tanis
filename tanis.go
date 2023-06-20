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
// XXX verify
func tanh(x float64) float64 {
	return math.Tanh(x)
}

// tanh derivative
// XXX verify
func tanhD1(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}

// relu activation function
// XXX verify
func relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// relu derivative
// XXX verify
func reluD1(x float64) float64 {
	if x < 0 {
		return 0
	}
	return 1
}

// linear activation function
func linear(x float64) float64 {
	return x
}

// linear derivative
func linearD1(x float64) float64 {
	return 1
}

// square activation function
func square(x float64) float64 {
	return x * x
}

// square derivative
func squareD1(x float64) float64 {
	return 2 * x
}

// square root activation function
func sqrt(x float64) float64 {
	// handle negative numbers
	if x < 0 {
		return -math.Sqrt(-x)
	}
	return math.Sqrt(x)
}

// square root derivative
func sqrtD1(x float64) float64 {
	// handle negative numbers
	if x < 0 {
		return -1 / (2 * math.Sqrt(-x))
	}
	return 1 / (2 * math.Sqrt(x))
}

// abs activation function
func abs(x float64) float64 {
	return math.Abs(x)
}

// abs derivative
func absD1(x float64) float64 {
	if x < 0 {
		return -1
	}
	if x > 0 {
		return 1
	}
	return 0
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
	Name        string
	InputCount  int
	InputNames  []string
	OutputNames []string
	Layers      []*Layer
	cost        float64 // most recent training cost
	lock        sync.Mutex
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
func (n *Network) Save() (out string) {
	n.lock.Lock()
	defer n.lock.Unlock()

	var buf []byte
	buf, err := json.MarshalIndent(n, "", "  ")
	if err != nil {
		n.showNaNs()
		Assert(false, "error marshaling network: %v", err)
	}
	out = string(buf)
	return
}

// Load deserializes a network configuration, weights, and biases from
// a JSON string.
func Load(txt string) (n *Network, err error) {
	defer Return(&err)
	n = &Network{}
	err = json.Unmarshal([]byte(txt), &n)
	Ck(err)
	n.init()
	return
}

// NewNetwork creates a new network with the given configuration.  The
// layerSizes arg is a slice of integers, where each integer is the
// number of nodes in a layer. The activation function defaults to
// sigmoid, and the default can be overriden for any layer or node by
// calling SetActivation().
func NewNetwork(name string, inputCount int, layerSizes ...int) (net *Network) {
	net = &Network{
		Name:       name,
		InputCount: inputCount,
		Layers:     make([]*Layer, len(layerSizes)),
	}
	layerInputCount := inputCount
	for layerNum := 0; layerNum < len(layerSizes); layerNum++ {
		if layerNum > 0 {
			layerInputCount = layerSizes[layerNum-1]
		}
		nodeCount := layerSizes[layerNum]
		layer := &Layer{
			Nodes: make([]*SimpleNode, nodeCount),
		}
		for n := 0; n < nodeCount; n++ {
			layer.Nodes[n] = newNode("sigmoid")
		}
		layer.init(layerInputCount, nil) // net.init() will populate upstream field
		net.Layers[layerNum] = layer
	}
	net.init()
	net.Randomize()
	return
}

// SetActivation sets the activation function for the given layer and
// node. The layerNum and nodeNum args are 0-based. If the layerNum is
// -1, then the activation function is set for all layers. If the
// nodeNum is -1, then the activation function is set for all nodes in
// the given layer. The activation function can be one of "sigmoid",
// "tanh", "relu", or "linear".
func (n *Network) SetActivation(layerNum, nodeNum int, activation string) {
	n.lock.Lock()
	defer n.lock.Unlock()
	if layerNum < 0 {
		for _, layer := range n.Layers {
			for _, node := range layer.Nodes {
				node.setActivation(activation)
			}
		}
		return
	}
	Assert(layerNum < len(n.Layers))
	layer := n.Layers[layerNum]
	if nodeNum < 0 {
		for _, node := range layer.Nodes {
			node.setActivation(activation)
		}
		return
	}
	Assert(nodeNum < len(layer.Nodes))
	node := layer.Nodes[nodeNum]
	node.setActivation(activation)
}

// GetName returns the name of the network.
func (n *Network) GetName() string {
	return n.Name
}

// GetCost returns the cost of the most recent call to Train() or Learn().
func (n *Network) GetCost() float64 {
	return n.cost
}

// SetInputNames sets the names of the inputs. The names are used in
// the arguments to LearnNamed() and PredictNamed().
func (n *Network) SetInputNames(names ...string) {
	n.lock.Lock()
	defer n.lock.Unlock()
	Assert(len(names) == n.InputCount)
	Assert(n.InputNames == nil)
	n.InputNames = names
}

// SetOutputNames sets the names of the outputs. The names are used in
// the arguments to LearnNamed() and PredictNamed().
func (n *Network) SetOutputNames(names ...string) {
	n.lock.Lock()
	defer n.lock.Unlock()
	Assert(len(names) == len(n.Layers[len(n.Layers)-1].Nodes))
	Assert(n.OutputNames == nil)
	n.OutputNames = names
}

// GetNames returns the names of the inputs and outputs.
func (n *Network) GetNames() (inputNames, outputNames []string) {
	n.lock.Lock()
	defer n.lock.Unlock()
	return n.InputNames, n.OutputNames
}

/*
// setNamedInputs sets the values of the named inputs.
func (n *Network) SetNamedInputs(inputs map[string]float64) {
	inputLayer := n.Layers[0]
	for i, name := range n.InputNames {
		inputLayer.inputs[i] = inputs[name]
	}
}

// getNamedOutputs returns the values of the named outputs.
func (n *Network) GetNamedOutputs() (outputs map[string]float64) {
	outputLayer := n.Layers[len(n.Layers)-1]
	outputs = make(map[string]float64)
	for i, name := range n.OutputNames {
		outputs[name] = outputLayer.Nodes[i].getOutput()
	}
	return
}
*/

// PredictNamed returns named outputs for the given named inputs.  It
// ignores named inputs which are not in the network, and sets to zero
// named inputs which are in the network but not in the given map.
func (n *Network) PredictNamed(inputMap map[string]float64) (outputMap map[string]float64) {
	n.lock.Lock()
	defer n.lock.Unlock()
	inputSlice := make([]float64, len(n.InputNames))
	for i, name := range n.InputNames {
		input, ok := inputMap[name]
		if !ok {
			continue
		}
		Assert(!math.IsNaN(input), "input %s is NaN", name)
		inputSlice[i] = input
	}
	outputSlice := n.predict(inputSlice)
	outputMap = make(map[string]float64)
	for i, name := range n.OutputNames {
		outputMap[name] = outputSlice[i]
	}
	return
}

// LearnNamed trains the network for one iteration with the given
// named input and target maps.  It ignores named inputs and targets
// which are not in the network, and assumes zero values for named
// inputs and targets which are in the network but not in the given
// maps.
func (n *Network) LearnNamed(inputMap, targetMap map[string]float64, rate float64) (cost float64) {
	n.lock.Lock()
	defer n.lock.Unlock()

	// special case: if we're calling this for the first time and the
	// input names and output names have not been set, then set them
	// now.
	if n.InputNames == nil && n.OutputNames == nil {
		Assert(len(inputMap) <= n.InputCount)
		Assert(len(targetMap) <= len(n.Layers[len(n.Layers)-1].Nodes))
		n.InputNames = make([]string, len(inputMap))
		n.OutputNames = make([]string, len(targetMap))
		i := 0
		for name, _ := range inputMap {
			n.InputNames[i] = name
			i++
		}
		i = 0
		for name, _ := range targetMap {
			n.OutputNames[i] = name
			i++
		}
	}
	Assert(len(n.InputNames) > 0)
	Assert(len(n.OutputNames) > 0)

	inputSlice := make([]float64, len(n.InputNames))
	targetSlice := make([]float64, len(n.OutputNames))
	// initialize with NaNs so that we can detect missing targets
	for i := range targetSlice {
		targetSlice[i] = math.NaN()
	}

	foundInput := false
	foundTarget := false
	for i, name := range n.InputNames {
		input, ok := inputMap[name]
		if !ok {
			continue
		}
		Assert(!math.IsNaN(input), "input %s is NaN", name)
		inputSlice[i] = input
		foundInput = true
	}
	for i, name := range n.OutputNames {
		target, ok := targetMap[name]
		if !ok {
			continue
		}
		Assert(!math.IsNaN(target), "target %s is NaN", name)
		targetSlice[i] = target
		foundTarget = true
	}
	Assert(foundInput, "%v %v", n.InputNames, inputMap)
	Assert(foundTarget, "%v %v", n.OutputNames, targetMap)
	cost = n.learn(inputSlice, targetSlice, rate)
	return
}

// Zero initializes the weights and biases to zero.
func (n *Network) Zero() {
	n.lock.Lock()
	defer n.lock.Unlock()
	for _, layer := range n.Layers {
		for _, node := range layer.Nodes {
			for i := range node.Weights {
				node.Weights[i] = 0
			}
			node.Bias = 0
		}
	}
}

// Add adds the weights and biases of the given network to this
// network. The given network must have the same structure as this
// network.
func (n *Network) Add(other *Network) {
	n.lock.Lock()
	defer n.lock.Unlock()
	other.lock.Lock()
	defer other.lock.Unlock()
	Assert(n.InputCount == other.InputCount)
	Assert(len(n.Layers) == len(other.Layers))
	for i := 0; i < len(n.Layers); i++ {
		layer := n.Layers[i]
		otherLayer := other.Layers[i]
		Assert(len(layer.Nodes) == len(otherLayer.Nodes))
		for j := 0; j < len(layer.Nodes); j++ {
			node := layer.Nodes[j]
			otherNode := otherLayer.Nodes[j]
			Assert(len(node.Weights) == len(otherNode.Weights))
			for k := 0; k < len(node.Weights); k++ {
				// if i+j+k == 0 { fmt.Println("adding", node.Weights[k], otherNode.Weights[k]) }
				node.Weights[k] += otherNode.Weights[k]
			}
			node.Bias += otherNode.Bias
		}
	}
	n.cost += other.cost
}

// Divide divides the weights and biases of this network by the given
// scalar.
func (n *Network) Divide(scalar float64) {
	n.lock.Lock()
	defer n.lock.Unlock()
	for _, layer := range n.Layers {
		for _, node := range layer.Nodes {
			for k := 0; k < len(node.Weights); k++ {
				// if i+j+k == 0 { fmt.Println("dividing", node.Weights[k], scalar) }
				node.Weights[k] /= scalar
			}
			node.Bias /= scalar
		}
	}
	n.cost /= scalar
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
	Assert(len(inputs) == n.InputCount)
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
		outputs = append(outputs, outputNode.Output())
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

// ShowNaNs shows the weights and biases of all nodes that are NaN.
func (n *Network) showNaNs() {
	for i, layer := range n.Layers {
		for j, node := range layer.Nodes {
			for k := 0; k < len(node.Weights); k++ {
				if math.IsNaN(node.Weights[k]) {
					Pf("layer %v node %v weight %v is NaN\n", i, j, k)
				}
			}
			if math.IsNaN(node.Bias) {
				Pf("layer %v node %v bias is NaN\n", i, j)
			}
		}
	}
}

// Layer represents a layer of nodes.
type Layer struct {
	Nodes    []*SimpleNode
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

type Node interface {
}

// SimpleNode represents a node in a neural network.
type SimpleNode struct {
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

// Activation runs the activation function of this node.
func (n *SimpleNode) Activation(x float64) float64 {
	return n.activation(x)
}

// ActivationD1 runs the derivative of the activation function of this
// node.
func (n *SimpleNode) ActivationD1(x float64) float64 {
	return n.activationD1(x)
}

// newNode creates a new node.  The arguments are the activation
// function and its derivative.
func newNode(activationName string) (n *SimpleNode) {
	n = &SimpleNode{ActivationName: activationName}
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
	case "relu":
		activation = relu
		activationD1 = reluD1
	case "linear":
		activation = linear
		activationD1 = linearD1
	case "square":
		activation = square
		activationD1 = squareD1
	case "sqrt":
		activation = sqrt
		activationD1 = sqrtD1
	case "abs":
		activation = abs
		activationD1 = absD1
	default:
		Assert(false, "unknown activation function: %s", name)
	}
	return
}

// setActivation sets the activation function and its derivative.
func (n *SimpleNode) setActivation(name string) {
	n.ActivationName = name
	n.activation, n.activationD1 = activationFuncs(name)
}

// init initializes a node by connecting it to the upstream layer and
// creating a weight slot for each upstream node.
func (n *SimpleNode) init(layer *Layer) {
	n.setActivation(n.ActivationName)
	n.layer = layer
	// allow for test cases to pre-set weights by only creating the
	// weights slice if it's not already set
	if n.Weights == nil {
		n.Weights = make([]float64, len(layer.getInputs()))
	}
}

// setWeights sets the weights of this node to the given values.
func (n *SimpleNode) setWeights(weights []float64) {
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
			inputs = append(inputs, upstreamNode.Output())
		}
	}
	return
}

// Output executes the forward function of a node and returns its
// output value.
func (n *SimpleNode) Output() (output float64) {
	// lock the node so that only one goroutine can access it at a time
	n.lock.Lock()
	defer n.lock.Unlock()
	if !n.cached {
		inputs := n.layer.getInputs()
		weightedSum := 0.0
		// add weighted inputs
		for i, input := range inputs {
			Assert(!math.IsNaN(input), "input: %v", input)
			weightedSum += input * n.Weights[i]
			// Assert(!math.IsNaN(weightedSum), "weightedSum: %v, input: %v, weight: %v", weightedSum, input, n.Weights[i])
		}
		// add bias
		weightedSum += n.Bias
		// Assert(!math.IsNaN(weightedSum), "weightedSum: %v", weightedSum)
		// apply activation function
		n.output = n.Activation(weightedSum)
		// handle overflow
		if math.IsNaN(n.output) || math.IsInf(n.output, 0) {
			Pf("overflow, randomizing node: weightedSum: %v, inputs: %v, weights: %v, bias: %v", weightedSum, inputs, n.Weights, n.Bias)
			n.randomize()
			n.output = rand.Float64()*2 - 1
		}
		Assert(!math.IsNaN(n.output), Spf("output: %v, weightedSum: %v, activation: %#v", n.output, weightedSum, n.Activation))
		n.cached = true
	}
	return n.output
}

func (n *SimpleNode) randomize() {
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
		n.cost = 0.0
		for _, trainingCase := range trainingSet.Cases {
			n.cost += n.trainOne(trainingCase, learningRate)
		}
		n.cost /= float64(len(trainingSet.Cases))
		if n.cost < maxCost {
			return n.cost, nil
		}
	}
	return n.cost, fmt.Errorf("max iterations reached")
}

func (n *Network) trainOne(trainingCase *TrainingCase, learningRate float64) (cost float64) {
	inputs := trainingCase.Inputs
	targets := trainingCase.Targets
	return n.learn(inputs, targets, learningRate)
}

// Learn runs one backpropagation iteration through the network. It
// takes inputs and targets returns the total error cost of
// the output nodes.
func (n *Network) Learn(inputs []float64, targets []float64, learningRate float64) (cost float64) {
	n.lock.Lock()
	defer n.lock.Unlock()
	return n.learn(inputs, targets, learningRate)
}

func (n *Network) learn(inputs []float64, targets []float64, learningRate float64) (cost float64) {
	Assert(len(inputs) == n.InputCount, "input count mismatch")
	outputLayer := n.Layers[len(n.Layers)-1]
	Assert(len(targets) == len(outputLayer.Nodes), "output count mismatch")

	// provide inputs, get outputs
	outputs := n.predict(inputs)
	n.cost = 0.0

	// initialize the error vector with the output errors
	errors := make([]float64, len(outputs))
	for i, target := range targets {
		// Assert(!math.IsNaN(target), "target is NaN")
		// skip this output if the target is NaN -- this is useful
		// when using LearnNamed, which may not have a target for
		// every output.
		if math.IsNaN(target) {
			continue
		}
		// populate the vector of errors for the output layer -- this is the
		// derivative of the cost function, which is just the difference
		// between the expected and actual output values.
		//
		// cost = 0.5 * (y - x)^2
		// dcost/dx = y - x
		// XXX DcostDoutput
		errors[i] = target - outputs[i]
		Assert(!math.IsNaN(errors[i]), Spf("error is NaN, target: %v, output: %v", target, outputs[i]))
		// accumulate total cost
		n.cost += 0.5 * math.Pow(target-outputs[i], 2)
	}
	// Pf("inputs: %v, outputs: %v, targets: %v, errors: %v, cost: %v\n", inputs, outputs, targets, errors, n.cost)

	// Backpropagate the errors through the network and update the
	// weights, starting from the output layer and working backwards.
	// Since Backprop is recursive, we only need to call it on the
	// output layer.
	outputLayer.backprop(errors, learningRate)

	return n.cost
}

// InputErrors returns the errors for the inputs of the given layer
func (l *Layer) InputErrors(outputErrors []float64) (inputErrors []float64) {
	Assert(len(outputErrors) == len(l.Nodes))
	inputErrors = make([]float64, len(l.getInputs()))
	for i, n := range l.Nodes {
		n.AddInputErrors(outputErrors[i], inputErrors)
	}
	return
}

// AddInputErrors adds the errors for the inputs of the given node to the
// given error vector.
func (n *SimpleNode) AddInputErrors(outputError float64, inputErrors []float64) {
	Assert(len(inputErrors) == len(n.Weights))
	for i, weight := range n.Weights {
		delta := outputError * n.ActivationD1(n.Output())
		inputErrors[i] += weight * delta
	}
}

// updateWeights updates the weights of this node
func (n *SimpleNode) updateWeights(outputError float64, inputs []float64, learningRate float64) {
	Assert(!math.IsNaN(outputError), "outputError is NaN")
	for j, input := range inputs {
		// update the weight for the j-th input to this node
		Assert(!math.IsNaN(input), "input is NaN")
		n.Weights[j] += learningRate * outputError * n.ActivationD1(n.Output()) * input
		// Assert(!math.IsNaN(n.Weights[j]), Spf("weight is NaN, outputError: %v, input: %v, activationD1: %v, weight: %v", outputError, input, n.ActivationName))
		// if overflow then randomize the weight
		if math.IsNaN(n.Weights[j]) || math.IsInf(n.Weights[j], 0) {
			Pf("overflow detected, randomizing weight: input: %v, outputError: %v, activation: %v, weight: %v\n", input, outputError, n.ActivationName, n.Weights[j])
			n.Weights[j] = rand.Float64()*2 - 1
		}
	}
	// update the bias
	n.Bias += learningRate * outputError * n.ActivationD1(n.Output())
	// randomize the bias if overflow
	if math.IsNaN(n.Bias) || math.IsInf(n.Bias, 0) {
		Pf("overflow detected, randomizing bias: outputError: %v, activation: %v, bias: %v\n", outputError, n.ActivationName, n.Bias)
		n.Bias = rand.Float64()*2 - 1
	}
	// Assert(!math.IsNaN(n.Bias), Spf("bias is NaN, outputError: %v, activationD1: %v", outputError, n.ActivationName))
	// Debug("Backprop: node %d errs %v weights %v, bias %v", i, outputErrs, node.Weights, node.Bias)

	// mark cache dirty last so we only use the old output value
	// in the above calculations
	n.cached = false
}

// updateWeights updates the weights of this layer
func (l *Layer) updateWeights(outputErrors []float64, learningRate float64) {
	Assert(len(outputErrors) == len(l.Nodes))
	for i, node := range l.Nodes {
		node.updateWeights(outputErrors[i], l.getInputs(), learningRate)
	}
}

// backprop performs backpropagation on a layer.  It takes a vector of
// errors as input, updates the weights of the nodes in the layer, and
// recurses to the upstream layer.
// XXX move errs into the node struct, have node calculate its own
// deltas and errors
func (l *Layer) backprop(outputErrs []float64, learningRate float64) {

	// handle overflows in the output errors
	newOutputErrs := make([]float64, len(outputErrs))
	for i, outputErr := range outputErrs {
		if math.IsNaN(outputErr) || math.IsInf(outputErr, 0) {
			Pf("overflow detected, randomizing output error: %v\n", outputErr)
			newOutputErrs[i] = rand.Float64()*2 - 1
		} else {
			newOutputErrs[i] = outputErr
		}
	}
	outputErrs = newOutputErrs

	// Pf("Backprop: outputErrs: %v layer: %#v\n", outputErrs, l)

	// get the errors for the inputs to this layer
	inputErrors := l.InputErrors(outputErrs)

	// update the input weights in this layer
	l.updateWeights(outputErrs, learningRate)

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
