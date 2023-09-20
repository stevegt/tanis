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
// takes inputs and targets and returns the total cost of
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

	// initialize the gradient vector with the output gradients
	gradients := make([]float64, len(outputs))
	for i, target := range targets {
		// Assert(!math.IsNaN(target), "target is NaN")
		// skip this output if the target is NaN -- this is useful
		// when using LearnNamed, which may not have a target for
		// every output.
		if math.IsNaN(target) {
			continue
		}
		// populate the vector of gradients for the output layer -- this is the
		// derivative of the cost function, which is just the difference
		// between the expected and actual output values.
		//
		// cost = 0.5 * (y - x)^2
		// dcost/dx = y - x
		// XXX DcostDoutput
		gradients[i] = target - outputs[i]
		Assert(!math.IsNaN(gradients[i]), Spf("gradient is NaN, target: %v, output: %v", target, outputs[i]))
		// accumulate total cost
		n.cost += 0.5 * math.Pow(gradients[i], 2)
	}

	// Backpropagate the gradients through the network and update the
	// weights, starting from the output layer and working backwards.
	// Since Backprop is recursive, we only need to call it on the
	// output layer.
	outputLayer.backprop(gradients, learningRate)

	return n.cost
}

// InputGradients returns the gradients for the inputs of the given layer.
func (l *Layer) InputGradients(outputGradients []float64) (inputGradients []float64) {
	Assert(len(outputGradients) == len(l.Nodes))
	inputGradients = make([]float64, len(l.getInputs()))
	for i, n := range l.Nodes {
		n.AddInputGradients(outputGradients[i], inputGradients)
	}
	return
}

// AddInputGradients adds the gradients for the inputs of this node to the
// given gradient vector.
func (n *SimpleNode) AddInputGradients(outputGradient float64, inputGradients []float64) {
	Assert(len(inputGradients) == len(n.Weights))
	for i, weight := range n.Weights {
		delta := outputGradient * n.ActivationD1(n.Output())
		inputGradients[i] += weight * delta
	}
}

// updateWeights updates the weights of this node
func (n *SimpleNode) updateWeights(outputGradient float64, inputs []float64, learningRate float64) {
	Assert(!math.IsNaN(outputGradient), "outputGradient is NaN")
	for j, input := range inputs {
		// update the weight for the j-th input to this node
		Assert(!math.IsNaN(input), "input is NaN")
		n.Weights[j] += learningRate * outputGradient * n.ActivationD1(n.Output()) * input
		// if overflow then randomize the weight
		if math.IsNaN(n.Weights[j]) || math.IsInf(n.Weights[j], 0) {
			Pf("overflow detected, randomizing weight: input: %v, outputGradient: %v, activation: %v, weight: %v\n", input, outputGradient, n.ActivationName, n.Weights[j])
			n.Weights[j] = rand.Float64()*2 - 1
		}
	}
	// update the bias
	n.Bias += learningRate * outputGradient * n.ActivationD1(n.Output())
	// randomize the bias if overflow
	if math.IsNaN(n.Bias) || math.IsInf(n.Bias, 0) {
		Pf("overflow detected, randomizing bias: outputGradient: %v, activation: %v, bias: %v\n", outputGradient, n.ActivationName, n.Bias)
		n.Bias = rand.Float64()*2 - 1
	}
	// Debug("Backprop: node %d errs %v weights %v, bias %v", i, outputErrs, node.Weights, node.Bias)

	// mark cache dirty last so we only use the old output value
	// in the above calculations
	n.cached = false
}

// updateWeights updates the weights of this layer
func (l *Layer) updateWeights(outputGradients []float64, learningRate float64) {
	Assert(len(outputGradients) == len(l.Nodes))
	for i, node := range l.Nodes {
		node.updateWeights(outputGradients[i], l.getInputs(), learningRate)
	}
}

// backprop performs backpropagation on a layer.  It takes a vector of
// gradients as input, updates the weights of the nodes in the layer, and
// recurses to the upstream layer.
// XXX move errs into the node struct, have node calculate its own
// deltas and gradients, and update its own weights
func (l *Layer) backprop(outputGradients []float64, learningRate float64) {
	// handle overflows in the output gradients
	newOutputGradients := make([]float64, len(outputGradients))
	for i, outputGradient := range outputGradients {
		if math.IsNaN(outputGradient) || math.IsInf(outputGradient, 0) {
			Pf("overflow detected, randomizing output gradient: %v\n", outputGradient)
			newOutputGradients[i] = rand.Float64()*2 - 1
		} else {
			newOutputGradients[i] = outputGradient
		}
	}
	outputGradients = newOutputGradients
	// get the gradients for the inputs to this layer
	inputGradients := l.InputGradients(outputGradients)
	// update the input weights in this layer
	l.updateWeights(outputGradients, learningRate)
	if l.upstream != nil {
		// recurse to the upstream layer
		l.upstream.backprop(inputGradients, learningRate)
	}
}

// Adam is an adaptive learning rate algorithm for gradient descent.
// See https://arxiv.org/pdf/1412.6980.pdf
type Adam struct {
	// learning rate
	learningRate float64
	// decay rate for first moment estimate
	beta1 float64
	// decay rate for second moment estimate
	beta2 float64
	// first moment estimate
	m []float64
	// second moment estimate
	v []float64
	// bias-corrected first moment estimate
	mHat []float64
	// bias-corrected second moment estimate
	vHat []float64
	// number of iterations
	t int
}

// NewAdam returns a new Adam optimizer
func NewAdam(learningRate, beta1, beta2 float64, nWeights int) *Adam {
	return &Adam{
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		m:            make([]float64, nWeights),
		v:            make([]float64, nWeights),
		mHat:         make([]float64, nWeights),
		vHat:         make([]float64, nWeights),
	}
}

// Update updates the weights of the given layer using Adam
func (a *Adam) Update(l *Layer, outputGradients []float64) {
	// get the gradients for the inputs to this layer
	inputGradients := l.InputGradients(outputGradients)
	// update the input weights in this layer
	a.updateWeights(l, outputGradients)
	if l.upstream != nil {
		// recurse to the upstream layer
		a.Update(l.upstream, inputGradients)
	}
}

// updateWeights updates the weights of this layer
func (a *Adam) updateWeights(l *Layer, outputGradients []float64) {
	Assert(len(outputGradients) == len(l.Nodes))
	for i, node := range l.Nodes {
		node.updateWeightsAdam(outputGradients[i], l.getInputs(), a)
	}
}

// updateWeightsAdam updates the weights of this node using Adam.  The Adam
// algorithm is described in https://arxiv.org/pdf/1412.6980.pdf.  The
// formulas for the algorithm are:
//
// m = beta1 * m + (1 - beta1) * gradient
// v = beta2 * v + (1 - beta2) * gradient^2
// mHat = m / (1 - beta1^t)
// vHat = v / (1 - beta2^t)
// weight = weight - learningRate * mHat / (sqrt(vHat) + epsilon)
func (n *SimpleNode) updateWeightsAdam(outputGradient float64, inputs []float64, a *Adam) {
	epsilon := 1e-8
	for j, input := range inputs {
		// update the first moment estimate
		a.m[j] = a.beta1*a.m[j] + (1-a.beta1)*outputGradient
		// update the second moment estimate
		a.v[j] = a.beta2*a.v[j] + (1-a.beta2)*outputGradient*outputGradient
		// bias-correct the first moment estimate
		a.mHat[j] = a.m[j] / (1 - math.Pow(a.beta1, float64(a.t)))
		// bias-correct the second moment estimate
		a.vHat[j] = a.v[j] / (1 - math.Pow(a.beta2, float64(a.t)))
		// update the weight
		n.Weights[j] -= a.learningRate * a.mHat[j] / (math.Sqrt(a.vHat[j]) + epsilon)
		// if overflow then randomize the weight
		if math.IsNaN(n.Weights[j]) || math.IsInf(n.Weights[j], 0) {
			Pf("overflow detected, randomizing weight: input: %v, outputGradient: %v, activation: %v, weight: %v\n", input, outputGradient, n.ActivationName, n.Weights[j])
			n.Weights[j] = rand.Float64()*2 - 1
		}
	}
	// update the bias
	n.Bias -= a.learningRate * a.mHat[len(inputs)] / (math.Sqrt(a.vHat[len(inputs)]) + epsilon)
	// randomize the bias if overflow
	if math.IsNaN(n.Bias) || math.IsInf(n.Bias, 0) {
		Pf("overflow detected, randomizing bias: input: %v, outputGradient: %v, activation: %v, bias: %v\n", inputs, outputGradient, n.ActivationName, n.Bias)
		n.Bias = rand.Float64()*2 - 1
	}
}
