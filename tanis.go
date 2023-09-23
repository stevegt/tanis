package tanis

import (
	"encoding/json"
	"math"
	"math/rand"
	"strings"
	"sync"

	. "github.com/stevegt/goadapt"
	"github.com/stevegt/tanis/dna"
	"github.com/stevegt/tanis/shape"
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

// Network represents a neural network
type Network struct {
	Name string
	// InputCount  int
	InputNames  []string
	OutputNames []string
	Layers      []*Layer
	// most recent fitness
	// - fitness is equal to the square root of the mean squared error
	// - lower fitness is better
	fitness float64
	lock    sync.Mutex
}

// XXXShapeString returns a string representation of the network's shape.
func (n *Network) XXXShapeString() (shape string) {
	// layerShapes is a slice of strings, one for each layer
	layerShapes := make([]string, len(n.Layers))
	for i, layer := range n.Layers {
		// count nodes by activation function
		counts := make(map[string]int)
		for _, node := range layer.Nodes {
			counts[node.ActivationName]++
		}
		// convert counts to a slice of strings
		shapes := make([]string, 0, len(counts))
		for activation, count := range counts {
			shapes = append(shapes, Spf("%s(%d)", activation, count))
		}
		// join the shapes for this layer
		layerShapes[i] = Spf("(%s)", strings.Join(shapes, " "))
	}
	shape = Spf("%s((%s) %s (%s))", n.Name, strings.Join(n.InputNames, " "), layerShapes, strings.Join(n.OutputNames, " "))
	return
}

// ShapeString returns a string representation of the network.
// This is essentially everything except the weights and biases, and
// is suitable for feeding into NewNetwork().
func (n *Network) ShapeString() (out string) {
	s := shape.Shape{
		Name:        n.Name,
		InputNames:  n.InputNames,
		OutputNames: n.OutputNames,
	}
	for i, layer := range n.Layers {
		layerShape := &shape.LayerShape{}
		for _, node := range layer.Nodes {
			nodeShape := &shape.NodeShape{
				ActivationName: node.ActivationName,
			}
			layerShape.Nodes = append(layerShape.Nodes, nodeShape)
		}
		// set output node names
		if i == len(n.Layers)-1 {
			// output layer
			Assert(len(layer.Nodes) == len(n.OutputNames), "n.OutputNames %v layer.Nodes %v", n.OutputNames, layer.Nodes)
			for j, nodeShape := range layerShape.Nodes {
				nodeShape.Name = n.OutputNames[j]
			}
		}
		s.LayerShapes = append(s.LayerShapes, layerShape)
	}
	out = s.String()
	return
}

// NetworkFromDNA returns a new network from a DNA object.  It is the
// inverse of Network.DNA().
func NetworkFromDNA(D *dna.DNA) (net *Network, err error) {
	defer Return(&err)
	net = &Network{
		Name:        D.Name,
		InputNames:  D.InputNames,
		OutputNames: D.OutputNames,
	}
	err = net.ExecDNA(D)
	Ck(err)
	return
}

// ExecDNA executes the statements in a DNA object to change the network.
func (net *Network) ExecDNA(D *dna.DNA) (err error) {
	defer Return(&err)
	var layer *Layer
	var node *SimpleNode
	for _, statement := range D.Statements {
		if statement.Opcode == dna.OpHalt {
			break
		}
		layer, node = net.execStatement(layer, node, statement)
	}
	return
}

// execStatement executes a single statement on a network.
func (net *Network) execStatement(layer *Layer, node *SimpleNode, statement *dna.Statement) (*Layer, *SimpleNode) {
	arg := statement.Arg
	opcode := statement.Opcode % dna.OpLast
	switch opcode {
	case dna.OpAddLayer:
		// add layer
		layer = &Layer{}
		net.Layers = append(net.Layers, layer)
		node = nil
	case dna.OpAddNode:
		// add node to most recent layer
		if layer != nil {
			node = &SimpleNode{}
			layer.Nodes = append(layer.Nodes, node)
		}
	case dna.OpSetActivation:
		// set activation on most recent node
		if layer != nil && node != nil {
			length := len(dna.ActivationName)
			// convert from float64 to int in multiple steps so we
			// handle overflows in th conversion to smaller types
			absArg := math.Abs(arg)
			signedInt := int8(absArg)
			absInt := int(math.Abs(float64(signedInt)))
			actNum := absInt % length
			node.ActivationName = dna.ActivationName[actNum]
			Assert(node.ActivationName != "", "invalid activation number: length=%v, arg=%v, actNum=%v", length, arg, actNum)
		}
	case dna.OpSetBias:
		// set bias on most recent node
		if layer != nil && node != nil {
			node.Bias = arg
		}
	case dna.OpAddWeight:
		// add weight to most recent node
		if layer != nil && node != nil {
			node.Weights = append(node.Weights, arg)
		}
	default:
		Assert(false, "unknown opcode: %v", opcode)
	}
	return layer, node
}

// DNA returns a DNA object for the network.  The DNA object is
// suitable for feeding into NetworkFromDNA(). It is intended to be
// used in crossover and mutation.  It should never be used for
// storage or transmission.
func (net *Network) DNA() (D *dna.DNA) {
	D = dna.New()
	D.Name = net.Name
	D.InputNames = net.InputNames
	D.OutputNames = net.OutputNames
	// add layers
	for _, layer := range net.Layers {
		D.AddOp(dna.OpAddLayer, 0)
		for _, node := range layer.Nodes {
			// add node
			D.AddOp(dna.OpAddNode, 0)
			// set activation
			D.AddOp(dna.OpSetActivation, float64(dna.ActivationNum[node.ActivationName]))
			// set bias
			D.AddOp(dna.OpSetBias, node.Bias)
			for _, weight := range node.Weights {
				// add weight
				D.AddOp(dna.OpAddWeight, weight)
			}
		}
	}
	return
}

/*
// parseShape returns a Shape struct containing the network's
// configuration from a shape string.
func parseShape(in string) (shape Shape, err error) {
	defer Return(&err)
	in = strings.TrimSpace(in)
	// get name and body
	// name((inputNames...) (activation(count)...)... (outputNames...))
	re := regexp.MustCompile(`^(\w+)\((.*)\)$`)
	m := re.FindStringSubmatch(in)
	Assert(len(m) == 3, "invalid shape: name not found: %s", in)
	shape.Name = m[1]
	body := m[2]
	// get body parts
	// (inputNames...) (activation(count)...)... (outputNames...)
	re = regexp.MustCompile(`\(([^)]+)\)`)
	mbody := re.FindAllStringSubmatch(body, -1)
	Assert(len(mbody) >= 3, "invalid shape: missing body parts: %s", body)
	shape.InputNames = strings.Split(mbody[0][1], " ")
	shape.OutputNames = strings.Split(mbody[len(mbody)-1][1], " ")
	shape.LayerShapes = make([]*LayerShape, len(mbody)-2)
	for i := 1; i < len(mbody)-1; i++ {
		layer := mbody[i][1]
		// activation(count)...
		re = regexp.MustCompile(`(\w+)\((\d+)\)`)
		mlayer := re.FindAllStringSubmatch(layer, -1)
		Assert(len(mlayer) > 0, "invalid shape: garbled layer: %s", layer)
		layerShape := shape.LayerShapes[i-1]
		for _, m := range mlayer {
			activation := m[1]
			count, err := strconv.Atoi(m[2])
			Assert(err == nil, "invalid shape: invalid count: %s", m[2])
			(*layerShape)[activation] = count
		}
	}
	return
}
*/

// clean readies a network for the next training run by clearing
// temporary values and ensuring structures are initialized.
func (n *Network) clean() (err error) {
	defer Return(&err)
	/*
		defer func() {
			// show network configuration on panic, then re-raise
			if r := recover(); r != nil {
				Pprint(n)
				panic(r)
			}
		}()
	*/
	Assert(len(n.Layers) > 1)
	inputCount := len(n.InputNames)
	Assert(inputCount > 0)
	Assert(len(n.OutputNames) > 0)
	n.fitness = 0
	Assert(len(n.Layers) > 1)
	layer0 := n.Layers[0]
	err = layer0.clean(inputCount, nil, nil)
	Ck(err)
	for i := 1; i < len(n.Layers); i++ {
		layer := n.Layers[i]
		upstreamLayer := n.Layers[i-1]
		inputCount := len(upstreamLayer.Nodes)
		if i == len(n.Layers)-1 {
			// last layer
			err = layer.clean(inputCount, n.OutputNames, upstreamLayer)
			Ck(err)
		} else {
			err = layer.clean(inputCount, nil, upstreamLayer)
			Ck(err)
		}
	}
	return
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
	return
}

// NewNetwork creates a new network with the given shape.
func NewNetwork(shapeStr string) (net *Network, err error) {
	defer Return(&err)
	shape, err := shape.Parse(shapeStr)
	Ck(err)
	net, err = NetworkFromShape(shape)
	Ck(err)
	err = net.clean()
	Ck(err)
	return
}

func NetworkFromShape(s *shape.Shape) (net *Network, err error) {
	defer Return(&err)
	D := DNAFromShape(s)
	net, err = NetworkFromDNA(D)
	Ck(err)
	return
}

func DNAFromShape(shape *shape.Shape) (D *dna.DNA) {
	name := shape.Name
	inputNames := shape.InputNames
	outputNames := shape.OutputNames
	layerShapes := shape.LayerShapes
	Assert(len(layerShapes) > 1)
	Assert(len(inputNames) > 0)
	Assert(len(outputNames) > 0)
	// create DNA
	D = dna.New()
	// add name
	D.Name = name
	D.InputNames = inputNames
	D.OutputNames = outputNames
	// add layers
	inputCount := len(inputNames)
	for _, layerShape := range layerShapes {
		// add layer
		D.AddOp(dna.OpAddLayer, 0)
		nodeCount := len(layerShape.Nodes)
		for _, node := range layerShape.Nodes {
			// add node
			D.AddOp(dna.OpAddNode, 0)
			// set activation
			actName := node.ActivationName
			D.AddOp(dna.OpSetActivation, float64(dna.ActivationNum[actName]))
			// set bias
			D.AddOp(dna.OpSetBias, rand.Float64()*2-1)
			for j := 0; j < inputCount; j++ {
				// add weight
				D.AddOp(dna.OpAddWeight, rand.Float64()*2-1)
			}
		}
		inputCount = nodeCount
	}
	return
}

// GetName returns the name of the network.
func (n *Network) GetName() string {
	return n.Name
}

// GetNames returns the names of the inputs and outputs.
func (n *Network) GetNames() (inputNames, outputNames []string) {
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

// Zero initializes the weights and biases to zero.
func (n *Network) Zero() {
	for _, layer := range n.Layers {
		for _, node := range layer.Nodes {
			for i := range node.Weights {
				node.Weights[i] = 0
			}
			node.Bias = 0
		}
	}
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
	outputs = n.predict(inputs)
	return
}

func (n *Network) predict(inputs []float64) (outputs []float64) {
	Assert(len(inputs) == len(n.InputNames))
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

// clean readies a layer for the next training run by clearing
// temporary values and ensuring structures are initialized.
func (l *Layer) clean(inputCount int, outputNames []string, upstream *Layer) (err error) {
	defer Return(&err)
	Assert(inputCount > 0)
	l.upstream = upstream
	// ensure inputs slice is initialized
	if l.inputs == nil || len(l.inputs) != inputCount {
		l.inputs = make([]float64, inputCount)
	}
	// if this is an output layer, then outputNames will be populated.
	// make sure we have the right number of nodes
	if len(outputNames) > 0 {
		Assert(len(l.Nodes) == len(outputNames))
	}
	for _, node := range l.Nodes {
		err = node.clean(inputCount, l)
		Ck(err)
	}
	return
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
func newNode(inputCount int, activationName string, layer *Layer) (n *SimpleNode) {
	n = &SimpleNode{ActivationName: activationName}
	err := n.clean(inputCount, layer)
	Ck(err)
	n.randomize()
	return
}

type actFuncs struct {
	activation   func(float64) float64
	activationD1 func(float64) float64
}

var Activations = map[string]actFuncs{
	"sigmoid": actFuncs{sigmoid, sigmoidD1},
	"tanh":    actFuncs{tanh, tanhD1},
	"relu":    actFuncs{relu, reluD1},
	"linear":  actFuncs{linear, linearD1},
	"square":  actFuncs{square, squareD1},
	"sqrt":    actFuncs{sqrt, sqrtD1},
	"abs":     actFuncs{abs, absD1},
}

// activationFuncs returns the activation function and its derivative
// for the given name.
func activationFuncs(name string) (activation, activationD1 func(float64) float64) {
	afs, ok := Activations[name]
	Assert(ok, "unknown activation function: %s", name)
	return afs.activation, afs.activationD1
}

// setActivation sets the activation function and its derivative.
func (n *SimpleNode) setActivation(name string) {
	n.ActivationName = name
	n.activation, n.activationD1 = activationFuncs(name)
}

func (n *SimpleNode) clean(inputCount int, layer *Layer) (err error) {
	defer Return(&err)
	n.cached = false
	if n.ActivationName == "" {
		n.ActivationName = "sigmoid"
	}
	n.setActivation(n.ActivationName)
	n.layer = layer
	if n.Weights == nil || len(n.Weights) != inputCount {
		n.Weights = make([]float64, inputCount)
		n.randomize()
	}
	return
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
		// Assert(len(l.inputs) == 0, Spf("layer: %#v", l))
		for _, upstreamNode := range l.upstream.Nodes {
			inputs = append(inputs, upstreamNode.Output())
		}
	}
	return
}

// Output executes the forward function of a node and returns its
// output value.
func (n *SimpleNode) Output() (output float64) {
	if !n.cached {
		inputs := n.layer.getInputs()
		weightedSum := 0.0
		// add weighted inputs
		Debug("weights: %v\n", n.Weights)
		for i, input := range inputs {
			Assert(!math.IsNaN(input), "input: %v", input)
			Debug("i = %v, input = %v\n", i, input)
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
			// Pf("overflow, randomizing node: weightedSum: %v, inputs: %v, weights: %v, bias: %v", weightedSum, inputs, n.Weights, n.Bias)
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
