package tanis

// Many of the following test cases were adapted from
// https://github.com/zserge/nanonn/blob/master/go/nn_test.go,
// licensed under the Apache 2.0 license by Serge Zaitsev.  The nanonn
// project is a good Rosetta Stone, with readable implementations of
// neural networks in several languages.

import (
	"encoding/csv"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"

	. "github.com/stevegt/goadapt"
)

func init() {
	rand.Seed(1)
}

func TestSigmoid(t *testing.T) {
	if y := sigmoid(0); y != 0.5 {
		t.Error(y)
	}
	if y := sigmoid(2); math.Abs(y-0.88079708) > 0.0001 {
		t.Error(y)
	}
}

// Dense returns a new dense fully-connected layer with sigmoid
// activation function and the given number of inputs and nodes.
func Dense(nodes, inputs int) (layer *Layer) {
	layer = &Layer{}
	for i := 0; i < nodes; i++ {
		node := newNode("sigmoid")
		layer.Nodes = append(layer.Nodes, node)
	}
	layer.init(inputs, nil)
	layer.randomize()
	return
}

// Forward takes a vector of inputs, creates a network
// with the given layer, runs the inputs through the network
// and returns the outputs.
func (l *Layer) Forward(x []float64) (z []float64) {
	n := &Network{InputCount: len(x), Layers: []*Layer{l}}
	// Pprint(n)
	n.init()
	// Pprint(n)
	// os.Exit(44)
	z = n.Predict(x)
	return
}

// New returns a new sequential network constructed from the given layers. An
// error is returned if the number of inputs and outputs in two adjacent layers
// is not the same.
func New(layers ...*Layer) (*Network, error) {
	Assert(len(layers) > 0, "no layers")
	Assert(len(layers[0].inputs) > 0, "no inputs")
	for i := 1; i < len(layers); i++ {
		Assert(len(layers[i].getInputs()) == len(layers[i-1].Nodes), "layer mismatch")
	}
	n := &Network{InputCount: len(layers[0].inputs), Layers: layers}
	n.init()
	// Pprint(n)
	// Pf("l2: %#v\n", layers[1])
	// os.Exit(55)
	return n, nil
}

func TestForward(t *testing.T) {
	l := Dense(1, 3)
	l.setWeights([][]float64{{1.74481176, -0.7612069, 0.3190391}})
	l.setBiases([]float64{-0.24937038})
	x1 := []float64{1.62434536, -0.52817175, 0.86540763}
	y1 := 0.96313579
	z1 := l.Forward(x1)
	if math.Abs(z1[0]-y1) > 0.001 {
		t.Error(z1, y1)
	}
	x2 := []float64{-0.61175641, -1.07296862, -2.3015387}
	y2 := 0.22542973
	z2 := l.Forward(x2)
	if math.Abs(z2[0]-y2) > 0.001 {
		t.Error(z2, y2)
	}
}

// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
func TestWeights(t *testing.T) {
	l1 := Dense(2, 2)
	l2 := Dense(2, 2)
	n, _ := New(l1, l2)

	// Initialise weights
	l1.setWeights([][]float64{{0.15, 0.2}, {0.25, 0.3}})
	l1.setBiases([]float64{0.35, 0.35})
	// Pprint(l1)
	// os.Exit(87)
	l2.setWeights([][]float64{{0.4, 0.45}, {0.5, 0.55}})
	l2.setBiases([]float64{0.6, 0.6})
	// Pprint(l2)
	// os.Exit(91)

	// Ensure forward propagation works for both layers
	// Pprint(n)
	// Pprint(n.Layers[1].upstream)
	// Pf("l2: %#v\n", l2)
	z := n.Predict([]float64{0.05, 0.1})
	// Pprint(n)
	// os.Exit(99)
	if e := math.Abs(z[0] - 0.75136507); e > 0.0001 {
		t.Error(e, z)
	}
	if e := math.Abs(z[1] - 0.772928465); e > 0.0001 {
		t.Error(e, z)
	}

	// Ensure that squared error is calculated correctly (use rate=0 to avoid training)
	case1 := NewTrainingCase([]float64{0.05, 0.1}, []float64{0.01, 0.99})
	e := n.trainOne(case1, 0)
	// Pf("error: %v\n", e)
	if math.Abs(e-0.298371109) > 0.0001 {
		t.Log(e)
	}

	// Backpropagtion with rate 0.5
	case2 := NewTrainingCase([]float64{0.05, 0.1}, []float64{0.01, 0.99})
	e = n.trainOne(case2, 0.5)
	// Pf("error: %v\n", e)
	// os.Exit(125)

	// check l2 weights and biases
	// for i, w := range []float64{0.35891648, 0.408666186, 0.530751, 0.511301270, 0.561370121, 0.619049} {
	for nodenum, wb := range [][]float64{{0.35891648, 0.408666186, 0.530751}, {0.511301270, 0.561370121, 0.619049}} {
		weights := wb[:2]
		bias := wb[2]
		for i, w := range weights {
			e := math.Abs(w - l2.Nodes[nodenum].Weights[i])
			if e > 0.001 {
				t.Error(nodenum, i, w, l2.Nodes[nodenum].Weights[i], e)
			}
		}
		e := math.Abs(bias - l2.Nodes[nodenum].Bias)
		if e > 0.001 {
			t.Error(nodenum, bias, l2.Nodes[nodenum].Bias, e)
		}
	}

	// check l1 weights
	// for i, w := range []float64{0.149780716, 0.19956143, 0.345614, 0.24975114, 0.29950229, 0.345614} {
	for nodenum, wb := range [][]float64{{0.149780716, 0.19956143, 0.345614}, {0.24975114, 0.29950229, 0.345614}} {
		weights := wb[:2]
		bias := wb[2]
		for i, w := range weights {
			e := math.Abs(w - l1.Nodes[nodenum].Weights[i])
			if e > 0.001 {
				t.Error(nodenum, i, w, e, l1.Nodes[nodenum].Weights[i])
			}
		}
		e := math.Abs(bias - l1.Nodes[nodenum].Bias)
		if e > 0.001 {
			t.Error(nodenum, bias, l1.Nodes[nodenum].Bias, e)
		}
	}

}

// Backward calls Backprop for layer l, passing vectors for inputs and
// errors and a learning rate.
func (l *Layer) Backward(inputs, errors []float64, rate float64) {
	l.setInputs(inputs)
	l.backprop(errors, rate)
}

// Use single unit layer to predict OR function
func TestLayerOr(t *testing.T) {
	x := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := [][]float64{{0}, {1}, {1}, {1}}
	e := make([]float64, 1, 1)
	// A layer of a single unit with two x and one output
	l := Dense(1, 2)
	// Train layer for several epochs
	for epoch := 0; epoch < 1000; epoch++ {
		for i, x := range x {
			z := l.Forward(x)
			e[0] = y[i][0] - z[0]
			l.Backward(x, e, 1)
		}
	}
	// Predict the outputs, expecting only a small error
	for i, x := range x {
		z := l.Forward(x)
		if math.Abs(z[0]-y[i][0]) > 0.1 {
			t.Error(x, z, y[i])
		}
	}
}

// Use hidden layer to predict XOR function
func TestNetworkXor(t *testing.T) {
	x := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := [][]float64{{0}, {1}, {1}, {0}}
	// n, _ := New(Dense(4, 2), Dense(1, 4))
	n := NewNetwork("foo", 2, 4, 1)
	// create training set
	set := NewTrainingSet()
	for i, _ := range x {
		set.Add(x[i], y[i])
	}
	// Train at a learning rate of 1 for 10000 iterations or until cost is less than 2%
	cost, err := n.Train(set, 1, 10000, 0.02)
	Tassert(t, err == nil, "cost too high: %v", cost)

	// Pl(n.Save())
}

// Use multiple hidden layers to predict sinc(x) function.
func TestNetworkSinc(t *testing.T) {
	sinc := func(x float64) float64 {
		if x == 0 {
			return 1
		}
		return math.Sin(x) / x
	}
	n, _ := New(Dense(5, 1), Dense(10, 5), Dense(1, 10))
	var e float64
	for i := 0; i < 1000; i++ {
		e = 0.0
		for j := 0; j < 100; j++ {
			x := rand.Float64()*10 - 5
			case1 := NewTrainingCase([]float64{x}, []float64{sinc(x)})
			e = e + n.trainOne(case1, 0.5)/100
		}
		if e < 0.01 {
			return
		}
	}
	t.Error("failed to train", e)
}

// Train and test on Iris dataset
func TestIris(t *testing.T) {
	x, y := loadCSV("../testdata/iris.csv")
	n, _ := New(Dense(10, 4), Dense(3, 10))
	k := len(x) * 9 / 10 // use 90% for training, 10% for testing
	// replace Y with a 3-item vector for classification
	for i := range y {
		n := y[i][0]
		y[i] = []float64{0, 0, 0}
		y[i][int(n)] = 1
		x[i] = x[i][1:]
	}
	rand.Shuffle(len(x), func(i, j int) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	})
	maxind := func(x []float64) int {
		m := -1
		for i := range x {
			if m < 0 || x[i] > x[m] {
				m = i
			}
		}
		return m
	}

	e := 0.0
	for epoch := 0; epoch < 10000; epoch++ {
		e = 0.0
		for i := 0; i < k; i++ {
			case1 := NewTrainingCase(x[i], y[i])
			e = e + n.trainOne(case1, 0.4)/float64(k)
		}
		if e < 0.01 {
			// Classify all data and print failures
			for i := 0; i < len(x); i++ {
				z := n.Predict(x[i])
				if maxind(z) != maxind(y[i]) {
					t.Log(x[i], y[i], z)
				}
			}
			return
		}
	}
	t.Error("failed to train", e)
}

func trainingBench(b *testing.B) {
	width := 200
	n, _ := New(Dense(width, 1), Dense(width, width), Dense(width, width), Dense(1, width))
	x := []float64{0}
	y := []float64{0}
	for i := 0; i < b.N; i++ {
		x[0] = rand.Float64()*10 - 5
		y[0] = math.Sin(x[0])
		case1 := NewTrainingCase(x, y)
		n.trainOne(case1, 0.5)
	}
}

func BenchmarkSingleThreaded(b *testing.B) {
	trainingBench(b)
}

/*
func BenchmarkMultiThreaded(b *testing.B) {
	StartPool(3, 10)
	trainingBench(b)
}
*/

func loadCSV(filename string) (x [][]float64, y [][]float64) {
	f, _ := os.Open(filename)
	defer f.Close()
	rows, _ := csv.NewReader(f).ReadAll()
	for _, row := range rows {
		nums := []float64{}
		for _, s := range row {
			n, _ := strconv.ParseFloat(s, 64)
			nums = append(nums, n)
		}
		x = append(x, nums[0:len(nums)-1])
		y = append(y, nums[len(nums)-1:])
	}
	return x, y
}

func TestClone(t *testing.T) {
	// Train an XOR network
	x := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := [][]float64{{0}, {1}, {1}, {0}}
	n, _ := New(Dense(4, 2), Dense(1, 4))
	// Pprint(n)
	// os.Exit(200)
	// Train for several epochs, or until the error is less than 2%
	var e float64
	for epoch := 0; epoch < 10000; epoch++ {
		e = 0.0
		for i := range x {
			case1 := NewTrainingCase(x[i], y[i])
			e = e + n.trainOne(case1, 1)
		}
		if e < 0.02 {
			break
		}
	}
	if e > 0.02 {
		t.Error("failed to train", e)
	}

	// Clone the network
	n2 := n.Clone("foo")

	Tassert(t, n2.Name == "foo", n2.Name)

	// Tassert(t, err == nil, "Load failed", err)
	// make some predictions
	for i := range x {
		z := n2.Predict(x[i])
		if z[0] < 0.5 {
			z[0] = 0
		} else {
			z[0] = 1
		}
		if z[0] != y[i][0] {
			t.Error("failed to predict", x[i], z[0], y[i][0])
		}
	}
}

func TestActivations(t *testing.T) {
	net := NewNetwork("foo", 3, 4, 5, 6)
	net.SetActivation(-1, -1, "tanh")
	net.SetActivation(1, -1, "sigmoid")
	net.SetActivation(2, 1, "relu")

	Tassert(t, net.Layers[1].Nodes[0].ActivationName == "sigmoid", net.Save())
	Tassert(t, net.Layers[2].Nodes[0].ActivationName == "tanh", net.Save())
	Tassert(t, net.Layers[2].Nodes[1].ActivationName == "relu", net.Save())
	Tassert(t, net.Layers[2].Nodes[2].ActivationName == "tanh", net.Save())
}

// Test named inputs and outputs
func TestNamed(t *testing.T) {
	net := NewNetwork("foo", 3, 4, 5, 6)
	net.SetActivation(2, -1, "linear")
	net.SetInputNames("a", "b", "c")
	net.SetOutputNames("x", "y", "z", "w", "v", "u")

	// create some named inputs and targets
	inputs := map[string]float64{"a": 1, "b": 2, "c": 3}
	targets := map[string]float64{"x": 1, "y": 2, "z": 3, "w": 4, "v": 5, "u": 6}

	// train the network
	for i := 0; i < 1000; i++ {
		net.LearnNamed(inputs, targets, 0.1)
	}

	// make some predictions
	outputs := net.PredictNamed(inputs)
	for k, v := range outputs {
		Tassert(t, math.Abs(v-targets[k]) < 0.1, k, v, targets[k])
	}

}
