package tanis

// Many of the following test cases were adapted from
// https://github.com/zserge/nanonn/blob/master/go/nn_test.go,
// licensed under the Apache 2.0 license by Serge Zaitsev.  The nanonn
// project is a good Rosetta Stone, with readable implementations of
// neural networks in several languages.

import (
	"math"
	"math/rand"
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

func TestActivations(t *testing.T) {
	shape := "(foo a b c (tanh 4) (sigmoid 5) (+ (tanh x y z w v) (relu u)))"
	net, err := NewNetwork(shape)
	Tassert(t, err == nil, err)
	Tassert(t, net.Layers[1].Nodes[0].ActivationName == "sigmoid", net.Save())
	Tassert(t, net.Layers[2].Nodes[0].ActivationName == "tanh", net.Save())
	Tassert(t, net.Layers[2].Nodes[4].ActivationName == "tanh", net.Save())
	Tassert(t, net.Layers[2].Nodes[5].ActivationName == "relu", net.Save())
}

// Test named inputs and outputs
func TestNamed(t *testing.T) {
	// ape := "foo((a b c) (sigmoid(4)) (sigmoid(5)) (linear(6)) (x y z w v u))"
	shape := "(foo a b c (sigmoid 4) (sigmoid 5) (linear x y z w v u))"
	net, err := NewNetwork(shape)
	Tassert(t, err == nil, err)

	// check that the names are set
	Tassert(t, net.InputNames[0] == "a", net.InputNames)
	Tassert(t, net.InputNames[1] == "b", net.InputNames)
	Tassert(t, net.InputNames[2] == "c", net.InputNames)
	Tassert(t, net.OutputNames[0] == "x", net.OutputNames)
	Tassert(t, net.OutputNames[1] == "y", net.OutputNames)
	Tassert(t, net.OutputNames[2] == "z", net.OutputNames)
	Tassert(t, net.OutputNames[3] == "w", net.OutputNames)
	Tassert(t, net.OutputNames[4] == "v", net.OutputNames)
	Tassert(t, net.OutputNames[5] == "u", net.OutputNames)

	// check that all the layers are hooked up
	Tassert(t, len(net.Layers) == 3, net.Layers)
	Tassert(t, net.Layers[0].upstream == nil, net.Layers[0].upstream)
	Tassert(t, net.Layers[1].upstream == net.Layers[0], net.Layers[1].upstream)
	Tassert(t, net.Layers[2].upstream == net.Layers[1], net.Layers[2].upstream)

	// check the node weights
	Tassert(t, len(net.Layers[0].Nodes) == 4, net.Layers[0].Nodes)
	Tassert(t, len(net.Layers[1].Nodes) == 5, net.Layers[1].Nodes)
	Tassert(t, len(net.Layers[2].Nodes) == 6, net.Layers[2].Nodes)
	for _, node := range net.Layers[0].Nodes {
		Tassert(t, len(node.Weights) == 3, node.Weights)
	}
	for _, node := range net.Layers[1].Nodes {
		Tassert(t, len(node.Weights) == 4, node.Weights)
	}
	for _, node := range net.Layers[2].Nodes {
		Tassert(t, len(node.Weights) == 5, node.Weights)
	}

}

func TestPredictNamed(t *testing.T) {
	// linear activation with all weights = 1 and all biases = 0
	shape := "(foo in (linear 1) (linear out))"
	net, err := NewNetwork(shape)
	Tassert(t, err == nil, err)
	for _, layer := range net.Layers {
		for _, node := range layer.Nodes {
			// set all weights to 1
			for i, _ := range node.Weights {
				node.Weights[i] = 1
			}
			// set all biases to 0
			node.Bias = 0
		}
	}
	inputs := map[string]float64{"in": 1}
	outputs := net.PredictNamed(inputs)
	Tassert(t, outputs["out"] == 1, outputs)

	// same but with more layers and nodes
	shape = "(foo a b c (linear 4 ) (linear 1 ) (linear x y z w v u))"
	net, err = NewNetwork(shape)
	for _, layer := range net.Layers {
		for _, node := range layer.Nodes {
			// set all weights to 1
			for i, _ := range node.Weights {
				node.Weights[i] = 1
			}
			// set all biases to 0
			node.Bias = 0
		}
	}
	inputs = map[string]float64{"a": 1, "b": 1, "c": 1}
	outputs = net.PredictNamed(inputs)
	Tassert(t, outputs["y"] == 12, outputs)
}

func TestCopy(t *testing.T) {
	// net := NewNetwork("foo", 3, 4, 5, 2)
	// 	net.SetActivation(2, -1, "linear")
	// 	net.SetInputNames("a", "b", "c")
	// 	net.SetOutputNames("x", "y")
	shape := "(foo a b c (sigmoid 3) (sigmoid 4) (linear 5) (linear x y))"
	net, err := NewNetwork(shape)
	Tassert(t, err == nil, err)
	net2 := net.cp()
	Tassert(t, net2.Name == net.Name, net2.Name)

	for i, _ := range net.InputNames {
		Tassert(t, net2.InputNames[i] == net.InputNames[i], net2.InputNames[i])
	}
	for i, _ := range net.OutputNames {
		Tassert(t, net2.OutputNames[i] == net.OutputNames[i], net2.OutputNames[i])
	}

	Tassert(t, len(net2.Layers) == len(net.Layers), net2.Layers)
	for i, _ := range net.Layers {
		Tassert(t, len(net2.Layers[i].Nodes) == len(net.Layers[i].Nodes), net2.Layers[i].Nodes)
		if i == 0 {
			Tassert(t, net2.Layers[i].upstream == nil, net2.Layers[i].upstream)
		} else {
			Tassert(t, net2.Layers[i].upstream == net2.Layers[i-1], net2.Layers[i].upstream)
		}
		for j, _ := range net.Layers[i].Nodes {
			Tassert(t, len(net2.Layers[i].Nodes[j].Weights) == len(net.Layers[i].Nodes[j].Weights), net2.Layers[i].Nodes[j].Weights)
			Tassert(t, net2.Layers[i].Nodes[j].Bias == net.Layers[i].Nodes[j].Bias, net2.Layers[i].Nodes[j].Bias)
			Tassert(t, net2.Layers[i].Nodes[j].ActivationName == net.Layers[i].Nodes[j].ActivationName, net2.Layers[i].Nodes[j].ActivationName)
			Tassert(t, net2.Layers[i].Nodes[j].activation != nil)
			Tassert(t, net2.Layers[i].Nodes[j].activationD1 != nil)
			for k, _ := range net.Layers[i].Nodes[j].Weights {
				Tassert(t, net2.Layers[i].Nodes[j].Weights[k] == net.Layers[i].Nodes[j].Weights[k], net2.Layers[i].Nodes[j].Weights[k])
			}
		}
	}
}

func TestMutations(t *testing.T) {
	rand.Seed(1)
	// net := NewNetwork("foo", 2, 4, 1)
	// net.SetActivation(-1, -1, "linear")
	// net.SetInputNames("a", "b")
	// net.SetOutputNames("y")
	shape := "(foo a b (linear 2) (linear 4) (linear y))"
	net, err := NewNetwork(shape)
	Tassert(t, err == nil, err)
	for i, layer := range net.Layers {
		for j, node := range layer.Nodes {
			// set all weights to layer*100 + node*10 + weight
			for k, _ := range node.Weights {
				node.Weights[k] = float64(i*100) + float64(j*10) + float64(k)
			}
			// set all biases to layer*100 + node*10
			node.Bias = float64(i*100) + float64(j*10)
		}
	}

	net2, err := NewMutatedNetwork(net, 0.1)
	Tassert(t, err == nil, err)
	Tassert(t, net2 != nil, net2)
	Tassert(t, net2.Name == net.Name, net2.Name)
	Tassert(t, len(net2.InputNames) == len(net.InputNames), net2.InputNames)
	Tassert(t, len(net2.OutputNames) == len(net.OutputNames), net2.OutputNames)
	for i, _ := range net.InputNames {
		Tassert(t, net2.InputNames[i] == net.InputNames[i], net2.InputNames[i])
	}
	for i, _ := range net.OutputNames {
		Tassert(t, net2.OutputNames[i] == net.OutputNames[i], net2.OutputNames[i])
	}

	net1string := net.DNA().String()
	net2string := net2.DNA().String()
	Tassert(t, net1string != net2string, net1string, net2string)
}

func TestMutationsStress(t *testing.T) {
	rand.Seed(1)
	shape := "(foo a b (linear 2) (linear 4) (linear x y))"
	net, err := NewNetwork(shape)
	Tassert(t, err == nil, err)
	cases := &TrainingCases{
		TrainingCase{
			Inputs:  map[string]float64{"a": 1, "b": 1, "c": 1},
			Targets: map[string]float64{"x": 4, "y": 12},
		},
	}
	ok := 0
	for i := 0; i < 1000; i++ {
		mutated, err := NewMutatedNetwork(net, 0.1)
		if err != nil {
			continue
		}
		err = mutated.clean()
		if err != nil {
			continue
		}
		Tassert(t, mutated != nil, mutated)
		Tassert(t, mutated.Name == net.Name, mutated.Name)
		Tassert(t, len(mutated.InputNames) == len(net.InputNames), mutated.InputNames)
		Tassert(t, len(mutated.OutputNames) == len(net.OutputNames), mutated.OutputNames)
		for i, _ := range net.InputNames {
			Tassert(t, mutated.InputNames[i] == net.InputNames[i], mutated.InputNames[i])
		}
		for i, _ := range net.OutputNames {
			Tassert(t, mutated.OutputNames[i] == net.OutputNames[i], mutated.OutputNames[i])
		}
		Pf("%d mutated: %v\n", i, mutated.ShapeString())
		verifyNetReadiness(t, mutated)
		_ = mutated.PredictNamed((*cases)[0].Inputs)
		w := NewWorld(mutated, cases, 4, 0.1)
		mutated.fitness = 0
		_ = w.Fitness(mutated)
		ok++
	}
	Tassert(t, ok == 710, ok)
}

func TestBreeding(t *testing.T) {
	rand.Seed(1)
	shape := "(foo a b (linear 2) (linear 4) (linear x y))"
	net, err := NewNetwork(shape)
	Tassert(t, err == nil, err)
	cases := &TrainingCases{
		TrainingCase{
			Inputs:  map[string]float64{"a": 1, "b": 1, "c": 1},
			Targets: map[string]float64{"x": 4, "y": 12},
		},
	}
	ok := 0
	for i := 0; i < 1000; i++ {
		parent1, err := NewMutatedNetwork(net, 0.1)
		if err != nil {
			continue
		}
		parent2, err := NewMutatedNetwork(net, 0.1)
		if err != nil {
			continue
		}
		child, err := parent1.Breed(parent2)
		if err != nil {
			continue
		}
		err = child.clean()
		if err != nil {
			continue
		}
		Pf("%d parent1: %v\n", i, parent1.ShapeString())
		Pf("%d parent2: %v\n", i, parent2.ShapeString())
		Pf("%d   child: %v\n", i, child.ShapeString())
		verifyNetReadiness(t, child)
		_ = child.PredictNamed((*cases)[0].Inputs)
		w := NewWorld(child, cases, 4, 0.1)
		child.fitness = 0
		_ = w.Fitness(child)
		ok++
	}
	Tassert(t, ok == 435, ok)
}

// verifyWorldReadiness checks that the population is ready for training
func verifyWorldReadiness(t *testing.T, w *World) {
	pop := w.pop
	for _, net := range pop {
		verifyNetReadiness(t, net)
	}
}

// verifyNetReadiness checks that the network is ready for training
func verifyNetReadiness(t *testing.T, net *Network) {
	Tassert(t, net.fitness == 0, net.fitness)
	for i := 0; i < len(net.Layers); i++ {
		layer := net.Layers[i]
		// check inputs and upstream
		var inputCount int
		if i == 0 {
			inputCount = len(net.InputNames)
			Tassert(t, layer.upstream == nil, layer.upstream)
		} else {
			inputCount = len(net.Layers[i-1].Nodes)
			Tassert(t, layer.upstream == net.Layers[i-1], layer.upstream)
		}
		// check outputs
		if i == len(net.Layers)-1 {
			Tassert(t, len(layer.Nodes) == len(net.OutputNames), len(layer.Nodes))
		}
		Tassert(t, len(layer.inputs) == inputCount, len(layer.inputs))
		// check nodes
		for j := 0; j < len(layer.Nodes); j++ {
			node := layer.Nodes[j]
			Tassert(t, node.cached == false, node.cached)
			// check weights
			Tassert(t, len(node.Weights) == inputCount, "len(node.Weights)=%v, inputCount=%v", len(node.Weights), inputCount)
			// check layer
			Tassert(t, node.layer == layer, node.layer)
			// test activation
			activation, activationD1 := activationFuncs(node.ActivationName)
			expect := activation(0.5)
			got := node.activation(0.5)
			Tassert(t, got == expect, "expect %v, got %v", expect, got)
			expect = activationD1(0.5)
			got = node.activationD1(0.5)
			Tassert(t, got == expect, "expect %v, got %v", expect, got)
		}
	}
}

func TestWorld(t *testing.T) {
	rand.Seed(1)
	// net := NewNetwork("foo", 3, 4, 1)
	// net.SetActivation(-1, -1, "linear")
	// net.SetInputNames("a", "b", "c")
	// net.SetOutputNames("y")
	txt := "(foo a b c (linear 4) (linear y))"
	net, err := NewNetwork(txt)
	Tassert(t, err == nil, err)
	err = net.clean()
	Ck(err)
	// set all weights to 1 and all biases to 0
	for _, layer := range net.Layers {
		for _, node := range layer.Nodes {
			// set all weights to 1
			for i, _ := range node.Weights {
				node.Weights[i] = 1
			}
			// set all biases to 0
			node.Bias = 0
		}
	}

	cases := &TrainingCases{
		TrainingCase{
			Inputs:  map[string]float64{"a": 1, "b": 1, "c": 1},
			Targets: map[string]float64{"y": 12},
		},
	}
	outputs := net.PredictNamed((*cases)[0].Inputs)
	Tassert(t, outputs["y"] == 12, outputs)

	w := NewWorld(net, cases, 100, 0.1)

	top := w.pop[0]
	outputs = top.PredictNamed((*cases)[0].Inputs)
	Tassert(t, outputs["y"] == 12, outputs)

	bottom := w.pop[len(w.pop)-1]
	outputs = bottom.PredictNamed((*cases)[0].Inputs)
	Pl("bottom outputs", outputs)

	topFitness := w.Fitness(top)
	Tassert(t, math.Abs(topFitness) < 0.0001, topFitness)

	bottomFitness := w.Fitness(bottom)
	Tassert(t, math.Abs(bottomFitness)-12.785167418344809 < 0.0001, bottomFitness)

	for i := 0; i < len(w.pop); i++ {
		fitness := w.Fitness(w.pop[i])
		Pl("fitness", i, fitness, w.pop[i].ShapeString())
	}

	w.sort()
	w.cull()

	bottom = w.pop[len(w.pop)-1]
	bottomFitness = w.Fitness(bottom)
	Tassert(t, math.Abs(bottomFitness)-132 < 0.0001, bottomFitness)

	Pl("top", top.ShapeString())
	Pl("bottom", bottom.ShapeString())
	child, err := top.Breed(bottom)
	Tassert(t, err == nil, err)
	err = child.clean()
	Tassert(t, err == nil, err)
	Pl("child", child.ShapeString())

	// try running predict on each member of the population
	for i := 0; i < len(w.pop); i++ {
		ind := w.pop[i]
		_ = ind.PredictNamed((*cases)[0].Inputs)
	}

	// try running Fitness on each member of the population
	for i := 0; i < len(w.pop); i++ {
		ind := w.pop[i]
		ind.fitness = 0
		_ = w.Fitness(ind)
	}

	// try running Generation bits
	w.cull()
	w.clean()
	Pl("after resetRun")
	verifyWorldReadiness(t, w)
	w.breed()
	Pl("after breed")
	verifyWorldReadiness(t, w)
	w.sort()

	w.Generation(false)

}

// Test training
func TestTrainGA(t *testing.T) {
	rand.Seed(1)
	// net := NewNetwork("foo", 3, 4, 1)
	// net.SetActivation(-1, -1, "linear")
	// net.SetInputNames("a", "b", "c")
	// net.SetOutputNames("y")
	txt := "(foo a b c (linear 4) (linear y))"
	net, err := NewNetwork(txt)
	Tassert(t, err == nil, err)

	cases := &TrainingCases{
		TrainingCase{
			Inputs:  map[string]float64{"a": 1, "b": 1, "c": 1},
			Targets: map[string]float64{"y": 12},
		},
	}

	// train the network
	parms := TrainingParms{
		Generations:    10000,
		PopulationSize: 100,
		MutationRate:   0.1,
		MaxError:       0.1,
		Verbose:        false,
	}
	bestNet, meanError, err := net.TrainGA(cases, parms)
	Tassert(t, err == nil, "meanError: %v, err: %v", meanError, err)
	Tassert(t, bestNet != nil, "bestNet is nil")
	Tassert(t, meanError < 0.1, "mean error too high: %v", meanError)

	// make some predictions
	outputs := bestNet.PredictNamed((*cases)[0].Inputs)
	Tassert(t, outputs["y"]-12 < 0.1, outputs)

}

// predict OR function with no hidden layers
func TestOr(t *testing.T) {
	// net := NewNetwork("foo", 2, 2, 1)
	// net.SetActivation(-1, -1, "sigmoid")
	// net.SetInputNames("a", "b")
	// net.SetOutputNames("y")
	txt := "(foo a b (sigmoid 2) (sigmoid y))"
	net, err := NewNetwork(txt)
	Tassert(t, err == nil, err)
	Tassert(t, len(net.Layers) == 2, len(net.Layers))

	cases := &TrainingCases{
		TrainingCase{Inputs: map[string]float64{"a": 0, "b": 0}, Targets: map[string]float64{"y": 0}},
		TrainingCase{Inputs: map[string]float64{"a": 0, "b": 1}, Targets: map[string]float64{"y": 1}},
		TrainingCase{Inputs: map[string]float64{"a": 1, "b": 0}, Targets: map[string]float64{"y": 1}},
		TrainingCase{Inputs: map[string]float64{"a": 1, "b": 1}, Targets: map[string]float64{"y": 1}},
	}

	// train the network
	maxError := 0.2
	parms := TrainingParms{
		Generations:    10000,
		PopulationSize: 100,
		MutationRate:   0.1,
		MaxError:       maxError,
	}
	rand.Seed(1)
	bestNet, meanError, err := net.TrainGA(cases, parms)
	Tassert(t, err == nil, "meanError: %v, err: %v", meanError, err)
	Tassert(t, bestNet != nil, "bestNet is nil")
	Tassert(t, meanError < maxError, "mean error too high: %v", meanError)

	// test the network
	for _, tcase := range *cases {
		outputs := bestNet.PredictNamed(tcase.Inputs)
		Tassert(t, outputs["y"]-tcase.Targets["y"] < maxError*2, "inputs: %v, targets: %v, outputs: %v", tcase.Inputs, tcase.Targets, outputs)
	}

}

// Use hidden layer to predict XOR function
func TestXor(t *testing.T) {
	rand.Seed(1)
	txt := "(foo a b (sigmoid 2) (sigmoid 2) (sigmoid y))"
	net, err := NewNetwork(txt)
	Tassert(t, err == nil, err)
	Tassert(t, len(net.Layers) == 3, len(net.Layers))

	cases := &TrainingCases{
		TrainingCase{Inputs: map[string]float64{"a": 0, "b": 0}, Targets: map[string]float64{"y": 0}},
		TrainingCase{Inputs: map[string]float64{"a": 0, "b": 1}, Targets: map[string]float64{"y": 1}},
		TrainingCase{Inputs: map[string]float64{"a": 1, "b": 0}, Targets: map[string]float64{"y": 1}},
		TrainingCase{Inputs: map[string]float64{"a": 1, "b": 1}, Targets: map[string]float64{"y": 0}},
	}

	// train the network
	maxError := 0.1
	parms := TrainingParms{
		Generations:    10000,
		PopulationSize: 100,
		MutationRate:   0.1,
		MaxError:       maxError,
	}
	bestNet, meanError, err := net.TrainGA(cases, parms)
	Tassert(t, err == nil, "meanError: %v, err: %v", meanError, err)
	Tassert(t, bestNet != nil, "bestNet is nil")
	Tassert(t, meanError < maxError, "mean error too high: %v", meanError)

	// test the network
	for _, tcase := range *cases {
		outputs := bestNet.PredictNamed(tcase.Inputs)
		Tassert(t, outputs["y"]-tcase.Targets["y"] < maxError*2, "inputs: %v, targets: %v, outputs: %v", tcase.Inputs, tcase.Targets, outputs)
	}
}

// Use multiple hidden layers to predict sinc(x) function.
func TestNetworkSinc(t *testing.T) {
	rand.Seed(1)
	sinc := func(x float64) float64 {
		if x == 0 {
			return 1
		}
		return math.Sin(x) / x
	}
	txt := "(foo x (sigmoid 5) (sigmoid 10) (sigmoid 10) (sigmoid y))"
	net, err := NewNetwork(txt)
	Tassert(t, err == nil, err)
	Tassert(t, len(net.Layers) == 4, len(net.Layers))

	// build training cases
	cases := &TrainingCases{}
	for j := 0; j < 100; j++ {
		x := rand.Float64()*10 - 5
		y := sinc(x)
		tcase := TrainingCase{Inputs: map[string]float64{"x": x}, Targets: map[string]float64{"y": y}}
		*cases = append(*cases, tcase)
	}

	// train the network
	maxError := 0.01
	parms := TrainingParms{
		Generations:    10000,
		PopulationSize: 100,
		MutationRate:   0.5,
		MaxError:       maxError,
		Verbose:        true,
	}
	bestNet, meanError, err := net.TrainGA(cases, parms)
	Tassert(t, err == nil, "meanError: %v, err: %v", meanError, err)
	Tassert(t, bestNet != nil, "bestNet is nil")
	Tassert(t, meanError < maxError, "mean error too high: %v", meanError)
}

/*
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
*/

/*
// Test nested networks
func TestNested(t *testing.T) {
	// create a foo network with a bar network inside it
	foo := NewNetwork("foo", 3, 2, 1)
	foo.SetInputNames("a", "b", "c")
	foo.SetOutputNames("y")
	bar := NewNetwork("bar", 1, 2, 1)
	// replace the first node of the first layer of foo with bar
	foo.ReplaceNode(0, 0, bar)

	// create some named inputs and targets
	inputs := map[string]float64{"a": 1, "b": 2, "c": 3}
	targets := map[string]float64{"y": 2}

	// train the network
	for i := 0; i < 1000; i++ {
		foo.LearnNamed(inputs, targets, 0.1)
	}

	// make some predictions
	outputs := foo.PredictNamed(inputs)
	for k, v := range outputs {
		Tassert(t, math.Abs(v-targets[k]) < 0.1, k, v, targets[k])
	}
}
*/
