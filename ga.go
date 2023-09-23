package tanis

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	. "github.com/stevegt/goadapt"
	"github.com/stevegt/tanis/dna"
)

// TrainingCase is a set of two maps representing the named inputs and
// targets of a training case.
type TrainingCase struct {
	Inputs  map[string]float64
	Targets map[string]float64
}

type TrainingCases []TrainingCase

// TrainingParms contains the parameters for training a network.
type TrainingParms struct {
	Generations    int
	PopulationSize int
	MutationRate   float64
	MaxError       float64
	Verbose        bool
}

// TrainGA trains the network using a genetic algorithm, with
// named inputs and targets in the given set of training cases.  The
// training stops when the mean error of the population is less than
// the given maxError, or when the given number of generations is
// reached.  The mean error is equal to the square root of the mean
// squared error of the population.
func (net *Network) TrainGA(cases *TrainingCases, parms TrainingParms) (bestNet *Network, meanError float64, err error) {
	w := NewWorld(net, cases, parms.PopulationSize, parms.MutationRate)
	for i := 0; i < parms.Generations; i++ {
		Pf("generation %d\n", i)
		bestNet = w.Generation(parms.Verbose)
		meanError = w.Fitness(bestNet)
		if parms.Verbose {
			Pf("best: %s\n", bestNet.ShapeString())
			Pf("\tfitness: %f\n", bestNet.fitness)
			Pf("\tmean error: %f\n", meanError)
			worstNet := w.pop[len(w.pop)-1]
			Pf("worst: %s\n", worstNet.ShapeString())
			Pf("\tfitness: %f\n", worstNet.fitness)
			Pf("\tmean error: %f\n", w.Fitness(worstNet))
		}
		if meanError < parms.MaxError {
			return
		}
	}
	return bestNet, meanError, fmt.Errorf("max generations reached")
}

// World contains the population of individuals along with environmental parameters.
type World struct {
	cases        *TrainingCases
	pop          []*Network
	popSize      int
	mutationRate float64
}

// NewWorld creates a world of individuals from the given network.
func NewWorld(net *Network, cases *TrainingCases, popSize int, mutationRate float64) (w *World) {
	w = &World{
		cases:        cases,
		popSize:      popSize,
		mutationRate: mutationRate,
	}

	// add an exact copy of the original network to the population
	mutated := net.cp()
	err := mutated.clean()
	Ck(err)
	w.pop = append(w.pop, mutated)

	// add progressively mutated copies of the original network to the population
	for i := 0; i < w.popSize-1; i++ {
		mutated, err := NewMutatedNetwork(mutated, mutationRate)
		if err != nil {
			// mutation failed, so just add a copy of the original network
			mutated = net.cp()
		}
		err = mutated.clean()
		if err != nil {
			// individual is not viable, skip it
			continue
		}
		w.pop = append(w.pop, mutated)
	}
	return
}

// recoverNet recovers and dumps the network if a panic occurs.
func recoverNet(net *Network) {
	if r := recover(); r != nil {
		Pprint(net)
		panic(r)
	}
}

// Fitness returns the fitness of the given network.
func (w *World) Fitness(net *Network) (fitness float64) {
	// defer recoverNet(net)
	// check cache
	if net.fitness == 0.0 {
		// calculate mean error
		squaredError := 0.0
		for _, tcase := range *w.cases {
			outputs := net.PredictNamed(tcase.Inputs)
			for name, target := range tcase.Targets {
				output := outputs[name]
				squaredError += math.Pow(target-output, 2.0)
			}
		}
		meanSquaredError := squaredError / float64(len(*w.cases))
		meanError := math.Sqrt(meanSquaredError)
		// add infintesimal to avoid zero
		meanError += math.SmallestNonzeroFloat64
		// calculate fitness
		net.fitness = meanError
	}
	return net.fitness
}

// sort sorts the population by fitness -- lower fitness is good.
func (w *World) sort() {
	sort.Slice(w.pop, func(i, j int) bool {
		return w.Fitness(w.pop[i]) < w.Fitness(w.pop[j])
	})
}

// Generation trains the population for one generation of the genetic algorithm.
func (w *World) Generation(verbose bool) (bestNet *Network) {
	w.cull()
	w.clean()
	w.breed()
	w.clean()
	w.sort()
	bestNet = w.pop[0]
	return
}

// clean resets each net for a new run.
func (w *World) clean() {
	for i := 0; i < len(w.pop); i++ {
		net := w.pop[i]
		// XXX log and summarize errors
		// XXX remove clean() calls from everywhere else
		err := net.clean()
		if err != nil {
			// individual is not viable, remove it
			w.pop = append(w.pop[:i], w.pop[i+1:]...)
			i--
		}
	}
}

// cull removes the worst half of the population. The population must be sorted first.
func (w *World) cull() {
	w.pop = w.pop[:len(w.pop)/2]
}

// breed creates new individuals from the existing population until the population is full.
func (w *World) breed() {
	for len(w.pop) < w.popSize {
		// select two parents from the population
		topten := len(w.pop) / 10
		parent1 := w.pop[rand.Intn(topten)]
		parent2 := w.selectParent()
		// create a child from the parents
		child, err := parent1.Breed(parent2)
		if err != nil {
			// breeding failed, so just add a copy of the original network
			child = parent1.cp()
		}
		// mutate the child
		if true || rand.Float64() < w.mutationRate {
			newChild, err := NewMutatedNetwork(child, w.mutationRate)
			if err == nil {
				child = newChild
			}
		}
		// add the child to the population
		w.pop = append(w.pop, child)
	}
}

// NewMutatedNetwork mutates the network by the given rate, returning a new network.
func NewMutatedNetwork(net *Network, rate float64) (newNet *Network, err error) {
	defer Return(&err)
	// get dna
	D := net.DNA()
	layersDNA := D.StatementsAsBytes()
	// randomize bytes
	for i := range layersDNA {
		if rand.Float64() < rate {
			layersDNA[i] = byte(rand.Intn(256))
		}
	}
	// append random junk
	for i := 0; i < rand.Intn(100); i++ {
		layersDNA = append(layersDNA, byte(rand.Intn(256)))
	}
	D.StatementsFromBytes(layersDNA)
	// create a new network from the dna
	newNet, err = NetworkFromDNA(D)
	Ck(err)
	return
}

// Breed creates a child from the given parents.
func (parent1 *Network) Breed(parent2 *Network) (child *Network, err error) {
	defer Return(&err)
	// get dna
	parent1Layers := parent1.DNA().StatementsAsBytes()
	parent2Layers := parent2.DNA().StatementsAsBytes()
	// pick crossover points
	if len(parent1Layers) == 0 || len(parent2Layers) == 0 {
		err = fmt.Errorf("cannot breed empty network")
		return
	}
	cross1 := rand.Intn(len(parent1Layers))
	cross2 := rand.Intn(len(parent2Layers))
	// round to nearest statement
	cross1 = (cross1 / dna.StatementSize) * dna.StatementSize
	cross2 = (cross2 / dna.StatementSize) * dna.StatementSize
	// combine dna
	childLayers := append(parent1Layers[:cross1], parent2Layers[cross2:]...)
	// create child
	childDNA := &dna.DNA{
		Name:        parent1.Name,
		InputNames:  append([]string{}, parent1.InputNames...),
		OutputNames: append([]string{}, parent2.OutputNames...),
	}
	childDNA.StatementsFromBytes(childLayers)
	child, err = NetworkFromDNA(childDNA)
	Ck(err)
	return
}

// mutateWeights mutates the weights of the network by the given rate.
func (net *Network) mutateWeights(rate float64) {
	for _, layer := range net.Layers {
		for _, node := range layer.Nodes {
			for i := range node.Weights {
				if rand.Float64() < rate {
					node.Weights[i] = rand.Float64()
				}
			}
		}
	}
}

// mutateBiases mutates the biases of the network by the given rate.
func (net *Network) mutateBiases(rate float64) {
	for _, layer := range net.Layers {
		for _, node := range layer.Nodes {
			if rand.Float64() < rate {
				node.Bias = rand.Float64()
			}
		}
	}
}

// mutateActivation mutates the activation functions of the network by the given rate.
func (net *Network) mutateActivation(rate float64) {
	for _, layer := range net.Layers {
		for _, node := range layer.Nodes {
			if rand.Float64() < rate {
				name := randActivationName()
				node.setActivation(name)
			}
		}
	}
}

// randActivationName returns the name of a random activation function.
func randActivationName() string {
	k := rand.Intn(len(Activations))
	for name, _ := range Activations {
		if k == 0 {
			return name
		}
		k--
	}
	panic("unreachable")
}

// cp returns a new network with the same name, structure, and
// weights as this network.
func (src *Network) cp() (dst *Network) {
	dna := src.DNA()
	dst, err := NetworkFromDNA(dna)
	Ck(err)
	return
}

// dumpBiases prints the biases of the first node in each layer.
func dumpBiases(msg string, layers []*Layer) {
	Pf("%s: ", msg)
	for _, layer := range layers {
		Pf("%.2f ", layer.Nodes[0].Bias)
	}
	Pf("\n")
}

// selectParent selects a parent from the population.
func (w *World) selectParent() (parent *Network) {
	// select a random individual from the population
	index := rand.Intn(len(w.pop))
	parent = w.pop[index]
	return
}
