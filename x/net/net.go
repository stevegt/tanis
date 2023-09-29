package main

import (
	// . "github.com/stevegt/goadapt"

	"github.com/stevegt/tanis/x/node"
)

// Neuron is a single neuron in a neural network.
type Neuron struct {
	node.OldNode
}

// Net is a layerless neural network that supports genetic
// algorithm-based training.
type Net struct {
}
