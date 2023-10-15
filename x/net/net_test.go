package net

import (
	"math"
	"math/rand"
	"testing"

	. "github.com/stevegt/goadapt"
)

// TestRandbytes tests the Randbytes function.
func TestRandbytes(t *testing.T) {
	rand.Seed(1)
	n := 100000
	buf := Randbytes(n)
	if len(buf) != n {
		t.Errorf("Randbytes returned a slice of length %d, expected %d", len(buf), n)
	}
}

func TestRandInts(t *testing.T) {
	rand.Seed(1)
	size := 100000
	buf := Randbytes(size)

	ints, n := RandInts(buf)
	Pf("ints: %v\n", ints)
	Pf("n: %v\n", n)
}

func TestRandFloats(t *testing.T) {
	rand.Seed(43)
	size := 100000
	buf := Randbytes(size)

	floats, n := RandFloats(buf)
	Pf("floats: %v\n", floats)
	Pf("n: %v\n", n)

	m1 := math.Mod(floats[0], 3)
	Pf("m1: %v\n", m1)
}

func TestRandInstruction(t *testing.T) {
	rand.Seed(1)
	size := 100000
	buf := Randbytes(size)
	instruction, n := RandInstruction(buf)
	Pf("instruction: %v\n", instruction)
	Pf("n: %v\n", n)
}
