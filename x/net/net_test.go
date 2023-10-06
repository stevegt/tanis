package net

import (
	"testing"

	. "github.com/stevegt/goadapt"
)

// TestRandbytes tests the Randbytes function.
func TestRandbytes(t *testing.T) {
	n := 10
	buf := Randbytes(n)
	if len(buf) != n {
		t.Errorf("Randbytes returned a slice of length %d, expected %d", len(buf), n)
	}
}

func TestRandFloats(t *testing.T) {
	size := 100000
	buf := Randbytes(size)

	floats, n := RandFloats(buf)
	Pf("floats: %v\n", floats)
	Pf("n: %v\n", n)
}
