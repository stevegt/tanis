package net

import (
	"encoding/binary"
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

func TestBytesToInstBuf(t *testing.T) {
	rand.Seed(1)
	size := 100000
	buf := Randbytes(size)

	ints, n := BytesToInstBuf(buf)
	Pf("ints: %v\n", ints)
	Pf("n: %v\n", n)

	// fill the buffer with sequential ints
	buf = []byte{}
	for i := 0; i < size/8; i++ {
		binary.BigEndian.AppendUint64(buf, uint64(i))
	}
	// read the ints back out
	ints, n = BytesToInstBuf(buf)
	for i, v := range ints {
		Tassert(t, v == uint64(i), "v: %v != %v", v, i)
	}

}

func TestRandFloats(t *testing.T) {
	rand.Seed(45)
	size := 100000
	buf := Randbytes(size)

	floats, n := RandFloats(buf)
	Pf("floats: %v\n", floats)
	Pf("n: %v\n", n)

	m1 := math.Mod(floats[0], 3)
	Pf("m1: %v\n", m1)
}

func TestAbsUintMod(t *testing.T) {
	cases := []struct {
		x, y, z uint64
	}{
		{0, 3, 0},
		{1, 3, 1},
		{2, 3, 2},
		{3, 3, 0},
		{4, 3, 1},
		{5, 3, 2},
		{6, 3, 0},
		{27, 3, 0},
		{28, 3, 1},
		{42, 3, 0},
		{43, 3, 1},
	}
	for _, c := range cases {
		z := AbsUintMod(c.x, c.y)
		Tassert(t, z == c.z, "z: %v != %v", z, c.z)
	}
}

func randuint64s(n int) []uint64 {
	var ints []uint64
	for i := 0; i < n; i++ {
		ints = append(ints, uint64(rand.Int63()))
	}
	return ints
}

func TestOpConversion(t *testing.T) {
	cases := []struct {
		randOp   uint64
		randSize uint64
		randArgs []uint64
	}{
		{1, 10, randuint64s(999)},
		{0, 0, randuint64s(999)},
		{2, 20, randuint64s(999)},
		{3, 30, randuint64s(999)},
		{27, 82, randuint64s(999)},
		{42, 898, randuint64s(999)},
	}

	for _, c := range cases {
		Pf("c: %v %v ...\n", c.randOp, c.randSize)
		// prepend marker
		buf := []byte{0}
		buf = binary.BigEndian.AppendUint64(buf, c.randOp)
		buf = binary.BigEndian.AppendUint64(buf, c.randSize)
		for _, arg := range c.randArgs {
			buf = binary.BigEndian.AppendUint64(buf, arg)
		}
		Assert(len(buf) > 900, len(buf))
		Pf("buf: %v\n", buf)
		instruction, n := RandInstruction(buf)
		op, size, args := instruction.OpCode, instruction.Size, instruction.Args
		Tassert(t, op == c.randOp, "op: %v != %v", op, c.randOp)
		Tassert(t, size == c.randSize, "size: %v != %v", size, c.randSize)
		// XXX marker
		Tassert(t, uint64(n) == size+16, "n: %v != %v", n, size+16)
		Tassert(t, len(args) == len(c.randArgs), "len(args): %v != %v", len(args), len(c.randArgs))
		for i, arg := range args {
			Tassert(t, arg == c.randArgs[i], "arg: %v != %v", arg, c.randArgs[i])
		}
	}
}

func TestRandInstruction(t *testing.T) {
	rand.Seed(1)
	size := 100000
	buf := Randbytes(size)
	instruction, n := RandInstruction(buf)
	Pf("instruction: %v\n", instruction)
	Pf("n: %v\n", n)
}
