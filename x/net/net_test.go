package net

import (
	"encoding/binary"
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

func TestBytesToInst(t *testing.T) {
	var argc uint16 = 100
	var buf []byte
	var inst *Instruction
	var n int

	// fill the buffer with marker, opcode 0, argc, sequential ints
	buf = []byte{0, 0}
	buf = binary.BigEndian.AppendUint16(buf, argc)
	for i := 0; i < int(argc); i++ {
		buf = binary.BigEndian.AppendUint64(buf, uint64(i))
	}
	inst, n = FindInstruction(buf)
	Pf("n: %v\n", n)
	Pf("inst: %v\n", inst)
	Tassert(t, inst.OpCode == 0, "inst.OpCode: %v != %v", inst.OpCode, 0)
	Tassert(t, inst.Argc == 100, "inst.Argc: %v != %v", inst.Argc, argc)
	Tassert(t, len(inst.Argv) == int(argc), "len(inst.Args): %v != %v", len(inst.Argv), argc)
	for i, arg := range inst.Argv {
		Tassert(t, arg == uint64(i), "v: %v != %v", arg, i)
	}

}

/*
func TestRandFloats(t *testing.T) {
	rand.Seed(45)
	size := 100000
	buf := Randbytes(size)

	floats, n := XXXRandFloats(buf)
	Pf("floats: %v\n", floats)
	Pf("n: %v\n", n)

	m1 := math.Mod(floats[0], 3)
	Pf("m1: %v\n", m1)
}
*/

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
	rand.Seed(1)
	cases := []struct {
		randOp   uint8
		randArgc uint16
		randArgv []uint64
	}{
		{1, 10, randuint64s(9999)},
		{0, 0, randuint64s(9999)},
		{2, 20, randuint64s(9999)},
		{3, 30, randuint64s(9999)},
		{27, 82, randuint64s(9999)},
		{42, 898, randuint64s(9999)},
	}

	for _, c := range cases {
		Pf("c: %v %v ...\n", c.randOp, c.randArgc)
		inst := &Instruction{c.randOp, c.randArgc, c.randArgv}
		buf := inst.Bytes()
		Assert(len(buf) > 9000, len(buf))
		instruction, _ := FindInstruction(buf)
		op, size, args := instruction.OpCode, instruction.Argc, instruction.Argv
		Tassert(t, op == c.randOp, "op: %v != %v", op, c.randOp)
		Tassert(t, size == c.randArgc, "size: %v != %v", size, c.randArgc)
		Tassert(t, len(args) == len(c.randArgv), "len(args): %v != %v", len(args), len(c.randArgv))
		for i, arg := range args {
			Tassert(t, arg == c.randArgv[i], "arg: %v != %v", arg, c.randArgv[i])
		}
	}
}
