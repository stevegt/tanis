package net

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"

	// . "github.com/stevegt/goadapt"
	"github.com/stevegt/tanis/x/node"
)

// Net is a layerless neural network that supports genetic
// algorithm-based training.
type Net struct {
	node.Graph
}

// Randbytes returns a slice of random bytes. The length of the
// slice is specified by the parameter.
func Randbytes(n int) (buf []byte) {
	buf = make([]byte, n)
	for i := 0; i < n; i++ {
		buf[i] = byte(rand.Intn(256))
	}
	return
}

// NetFromBytes given a slice of arbitrary bytes returns a valid Net.
func NetFromBytes(buf []byte) (net *Net) {
	net = new(Net)
	// XXX see node_test.go

	return
}

// Instruction is a single instruction that is a step in constructing
// a Net.
type Instruction struct {
	OpCode uint8
	Argc   uint16 // number of args
	Argv   []uint64
}

type Opcode int

const (
	// AddNode adds a node to the net.
	OpAddNode Opcode = iota
	// AddConst adds a constant node to the net.
	OpAddConst
	// OpLast must be the last opcode.
	OpLast
)

/*
// RandInstruction given a slice of random bytes returns an
func RandInstruction(randBytes []byte) (inst Instruction, read int) {
	ints, read := FindInstruction(randBytes)
	Assert(len(ints) >= 2, "not enough ints")
	inst.OpCode = AbsUintMod(ints[0], uint64(OpLast))
	inst.Size = AbsUintMod(ints[1], 256)
	stop := int(math.Min(float64(inst.Size), float64(len(ints))))
	Pl("stop", stop)
	inst.Args = make([]uint64, stop)
	for i := 2; i < stop; i++ {
		inst.Args[i] = uint64(ints[i])
	}
	return
}
*/

// Bytes returns a slice of bytes that represents the Instruction.
func (inst *Instruction) Bytes() (buf []byte) {
	// marker is a single byte that indicates the start of an
	// instruction
	marker := byte(0)

	// opcode is a single byte
	opcode := inst.OpCode

	// argc is two bytes
	argc := inst.Argc

	// argv is a slice of uint64s
	argv := inst.Argv

	// buffer for bytes
	var wbuf bytes.Buffer

	// write marker
	wbuf.WriteByte(marker)

	// write opcode
	wbuf.WriteByte(opcode)

	// write argc
	binary.Write(&wbuf, binary.BigEndian, argc)

	// write argv
	for _, word := range argv {
		binary.Write(&wbuf, binary.BigEndian, word)
	}

	// return bytes
	buf = wbuf.Bytes()
	return
}

// FindInstruction given a slice of bytes finds and returns an
// Instruction along with the number of bytes consumed.
func FindInstruction(buf []byte) (inst *Instruction, read int) {
	// marker is a single byte that indicates the start of an
	// instruction
	marker := byte(0)

	// word buffer
	var wbuf bytes.Buffer

	// empty instruction
	inst = new(Instruction)

	type State int
	const (
		// look for marker
		SEARCH State = iota
		// get opcode
		OPCODE
		// get argc
		ARGC
		// get argv
		ARGV
	)
	var state State

	var b byte
	for read, b = range buf {

		// add byte to word buffer
		wbuf.WriteByte(b)

		switch state {
		case SEARCH:
			// look for marker
			if wbuf.Bytes()[0] == marker {
				state = OPCODE
			}
			// drain buffer
			wbuf.Truncate(0)
			continue
		case OPCODE:
			// first byte after marker is opcode
			inst.OpCode = uint8(wbuf.Bytes()[0])
			state = ARGC
			// drain buffer
			wbuf.Truncate(0)
			continue
		case ARGC:
			// next two bytes after marker is argc
			if wbuf.Len() < 2 {
				// need more bytes
				continue
			}
			// got argc
			inst.Argc = uint16(binary.BigEndian.Uint16(wbuf.Bytes()))
			state = ARGV
			// drain buffer
			wbuf.Truncate(0)
			continue
		case ARGV:
			// next ARGC words are argv
			if wbuf.Len() < 8 {
				// need more bytes
				continue
			}
			// got a word
			word := binary.BigEndian.Uint64(wbuf.Bytes())
			inst.Argv = append(inst.Argv, word)
			// drain buffer
			wbuf.Truncate(0)
			if len(inst.Argv) < int(inst.Argc) {
				// need more words
				continue
			}
			// got all argv; break out of for loop
			break
		}
	}

	return
}

/*
// XXXRandFloats given a slice of random bytes returns a slice of random
// float64 values along with the number of bytes consumed.
func XXXRandFloats(randBytes []byte) (floats []float64, read int) {
	// get uints
	uints, read := FindInstruction(randBytes)
	for _, word := range uints {
		// convert to float64
		float := math.Float64frombits(word)
		floats = append(floats, float)
	}
	return
}
*/

// AbsUintMod returns the absolute value of the modulus of two uint64
// values.
func AbsUintMod(a, b uint64) uint64 {
	return uint64(math.Abs(float64(a % b)))
}
