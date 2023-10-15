package net

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"

	"github.com/stevegt/tanis/x/node"
)

// . "github.com/stevegt/goadapt"

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

// RandNet given a slice of random bytes, returns a random but valid Net.
func RandNet(buf []byte) (net *Net) {
	return

}

// Instruction is a single instruction that is a step in constructing
// a Net.
type Instruction struct {
	OpCode int
	Size   int
	Args   []uint64
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

// RandInstruction given a slice of random bytes returns an instruction
// along with the number of bytes consumed.
func RandInstruction(randBytes []byte) (inst Instruction, read int) {
	ints, read := RandInts(randBytes)
	inst.OpCode = int(UintMod(ints[0], uint64(OpLast)))
	inst.Size = int(UintMod(ints[1], uint64(256)))
	inst.Args = make([]uint64, inst.Size)
	for i := 2; i < inst.Size; i++ {
		inst.Args[i] = uint64(ints[i])
	}
	return
}

// RandInts given a slice of random bytes returns a slice of random
// uint64 values along with the number of bytes consumed.
func RandInts(randBytes []byte) (uints []uint64, read int) {
	// marker is a single byte that indicates the start of an
	// instruction
	marker := byte(0)

	// word buffer
	var wbuf bytes.Buffer

	var parsing bool
	var size uint8
	var b byte
	for read, b = range randBytes {

		// add byte to word buffer
		wbuf.WriteByte(b)

		if !parsing {
			// look for marker
			if wbuf.Bytes()[0] == marker {
				parsing = true
			}
			// drop first byte (marker) from buffer
			wbuf.ReadByte()
			continue
		}

		// first byte is number of words in instruction
		if size == 0 {
			size = uint8(wbuf.Bytes()[0])
			// drop first byte from buffer
			wbuf.ReadByte()
			continue
		}

		if wbuf.Len() < 8 {
			// need more bytes
			continue
		}

		// got a word
		word := binary.BigEndian.Uint64(wbuf.Bytes())
		uints = append(uints, word)

		// drop word from buffer
		wbuf.Truncate(0)

		if len(uints) < int(size) {
			// need more words
			continue
		}

		// got a complete instruction
		break
	}

	return
}

// RandFloats given a slice of random bytes returns a slice of random
// float64 values along with the number of bytes consumed.
func RandFloats(randBytes []byte) (floats []float64, read int) {
	// get uints
	uints, read := RandInts(randBytes)
	for _, word := range uints {
		// convert to float64
		float := math.Float64frombits(word)
		floats = append(floats, float)
	}
	return
}

// UintMod returns the modulus of two uint64 values.
func UintMod(a, b uint64) (mod uint64) {
	mod = a % b
	return
}
