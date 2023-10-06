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
	OpCode byte
	Size   int
	Args   []int64
}

// RandInstruction given a slice of random bytes returns an instruction
// along with the number of bytes consumed.
// func RandInstruction(randBytes []byte) (inst Instruction, read int) {

// RandFloats given a slice of random bytes returns a slice of random
// float64 values along with the number of bytes consumed.
func RandFloats(randBytes []byte) (floats []float64, read int) {

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

		/*
			// try to read varint
			word, n := binary.Uvarint(wbuf.Bytes())
			if n == 0 {
				// need more bytes
				continue
			}
			if n < 0 {
				// overflow; discard first byte from buffer
				// ignore invalid varints
				wbuf.ReadByte()
				continue
			}
		*/

		if wbuf.Len() < 8 {
			// need more bytes
			continue
		}

		// got a word
		word := binary.BigEndian.Uint64(wbuf.Bytes())
		// convert to float64
		float := math.Float64frombits(word)

		floats = append(floats, float)
		// drop word from buffer
		wbuf.Truncate(0)

		if len(floats) < int(size) {
			// need more words
			continue
		}

		// got a complete instruction
		break
	}

	return
}
