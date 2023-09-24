package main

import (
	"bytes"
	"encoding/binary"
	"math/rand"

	"github.com/robskie/fibvec"
	. "github.com/stevegt/goadapt"
)

// fib experiment with fibonacci coding
func fib() {
	v := fibvec.NewVector()
	v.Add(1)
	v.Add(2)
	v.Add(999)
	Pl(v.Size())
}

// varintBuf experiments with variable length integers
// - demonstrates conversion from random to varintBuf
func varintBuf() {

	var max uint64
	var mark, pass, smallBuf, overflow int64
	for i := int64(1); i < 1e10; i++ {

		// create random byte array
		size := rand.Intn(13)
		b := make([]byte, size)
		rand.Read(b)

		// read bits as if varint
		var1, n := binary.Uvarint(b)
		if n == 0 {
			smallBuf++
			continue
		}
		if n < 0 {
			overflow++
			continue
		}

		if b[0] == 0b10101010 {
			mark++
		}

		pass++
		if var1 > max {
			// diff := math.MaxUint64 - var1
			Pf("%4d %4d %20d %10d %10d %10d %10d %x\n", size, n, var1, mark, pass, smallBuf, overflow, b)
			max = var1
		}

	}
}

// varintStream experiments with variable length integers from a
// stream of random bytes, using a marker
func varintStream() {
	// marker := byte(0b10101010)
	marker := []byte{0}
	// marker := []byte{0, 0}
	var max uint64
	var read, mark, pass, smallBuf, overflow int64

	// create random bytes in a channel
	randBytes := make(chan byte)
	go func() {
		for {
			b := make([]byte, 1)
			rand.Read(b)
			randBytes <- b[0]
		}
	}()

	// instruction buffer
	var ibuf bytes.Buffer

	// read random bytes from channel
	for randByte := range randBytes {
		read++

		// add byte to buffer
		ibuf.WriteByte(randByte)

		// wait for buffer length to equal marker length
		if ibuf.Len() < len(marker) {
			continue
		}

		// look for marker
		if !bytes.HasPrefix(ibuf.Bytes(), marker) {
			// drop first byte from buffer
			ibuf.ReadByte()
			continue
		}
		mark++

		// don't include marker in varint
		start := len(marker)

		// try to read varint
		var1, n := binary.Uvarint(ibuf.Bytes()[start:])
		if n == 0 {
			smallBuf++
			// add byte to buffer
			continue
		}
		if n < 0 {
			overflow++
			// drop first byte from buffer
			ibuf.ReadByte()
			continue
		}
		pass++

		// convert to float64
		// f := math.Float64frombits(var1)

		if var1 > max {
			// Pf("mark %10d  smallBuf %10d  overflow %10d  ibuf len %3d  varint len %3d  pass %10d  var1 %20d  float64 %25.15e\n", mark, smallBuf, overflow, ibuf.Len(), n, pass, var1, f)
			prob := float64(mark) / float64(read)
			Pf("read %20d mark %10d  prob %10.8f smallBuf %10d  overflow %10d  ibuf len %3d  varint len %3d  pass %10d\n", read, mark, prob, smallBuf, overflow, ibuf.Len(), n, pass)
			max = var1
		}

		// empty buffer
		ibuf.Truncate(0)

	}
}

// varintInstructions experiments with reading instructions from a
// stream of random bytes.  Each instruction is a single-byte marker
// followed by a 1-byte size, followed by the elements as varints.
func varintInstructions() {
	// marker := byte(0b10101010)
	marker := byte(0)
	var read, mark, pass, smallBuf, overflow, instructions int64

	// create random bytes in a channel
	src := make(chan byte)
	go func() {
		for {
			b := make([]byte, 1)
			rand.Read(b)
			src <- b[0]
		}
	}()

	// word buffer
	var wbuf bytes.Buffer
	// parsed instruction
	var instr []uint64

	var parsing bool
	var size uint8
	for {
		// get next byte from channel
		b := <-src
		read++

		// add byte to word buffer
		wbuf.WriteByte(b)

		if !parsing {
			// look for marker
			if wbuf.Bytes()[0] == marker {
				mark++
				parsing = true
			}
			// drop first byte from buffer
			wbuf.ReadByte()
			continue
		}

		if size == 0 {
			size = uint8(wbuf.Bytes()[0])
			// drop first byte from buffer
			wbuf.ReadByte()
			continue
		}

		// try to read varint
		word, n := binary.Uvarint(wbuf.Bytes())
		if n == 0 {
			smallBuf++
			// need more bytes
			continue
		}
		if n < 0 {
			overflow++
			// ignore invalid varints
			// drop first byte from buffer
			wbuf.ReadByte()
			continue
		}

		// got a word
		pass++
		instr = append(instr, word)
		// drop word from buffer
		wbuf.Truncate(0)

		if len(instr) < int(size) {
			// need more words
			continue
		}

		// got a complete instruction
		instructions++

		// opcode is low 4 bits of first word
		// opcode := instr[0] & 0b1111

		// opcode is first word mod 7
		// opcode := instr[0] % 7
		// Pl("opcode", opcode)

		prob := float64(mark) / float64(read)
		Pf("read %20d mark %10d  prob %10.8f smallBuf %10d  overflow %10d  pass %10d  instructions %5d  instr len %3d\n", read, mark, prob, smallBuf, overflow, pass, instructions, len(instr))

		parsing = false
		size = 0
		instr = nil

	}
}
func main() {

	// fib()

	// varintBuf()
	// varintStream()
	varintInstructions()
}
