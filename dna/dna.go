package dna

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"

	. "github.com/stevegt/goadapt"
)

// The DNA language is:
//
// name,inputNames,outputNames|layersDNA
//
// where:
//
// - name is the network's name
// - inputNames is a space-separated list of input names
// - outputNames is a space-separated list of output names
// - layersDNA is a space-separated list of layer DNA strings
//
// The layer DNA language is a binary byte array. Statements are 72
// bits -- 8 bits for the opcode, and 64 bits for the argument.  The
// opcodes are the OpCodes constants.  The arguments are float64s.

type Statement struct {
	Opcode Opcode
	Arg    float64
}

// String returns a string representation of the statement.
func (statement *Statement) String() string {
	return Spf("%v %f", statement.Opcode, statement.Arg)
}

type DNA struct {
	Name        string
	InputNames  []string
	OutputNames []string
	Statements  []*Statement
}

// String returns a string representation of the DNA.
func (dna *DNA) String() string {
	var buf bytes.Buffer
	buf.WriteString(Spf("DNA %s\n", dna.Name))
	buf.WriteString(Spf("Inputs: %s\n", strings.Join(dna.InputNames, " ")))
	buf.WriteString(Spf("Outputs: %s\n", strings.Join(dna.OutputNames, " ")))
	for _, statement := range dna.Statements {
		buf.WriteString(Spf("%s\n", statement))
	}
	return buf.String()
}

// Clone returns a copy of the DNA.
func (dna *DNA) Clone() (clone *DNA) {
	clone = &DNA{}
	clone.Name = dna.Name
	clone.InputNames = make([]string, len(dna.InputNames))
	copy(clone.InputNames, dna.InputNames)
	clone.OutputNames = make([]string, len(dna.OutputNames))
	copy(clone.OutputNames, dna.OutputNames)
	clone.Statements = make([]*Statement, len(dna.Statements))
	for i, statement := range dna.Statements {
		clone.Statements[i] = &Statement{
			Opcode: statement.Opcode,
			Arg:    statement.Arg,
		}
	}
	return
}

func New() (dna *DNA) {
	dna = &DNA{}
	return
}

// AddOp appends a statement to the DNA given an opcode and argument.
func (dna *DNA) AddOp(opcode Opcode, arg float64) {
	statement := &Statement{
		Opcode: opcode,
		Arg:    arg,
	}
	dna.Statements = append(dna.Statements, statement)
}

// AddBytes appends a statement to the DNA given a byte slice.
func (dna *DNA) AddBytes(buf []byte) {
	// get opcode and argument
	opcode := uint(buf[0])
	argbytes := buf[1:]
	arg := Float64FromBytes(argbytes)
	dna.AddOp(Opcode(opcode), arg)
	return
}

// AsBytes returns the DNA as a byte slice.
func (dna *DNA) AsBytes() (out []byte) {
	var buf bytes.Buffer
	head := Spf("%s,%s,%s|", dna.Name, strings.Join(dna.InputNames, " "), strings.Join(dna.OutputNames, " "))
	_, err := buf.WriteString(head)
	Ck(err)
	layersDNA := dna.StatementsAsBytes()
	_, err = buf.Write(layersDNA)
	Ck(err)
	out = buf.Bytes()
	return
}

// FromBytes creates a new DNA object from a byte slice.
func FromBytes(buf []byte) (dna *DNA) {
	dna = &DNA{}
	parts := strings.Split(string(buf), "|")
	Assert(len(parts) == 2, "invalid dna %x", dna)
	head := parts[0]
	layersDNA := []byte(parts[1])
	parts = strings.Split(head, ",")
	Assert(len(parts) == 3, "invalid dna head %s", head)
	dna.Name = parts[0]
	dna.InputNames = strings.Split(parts[1], " ")
	dna.OutputNames = strings.Split(parts[2], " ")
	dna.StatementsFromBytes(layersDNA)
	return
}

// StatementsFromBytes replaces the DNA statements from a byte slice.
func (dna *DNA) StatementsFromBytes(buf []byte) {
	dna.Statements = make([]*Statement, 0)
	for i := 0; i < len(buf); i += 9 {
		statementEnd := i + 9
		if statementEnd > len(buf) {
			// skip partial statement at end
			break
		}
		sbuf := buf[i:statementEnd]
		dna.AddBytes(sbuf)
	}
	return
}

// StatementsAsBytes returns the DNA statements as a byte slice.
func (dna *DNA) StatementsAsBytes() (outbuf []byte) {
	var buf bytes.Buffer
	for _, statement := range dna.Statements {
		// write opcode
		err := buf.WriteByte(byte(statement.Opcode))
		Ck(err)
		// write argument
		argbytes := Float64ToBytes(statement.Arg)
		n, err := buf.Write(argbytes)
		Ck(err)
		Assert(n == len(argbytes), "short write")
	}
	outbuf = buf.Bytes()
	return
}

type Opcode uint

const (
	// add layer
	OpAddLayer Opcode = iota
	// add node to most recent layer
	OpAddNode
	// set activation on most recent node
	OpSetActivation
	// set bias on most recent node
	OpSetBias
	// add weight to most recent node
	OpAddWeight
	// jump to layer
	OpJumpLayer
	// jump to node
	OpJumpNode
	// jump to weight
	OpJumpWeight
	// set most recent weight
	OpSetWeight
	// stop processing
	OpHalt
	// keep this last
	OpLast
)

var ActivationNum = map[string]int{
	"sigmoid": 0,
	"tanh":    1,
	"relu":    2,
	"linear":  3,
	"square":  4,
	"sqrt":    5,
	"abs":     6,
}

var ActivationName = map[int]string{
	0: "sigmoid",
	1: "tanh",
	2: "relu",
	3: "linear",
	4: "square",
	5: "sqrt",
	6: "abs",
}

func Float64FromBytes(bytes []byte) float64 {
	bits := binary.BigEndian.Uint64(bytes)
	float := math.Float64frombits(bits)
	return float
}

func Float64ToBytes(float float64) []byte {
	bits := math.Float64bits(float)
	bytes := make([]byte, 8)
	binary.BigEndian.PutUint64(bytes, bits)
	return bytes
}
