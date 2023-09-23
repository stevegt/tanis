package dna

import (
	"testing"

	. "github.com/stevegt/goadapt"
)

var testString = `DNA test
Inputs: a b
Outputs: c d
0 0.100000
1 0.200000
`

func TestString(t *testing.T) {
	dna := &DNA{
		Name:        "test",
		InputNames:  []string{"a", "b"},
		OutputNames: []string{"c", "d"},
		Statements: []*Statement{
			{Opcode: OpAddLayer, Arg: 0.1},
			{Opcode: OpAddNode, Arg: 0.2},
		},
	}

	Tassert(t, dna.String() == testString, dna.String())
}

func TestClone(t *testing.T) {
	dna1 := &DNA{
		Name:        "test",
		InputNames:  []string{"a", "b"},
		OutputNames: []string{"c", "d"},
		Statements: []*Statement{
			{Opcode: OpAddLayer, Arg: 0.1},
			{Opcode: OpAddNode, Arg: 0.2},
		},
	}

	dna2 := dna1.Clone()

	Tassert(t, dna1.String() == dna2.String(), "\n%s\n\n%s\n", dna1.String(), dna2.String())
}

func TestStatements(t *testing.T) {
	dna1 := &DNA{
		Name:        "test",
		InputNames:  []string{"a", "b"},
		OutputNames: []string{"c", "d"},
		Statements: []*Statement{
			{Opcode: OpAddLayer, Arg: 0.1},
			{Opcode: OpAddNode, Arg: 0.2},
		},
	}

	dna2 := dna1.Clone()
	buf := dna1.StatementsAsBytes()
	dna2.Statements = nil
	dna2.StatementsFromBytes(buf)

	Tassert(t, dna1.String() == dna2.String(), "\n%s\n\n%s\n", dna1.String(), dna2.String())
}

func TestGenome(t *testing.T) {
	dna1 := &DNA{
		Name:        "test",
		InputNames:  []string{"a", "b"},
		OutputNames: []string{"c", "d"},
		Statements: []*Statement{
			{Opcode: OpAddLayer, Arg: 0.1},
			{Opcode: OpAddNode, Arg: 0.2},
		},
	}

	genome := dna1.AsBytes()
	dna2 := FromBytes(genome)

	str1 := dna1.String()
	str2 := dna2.String()
	Tassert(t, str1 == str2, "\n%s\n\n%s\n", str1, str2)
}
