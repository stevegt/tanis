package shape

import (
	"testing"

	. "github.com/stevegt/goadapt"
)

func TestParse(t *testing.T) {
	txt := "(foo a b c (tanh 4) (sigmoid 5) (+(tanh 2)(relu 1)) (+(tanh x y z v w) (relu u)) )"
	s, err := Parse(txt)
	Tassert(t, err == nil, err)
	Tassert(t, s.Name == "foo", "name %s", s.Name)
	Tassert(t, len(s.InputNames) == 3, s.InputNames)
	Tassert(t, s.InputNames[0] == "a", s.InputNames[0])
	Tassert(t, s.InputNames[1] == "b", s.InputNames[1])
	Tassert(t, s.InputNames[2] == "c", s.InputNames[2])
	Tassert(t, len(s.OutputNames) == 6, s.OutputNames)
	Tassert(t, s.OutputNames[0] == "x", s.OutputNames[0])
	Tassert(t, s.OutputNames[1] == "y", s.OutputNames[1])
	Tassert(t, s.OutputNames[2] == "z", s.OutputNames[2])
	Tassert(t, s.OutputNames[3] == "v", s.OutputNames[3])
	Tassert(t, s.OutputNames[4] == "w", s.OutputNames[4])
	Tassert(t, s.OutputNames[5] == "u", s.OutputNames[5])
	Tassert(t, len(s.LayerShapes) == 4, s.LayerShapes)
	Tassert(t, len(s.LayerShapes[0].Nodes) == 4, s.LayerShapes[0].Nodes)
	Tassert(t, s.LayerShapes[0].Nodes[0].ActivationName == "tanh", s.LayerShapes[0].Nodes[0].ActivationName)
	Tassert(t, s.LayerShapes[0].Nodes[1].ActivationName == "tanh", s.LayerShapes[0].Nodes[1].ActivationName)
	Tassert(t, s.LayerShapes[0].Nodes[2].ActivationName == "tanh", s.LayerShapes[0].Nodes[2].ActivationName)
	Tassert(t, s.LayerShapes[0].Nodes[3].ActivationName == "tanh", s.LayerShapes[0].Nodes[3].ActivationName)
	Tassert(t, len(s.LayerShapes[1].Nodes) == 5, s.LayerShapes[1].Nodes)
	Tassert(t, s.LayerShapes[1].Nodes[0].ActivationName == "sigmoid", s.LayerShapes[1].Nodes[0].ActivationName)
	Tassert(t, s.LayerShapes[1].Nodes[1].ActivationName == "sigmoid", s.LayerShapes[1].Nodes[1].ActivationName)
	Tassert(t, s.LayerShapes[1].Nodes[2].ActivationName == "sigmoid", s.LayerShapes[1].Nodes[2].ActivationName)
	Tassert(t, s.LayerShapes[1].Nodes[3].ActivationName == "sigmoid", s.LayerShapes[1].Nodes[3].ActivationName)
	Tassert(t, s.LayerShapes[1].Nodes[4].ActivationName == "sigmoid", s.LayerShapes[1].Nodes[4].ActivationName)
	Tassert(t, len(s.LayerShapes[2].Nodes) == 3, s.LayerShapes[2].Nodes)
	Tassert(t, s.LayerShapes[2].Nodes[0].ActivationName == "tanh", s.LayerShapes[2].Nodes[0].ActivationName)
	Tassert(t, s.LayerShapes[2].Nodes[1].ActivationName == "tanh", s.LayerShapes[2].Nodes[1].ActivationName)
	Tassert(t, s.LayerShapes[2].Nodes[2].ActivationName == "relu", s.LayerShapes[2].Nodes[2].ActivationName)
	Tassert(t, len(s.LayerShapes[3].Nodes) == 6, s.LayerShapes[3].Nodes)
	Tassert(t, s.LayerShapes[3].Nodes[0].ActivationName == "tanh", s.LayerShapes[3].Nodes[0].ActivationName)
	Tassert(t, s.LayerShapes[3].Nodes[1].ActivationName == "tanh", s.LayerShapes[3].Nodes[1].ActivationName)
	Tassert(t, s.LayerShapes[3].Nodes[2].ActivationName == "tanh", s.LayerShapes[3].Nodes[2].ActivationName)
	Tassert(t, s.LayerShapes[3].Nodes[3].ActivationName == "tanh", s.LayerShapes[3].Nodes[3].ActivationName)
	Tassert(t, s.LayerShapes[3].Nodes[4].ActivationName == "tanh", s.LayerShapes[3].Nodes[4].ActivationName)
	Tassert(t, s.LayerShapes[3].Nodes[5].ActivationName == "relu", s.LayerShapes[3].Nodes[5].ActivationName)
}

func TestParse2(t *testing.T) {
	txt := "(foo a b c (sigmoid 4) (sigmoid 5) (linear x y z w v u))"
	s, err := Parse(txt)
	Tassert(t, err == nil, err)

	Tassert(t, len(s.LayerShapes[0].Nodes) == 4, s.LayerShapes[0].Nodes)
	Tassert(t, len(s.LayerShapes[1].Nodes) == 5, s.LayerShapes[0].Nodes)
	Tassert(t, len(s.LayerShapes[2].Nodes) == 6, s.LayerShapes[0].Nodes)
}

func TestString(t *testing.T) {
	txt := "(foo a b c (tanh 4) (sigmoid 5) (+ (tanh 2) (relu 1)) (+ (tanh x y z v w) (relu u v)))"
	s, err := Parse(txt)
	Tassert(t, err == nil, err)
	got := s.String()
	Tassert(t, got == txt, "\nwant %s\ngot  %s", txt, got)
}
