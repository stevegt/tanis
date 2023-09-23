package shape

import (
	"strconv"
	"strings"

	. "github.com/stevegt/goadapt"
	"github.com/xiam/sexpr/ast"
	"github.com/xiam/sexpr/parser"
)

// Shape is a representation of the network's shape.
type Shape struct {
	Name        string
	InputNames  []string
	OutputNames []string
	LayerShapes []*LayerShape
}

func (s *Shape) String() (out string) {
	inputNames := strings.Join(s.InputNames, " ")
	layersParts := []string{}
	for _, layer := range s.LayerShapes {
		layersParts = append(layersParts, layer.String())
	}
	layers := strings.Join(layersParts, " ")
	out = Spf("(%s %s %s)", s.Name, inputNames, layers)
	return
}

func (s *Shape) SetOutputNames() {
	// output names are the last layer's node names
	lastLayer := s.LayerShapes[len(s.LayerShapes)-1]
	s.OutputNames = lastLayer.OutputNames()
	return
}

type LayerShape struct {
	Nodes []*NodeShape
}

func (s *LayerShape) String() (out string) {
	hiddenGroups := make(map[string]int)
	outputGroups := make(map[string][]string)
	for _, node := range s.Nodes {
		actName := node.ActivationName
		if node.Name != "" {
			// output node
			outputGroups[actName] = append(outputGroups[actName], node.Name)
		} else {
			// hidden node
			hiddenGroups[actName]++
		}
	}
	isOutput := len(outputGroups) > 0
	isHidden := len(hiddenGroups) > 0
	// xor
	Assert(isHidden != isOutput, "layer has both hidden and output nodes")

	groups := []string{}
	if isHidden {
		for actName, count := range hiddenGroups {
			groups = append(groups, Spf("(%s %d)", actName, count))
		}
	}
	if isOutput {
		for actName, names := range outputGroups {
			groups = append(groups, Spf("(%s %s)", actName, strings.Join(names, " ")))
		}
	}

	if len(groups) == 1 {
		out = groups[0]
	} else {
		out = Spf("(+ %s)", strings.Join(groups, " "))
	}
	return
}

func (s *LayerShape) OutputNames() (names []string) {
	for i := 0; i < len(s.Nodes); i++ {
		names = append(names, s.Nodes[i].Name)
	}
	return
}

type NodeShape struct {
	Name           string
	ActivationName string
}

// SyntaxError is a syntax error.
type SyntaxError struct {
	msg  string
	node *ast.Node
}

func (e *SyntaxError) Error() string {
	return Spf("[shape:%s] %s:\n%s", e.node.Token().Pos, e.msg, e.node.String())
}

// synck raises a syntax err if cond is false.
func synck(node *ast.Node, cond bool, args ...interface{}) {
	if !cond {
		msg := FormatArgs(args...)
		panic(&SyntaxError{msg, node})
	}
}

func walk(node *ast.Node, f func(*ast.Node)) {
	f(node)
	for _, child := range node.List() {
		switch child.Type() {
		case ast.NodeTypeList, ast.NodeTypeExpression:
			walk(child, f)
		case ast.NodeTypeSymbol, ast.NodeTypeInt, ast.NodeTypeFloat, ast.NodeTypeString:
			f(child)
		default:
			Assert(false, "unknown node type %v", child.Type())
		}
	}
}

func dump(root *ast.Node) {
	ast.Print(root)
	f := func(node *ast.Node) {
		Pl("======")
		Pf("encoded: %s\n", node.Encode())
		Pf("token: %s\n", node.Token())
		Pf("type: %s\n", node.Type())
		Pf("value: %s\n", node.Value())
	}
	walk(root, f)
}

func Parse(txt string) (s *Shape, err error) {
	defer Return(&err)
	root, err := parser.Parse([]byte(txt))
	Ck(err)
	// dump(root)

	// (foo a b c (tanh 4) (sigmoid 5) [(tanh 2)(relu 1)] [(tanh x y z v w) (relu u)] )

	// root is a list
	synck(root, root.Type() == ast.NodeTypeList, "root is not a list")
	// root has one child
	children := root.List()
	synck(root, len(children) == 1, "root has %d children", len(children))
	// root's child is an expression
	expr := children[0]
	synck(expr, expr.Type() == ast.NodeTypeExpression, "root's child is not an expression")
	// the shape code is the several children of the expression
	s, err = parseShape(expr)
	Ck(err)

	return
}

type Expr struct {
	Op   string
	Args []Expr
}

func parseShape(n *ast.Node) (s *Shape, err error) {
	defer Return(&err)

	s = &Shape{}

	expr, err := parseExpr(n)
	Ck(err)
	s.Name = expr.Op
	for _, arg := range expr.Args {
		if len(arg.Args) == 0 {
			s.InputNames = append(s.InputNames, arg.Op)
		} else {
			// layer
			layerShape := parseLayer(arg)
			s.LayerShapes = append(s.LayerShapes, layerShape)
		}
	}
	s.SetOutputNames()
	Ck(err)
	return
}

func parseLayer(arg Expr) (layerShape *LayerShape) {
	layerShape = &LayerShape{}
	if arg.Op == "+" {
		// node groups
		for _, groupExpr := range arg.Args {
			subLayerShape := parseLayer(groupExpr)
			layerShape.Nodes = append(layerShape.Nodes, subLayerShape.Nodes...)
		}
	} else {
		actName := arg.Op
		for _, nodeExpr := range arg.Args {
			// nodeExpr.Op is either a node count or an output name
			count, err := strconv.Atoi(nodeExpr.Op)
			if err != nil {
				// it's an output name
				node := &NodeShape{}
				node.Name = nodeExpr.Op
				node.ActivationName = actName
				layerShape.Nodes = append(layerShape.Nodes, node)
			} else {
				// it's a node count
				for i := 0; i < count; i++ {
					node := &NodeShape{}
					node.ActivationName = actName
					layerShape.Nodes = append(layerShape.Nodes, node)
				}
			}
		}
	}
	return
}

func parseExpr(n *ast.Node) (expr *Expr, err error) {
	defer Return(&err)
	children := n.List()
	synck(n, len(children) > 0, "missing opcode")
	synck(n, children[0].Type() == ast.NodeTypeSymbol, "first word is not a symbol")
	expr = &Expr{}
	expr.Op = children[0].Encode()
	for i := 1; i < len(children); i++ {
		switch children[i].Type() {
		case ast.NodeTypeSymbol, ast.NodeTypeInt, ast.NodeTypeFloat, ast.NodeTypeString:
			expr.Args = append(expr.Args, Expr{children[i].Encode(), nil})
		case ast.NodeTypeExpression:
			arg, err := parseExpr(children[i])
			Ck(err)
			expr.Args = append(expr.Args, *arg)
		default:
			synck(children[i], false, "unknown node type %v", children[i].Type())
		}
	}
	return
}

/*



	switch op {
	case "+":
		err = parseNodeGroups(n, s)
	default:
		if s.Name == "" {
			s.Name = first
		} else {

	// first child is the net name
	children := n.List()
	synck(n, len(children) > 0, "missing net name")
	netName := children[0]
	synck(netName, netName.Type() == ast.NodeTypeSymbol, "net name is not a symbol")
	s.Name = netName.Encode()

	// next children are the input name symbols
	synck(n, len(children) > 1, "missing input names")
	i := 1
	for ; i < len(children); i++ {
		switch children[i].Type() {
		case ast.NodeTypeSymbol:
			s.InputNames = append(s.InputNames, children[i].Encode())
		default:
			break
		}
	}

	// remaining children are the layer expressions
	var upstream *LayerShape
	for ; i < len(children); i++ {
		switch children[i].Type() {
		case ast.NodeTypeExpression, ast.NodeTypeList:
			ls, err := parseLayer(children[i], upstream)
			Ck(err)
			s.LayerShapes = append(s.LayerShapes, ls)
		default:
			synck(children[i], false, "unexpected child")
		}
	}
	s.OutputNames = upstream.OutputNames()

	return
}

func parseLayer(s *Shape, n *ast.Node, upstream *LayerShape) (ls *LayerShape, err error) {
	defer Return(&err)
	ls = &LayerShape{}

	return
}

	switch n.Type() {
		case ast.NodeTypeExpression:
			// single node group
			ls, err = parseNodeGroup(n)
			Ck(err)



	// consume node groups
	children := n.List()
	for i := 0; i < len(children); i++ {



	return
}
*/

/*

	ls = &LayerShape{}

	// first child is either the activation function name or a list
	children := n.List()
	synck(n, len(children) > 0, "missing activation function name")
	af := children[0]
	synck(af, af.Type() == ast.NodeTypeSymbol, "activation function name is not a symbol")
	actName := af.Value()
	// second child is either the number of neurons or the first
	// output name
	synck(n, len(children) > 1, "missing number of neurons or first output name")
	child2 := children[1]
	nodeCount := 0
	switch child2.Type() {
	case ast.NodeTypeInt:
		// number of neurons
		nodeCount, err = strconv.Atoi(child2.Encode())
		Ck(err)
		synck(n, nodeCount > 0, "number of neurons is not positive")
		// should be no more children
		synck(n, len(children) == 2, "too many children")
	case ast.NodeTypeSymbol:
		// first output name




	synck(n, len(children) > 1, "missing number of neurons")
	nc := children[1]
	synck(nc, nc.Type() == ast.NodeTypeInt, "number of neurons is not an integer")
	nodeCount, err := strconv.Atoi(nc.Encode())
	// emit nodes
	for i := 0; i < nodeCount; i++ {
		D.AddOp(dna.OpAddNode, 0)
		// emit activation function
		D.AddOp(dna.OpSetActivation, dna.ActivationNum(actName))
		// emit bias
		D.AddOp(dna.OpSetBias, rand.Float64()*2-1)
		// emit weights
		for j := 0; j < inputCount; j++ {
			D.AddOp(dna.OpSetWeight, rand.Float64()*2-1)
		}
	}
	return
}
*/

/*
example AST:

(:list ()
    (:expression (:open_expression "(" [1 1])
        (:symbol (:word "foo" [1 2]))
        (:symbol (:word "a" [1 6]))
        (:symbol (:word "b" [1 8]))
        (:symbol (:word "c" [1 10]))
        (:expression (:open_expression "(" [1 12])
            (:symbol (:word "tanh" [1 13]))
            (:int (:integer "4" [1 18]))
        )
        (:expression (:open_expression "(" [1 21])
            (:symbol (:word "sigmoid" [1 22]))
            (:int (:integer "5" [1 30]))
        )
        (:expression (:open_expression "(" [1 33])
            (:expression (:open_expression "(" [1 34])
                (:symbol (:word "tanh" [1 35]))
                (:int (:integer "2" [1 40]))
            )
            (:expression (:open_expression "(" [1 42])
                (:symbol (:word "relu" [1 43]))
                (:int (:integer "1" [1 48]))
            )
        )
        (:expression (:open_expression "(" [1 52])
            (:expression (:open_expression "(" [1 53])
                (:symbol (:word "tanh" [1 54]))
                (:symbol (:word "x" [1 59]))
                (:symbol (:word "y" [1 61]))
                (:symbol (:word "z" [1 63]))
                (:symbol (:word "v" [1 65]))
                (:symbol (:word "w" [1 67]))
            )
            (:expression (:open_expression "(" [1 70])
                (:symbol (:word "relu" [1 71]))
                (:symbol (:word "u" [1 76]))
            )
        )
    )
)

*/
