
when using multiple named outputs in tanis, what happens if i perform
backprop training while providing only one named output in the
targetMap?  will it train properly?


When using multiple named outputs in Tanis and performing backprop
training while providing only one named output in the targetMap, the
training will still occur, but it will only consider the provided
target output for training. The other outputs will be treated as
having a target value of zero. This may not result in the desired
training results if the actual targets for the other outputs are not
zero.

oh, that's quite bad.  we want backprop to instead only train using
the given outputs.  suggest a fix.


To modify the backpropagation process so that it only trains using the
given outputs, you can update the `LearnNamed` function in tanis.go.
Specifically, you need to change how the `targetSlice` is constructed,
so that it only includes the provided named outputs. Here's a
suggestion:

1. Initialize the `targetSlice` with NaN values:

```go
targetSlice := make([]float64, len(n.OutputNames))
for i := range targetSlice {
    targetSlice[i] = math.NaN()
}
```

2. Update the loop that populates the `targetSlice`:

```go
for i, name := range n.OutputNames {
    target, ok := targetMap[name]
    if !ok {
        continue
    }
    Assert(!math.IsNaN(target), "target %s is NaN", name)
    targetSlice[i] = target
}
```

3. Modify the `learn` function to handle NaN values in the
   `targetSlice`. You can achieve this by skipping the backpropagation
   updates for the outputs that have NaN as the target. In the part
   where you calculate the `errors` vector in tanis.go, you can add a
   check for NaN values:

```go
errors[i] = target - outputs[i]
if math.IsNaN(target) {
    errors[i] = 0
}
```

That's it! With these changes, the network will only update weights
concerning the provided outputs during backpropagation and ignore the
ones with NaN values.

Does it also make sense to handle missing inputs the same way?  In
what cases might that be useful?

Yes, handling missing inputs similarly by assigning NaN or a specific
default value can be useful in certain situations. This approach might
be helpful in the following cases:

1. Incomplete or inconsistent data: When the data you're working with
   has missing input features, assigning a default value like NaN or a
   specific pre-defined value can help the network ignore those inputs
   during training while still processing available data.

2. Transfer learning or domain adaptation: If you're trying to adapt a
   pre-trained network to a new dataset or domain with different input
   feature sets, handling missing inputs gracefully can help during
   the fine-tuning process by avoiding potential errors due to
   inconsistent input dimensions.

3. Real-world applications with sensor data: In applications involving
   sensor data, it is common to have missing or unreliable readings
   due to sensor failures or interference. In such cases, handling
   missing inputs can help the network to make predictions using the
   available data without any issues.

It is important to consider the specific use case and the
characteristics of the data when deciding how to handle missing
inputs. Using NaN values might not be suitable for all situations, and
some cases may require a different default value or a more
sophisticated approach to missing data imputation.


Suggest code modifications that would allow handling missing named inputs.

To handle missing named inputs, you can make the following code
modifications in tanis.go:

1. Update the loop that populates the `inputSlice`:

```go
inputSlice := make([]float64, len(n.InputNames))
for i, name := range n.InputNames {
    input, ok := inputMap[name]
    if !ok {
        inputSlice[i] = math.NaN()
        continue
    }
    Assert(!math.IsNaN(input), "input %s is NaN", name)
    inputSlice[i] = input
}
```

2. Modify the `getInputs()` function in the `Layer` struct to handle
   NaN values. You can add a check for NaN values and replace them
   with a default value, like zero:

```go
func (l *Layer) getInputs() (inputs []float64) {
    if l.upstream == nil {
        Assert(len(l.inputs) > 0, "layer: %#v", l)
        inputs = make([]float64, len(l.inputs))
        for i, input := range l.inputs {
            if math.IsNaN(input) {
                inputs[i] = 0
            } else {
                inputs[i] = input
            }
        }
    } else {
        Assert(len(l.inputs) == 0, Spf("layer: %#v", l))
        for _, upstreamNode := range l.upstream.Nodes {
            inputs = append(inputs, upstreamNode.Output())
        }
    }
    return
}
```

With these modifications, the network will handle missing named inputs
by setting them to NaN, and the `getInputs()` function will replace
NaN values with zeros to prevent issues during computation.


in tanis.go, where is the gradient of the loss function being
computed?  what is its variable name?

The gradient of the loss function is computed in the `learn` function
in tanis.go. The variable name for the computed gradient is not
explicit, but the difference between target and network output, which
forms the basis for the computation of gradients during
backpropagation, is stored in the `errors` slice.

what modifications do i need to make to tanis.go to explicitly
compute the gradient of the loss function?

To make modifications in `tanis.go` for explicitly computing the
gradient of the loss function: 

1. Specific calculation for the gradient of the loss function needs to
   be done in the `learn` function where the error is calculated as
   the difference between the target and network output. The `errors`
   slice stores this information.

2. You need to add explicit code here to compute the gradients. For a
   neural network using mean square error as loss function and sigmoid
   as activation in neurons, the gradient of the loss function with
   respect to a weight can be computed using the chain rule as the
   derivative of the loss function times the derivative of activation
   function times the input to that weight. 



am i computing the derivative of the loss function anywhere in tanis.go?

Yes, the derivative of the loss function is being computed in the
`learn` function in the `tanis.go` file. Specifically, the difference
between the target and the network output, which acts as the
derivative of the loss function during backpropagation, is being
calculated and stored in the `errors` slice.


should "errors" be renamed to "lossDerivatives"?

Yes, the term "errors" should be renamed to "lossDerivatives". The
variable "errors" is storing the difference between the target and the
network output, which acts as the derivative of the loss function
during backpropagation. So, the variable is in fact holding the value
of the loss function's derivatives, not just errors, and therefore
"lossDerivatives" could be a more appropriate name.

what about the cost function for the network?  where is that being
computed?

The cost function for the network is being calculated in the `learn`
function in the `tanis.go` file as well. After calculating the
`errors` (or `lossDerivatives`), you are looping through the targets
and outputs, and for each pair, you add `0.5 * (target - output)^2` to
the total cost. This is represented in the line `n.cost += 0.5 *
math.Pow(target-outputs[i], 2)`. Here, `n.cost` represents the total
cost of the network.

what's the usual formula for the cost function?

The usual formula for the cost function in a neural network,
especially when using Mean Squared Error (MSE), is `0.5 * (target -
output)^2`. This costs function essentially calculates the difference
between the target (what we want the network to output) and the actual
output of the network. This difference is then squared and divided by
2 to ensure a positive value. The lower the result, the better the
neural network's performance. The '0.5' coefficient is used to make
the derivative of the cost function simpler during backpropagation.

what's the usual formula for the gradient of the cost function?

The usual formula for the gradient of the cost function in a neural
network, particularly when using Mean Squared Error (MSE) as the loss
function and Sigmoid as the activation function, is computed using the
chain rule as the derivative of the loss function times the derivative
of the activation function times the input to that weight. In code,
this is represented by computing the difference between the target and
the output (which forms the basis for the computation of gradients
during the backpropagation), stored in the `errors` slice.


is "gradient" just "first derivative"?

Yes, in the context of mathematical functions and calculus, the
"gradient" is often referred to as the "first derivative".
Specifically, in the context of optimizing a function, as in machine
learning models like neural networks, the gradient refers to the
vector of first order derivatives (partial derivatives) with respect
to the parameters. Each component of the gradient tells us the slope
of the function at a specific point with respect to each parameter.
So, in essence, the gradient is a generalization of the first
derivative to functions with multiple parameters.

is "loss function" just "cost function"?

Yes, in the context of machine learning and neural networks, the terms
"loss function" and "cost function" are often used interchangeably.
Both refer to a function that computes the difference between the
network's predicted output and the actual target output for a given
input data. This difference or "cost" is what the network aims to
minimize during the learning process.

so how should I change this code if I want to explicitly name the
gradient of the loss function?
```go
		errors[i] = target - outputs[i]
		Assert(!math.IsNaN(errors[i]), Spf("error is NaN, target: %v, output: %v", target, outputs[i]))
		// accumulate total cost
		n.cost += 0.5 * math.Pow(target-outputs[i], 2)
```

If you want to explicitly name the gradient of the loss function, you
can introduce a new variable. The gradient of the mean square error
loss function is the derivative of `0.5 * (target-output)^2` with
respect to `output`, and it equals `output - target`. 

Here is how you can modify your code:

```go
gradient := target - outputs[i]	
errors[i] = gradient
Assert(!math.IsNaN(errors[i]), Spf("error is NaN, target: %v, output: %v", target, outputs[i]))
// accumulate total cost
n.cost += 0.5 * math.Pow(target-outputs[i], 2)
```

Now the variable `gradient` explicitly represents the gradient of the loss function.


the Layer backprop() method in tanis.go has outputErrs as a parameter.
what should I rename that to?

You can rename the `outputErrs` parameter to `outputGradients` or
`lossGradients`. This is because during backpropagation, what is being
passed around are not just errors but gradients of the loss function
with respect to outputs. Thus, these names would be more descriptive
of the data being held in the variable.




recommend API improvements and simplifications in x/node/node.go.  see
the test cases in x/node/node_test.go as an example of the complexity
i'd like to simplify.  for example, i'm thinking of some combination
of these things:
- move topic and aggregator creation and management into the node
- add a node.Subscribe() method that returns an output channel
- have all node i/o be channels

Based on the provided context, here are some suggestions to improve
and simplify the API in `x/node/node.go`:

1. Encapsulation: You can create a more organized and simplified
   interface by handling the internal actions such as topic and
   aggregator creation and management inside the node. It achieves
   better data encapsulation, which would help to prevent the misuse
   of the data by the rest of the program.

3. Node Subscription Method: Adding a `node.Subscribe()` method that
   returns an output channel would provide a cleaner way to manage
   output data. This approach will also make it easier for other
   components that use the node to handle its output without needing
   to understand the internal workings of the node.

4. Channels for I/O: Making all input and output for nodes to be
   channels would simplify the interaction with nodes. It is an
   effective way to synchronize and exchange data between goroutines.
   By having all the node I/O be channels, you would allow
   asynchronous executions and it would make the node self-contained.


proposal:  
- refactor to layerless network
- no initial shape input -- get rid of ./shape
- nodes are persistent goroutines
- a net is represented by its top-level node and is garbage-collected
  when the node is removed from the population

- we break up the components of a neuron into separate functions
    - activation function
    - bias addition
    - weight multiplication
    - summing
- the generic function is func([]float64) []float64
- every function must accept a slice of inputs and return a slice of outputs


- each topic is a pub/sub topic
- each topic is a goroutine that reads from its input channel and fans out to its output channels
- dna creates topics, one topic for each node
    - each node has one output, connected to one topic
    - a node is a topic
- a subscriber connects each input to a topic

- each node publishes a pub/sub topic
    - the topic number has a unique uint64 in the dna
- each node fans its output out to its output channels
- a node subscribes to one or more topics for each input
    - each input has a topic number in the dna
    - this is the topic number
- create all the nodes (topics) first, put them all in a map
- then tell each node to subscribe to its inputs, passing in the map
    - if a topic number is not in the map, then abort subscriber startup 
    - a subscriber reads a channel from the publishing node
- then tell each node to start publishing


- word generator reads src, has these methods:
    - ReadByte()
    - ReadInt64()



- an instruction starts with a marker.  this is followed by N varints
  that are the opcode and operands.  
  that is the length N of the instruction.  this is followed by N
  varints that are the 

  is followed by a byte that is the function code.  this is followed
  by a float64 that is the parameter.  this is followed by a
  compressed bitmask of connections to prior nodes' outputs.

- each dna instruction includes a bitmask of connections to prior nodes' outputs
    - easy to verify that the dna is valid by just making sure that
      the bitmask is non-zero

- every instruction is a node creation
- instruction contents:
    - function (activation function, bias addition, weight multiplication, summing)
        - byte
        - mod this to get opcode
    - parameter 
        - float64 
    - compressed bitmask of connections to prior nodes' outputs
        - compression needs to support mutation
        - a sequence of ones represents a one in the uncompressed data
            - the location of the one = 
                (current node) - (length of the sequence of ones)^10 mod (number of prior nodes)
                XXX nope, too dependent on the number of prior nodes
        - maybe we don't compress this at all
            






- dna is a matrix of connections
    - each cell is a function with a number of parameters





- each edge is a pub/sub channel
- each edge is a goroutine that reads from its input channel and fans out to its output channels

- each node is a goroutine that reads from its input channels, does math, and writes to its output channel

- a nodedge is a goroutine that reads from its input channels, does math, and fans out to its output channels

- a net is y... = f(x...)
- a subnet is a goroutine that reads x... from its input channels, does math, and fans out y... to its output channels

- the generic function is func([]float64) []float64
- a generic node is NewNode(func([]float64) []float64) ([]chan float64, []chan []float64, chan error)


- functions can be passed through channels
- currying can be used to pass parameters through channels
- a dna message is a curry of a function and its parameters

- a generic node is  a configuration message and returns a function
    - NewNode(conf chan func([]float64) float64) (chan []float64, chan error)



- we assume that every node added gets output from every existing node
- the dna is a list of node additions and edge deletions


- dna is a fluent-style program, each instruction referencing the
  most recent node mentioned


- we assume that all node inputs are connected to all node outputs
- dna is a set of deletions

- dna is a list of edges and nodes








- the first N nodes are the inputs, the last M nodes are the outputs
    - minimum DNA size is (N+M)x4






- each node spawns a goroutine that then executes its own dna after startup

    - func NewNode(dna []byte, outputs chan []float64) (inputs []chan float64, errs chan error)
    - node gets dna, starts upstream nodes, giving them their dna along with new channels for
      their outputs, waits for the startup to return their input channels, then starts goroutine and returns its
      own input channels to caller
    - node might not keep all output channels it was given at
      startup; it might pass some of them on to other nodes it
      creates
    - node starts at most one upstream node for each output channel it was given at startup
    - goroutine waits for a message on each input channel, does math, sends copy of output
      message to each output channel

    - func NewNode(dna []byte, inputCount int, inputs []chan float64, outputs []chan float64, errs chan error)
    - goroutine accepts one input value on each input channel, does math, sends copy of output
      value to each output channel
    - constructor might not keep all output channels it was given;
      it might pass some of them upstream


    - constructor gets dna, starts one upstream node for each output channel, giving them their dna along with new channels, 
      waits for their startups to return, then starts goroutine.


- as a node, at startup we know:
    - how many input channels we have been given (one for each input value)
    - how many output channels we have been given (one for each
      downstream node)
    - how many nodes we need to create (one for each output channel)

- as a node, from dna we need to know:
    - which input channels we keep
    - activation, weight, and bias for each input we keep
    - which upstream nodes get which input channels we don't keep
    - which upstream nodes get how many outputs 


- each node starts a goroutine
    - func NewNode(inputs []chan float64) (outputs []chan float64, errors chan error)
- each node is given one channel for each input
- each node returns one channel for output
- each node is given a channel for error output
- end of input is signaled by closing the input channel
- a node waits for a message on each input before firing a message on
  the output channel, then closes the output channel
- the output collector waits for a message on each output channel
  before returning the outputs
- genome is stored on disk -- no more json
- each input has its own weight and bias
    - weight defaults to 1.0, bias defaults to 0.0
    - later instructions can change the weight and bias
- there is no initial net -- world starts with totally random dna
- the first N nodes are the inputs, the last M nodes are the outputs
    - minimum DNA size is (N+M)x4
- breed needs to crossover at statement boundaries
- there is no clean()
- the dna language is incapable of creating an invalid net
    - no node can have more than one input from the same node
- statements are XXX bytes long



- the dna language has an (OpInput nodeNum) opcode
    - creates a new input and connects it to an earlier node 
    - the nodeNum is the index of the node in the dna sequence, modulo
      the number of nodes

- first dna byte is version number
    - there is no other header in v1



- the dna language uses 256-bit words
    - first byte is opcode

    - each opcode is one byte
    - each operand is one byte
    - each operand is a signed int
    - the dna language includes an (OpSkip int) opcode
    - jumps forward in the dna sequence by int bytes or end, whichever
      comes first
    - allows keeping junk dna in the genome
- the dna language includes an (OpSkip int) opcode
    - jumps forward in the dna sequence by int bytes or end, whichever
      comes first
    - allows keeping junk dna in the genome

- OpInput
    - 0-0: opcode
    - 1-2: node number
    - 3-4: input node number
    - 5-5: 
    - 4-11: weight
    - 12-15: bias
    - first byte is the opcode
    - second byte is the node number
    - third byte is the input number
    - fourth byte is the weight
    



    



