package tanis

/*

XXX reconcile the following with the code

	// We use the chain rule to calculate the partial derivative of the cost
	// function with respect to the weight:
	//
	// XXX
	// dcost/dweight = dcost/doutput * doutput/dweight
	//
	// The cost function is the sum of the squares of the errors.  We
	// multiply by 0.5 to simplify the derivative.
	//
	// cost = 0.5 * (target - output)^2
	//
	// The derivative of the cost function with respect to the output
	// is simply (target - output).
	//
	// dcost/doutput = target - output

	// The derivative of the output with respect to the weighted input
	// is the derivative of the activation function. Why?  Because
	// the output is the weighted sum of the inputs, which is the
	// XXX



	// XXX
	// doutput/dweight = activationD1(output)

	// The derivative of the weighted sum with respect to a weight is
	// the output of the related upstream node.  Why?  Because the
	// weighted sum is the sum of the products of each weight and the
	// output of the correspending upstream node.  To simplify, if for
	// example we only had one upstream node, then the weighted sum
	// would be:
	//
	// weightedsum = weight * input
	//
	// We can't change the upstream node output, so we can only change
	// the weight.  So we can think of the above equation in the form
	// of:
	//
	// y = mx + b
	//
	// where y is the weighted sum, m is the weight, x is the upstream
	// node output, and b is the bias.
	//
	// The derivative of that equation with respect to m is x:
	//
	// dy/dm = x
	//
	// So the derivative of the weighted sum with respect to a weight
	// is simply the input to the node.
	//
	// dweightedsum/dweight = input
	//
	// So going back to the original equation using the partial
	// derivatives in the chain rule:
	//
	// dcost/dweight = dcost/doutput * doutput/dweightedsum * dweightedsum/dweight
	//               = (target - output) * activationD1(output) * input
	//
	// Let's do it in code:
	for i, upstreamNode := range n.Upstream {
		dweightedsum_dweight := upstreamNode.Output()
		dcost_dweight := dcost_doutput * doutput_dweightedsum * dweightedsum_dweight
		// adjust weight
		n.Weights[i] += dcost_dweight

		// Now we need to adjust the upstream node weights.  We do
		// this by calling the upstream node's backprop function. but
		// we need a target to pass to Backprop().  Let's ask the upstream
		// node what its target is, passing in what we already know:
		targetUpstream := upstreamNode.Target(dweightedsum_dweight)
		// Now we can call the upstream node's backprop function:
		upstreamNode.Backprop(targetUpstream)
	}
*/
