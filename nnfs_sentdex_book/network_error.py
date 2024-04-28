# Loss Function

'''

The loss function, also referred to as the cost function, is the algorithm
that quantifies how wrong a model is. Loss is the measure of this metric. Since loss is the model’s
error, we ideally want it to be 0.

'''

# Categorical Cross-Entropy Loss

'''

If you’re familiar with linear regression, then you already know one of the loss functions used
with neural networks that do regression: squared error (or mean squared error with neural
networks).

Categorical cross-entropy is explicitly used to
compare a “ground-truth” probability (y or “targets”) and some predicted distribution (y-hat or
“predictions”), so it makes sense to use cross-entropy here. It is also one of the most commonly
used loss functions with a softmax activation on the output layer.

'''

# Basic Idea

import math

# An example output from the output layer of the neural network
softmax_output = [0.7, 0.1, 0.2]

# Ground truth
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss) # it should be 0.35667494393873245

#pg 116