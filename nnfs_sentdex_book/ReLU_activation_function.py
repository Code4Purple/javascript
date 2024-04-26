
# Basic Example

'''

Despite the fancy sounding name, the rectified linear activation function is straightforward to
code. Most closely to its definition:

'''
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)

print(output) # it should be [0, 2, 0, 3.3, 0, 1.1, 2.2, 0]

'''

We made up a list of values to start. The ReLU in this code is a loop where we’re checking if the
current value is greater than 0. If it is, we’re appending it to the output list, and if it’s not, we’re
appending 0. This can be written more simply, as we just need to take the largest of two values: 0
or neuron value. For example:

'''

# Using the same inputs from above
output2 = []
for i in inputs: 
    output2.append(max(0, i))

print(output2) # it should be [0, 2, 0, 3.3, 0, 1.1, 2.2, 0]

# This Code Below adds to the tranning_data.py file :
print() # spacing the outputs

import numpy as np
# Where we are getting the data from
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()

# Dense Layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

# Let's see output of the first few samples:
print("The print out from tranning_data.py:")
print(dense1.output[:5])
print("")
print("The print after the activation function happens:")
print(activation1.output[:5])

'''

As you can see, negative values have been clipped (modified to be zero). That’s all there is to the
rectified linear activation function used in the hidden layer. Let’s talk about the activation function
that we are going to use on the output of the last layer.

'''