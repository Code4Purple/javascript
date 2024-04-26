
import numpy as np
# Where we are getting the data from
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()

'''
    So far, we’ve only used what’s called a dense or fully-
connected layer. These layers are commonly referred to as “dense” layers in papers, literature,
and code, but you will occasionally see them called fully-connected or “fc” for short in code. Our
dense layer class will begin with two methods.

'''

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

'''

    Next, we have the forward method. When we pass data through a model from beginning to end, this is called a
    forward pass. Just like everything else, however, this is not the only way to do things. You can
    have the data loop back around and do other interesting things. We’ll keep it common and perform
    a regular forward pass.

'''

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(dense1.output[:5]) # the output should be..

'''
Output should be...

[[ 0.0000000e+00 0.0000000e+00 0.0000000e+00 ]
 [-1.0475188e-04 1.1395361e-04 -4.7983500e-05]
 [-2.7414842e-04 3.1729150e-04 -8.6921798e-05]
 [-4.2188365e-04 5.2666257e-04 -5.5912682e-05]
 [-5.7707680e-04 7.1401405e-04 -8.9430439e-05]]                      

'''


