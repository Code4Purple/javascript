
'''

We’re looking to get this model to be a classifier, so we want an activation function
meant for classification. One of these is the Softmax activation function

In the case of classification,
what we want to see is a prediction of which class the network “thinks” the input represents. This
distribution returned by the softmax activation function represents confidence scores for each
class and will add up to 1. The predicted class is associated with the output neuron that returned
the largest confidence score. Still, we can also note the other confidence scores in our overarching
algorithm/program that uses this network. For example, if our network has a confidence
distribution for two classes: [0.45, 0.55], the prediction is the 2nd class, but the confidence in
this prediction isn’t very high. Maybe our program would not act in this case since it’s not very
confident.

'''

# Good classification Softmax ( without numpy )
print('-- Without Numpy --')
# Values from the previous output when we described
# what a neural network is
layer_outputs = [4.8, 1.21, 2.385]

# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercased
E = 2.71828182846 # you can also use math.e

# For each value in a vector, calculate the exponential value
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output) # ** - power operator in Python
print('exponentiated values:           ',exp_values)
#print(exp_values) # the valus should be [121.51041751893969, 3.3534846525504487, 10.85906266492961]

# Now normalize values
norm_base = sum(exp_values) # We sum all values

norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)

print('Normalized exponentiated values:', norm_values)
#print(norm_values)
print('Sum of normalized values:       ', sum(norm_values))

# Good Classification Softmax ( with numpy )
import numpy as np
# Values from the earlier previous when we described

# what a neural network is
layer_outputs = [4.8, 1.21, 2.385]

# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('\n-- With Numpy --')
print('exponentiated values:            ', exp_values)
#print(exp_values)

# Now normalize values
norm_values = exp_values / np.sum(exp_values)
print('normalized exponentiated values: ', norm_values)
#print(norm_values)
print('sum of normalized values:        ', np.sum(norm_values))

'''

exponentiated values:
[121.51041752 3.35348465 10.85906266]
normalized exponentiated values:
[0.89528266 0.02470831 0.08000903]
sum of normalized values: 0.9999999999999999

'''

# Let’s see some examples of how axis affects the sum using NumPy. First, we will just show the default, which is None.
 
#import numpy as np
layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])
print('\nSum without axis:')
print(np.sum(layer_outputs))
print('This will be identical to the above since default is None:')
print(np.sum(layer_outputs, axis=None))

'''

Sum without axis
18.172
This will be identical to the above since default is None:
18.172

'''

print('\nAnother way to think of it w/ a matrix == axis 0: columns:')
print(np.sum(layer_outputs, axis=0))

'''

Another way to think of it w/ a matrix == axis 0: columns:
[15.11 0.451 2.611]

'''

print('\nBut we want to sum the rows instead, like this w/ raw py:')
for i in layer_outputs:
    print(sum(i))

'''

But we want to sum the rows instead, like this w/ raw py:
8.395
7.29
2.4869999999999997

'''

print('\nSo we can sum axis 1, but note the current shape:')
print(np.sum(layer_outputs, axis=1))

'''

So we can sum axis 1, but note the current shape:
[8.395 7.29 2.487]

'''

print('\nSum axis 1, but keep the same dimensions as input:')
print(np.sum(layer_outputs, axis=1, keepdims=True))

'''

Sum axis 1, but keep the same dimensions as input:
[[8.395]
[7.29 ]
[2.487]]

'''

# Entering Softmax into the tranning_data.py & ReLU_activation_function.py

#import numpy as np
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

# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
   
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values  
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

#Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print('')
print(activation2.output[:5])

'''
Output :

[[0.33333334 0.33333334 0.33333334]
[0.33333316 0.3333332 0.33333364]
[0.33333287 0.3333329 0.33333418]
[0.3333326 0.33333263 0.33333477]
[0.33333233 0.3333324 0.33333528]]

'''

# Thus, our next step is to quantify how wrong the model is through what’s defined as a loss function.

#pg. 111 - Chpater 7 