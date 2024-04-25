# A Layer of Neurons with NumPy

''' In plain Python, we wrote this as a list of lists. With NumPy, this will be a 2-dimensional array, which we’ll call a matrix. Previously with the 3-neuron 
example, we performed a multiplication of those weights with a list containing inputs, which resulted in a list of output values — one per each neuron. 
We also described the dot product of two vectors, but the weights are now a matrix, and we need to perform a dot product of them and the input vector. 
NumPy makes this very easy for us — treating this matrix as a list of vectors and performing the dot product one by one with the vectorof inputs, 
returning a list of dot products.'''

import numpy as np 

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layered_outputs = np.dot(weights, inputs) + biases
print(layered_outputs) #array([4.8   1.21  2.385])