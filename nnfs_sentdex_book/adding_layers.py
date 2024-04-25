# Adding Layers

import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

'''
    As previously stated, inputs to layers are either inputs from the actual dataset you’re training with
or outputs from a previous layer. That’s why we defined 2 versions of weights and biases but only
1 of inputs — because the inputs for layer 2 will be the outputs from the previous layer

'''
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs) # it should be...
                      # [[ 0.5031 -1.04185 -2.03875],
                      # [ 0.2434 -2.7332 -5.7633 ],
                      # [-0.99314 1.41254 -0.35655]])

# pg 62 bookmark on tranning data