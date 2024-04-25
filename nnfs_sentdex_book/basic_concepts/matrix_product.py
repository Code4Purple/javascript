
'''
The matrix product is an operation in which we have 2 matrices, and we are performing dot
products of all combinations of rows from the first matrix and the columns of the 2nd matrix,
resulting in a matrix of those atomic dot products. 

'''
# Matrix Product Basic Idea
import numpy as np

inputs = [1,2,3]
weights = [2,
           3,
           4]

matrix_product = np.dot(inputs, weights)
print(matrix_product) # it should be 20


# Transposition for the Matrix Product
'''
    Here we introduce the concept of transposition. 
    Transpostion is simply modifies a matrix so that its 
    rows becomes columns and columns become rows.
'''

# Basic arrray a
a = [1, 2, 3]
a_output = np.array(a)
print(a_output) # Should be [1 2 3]

# Expand Dimensions of array a
expand_a_output = np.expand_dims(np.array(a), axis=0)
print(expand_a_output) # it should be [[1, 2, 3]]

# Transpose the array a
transpose_a_output = np.array(a) #np.transpose(a_output)
print(transpose_a_output) # it should be [1 2 3]

# Put it all together 
inputs_z = [[1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]
weights_z = [[0.2, 0.8, -0.5, 1.0],
             [0.5, -0.91, 0.26, -0.5],
             [-0.26, -0.27, 0.17, 0.87]]
biases_z = [2.0, 3.0, 0.5]

output_z = np.dot(inputs_z, np.array(weights_z).T) + biases_z
print(output_z) # it should be [[4.8 1.21 2.385],
                #               [8.9 -1.81 0.2],
                #               [1.41 1.05 0.025]]