# What are “tensors?”
'''
Tensors are closely-related to arrays. If you interchange tensor/array/matrix when it comes
to machine learning, people probably won’t give you too hard of a time. But there are subtle
differences, and they are primarily either the context or attributes of the tensor object. To
understand a tensor, let’s compare and describe some of the other data containers in Python
(things that hold data). Let’s start with a list. A Python list is defined by comma-separated objects
contained in brackets. So far, we’ve been using lists.
 '''


# Simple List Example
l = [1,5,6,2]

# A list of lists
lol = [[1,5,6,2],[3,2,1,3]]

# A list of lists of lists!
lolol = [[[1,5,6,2], [3,2,1,3]], [[5,2,1,2], [6,4,8,4]], [[2,8,5,3], [1,1,9,4]]]


'''A matrix is pretty simple. It’s a rectangular array. It has columns and rows. It is two dimensional.
So a matrix can be an array (a 2D array). Can all arrays be matrices? No. An array can be far more
than just columns and rows, as it could have four dimensions, twenty dimensions, and so on.'''

# A matrix
list_matrix_array = [[4,2],
                     [5,1],
                     [8,2]]

'''We need to learn one more notion — a vector. Put simply, a vector in math is what we call a list
in Python or a 1-dimensional array in NumPy. Of course, lists and NumPy arrays do not have
the same properties as a vector, but, just as we can write a matrix as a list of lists in Python, we
can also write a vector as a list or an array! Additionally, we’ll look at the vector algebraically
(mathematically) as a set of numbers in brackets. This is in contrast to the physics perspective,
where the vector’s representation is usually seen as an arrow, characterized by a magnitude and a
direction.'''

'''Let’s now address vector multiplication, as that’s one of the most important operations we’ll
perform on vectors. We can achieve the same result as in our pure Python implementation of
multiplying each element in our inputs and weights vectors element-wise by using a dot product,
which we’ll explain shortly. Traditionally, we use dot products for vectors (yet another name for
a container), and we can certainly refer to what we’re doing here as working with vectors just as
we can call them “tensors.”'''

# Let’s write out how a dot product is calculated in Python.

# Dot product
a = [1, 2, 3] # we can call this inputs vector
b = [2, 3, 4] # we can call this weights vector 

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
print(dot_product) # it should be 20