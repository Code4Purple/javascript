# A Single Neuron

# 3 Inputs , 3 Weights, 1 Bias
inputs = [1,2,3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = (inputs[0] * weights[0] + 
	  inputs[1] * weights[1] + 
	  inputs[2] * weights[2] + bias)

print(output) # it should be 2.3

# 4 Inputs, 4 Weights, 1 Bias
inputs2 = [1.0, 2.0, 3.0, 2.5]
weights2 = [0.2, 0.8, -0.5, 1.0]
bias2 = 2.0

output2 = (inputs2[0] * weights2[0] + 
           inputs2[1] * weights2[1] + 
	   inputs2[2] * weights2[2] + 
	   inputs2[3] * weights2[3] + bias2)

print(output2) # it should be 4.8


