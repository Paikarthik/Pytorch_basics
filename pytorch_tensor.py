"""
Tensor : Specialized data structure similar to array or matrices.
There are similar to Numpy nd-arrays, tensors are optimized to run on gpu
"""

import torch
import numpy as np

# Initializing a Tensor 
# There are several ways through which the tensors can be initialized 

# Directly from data 

data = [[1,2], 
	[3,4]]
x_data = torch.tensor(data) # data type is automaticaaly inferred 
print(x_data)
print(x_data.shape)

# From the numpy array
np_array = np.array(data) # converting list into numpy array
x_np = torch.from_numpy(np_array)

# From another tensor 
# Note: The new tensor retains the property such as
# data type and shape
x_ones = torch.ones_like(x_data) # retains propery of x_data like shape
print(f"Ones Tensor: \n {x_ones} \n")
# and data type unless explicitly overwritten
x_rand = torch.rand_like(x_data, dtype= torch.float) # overrides the data type of the x_data 
print(f"Random Tensor: \n {x_rand} \n")

# with random or constant values 
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# Attributes of the tensor 

"""
Attributes of tensor describe their shape, data type, and the device on which they are stored
"""
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


