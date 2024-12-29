"""
Operations On Tensors:
Over 100 tensor operations, including arithmetic, linear algebra, 
matrix manipulation (transposing, indexing, slicing), sampling and more 
"""
# By default the tensors are creared in the CPU,
# We can move the tenosr to the GPU using the 
# to method 

import torch 

tensor = torch.rand(2,3)

# check if the cuda is available
if torch.cuda.is_available():
	tensor = tensor.to("cuda")
	print("Tensor in GPU")

# Indexing and slicing of tensors 
data = [[1,2,5,6],
	[3,4,7,8]]
# convert data into tensor 
tensor = torch.tensor(data) 
print(f"First row: {tensor[0]}")

# Note :
# For Reference, Firts index is for row
# Second index is for the column

print(f"Last row:{tensor[-1]}")
print(f"First column:{tensor[:, 0]}")
print(f"Last column:{tensor[:, -1]}")
print(f"Second column:{tensor[:, 1]}")
