import numpy as np
from tensor import Tensor
from nn import Sequential, Linear


model = Sequential(
	Linear(5,5),
	Linear(5,4),
	Linear(4,2)
)

a = Tensor(np.zeros(5))

b = model(a)
print(a, b, sep='\n')
