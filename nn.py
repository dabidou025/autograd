import inspect
import numpy as np
from tensor import Tensor, relu, sigmoid


class Parameter(Tensor):

	def __init__(self, *shape):
		
		data = np.random.randn(*shape)
		
		super().__init__(data, rg=True)


class Layer():
	
	def __call__(self, *args, **kwargs):
		
		return self.forward(*args, **kwargs)


	def parameters_gen(self):
		
		for _, value in inspect.getmembers(self):
			if isinstance(value, Tensor) and value.rg:
				yield value

	
	def parameters(self):
		return list(self.parameters_gen())

	
class Linear(Layer):

	def __init__(self, in_shape, out_shape, bias=True):

		self.W = Parameter(in_shape, out_shape)
		self.b = Parameter(out_shape) if bias else None

	
	def forward(self, x):
		
		return x @ self.W + self.b if self.b is not None else x @ self.W


# class ReLU(Layer):

# 	def forward(self, x):
		
# 		return relu(x)


# class Sigmoid(Layer):
    
#     def forward(self, x):
        
#         return sigmoid(x)


class Model:

	def __call__(self, *args, **kwargs):
		
		return self.forward(*args, **kwargs)


	def parameters_gen(self):

		for _, value in inspect.getmembers(self):

			if isinstance(value, Tensor) and value.rg:
				yield value

			elif isinstance(value, Layer):
				yield from value.parameters_gen()

			elif isinstance(value, Model):
				yield from value.parameters_gen()


	def parameters(self):
		return list(self.parameters_gen())


class Sequential(Model):

	def __init__(self, *layers):

		self.layers_keys = range(len(layers))
		for i, layer in enumerate(layers):
			self.__dict__[str(i)] = layer


	def forward(self, x):

		for key in self.layers_keys:
			x = self.__dict__[str(key)](x)

		return x