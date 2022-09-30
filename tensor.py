import numpy as np
from collections import namedtuple

Dep = namedtuple('Dep', ['parent', 'grad_fun'])

class Tensor:

	def __init__(self, data, rg=False, deps=None):

		self.data = np.copy(data)
		self.shape = data.shape

		self.rg = rg
		self.grad = None

		self.deps = deps


	def __str__(self):
		
		ret = f"Tensor of shape {self.data.shape} :\n"
		ret += np.array2string(self.data)

		return ret

	
	def zero_grad(self):
		
		self.grad = None


	def backward(self, grad=None):

		assert self.rg, "Called backward on non-rg Tensor"

		if grad is None:
						
			assert self.shape == (), "You must specify grad for Tensors of shape ()"	
	
			grad = np.ones(self.shape)

		if self.grad is None:
			self.grad = np.copy(grad)
		else:
			self.grad += grad

		if self.deps is not None:
			for dep in self.deps:
				vjp = dep.grad_fun(self.grad)
				dep.parent.backward(vjp)


	
	def __add__(self, other):

		if isinstance(other, Tensor):
			
			data = self.data + other.data
			rg = self.rg or other.rg

			if rg:
				deps = []
				if self.rg:
					def grad_fun(grad):
						if self.shape == (1,):
							grad = grad.sum(keepdims=True)[0]
						return grad.sum(axis=tuple(range(grad.ndim - self.data.ndim)))
					deps.append(Dep(self, grad_fun))
				if other.rg:
					def grad_fun(grad):
						if other.shape == (1,):
							grad = grad.sum(keepdims=True)[0]
						return grad.sum(axis=tuple(range(grad.ndim - other.data.ndim)))
					deps.append(Dep(other, grad_fun))
			else:
				deps = None

		else:

			data = self.data + other
			rg = self.rg

			if rg:
				def grad_fun(grad):
					if self.shape == (1,):
						grad = grad.sum(keepdims=True)[0]
					return grad.sum(axis=tuple(range(grad.ndim - self.data.ndim)))
				deps = [Dep(self, grad_fun)]
			else:
				deps = None


		return Tensor(data, rg, deps)


	def __radd__(self, other):

		return self.__add__(other)


	def __sub__(self, other):

		if isinstance(other, Tensor):
			
			data = self.data - other.data
			rg = self.rg or other.rg

			if rg:
				deps = []
				if self.rg:
					def grad_fun(grad):
						if self.shape == (1,):
							grad = grad.sum(keepdims=True)[0]
						return grad.sum(axis=tuple(range(grad.ndim - self.data.ndim)))
					deps.append(Dep(self, grad_fun))
				if other.rg:
					def grad_fun(grad):
						if other.shape == (1,):
							grad = grad.sum(keepdims=True)[0]
						return -grad.sum(axis=tuple(range(grad.ndim - other.data.ndim)))
					deps.append(Dep(other, grad_fun))
			else:
				deps = None

		else:

			data = self.data - other
			rg = self.rg

			if rg:
				def grad_fun(grad):
					if self.shape == (1,):
						grad = grad.sum(keepdims=True)[0]
					return grad.sum(axis=tuple(range(grad.ndim - self.data.ndim)))
				deps = [Dep(self, grad_fun)]
			else:
				deps = None

		return Tensor(data, rg, deps)


	def __rsub__(self, other):
	
		if isinstance(other, Tensor):
			
			data = other.data - self.data
			rg = self.rg or other.rg

			if rg:
				deps = []
				if self.rg:
					def grad_fun(grad):
						if self.shape == (1,):
							grad = grad.sum(keepdims=True)[0]

						return -grad.sum(axis=tuple(range(grad.ndim - self.data.ndim)))
					deps.append(Dep(self, grad_fun))
				if other.rg:
					def grad_fun(grad):
						if other.shape == (1,):
							grad = grad.sum(keepdims=True)[0]
						
						return grad.sum(axis=tuple(range(grad.ndim - other.data.ndim)))
					deps.append(Dep(other, grad_fun))
			else:
				deps = None

		else:

			data = other - self.data
			rg = self.rg

			if rg:
				def grad_fun(grad):
					if self.shape == (1,):
						grad = grad.sum(keepdims=True)[0]
					return -grad.sum(axis=tuple(range(grad.ndim - self.data.ndim)))
				deps = [Dep(self, grad_fun)]
			else:
				deps = None

		return Tensor(data, rg, deps)


	def __mul__(self, other):
	
		if isinstance(other, Tensor):
				
			data = self.data * other.data
			rg = self.rg or other.rg

			if rg:
				deps = []
				if self.rg:
					def grad_fun(grad):
						grad = grad * other.data
						if self.shape == (1,):
							grad = grad.sum(keepdims=True)[0]
						return grad.sum(axis=tuple(range(grad.ndim - self.data.ndim)))
					deps.append(Dep(self, grad_fun))
				if other.rg:
					def grad_fun(grad):
						grad = grad * self.data
						if other.shape == (1,):
							grad = grad.sum(keepdims=True)[0]
						return grad.sum(axis=tuple(range(grad.ndim - other.data.ndim)))
					deps.append(Dep(other, grad_fun))
			else:
				deps = None

		else:

			data = self.data * other
			rg = self.rg

			if self.rg:
				def grad_fun(grad):
					grad = grad * other
					if self.shape == (1,):
						grad = grad.sum(keepdims=True)[0]
					return grad.sum(axis=tuple(range(grad.ndim - self.data.ndim)))
				deps = [Dep(self, grad_fun)]
			else:
				deps = None

		return Tensor(data, rg, deps)

	
	def __rmul__(self, other):
		
		return self.__mul__(other)


	def __truediv__(self, other):

		assert (not isinstance(other, Tensor)), "Can't divide a Tensor by another Tensor"

		return self.__mul__(1/other)

	
	def __neg__(self):

		return self.__mul__(-1)

	
	def __pow__(self, other):

		assert (not isinstance(other, Tensor)), "Can't raise a Tensor to the power of another Tensor"

		data = self.data ** other
		rg = self.rg
		
		if rg:
			def grad_fun(grad):
				return grad * other * (self.data)**(other-1)
			deps = [Dep(self, grad_fun)]

		else:
			deps = None

		return Tensor(data, rg, deps)

	
	def __matmul__(self, other):

		assert (isinstance(other, Tensor)), "Can't @ non-Tensors"

		data = self.data @ other.data
		rg = self.rg or other.rg

		if rg:
			deps = []
			if self.rg:
				def grad_fun(grad):
					return grad @ other.data.T
				deps.append(Dep(self, grad_fun))
			if other.rg:
				def grad_fun(grad):
					return self.data.T @ grad
				deps.append(Dep(other, grad_fun))
		else:
			deps = None

		return Tensor(data, rg, deps)


	def __getitem__(self, idxs):
		
		data = self.data[idxs]
		rg = self.rg

		if rg:
			def grad_fn(grad):
				bigger_grad = np.zeros_like(self.data)
				bigger_grad[idxs] = grad
				return bigger_grad

			deps = Dep(self, grad_fn)
		else:
			deps = None

		return Tensor(data, rg, deps)		


	def sum(self):

		data = self.data.sum()
		rg = self.rg
		deps = [Dep(self, lambda x : x * np.ones(self.shape))] if rg else None

		return Tensor(data, rg, deps)

	
	def mean(self):
		
		return self.sum() / np.prod(self.shape)
	
	
	def T(self):
		
		data = np.transpose(self.data)
		rg = self.rg
	
		if rg:
			def grad_fun(grad):
				return np.transpose(grad)
			deps = [Dep(self, grad_fun)]
		else:
			deps = None

		return Tensor(data, rg, deps)


def log(tensor):

	data = np.log(tensor.data)
	rg = tensor.rg

	if rg:
		def grad_fun(grad):
			return grad * (1 / tensor.data)
		deps = [Dep(tensor, grad_fun)]
	else:
		deps = None

	return Tensor(data, rg, deps)

	
def relu(tensor):

	data = np.maximum(0, tensor.data)
	rg = tensor.rg

	if rg:
		def grad_fun(grad):
			return grad * (tensor.data > 0).astype("float")
		deps = [Dep(tensor, grad_fun)]
	else:
		deps = None

	return Tensor(data, rg, deps)


def my_sigmoid(x):
	
	return 1 / (1 + np.exp(-x))


def sigmoid(tensor):
	
	data = my_sigmoid(tensor.data)
	rg = tensor.rg

	if rg:
		def grad_fun(grad):
			return grad * data * (1 - data)
		deps = [Dep(tensor, grad_fun)]
	else:
		deps = None

	return Tensor(data, rg, deps)


def clip(tensor, inf=None, sup=None):

	data = np.clip(tensor.data, inf, sup)
	rg = tensor.rg

	if rg:
		def grad_fun(grad):
			cond = np.ones(data.shape)
			if inf is not None:
				cond = (tensor.data >= inf).astype("float")
			if sup is not None:
				cond = cond * (tensor.data <= sup).astype("float")
			return grad * cond
		deps = [Dep(tensor, grad_fun)]
	else:
		deps = None

	return Tensor(data, rg, deps)


	