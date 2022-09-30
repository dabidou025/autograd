import numpy as np
from tensor import Tensor


class Optimizer:
	
	def __init__(self, params, lr):
		
		self.params = params
		self.lr = lr
		self.n_params = len(params)
		

	def reset_grads_and_deps(self):

		for param in self.params:

			param.grad = None
			param.deps = None


class SGD(Optimizer):

	def __init__(self, params, lr):

		super().__init__(params, lr)	

	
	def step(self):

		for param in self.params:
			
			param.data = param.data - self.lr*param.grad


class Adam(Optimizer):

	def __init__(self, params, lr, betas, eps):
		
		super().__init__(params, lr)

		self.betas = betas
		self.eps = eps

		self.m = [np.zeros(param.shape) for param in self.params]
		self.v = [np.zeros(param.shape) for param in self.params]
		
		self.t = 1

	def step(self):

		for i in range(self.n_params):

			self.m[i] = self.betas[0]*self.m[i] + (1-self.betas[0])*self.params[i].grad

			self.v[i] = self.betas[1]*self.v[i] + (1-self.betas[1])*(self.params[i].grad)**2

			self.params[i].data = self.params[i].data - self.lr * (self.m[i] / (1 - self.betas[0]**self.t)) / (np.sqrt(self.v[i] / (1 - self.betas[1]**self.t)) + self.eps)

		self.t += 1

