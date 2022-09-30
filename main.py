from time import sleep
import numpy as np
from tqdm import tqdm

from tensor import Tensor, log, relu, sigmoid, clip
from nn import Model, Sequential, Linear#, ReLU, Sigmoid
from optimizers import SGD, Adam


def MSE(x, y):
	
	return ((x - y)**2).mean()


def BCE(y_pred, y_true, inf_clip_val=0.00001, sup_clip_val=0.99999):
	
	return -(y_true * log(clip(y_pred, inf_clip_val, sup_clip_val)) + (1 - y_true) * log(clip(1 - y_pred, inf_clip_val, sup_clip_val))).mean()
		

class DNN(Model):
	
	def __init__(self, n_features, h):

		self.linear_1 = Linear(n_features, h)
		self.linear_2 = Linear(h, 1)

	def forward(self, x):
		
		x = self.linear_1(x)
		x = relu(x)
		x = self.linear_2(x)
	
		return x


if __name__ == "__main__":
	
	n_sample = 1000
	n_features = 25
	bs = 64
 
	to_one = np.random.choice(
     	np.arange(n_sample),
      	replace=False,
        size=int(n_sample * 0.5)
	)

	X = np.random.randn(n_sample, n_features)
	X[to_one] += 10
	X = Tensor(X)

	y_true = np.zeros((n_sample, 1))
	y_true[to_one] = 1
	y_true = Tensor(y_true)
	
	h = 10
 
	model = DNN(n_features, h)

	# model = Sequential(
	# 	Linear(n_features, h),
	# 	Linear(h, 1),
	# )

	optim = Adam(
		params=model.parameters(),
		lr=1e-1,
		betas=(0.9, 0.9),
		eps=1e-8
	)

	pbar = tqdm(range(1, 100))
	for epoch in pbar:
     
		epoch_loss = 0
		batchs = range(0, n_sample, bs)
		for i in batchs:

			optim.reset_grads_and_deps()
			
			y_pred = sigmoid(model(X[i:i+bs]))

			loss = BCE(y_pred, y_true[i:i+bs])

			loss.backward()
			optim.step()

			epoch_loss += loss.data.item()

		# pbar.set_postfix({"loss": round(epoch_loss/len(batchs), 3)})
		
	print(f"Loss = {loss.data.item():.2e}")
 
	y_pred = sigmoid(model(X))
	print("Accuracy =", ((y_true.data == 1) == (y_pred.data > 0.5)).mean())

