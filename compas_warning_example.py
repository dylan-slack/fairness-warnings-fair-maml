import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from Fairness_Warnings import *

import torch
from torch import nn
from torch.nn import functional as F

np.random.seed(12321)
torch.manual_seed(12321)

minibatch = 32
metric = "eo"

if metric == 'dp':
	gamma = .5
	reg = disparate_impact_reg
else:
	gamma = 1
	reg = equal_op_reg

PROTECTED_CLASS = 1
UNPROTECTED_CLASS = 0
POSITIVE_OUTCOME = 1
NEGATIVE_OUTCOME = 0 

class Model(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(Model, self).__init__()
		self.fc1 = nn.Linear(dim_in, 40)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(40, 40)
		self.relu2 = nn.ReLU()
		self.logits = nn.Linear(40, dim_out)

	def forward(self, x):
		l1 = self.relu1(self.fc1(x))
		l2 = self.relu2(self.fc2(l1))
		return self.logits(l2)

	def predict(self, x):
		if not isinstance(x, torch.Tensor):
			x = totorch(x, 'cpu').float()

		return torch.argmax(F.softmax(self.forward(x),dim=1),dim=1).data.numpy()

if __name__ == "__main__":
	X, y, categorical_cols = get_and_preprocess_compas_data() 
	features = list(X)
	race_indc = features.index('race')
	X = X.values

	xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.1)

	##
	# Train base model
	##

	xtrain = totorch(xtrain,'cpu',grad=True).float()
	xtest = totorch(xtest,'cpu').float()
	ytrain = totorch(ytrain,'cpu').long()

	classifier = Model(X.shape[1], 2)
	loss_f = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
	
	for _ in range(5):
		perm = np.random.permutation(xtrain.shape[0])
		for i in range(0,xtrain.shape[0],minibatch):

			loss = loss_f(classifier(xtrain[perm][i:i+minibatch]), ytrain[perm][i:i+minibatch]) + \
					gamma * reg(ytrain[perm][i:i+minibatch], 
												 classifier(xtrain[perm][i:i+minibatch]), 
												 xtrain[perm][i:i+minibatch][:,race_indc]) 

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	print ("Base Model:")
	print ("-----------")

	y_hat = torch.argmax(F.softmax(classifier(xtest),dim=1),dim=1).data.numpy()
	print("Accuracy:",np.sum(y_hat == ytest) / len(ytest))

	if metric == "dp":
		threshold=0.8
		print("DP:",disparate_impact(ytest, y_hat, xtest[:,race_indc].data.numpy()))
	elif metric == "eo":
		threshold=0.7
		print("EO:",equal_opp(ytest, y_hat, xtest[:,race_indc].data.numpy()))
	else:
		raise NotImplementedError("Metric {} not implemented yet.".format(metric))

	##
	# Train fairness warning
	##

	print ("Training Fainess Warning:")
	print ("-----------")

	fw = Fairness_Warnings(classifier,metric=metric)
	fw.fit(X, y, X[:,race_indc], features, categorical_features=categorical_cols, threshold=threshold, verbose=True)






	