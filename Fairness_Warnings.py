import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from copy import deepcopy
from utils import *

from interpretability import SLIM

class Fairness_Warnings(object):
	"""
	Fairness Warnings object. Accepts classifier to perform fairness warnings on

	Parameters:
	------------
	classifier: function
	"""
	def __init__(self, classifier, metric="dp"):
		self.classifier=classifier

		if metric not in ["dp", "eo"]:
			raise NameError("Metric must be dp or eo")
		
		self.metric = metric


	def fit(self, X, Y, A, features, threshold=0.8, categorical_features=[], perturbation_std=1, protected_class=1, num_shifts=2000, verbose=False, test_size=0.2):
		"""
		Trains fairness warning according to training data X, sensitive attribute A, and threshold value. Generates num_shifts perturabtion and assess
		fairness behavior.  We assume all categorical features are binary one-hot encoded. 

		Parameters:
		------------
		X: np.ndarray
		A: np.ndarray
		threshold: float
		categorical_features: list
		perturbation_std: float
		protected_class: int
		num_shifts: int 
		verbose: bool
		test_size: float

		Returns:
		------------
		Self
		"""

		total_shifts, total_fair_binary = [], []

		for nsh in range(num_shifts):
			scaler = StandardScaler()
			scaler.fit(X)

			mean = scaler.mean_
			stdv = scaler.scale_

			shifts = np.random.normal(0, perturbation_std, mean.shape) * stdv
			rands = np.tile(shifts, (X.shape[0], 1))
			p_X = X + rands
			m,n = X.shape

			for i in categorical_features:
				rands[:, i] = 0	

				percent_one = len(np.where(X[:,i]==1)[0]) / len(X[:,i])
				categorical_shift = np.random.normal(percent_one, perturbation_std)

				if categorical_shift < 0:
					categorical_shift = 0
				if categorical_shift > 1:
					categorical_shift = 1

				relative_shift = categorical_shift - percent_one
				shifts[i] = deepcopy(relative_shift)

				new_cat_col = np.random.choice([1,0], p=[abs(categorical_shift), 1-(abs(categorical_shift))], size=X.shape[0])
				p_X[:,i] = new_cat_col

			total_shifts.append(shifts)
			total_fair_binary.append(disparate_impact(Y, self.classifier.predict(p_X), A) >= threshold)

		total_shifts = np.vstack(total_shifts)
		total_fair_binary = np.array(total_fair_binary).astype(int)

		xtrain, xtest, ytrain, ytest = train_test_split(total_shifts, total_fair_binary, test_size=test_size)

		sl = SLIM(time_limit=100, dataset="shifts_training_data", bound=10, C0=1e-3)
		sl.fit(xtrain, ytrain, features, "two_year_recid", binning=False)

		if verbose:
			print ("Fairness warnings training score", round(np.sum(sl.predict(xtrain) == ytrain) / xtrain.shape[0],2))
			print ("Fairness warnings test score", round(np.sum(sl.predict(xtest) == ytest) / xtest.shape[0],2))
			print ("See SLIM logs for more details (1 is unfair).")

		self.slim_warning = sl

		return self

