import numpy as np
import pandas as pd

import torch
from torch import autograd as ag
from torch.nn import functional as F

import math

# Assuming 1 is positive outcome, 1 is the protected class
PROTECTED_CLASS = 1
UNPROTECTED_CLASS = 0
POSITIVE_OUTCOME = 1
NEGATIVE_OUTCOME = 0 

def disparate_impact(Y, Y_hat, A):
	dp = (np.sum(np.where(A==PROTECTED_CLASS, Y_hat, 0)) / np.sum(A==PROTECTED_CLASS))\
			 / (np.sum(np.where(A==UNPROTECTED_CLASS, Y_hat, 0) / np.sum(A==UNPROTECTED_CLASS)))

	if math.isnan(dp):
		return 0
	else:
		return dp

def equal_opp(Y, Y_hat, A):
	eo = (np.sum(Y_hat[np.logical_and(A==PROTECTED_CLASS,Y==POSITIVE_OUTCOME)]) / np.sum(A==PROTECTED_CLASS))\
				/ (np.sum(Y_hat[np.logical_and(A==UNPROTECTED_CLASS,Y==POSITIVE_OUTCOME)]) / np.sum(A==UNPROTECTED_CLASS))

	if math.isnan(eo):
		return 0
	else:
		return eo

def disparate_impact_reg(y, y_hat, s):
	pred_col = F.softmax(y_hat, dim=1)[:,1]
	disadvantaged_rate = (s.float() * pred_col).mean()
	return (1 - disadvantaged_rate) 

def equal_op_reg(y,y_hat,s):
	pred_col = F.softmax(y_hat, dim=1)[:,1]
	disadvanted_rate = (pred_col * s.float() * y.float()).mean()
	return (1 - disadvanted_rate)

def totorch(x, device,grad=False):
	return ag.Variable(torch.Tensor(x),requires_grad=grad).to(device)

def get_and_preprocess_compas_data():
	"""Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis
	
	Parameters
	----------
	params : Params
	Returns
	----------
	Pandas data frame X of processed data, np.ndarray y, and list of column names
	"""

	compas_df = pd.read_csv("data/compas-scores-two-years.csv", index_col=0)
	compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
							  (compas_df['days_b_screening_arrest'] >= -30) &
							  (compas_df['is_recid'] != -1) &
							  (compas_df['c_charge_degree'] != "O") &
							  (compas_df['score_text'] != "NA")]

	compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
	X = compas_df[['age','c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay']]

	# if person has high score give them the _negative_ model outcome
	y = np.array([NEGATIVE_OUTCOME if val == 1 else POSITIVE_OUTCOME for val in compas_df['two_year_recid']])

	sens = X.pop('race')

	# assign African-American as the protected class
	X = pd.get_dummies(X)
	sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
	X['race'] = sensitive_attr

	# make sure everything is lining up
	assert all((sens == 'African-American') == (X['race'] == PROTECTED_CLASS))
	cols = [col for col in X]
	
	return X, y, [cols.index(val) for val in cols if val not in ['age', 'priors_count']]

def get_MAML_config_cc():
	config = [
		('linear', [5, 96]),
        ('relu', [True]),
        ('linear', [5, 5]),
        ('relu', [True]),
        ('linear', [2, 5])
	]
	return config

import matplotlib.pyplot as plt

def visualize(wl_updated, wl_meta, num):


	# for wl in weight_list:
	wl_updated = [wl[4].cpu().data.numpy() for wl in wl_updated]
	wl_meta = wl_meta[4].cpu().data.numpy()

	plt.ion()

	for wlp in wl_updated:
		for i, wl in enumerate(wlp):
			plt.scatter([j for j in range(len(wl))],wl)

			break
			

	plt.scatter([j for j in range(len(wl_meta[i]))],wl_meta[0],marker='x')
	plt.savefig("pics/{}.png".format(num))
	plt.pause(0.0001)
	plt.clf()






