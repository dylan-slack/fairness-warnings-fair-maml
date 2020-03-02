import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings
import pickle

import time 
warnings.filterwarnings('ignore')

PROTECTED_CLASS = 1
UNPROTECTED_CLASS = 0
POSITIVE_OUTCOME = 1
NEGATIVE_OUTCOME = 0

high_violent_crimes_threshold = 50

def get_and_preprocess_communities_and_crime_data():
	full_cc = pd.read_csv("data/communites_and_crime/communities_and_crime.csv", index_col=0)
	y_col = 'ViolentCrimesPerPop numeric'
	binary_y_col = 'has_high_violent_crimes'
	protected = 's'

	full_cc = full_cc[full_cc[y_col] != "?"]
	full_cc[y_col] = full_cc[y_col].values.astype('float32')

	cols_with_missing_values = []
	for col in full_cc:
		if len(np.where(full_cc[col].values == '?')[0]) >= 1:
			cols_with_missing_values.append(col)	

	full_cc = full_cc.drop(cols_with_missing_values + ['communityname string', 'fold numeric', 'county numeric', 'community numeric'], axis=1)
	
	high_pct_black = np.array([PROTECTED_CLASS if full_cc['racepctblack numeric'].values[i] > full_cc['racePctWhite numeric'].values[i] or \
		(full_cc['racePctWhite numeric'].values[i] > full_cc['racepctblack numeric'].values[i] and full_cc['racepctblack numeric'].values[i] > full_cc['racePctHisp numeric'].values[i] and full_cc['racepctblack numeric'].values[i] > full_cc['racePctAsian numeric'].values[i]) \
		else UNPROTECTED_CLASS for i in range(full_cc.shape[0])])
	
	full_cc = full_cc.drop(['racepctblack numeric', 'racePctWhite numeric', 'racePctAsian numeric', 'racePctHisp numeric'], axis=1)
	
	full_cc[protected] = high_pct_black
	all_states = full_cc['state numeric'].unique()

	tasks = []
	lens = []

	all_states_good = []
	for s in all_states:
		lens.append(len(full_cc[full_cc['state numeric'] == s]))
		if len(full_cc[full_cc['state numeric'] == s]) >= 20:
			state_x = full_cc[full_cc['state numeric'] == s]
			state_y = state_x[y_col].values

			y_cutoff = np.percentile(state_y, high_violent_crimes_threshold)
			has_high_violent_crimes = np.array([NEGATIVE_OUTCOME if val > y_cutoff else POSITIVE_OUTCOME for val in state_y])

			state_s = state_x[protected].values
			state_df = state_x.drop([y_col,'state numeric',protected], axis=1)
			state_x = state_df.values

			cols = state_df.columns
			all_states_good.append(state_x)

			state_x = StandardScaler().fit_transform(state_x)
			tasks.append([state_x, has_high_violent_crimes, state_s])

	all_data = np.vstack(all_states_good)

	std = []
	for j in range(all_data.shape[1]):
		std.append(np.std(all_data[:,j]))

	return tasks, cols 

def get_splits(tasks=None,new=False,num_training_batches = 100, meta_batch_size=8, k=15):
	if tasks == None and new == True:
		raise NameError("Can't have tasks be none and new false.")

	if new:
		leave_out = np.random.choice([i for i in range(len(tasks))],size=5,replace=False)
		leave_in = np.array([i for i in range(len(tasks)) if i not in leave_out])

		with open('data/communites_and_crime/split.pkl', 'wb') as handle:
			pickle.dump({'out':leave_out,'in':leave_in,'tasks':tasks}, handle, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		with open('data/communites_and_crime/split.pkl', 'rb') as handle:
				b = pickle.load(handle)
		leave_out = b['out']
		leave_in = b['in']
		tasks = b['tasks']

	batched_training_tasks = []

	# get training batches
	for b in range(num_training_batches):
		new_batch = []
		x_spts, y_spts, x_qrys, y_qrys, c_spts, c_qrys = [], [], [], [], [], []
		for mb in range(meta_batch_size):
			t = np.random.choice(leave_in)
			task = tasks[t]

			# grap support and qry sets
			spts = np.random.choice(len(task[0]),size=k,replace=False)
			qrys = np.random.choice(len(task[0]),size=k,replace=False)
			x_spts.append(task[0][spts])
			y_spts.append(task[1][spts])
			qrs = task[0][qrys]
			x_qrys.append(qrs)
			y_qrys.append(task[1][qrys])
			c_spts.append(task[2][spts])
			c_qrys.append(task[2][qrys])

			for i in range(len(y_spts)):
				if np.random.choice([True, False]):
					pos_spt = [y_spts[i] == 1]
					neg_spt = [y_spts[i] == 0]

					pos_qry = [y_qrys[i] == 1]
					neg_qry = [y_qrys[i] == 0]

					y_spts[i][pos_spt] = 0
					y_spts[i][neg_spt] = 1

					y_qrys[i][pos_qry] = 0
					y_qrys[i][neg_qry] = 1

		batched_training_tasks.append([np.array(x_spts),np.array(y_spts),np.array(x_qrys),np.array(y_qrys),np.array(c_spts),np.array(c_qrys)])

	# get testing batches
	x_spts, y_spts, x_qrys, y_qrys, c_spts, c_qrys = [], [], [], [], [], []
	for b in leave_out:
		task = tasks[b]
		spts = np.random.choice(len(task[0]),size=k,replace=False)
		qrys = np.array([i for i in range(len(task[0])) if i not in spts])

		x_spts.append(task[0][spts])
		y_spts.append(task[1][spts])
		x_qrys.append(task[0][qrys])
		y_qrys.append(task[1][qrys])
		c_spts.append(task[2][spts])
		c_qrys.append(task[2][qrys])


	batched_testing_tasks = [np.array(x_spts),np.array(y_spts),np.array(x_qrys),np.array(y_qrys),np.array(c_spts),np.array(c_qrys)]
	return leave_out, leave_in, tasks, batched_training_tasks, batched_testing_tasks
