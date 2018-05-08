import pandas as pd
import numpy as np
from numpy.linalg import inv


def load_data():
		data = "/Users/salmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv"
		df_train = pd.read_csv(data)
		pos_train = df_train[df_train.y == 1]
		neg_train = df_train[df_train.y == 0]
		return pos_train, neg_train, df_train


def sample_covariance(X):
		means = sample_mean(X)
		X_centred = X - means
		covariance = X_centred.T @ X_centred
		return covariance / (len(X) - 1)


def sample_mean(X):
		n, d = X.shape
		return np.sum(X, axis=0) / n


def brute_force_search(cov, mean, query, n, d, ell=20):
		i_cov = inv(cov)
		q = query.T @ i_cov
		ell_lower = (-ell + n * mean) / n
		ell_upper = (ell + n * mean) / n

		print("Lower :", ell_lower)
		print("Upper :", ell_upper)

		def costFx(x):
				return 0.5 * (x.T @ i_cov @ x) + q.T @ x

		x1 = np.array([ell_lower[0], ell_upper[0]])
		x2 = np.array([ell_lower[1], ell_upper[1]])
		for lower in x1:
				for upper in x2:
						print("values:", lower, upper, costFx(np.array([lower, upper])))


def bnb_search(cov, mean, query, n, d, ell=20):
		i_cov = inv(cov)
		q = query.T @ i_cov
		ell_lower = (-ell + n * mean) / n
		ell_upper = (ell + n * mean) / n

		def costFx(x):
				return 0.5 * (x.T @ i_cov @ x) + q.T @ x

		level = 0
		x_selected = ell_lower
		sample_idx = np.random.choice(list(range(d)), d, replace=False)

		while level < d:
				leaf_idx = sample_idx[level]
				print(leaf_idx)
				x_selected[leaf_idx] = ell_lower[leaf_idx]
				right_leaf = costFx(x_selected)
				x_selected[leaf_idx] = ell_upper[leaf_idx]
				left_leaf = costFx(x_selected)
				if right_leaf > left_leaf:
						x_selected[leaf_idx] = ell_lower[leaf_idx]
				elif left_leaf > right_leaf:
						x_selected[leaf_idx] = ell_upper[leaf_idx]
				else:
						print('Shitttt why?')
				print("level:", level)
				level += 1

		print("optimal: ", x_selected, costFx(x_selected))


def costFx(x, cov, query):
		i_cov = inv(cov)
		q = query.T @ i_cov
		return 0.5 * (x.T @ i_cov @ x) + q.T @ x


pos, neg, _ = load_data()
X = pos.loc[:, ['x1', 'x2']].as_matrix()
n, d = X.shape
e_cov = sample_covariance(X)
e_mean = sample_mean(X)
query = np.array([0.830031, 0.108776])
bnb_search(e_cov, e_mean, query, n, d)

print("1:", costFx(np.array([2.10, -1.50]), e_cov, query))
brute_force_search(e_cov, e_mean, query, n, d)
