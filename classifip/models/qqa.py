import pandas as pd
import os
import numpy as np
from cvxopt import solvers, matrix, normal
from numpy.linalg import inv

current_dir = os.getcwd()

data = os.path.join(current_dir, "/Users/salmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv")
df_train = pd.read_csv(data)
pos_train = df_train[df_train.y == 1]
neg_train = df_train[df_train.y == 0]


def sample_covariance(X):
		means = sample_mean(X)
		X_centred = X - means
		covariance = X_centred.T @ X_centred
		return covariance / (len(X) - 1)


def sample_mean(X):
		n, d = X.shape
		return np.sum(X, axis=0) / n


X = pos_train.loc[:, ['x1', 'x2']].as_matrix()
n, d = X.shape
e_cov = sample_covariance(X)
e_mean = sample_mean(X)

query = np.array([0.830031, 0.108776])


def brute_force_search(cov, mean, query, n, d, ell=20):
		i_cov = inv(cov)
		q = query.T @ i_cov
		ell_lower = (-ell + n * mean) / n
		ell_upper = (ell + n * mean) / n

		def costFx(x):
				return 0.5 * (x.T @ i_cov @ x) + q.T @ x

		def forRecursive(lowers, uppers, level, L, optimal):
				for current in np.array([lowers[level], uppers[level]]):
						if level < L - 1:
								forRecursive(lowers, uppers, level + 1, L, np.append(optimal, current))
						else:
								print("optimal value cost:", costFx(np.append(optimal, current)))

		forRecursive(ell_lower, ell_upper, 0, d, np.array([]))


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
				level += 1

		print("optimal: ", x_selected, costFx(x_selected))


def maximum_Fx(cov, mean, query, n, d, ell=20):
		i_cov = matrix(inv(cov))
		b = matrix(query.T @ i_cov)

		def cOptFx(x=None, z=None):
				if x is None:
						return 0, matrix(0.0, (d, 1))
				f = -1 * (0.5 * (x.T * i_cov * x) + b.T * x)
				Df = -1 * (i_cov * x + b).T
				if z is None:
						return f, Df
				H = -1*z[0] * i_cov
				return f, Df, H

		ll_lower = matrix((-ell + n * mean) / n, (d, 1))
		ll_upper = matrix((ell + n * mean) / n, (d, 1))
		print("Q:", i_cov)
		print("q:", b)
		print("lower", ll_lower)
		print("upper", ll_upper)
		I = matrix(0.0, (d, d))
		I[::d + 1] = 1
		G = matrix([I, -I])
		h = matrix([ll_upper, -ll_lower])
		return solvers.cp(cOptFx, G=G, h=h, kktsolver='ldl', options={'kktreg': 1e-6})


def testingLargeDim(n, d):
		def costFx(x, cov, query):
				i_cov = inv(cov)
				q = query.T @ i_cov
				return 0.5 * (x.T @ i_cov @ x) + q.T @ x

		e_mean = np.random.normal(size=d)
		e_cov = normal(d, d)
		e_cov = e_cov.T * e_cov
		query = np.random.normal(size=d)
		q = maximum_Fx(e_cov, e_mean, query, n, d)
		print(q["x"], costFx(np.array(q["x"]), e_cov, query))
		bnb_search(e_cov, e_mean, query, n, d)
		brute_force_search(e_cov, e_mean, query, n, d)


testingLargeDim(20, 5)
# n, d = X.shape
# q = maximum_Fx(e_cov, e_mean, query, n, d)
# print("salmuz-->", q)
# print(q["x"])
# for i in range(1000):
# 		try:
# 			q = maximum_Fx(e_cov, e_mean, query, n, d)
# 			#d = min_convex_query(query, e_mean, e_cov, n, d)
# 			print("salmuz-->", q)
# 			print(q["x"])
# 		except ValueError:
# 				print("errrr")
