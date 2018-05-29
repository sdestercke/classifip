import pandas as pd
import os
import numpy as np
from cvxopt import solvers, matrix
from numpy.linalg import inv, det

from classifip.models.imprecise import Imprecise
import abc

import matlab.engine

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


class Plots:
    @staticmethod
    def plot2D_classification(X, y, colors={0: 'red', 1: 'blue'}):
        n_row, n_col = X.shape
        if n_col != 2: raise ValueError("Dimension in X matrix aren't corrected form.")
        if not isinstance(y, list): raise ValueError("Y isn't corrected form.")
        if n_row != len(y): raise ValueError("The number of column is not same in (X,y)")
        colList = [colors[c] for c in y]
        plt.scatter(X[:, 0], X[:, 1], c=colList, marker='+')

    @staticmethod
    def plot_cov_ellipse(points, nstd=1, col='r'):
        # Compute mean
        pos = [float(sum(l)) / len(l) for l in zip(*points)]
        # Compute covariance
        cov = np.cov(points, rowvar=False)
        # Descomposition Spectral
        Plots.plot_ellipse(pos, cov, nstd, col)

    @staticmethod
    def plot_ellipse(pos, cov, nstd=1, col='r'):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # Compute degree of rotation angle
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        # https://stats.stackexchange.com/questions/164741/how-to-find-the-maximum-axis-of-ellipsoid-given-the-covariance-matrix
        width, height = 2 * nstd * np.sqrt(vals)
        ellipse = Ellipse(xy=pos, width=width, height=height, angle=theta, facecolor='none', edgecolor=col, lw=1)
        ax = plt.gca()
        ax.add_artist(ellipse)
        return ellipse


class DiscriminantAnalysis(Imprecise, metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def costFx(self):
        pass

    def brute_force_search(self, query, lower, upper, d):
        def forRecursive(lowers, uppers, level, L, optimal):
            for current in np.array([lowers[level], uppers[level]]):
                if level < L - 1:
                    forRecursive(lowers, uppers, level + 1, L, np.append(optimal, current))
                else:
                    print("optimal value cost:", self.costFx(np.append(optimal, current), query))

        forRecursive(lower, upper, 0, d, np.array([]))

    def sample_covariance(self, X):
        means = self.sample_mean(X)
        X_centred = X - means
        covariance = X_centred.T @ X_centred
        return covariance / (len(X) - 1)

    @staticmethod
    def sample_mean(X):
        n, d = X.shape
        return np.sum(X, axis=0) / n


class LinearDiscriminant(DiscriminantAnalysis):

    def __init__(self, X, y, ell=20):
        """
        ell : be careful with negative ll_upper interval, it needs max(0 , ll_upper)
        :param ell:
        """
        self._X = X
        self._n, self._p = X.shape
        assert self._n == len(y), "Size X and y is not equals."
        self._y = np.array(y) if type(y) is list else y
        self.__eng = matlab.engine.start_matlab()
        self._mean = self.sample_mean(X)
        self._cov = self.sample_covariance(X)
        self._inv_cov = inv(self._cov)
        self._det_cov = det(self._cov)
        self._mean_lower = (-ell + self._n * self._mean) / self._n
        self._mean_upper = (ell + self._n * self._mean) / self._n
        self._ell = ell

    def fit_estimators(self):
        return self._mean

    def costFx(self, x, query):
        return 0.5 * (x.T @ self._inv_cov @ x) + query.T @ self._inv_cov @ x

    def inference(self, query, method="quadratic"):
        q = query.T @ self._inv_cov
        upper = self.__upper_probability(q)
        lower = self.__lower_probability(q)
        print(upper, lower, "shape")
        print("upper prob:", self.__compute_probability(upper, query))
        print("lower prob:", self.__compute_probability(lower, query))

    def __compute_probability(self, mean, query):
        _exp = -0.5 * (query - mean).T @ self._inv_cov @ (query - mean)
        _const = np.power(self._det_cov, -0.5) / np.power(2 * np.pi, self._p / 2)
        return _const * np.exp(_exp)

    def __upper_probability(self, q, method="quadratic"):
        if method == "quadratic":
            solution = self.__min_convex_qp(self._inv_cov, q,
                                            self._mean_lower,
                                            self._mean_upper, self._p)
        elif method == "nonlinear":
            solution = self.__min_convex_cp(self._inv_cov, q,
                                            self._mean_lower,
                                            self._mean_upper, self._p)
        else:
            raise Exception("Yet doesn't exist optimisation implemented")

        if solution['status'] != 'optimal':
            raise Exception("Not exist solution optimal!!")

        return [v for v in solution['x']]

    @staticmethod
    def __min_convex_qp(A, q, lower, upper, d):
        ell_lower = matrix(lower, (d, 1))
        ell_upper = matrix(upper, (d, 1))
        P = matrix(A)
        q = matrix(q)
        I = matrix(0.0, (d, d))
        I[::d + 1] = 1
        G = matrix([I, -I])
        h = matrix([ell_upper, -ell_lower])
        return solvers.qp(P=P, q=q, G=G, h=h)

    @staticmethod
    def __min_convex_cp(cov, q, lower, upper, d):
        i_cov = matrix(inv(cov))
        b = matrix(q)

        def cOptFx(x=None, z=None):
            if x is None: return 0, matrix(0.0, (d, 1))
            f = (0.5 * (x.T * i_cov * x) + b.T * x)
            Df = (i_cov * x + b)
            if z is None: return f, Df.T
            H = z[0] * i_cov
            return f, Df.T, H

        ll_lower = matrix(lower, (d, 1))
        ll_upper = matrix(upper, (d, 1))
        I = matrix(0.0, (d, d))
        I[::d + 1] = 1
        G = matrix([I, -I])
        h = matrix([ll_upper, -ll_lower])
        return solvers.cp(cOptFx, G=G, h=h)

    def __lower_probability(self, q):
        Q = matlab.double((-1*self._inv_cov).tolist())
        q = matlab.double((-1*q).tolist())
        LB = matlab.double(self._mean_lower.tolist())
        UB = matlab.double(self._mean_upper.tolist())
        A = matlab.double([])
        b = matlab.double([])
        Aeq = matlab.double([])
        beq = matlab.double([])
        x, f_val, time, stats = self.__eng.quadprogbb(Q, self.__eng.transpose(q), A, b, Aeq, beq,
                                                      self.__eng.transpose(LB), self.__eng.transpose(UB), nargout=4)
        return np.asarray(x).reshape((1, self._p))[0]

    def __repr__(self):
        print("mean upper:", self._mean_upper)
        print("mean lower:", self._mean_lower)
        print("cv:", self._inv_cov)
        print("s:", self._mean)
        return 'Wahaha!'

    @property
    def cov(self):
        return self._cov

    @property
    def mean(self):
        return self._mean


## Testing
current_dir = os.getcwd()
data = os.path.join(current_dir, "/Users/salmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv")
df_train = pd.read_csv(data)

pos_train = df_train[df_train.y == 1]
neg_train = df_train[df_train.y == 0]
X = pos_train.loc[:, ['x1', 'x2']].as_matrix()
y = pos_train.y.tolist()
lqa = LinearDiscriminant(X, y, ell=.5)
query = np.array([2, -2])
lqa.inference(query)
# print(lqa.mean, lqa.cov)
# query = np.array([2, -2])
# lqa.inference(query)
# Plots.plot2D_classification(X, y)
# Plots.plot_cov_ellipse(X)
# plt.show()

#
# n, d = X.shape
# e_cov = sample_covariance(X)
# e_mean = sample_mean(X)
#
# kktsolver='ldl', options={'kktreg': 1e-6})
# def testingLargeDim(n, d):
#     def costFx(x, cov, query):
#         i_cov = inv(cov)
#         q = query.T @ i_cov
#         return 0.5 * (x.T @ i_cov @ x) + q.T @ x
#
#     e_mean = np.random.normal(size=d)
#     e_cov = normal(d, d)
#     e_cov = e_cov.T * e_cov
#     query = np.random.normal(size=d)
#     q = maximum_Fx(e_cov, e_mean, query, n, d)
#     print("--->", q["x"], costFx(np.array(q["x"]), e_cov, query))
#     bnb_search(e_cov, e_mean, query, n, d)
#     brute_force_search(e_cov, e_mean, query, n, d)
# testingLargeDim(20, 5)
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
