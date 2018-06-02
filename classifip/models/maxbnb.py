import pandas as pd
import os
import numpy as np
from cvxopt import solvers, matrix
from numpy.linalg import inv, det

import abc

import matlab.engine

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

class Imprecise(metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def inference(self, query, method):
        pass

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
        # Compute covariancec
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
        # Starting matlab envirement
        self.__eng = matlab.engine.start_matlab()
        """
        ell : be careful with negative ll_upper interval, it needs max(0 , ll_upper)
        :param ell:
        """
        self._X = X
        self._n, self._p = X.shape
        assert self._n == len(y), "Size X and y is not equals."
        self._y = np.array(y) if type(y) is list else y
        self._data = pd.concat([pd.DataFrame(X), pd.Series(y, dtype="category")], axis=1)
        # create columns names
        self._columns = ["x" + i for i in map(str, range(self._p))]
        self._columns.extend('y')
        self._data.columns = self._columns
        self._ell = ell

    def fit_estimators(self):
        return self._mean

    def costFx(self, x, query):
        return 0.5 * (x.T @ self._inv_cov @ x) + query.T @ self._inv_cov @ x

    def __mean(self, clazz):
        return self._data[self._data.y == clazz].iloc[:, :-1].mean()

    def __cov(self, clazz):
        return self._data[self._data.y == clazz].iloc[:, :-1].cov()

    def inference(self, query, method="quadratic"):
        means, inv_cov, det_cov = dict(), dict(), dict()
        bounds = dict((clazz, dict()) for clazz in self._data.y.cat.categories.tolist())

        def __mean(_self, clazz):
            if clazz not in means:
                mean_clazz = _self.__mean(clazz)
                means[clazz] = mean_clazz
            return means[clazz]

        def __inv(_self, clazz):
            if clazz not in inv_cov:
                cov_clazz = _self.__cov(clazz)
                inv_cov[clazz] = inv(cov_clazz)
                det_cov[clazz] = det(cov_clazz)
            return inv_cov[clazz], det_cov[clazz]

        def __get_bounds(clazz, bound="lower"):
            return bounds[clazz][bound] if bound in bounds[clazz] else None

        def __put_bounds(probability, clazz, bound="lower"):
            bounds[clazz][bound] = probability

        def __prob_lower(_self, query, clazz_up, inv_up, mean_up, det_up):
            if __get_bounds(clazz_up) is None:
                q = query.T @ inv_up
                _mean_lower = (-_self.ell + _self.n * mean_up) / _self.n
                _mean_upper = (_self.ell + _self.n * mean_up) / _self.n
                fit_lower = _self.__lower_probability(inv_up, q, _mean_lower, _mean_upper)
                prob_lower = _self.__compute_probability(fit_lower, inv_up, det_up, query)
                __put_bounds(prob_lower, clazz_up)
            return __get_bounds(clazz_up)

        def __prob_upper(_self, query, clazz_down, inv_down, mean_down, det_down):
            if __get_bounds(clazz_down, "upper") is None:
                q = query.T @ inv_down
                _mean_lower = (-_self.ell + _self.n * mean_down) / _self.n
                _mean_upper = (_self.ell + _self.n * mean_down) / _self.n
                fit_upper = _self.__upper_probability(inv_down, q, _mean_lower, _mean_upper)
                prob_upper = _self.__compute_probability(fit_upper, inv_down, det_down, query)
                __put_bounds(prob_upper, clazz_down, "upper")
            return __get_bounds(clazz_down, "upper")

        def pairwise_comparison(_self, query, clazz_up, clazz_down, mean_up, mean_down,
                                inv_up, inv_down, det_up, det_down):

            numerator = __prob_lower(_self, query, clazz_up, inv_up, mean_up, det_up)
            denominator = __prob_upper(_self, query, clazz_down, inv_down, mean_down, det_down)
            return numerator / denominator





        # class Node:
        #     def __init__(self, clazz):
        #         self._clazz = clazz if type(clazz) == list() else [clazz]
        #
        #     @property
        #     def clazz(self):
        #         return self._clazz
        #
        #     def append(self, new_clazz):
        #         self._clazz.append(new_clazz)
        #
        #     def selection(self):
        #         return np.random.choice(self._clazz, 1)[0]
        #
        #     def __hash__(self):
        #         print(hash("_".join(str(x) for x in self._clazz)))
        #         return hash("_".join(str(x) for x in self._clazz))
        #
        # # First Step (computing clusters partial ordering)
        # clusters = [{Node(clazz): []} for clazz in self._data.y.cat.categories.tolist()]
        # while len(clusters) > 1:
        #     new_clusters = []
        #     while len(clusters) > 0:
        #         if len(clusters) > 1:
        #             index = np.random.choice(len(clusters), 2, replace=False)
        #             y1_compound = np.random.choice([*clusters[index[0]]], 1)[0]
        #             y2_compound = np.random.choice([*clusters[index[1]]], 1)[0]
        #             y1, y2 = y1_compound.selection(), y2_compound.selection()
        #             mean_y1 = __mean(self, y1)
        #             mean_y2 = __mean(self, y2)
        #             inv_y1, det_y1 = __inv(self, y1)
        #             inv_y2, det_y2 = __inv(self, y2)
        #             inference = pairwise_comparison(self, query, y1, y2, mean_y1, mean_y2,
        #                                             inv_y1, inv_y2, det_y1, det_y2, self._n, self._ell)
        #             new_cluster = clusters[index[0]]
        #             new_cluster.update(clusters[index[1]])
        #             if inference > 1:
        #                 new_cluster[y1_compound].append(y2_compound)
        #             else:
        #                 inference = pairwise_comparison(self, query, y2, y1, mean_y2, mean_y1,
        #                                                 inv_y2, inv_y1, det_y2, det_y1, self._n, self._ell)
        #                 if inference > 1:
        #                     new_cluster[y2_compound].append(y1_compound)
        #                 else:
        #                     children = set(new_cluster[y1_compound])
        #                     children.update(new_cluster[y2_compound])
        #                     del new_cluster[y1_compound]
        #                     del new_cluster[y2_compound]
        #                     compound = y1_compound.clazz
        #                     compound.append(y2_compound.clazz)
        #                     new_cluster[Node(compound)] = children
        #             new_clusters.append(new_cluster)
        #         else:
        #             index = [0]
        #             new_clusters.append(clusters[index[0]])
        #         [clusters.pop(idx) for idx in sorted(index, reverse=True)]
        #     clusters = new_clusters
        #
        # # Second step find connected components (transitive nodes)
        # graph = clusters[0]
        # components = dict()
        # visited = set()
        # from collections import deque
        # stack = deque(graph.keys())
        # while not stack.empty():
        #     if node is not visited:
        #         node = stack.pop()
        #         visited.add(node)
        #         components[node] = dict()
        #         for neighbor in graph[node]:
        #             stack.append(neighbor)

        # for node, children in :
        #     if len(children) > 0:
        #         seen.add(node)
        #         components.append(node)
        #
        #         while not stack.empty():
        #             child = stack.pop()


    def __compute_probability(self, mean, inv_cov, det_cov, query):
        _exp = -0.5 * (query - mean).T @ inv_cov @ (query - mean)
        _const = np.power(det_cov, -0.5) / np.power(2 * np.pi, self._p / 2)
        return _const * np.exp(_exp)

    def __upper_probability(self, Q, q, mean_lower, mean_upper, method="quadratic"):
        if method == "quadratic":
            solution = self.__min_convex_qp(Q, q, mean_lower, mean_upper, self._p)
        elif method == "nonlinear":
            solution = self.__min_convex_cp(Q, q, mean_lower, mean_upper, self._p)
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

    def __lower_probability(self, Q, q, mean_lower, mean_upper):
        Q = matlab.double((-1 * Q).tolist())
        q = matlab.double((-1 * q).tolist())
        LB = matlab.double(mean_lower.tolist())
        UB = matlab.double(mean_upper.tolist())
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

    def data_test(d):
        e_mean = np.random.normal(size=d)
        e_cov = normal(d, d)
        e_cov = e_cov.T * e_cov
        query = np.random.normal(size=d)
        return e_mean, e_cov, query

    @property
    def cov(self):
        return self._cov

    @property
    def mean(self):
        return self._mean


## Testing
current_dir = os.getcwd()
root_path = "/Users/salmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv"
# root_path = "/Volumes/Data/DBSalmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv"
data = os.path.join(current_dir, root_path)
df_train = pd.read_csv(data)

pos_train = df_train[df_train.y == 1]
neg_train = df_train[df_train.y == 0]
X = df_train.loc[:, ['x1', 'x2']].values
y = df_train.y.tolist()
lqa = LinearDiscriminant(X, y, ell=.5)
query = np.array([2, -2])
lqa.inference(query)
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
