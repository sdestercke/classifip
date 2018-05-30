import abc
import os

import matlab.engine
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
from numpy.linalg import inv, det


class DiscriminantAnalysis(metaclass=abc.ABCMeta):
    def __init__(self, init_matlab=False):
        # Starting matlab environment
        if init_matlab: self._eng = matlab.engine.start_matlab()
        self._N, self._p = None, None
        self._data = None
        self._clazz = None
        self._nb_clazz = None
        self.means, self.cov, self.inv_cov, self.det_cov = dict(), dict(), dict(), dict()

    def __mean_by_clazz(self, clazz):
        return self._data[self._data.y == clazz].iloc[:, :-1].mean().as_matrix()

    def __cov_by_clazz(self, clazz):
        return self._data[self._data.y == clazz].iloc[:, :-1].cov().as_matrix()

    def get_mean_by_clazz(self, clazz):
        if clazz not in self.means:
            self.means[clazz] = self.__mean_by_clazz(clazz)
        return self.means[clazz]

    def get_cov_by_clazz(self, clazz):
        if clazz not in self.inv_cov:
            cov_clazz = self.__cov_by_clazz(clazz)
            self.cov = cov_clazz
            self.inv_cov[clazz] = inv(cov_clazz)
            self.det_cov[clazz] = det(cov_clazz)
        return self.cov, self.inv_cov[clazz], self.det_cov[clazz]


class LinearDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    def __init__(self, ell=20, init_matlab=True):
        """
        ell : be careful with negative ll_upper interval, it needs max(0 , ll_upper)
        :param ell:
        """
        super(LinearDiscriminant, self).__init__(init_matlab=init_matlab)
        self.ell = ell
        self.means, self.inv_cov, self.det_cov = dict(), dict(), dict()

    def __cov_group_sample(self):
        cov = 0
        for clazz in self._clazz:
            covClazz, _, _ = self.get_cov_by_clazz(clazz)
            cov += covClazz * (self._nb_clazz - 1)  # biased estimator
        cov = cov / (self._N - self._nb_clazz)  # unbiased estimator group
        return cov, inv(cov), det(cov)

    def __nb_by_clazz(self, clazz):
        return len(self._data[self._data.y == clazz])

    def learn(self, X, y):
        self._N, self._p = X.shape
        assert self._N == len(y), "Size X and y is not equals."
        y = np.array(y) if type(y) is list else y
        self._data = pd.concat([pd.DataFrame(X), pd.Series(y, dtype="category")], axis=1)
        columns = ["x" + i for i in map(str, range(self._p))]  # create columns names
        columns.extend('y')
        self._data.columns = columns
        self._clazz = self._data.y.cat.categories.tolist()
        self._nb_clazz = len(self._clazz)

        for clazz in self._data.y.cat.categories.tolist():
            self.get_mean_by_clazz(clazz)
            self.get_cov_by_clazz(clazz)

    def evaluate(self, query, method="quadratic"):

        bounds = dict((clazz, dict()) for clazz in self._data.y.cat.categories.tolist())

        def __get_bounds(clazz, bound="inf"):
            return bounds[clazz][bound] if bound in bounds[clazz] else None

        def __put_bounds(probability, clazz, bound="inf"):
            bounds[clazz][bound] = probability

        def __probability(_self, query, clazz, inv, mean, det, n, bound="inf"):
            if __get_bounds(clazz, bound) is None:
                q = query.T @ inv
                _mean_lower = (-_self.ell + n * mean) / n
                _mean_upper = (_self.ell + n * mean) / n
                if bound == "inf":
                    estimator = _self.__infimum_estimation(inv, q, _mean_lower, _mean_upper)
                else:
                    estimator = _self.__supremum_estimation(inv, q, _mean_lower, _mean_upper, method)
                prob_lower = _self.__compute_probability(estimator, inv, det, query)
                __put_bounds(prob_lower, clazz, bound)
            return __get_bounds(clazz, bound)

        answers = []
        lower = []
        upper = []
        cov, inv, det = self.__cov_group_sample()
        for clazz in self._data.y.cat.categories.tolist():
            mean = self.get_mean_by_clazz(clazz)
            n_clazz = self.__nb_by_clazz(clazz)
            lower.append(__probability(self, query, clazz, inv, mean, det, n_clazz, bound='inf'))
            upper.append(__probability(self, query, clazz, inv, mean, det, n_clazz, bound='sup'))

        print("probabilities", np.row_stack((upper, lower)))
        from classifip.representations.intervalsProbability import IntervalsProbability
        answers.append(IntervalsProbability(np.row_stack((upper, lower))))
        return answers

    def __compute_probability(self, mean, inv_cov, det_cov, query):
        _exp = -0.5 * (query - mean).T @ inv_cov @ (query - mean)
        _const = np.power(det_cov, -0.5) / np.power(2 * np.pi, self._p / 2)
        return _const * np.exp(_exp)

    def __supremum_estimation(self, Q, q, mean_lower, mean_upper, method="quadratic"):

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

        if method == "quadratic":
            solution = __min_convex_qp(Q, q, mean_lower, mean_upper, self._p)
        elif method == "nonlinear":
            solution = __min_convex_cp(Q, q, mean_lower, mean_upper, self._p)
        else:
            raise Exception("Yet doesn't exist optimisation implemented")

        if solution['status'] != 'optimal':
            raise Exception("Not exist solution optimal!!")

        return [v for v in solution['x']]

    def __infimum_estimation(self, Q, q, mean_lower, mean_upper):
        Q = matlab.double((-1 * Q).tolist())
        q = matlab.double((-1 * q).tolist())
        LB = matlab.double(mean_lower.tolist())
        UB = matlab.double(mean_upper.tolist())
        A = matlab.double([])
        b = matlab.double([])
        Aeq = matlab.double([])
        beq = matlab.double([])
        x, f_val, time, stats = self._eng.quadprogbb(Q, self._eng.transpose(q), A, b, Aeq, beq,
                                                      self._eng.transpose(LB), self._eng.transpose(UB), nargout=4)
        return np.asarray(x).reshape((1, self._p))[0]

    def __fit_max_likelihood(self, query):
        pass

    def __repr__(self):
        print("mean upper:", self._mean_upper)
        print("mean lower:", self._mean_lower)
        print("cv:", self._inv_cov)
        print("s:", self._mean)
        return 'Wahaha!'

    ## Testing Optimal force brute
    def costFx(self, x, query, inv):
        return 0.5 * (x.T @ inv @ x) + query.T @ inv @ x

    def __brute_force_search(self, clazz, query, lower, upper, inv, det, d):
        def forRecursive(lowers, uppers, level, L, optimal):
            for current in np.array([lowers[level], uppers[level]]):
                if level < L - 1:
                    forRecursive(lowers, uppers, level + 1, L, np.append(optimal, current))
                else:
                    print("optimal value cost:", clazz,
                          self.__compute_probability(np.append(optimal, current), inv, det, query),
                          self.costFx(np.append(optimal, current), query, inv))

        forRecursive(lower, upper, 0, d, np.array([]))

    def supremum_bf(self, query):
        cov, inv, det =  self.__cov_group_sample()
        for clazz in self._data.y.cat.categories.tolist():
            mean = self.get_mean_by_clazz(clazz)
            n = self.__nb_by_clazz(clazz)
            _mean_lower = (-self.ell + n * mean) / n
            _mean_upper = (self.ell + n * mean) / n
            print("box", _mean_lower, _mean_upper)
            self.__brute_force_search(clazz, query, _mean_lower, _mean_upper, inv, det, self._p)


## Testing
current_dir = os.getcwd()
root_path = "/Users/salmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv"
# root_path = "/Volumes/Data/DBSalmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv"
data = os.path.join(current_dir, root_path)
df_train = pd.read_csv(data)
X = df_train.loc[:, ['x1', 'x2']].values
y = df_train.y.tolist()
lqa = LinearDiscriminant(ell=5)
query = np.array([0.830031, 0.108776])
lqa.learn(X, y)
test = lqa.evaluate(query)
lqa.supremum_bf(query)
print(test[0].getmaximaldecision())
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
