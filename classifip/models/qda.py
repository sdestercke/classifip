import abc
import matlab.engine
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
from numpy import linalg
import sys
import io


class DiscriminantAnalysis(metaclass=abc.ABCMeta):
    def __init__(self, init_matlab=False, DEBUG=False):
        # Starting matlab environment
        if init_matlab:
            self._eng = matlab.engine.start_matlab()
            print(matlab.engine.find_matlab())
        else:
            self._eng = None
        self._DEBUG = DEBUG
        self._N, self._p = None, None
        self._data = None
        self.ell = None
        self._clazz = None
        self._nb_clazz = None
        self.means, self.cov, self.inv_cov, self.det_cov = dict(), dict(), dict(), dict()

    def getData(self):
        return self._data

    def getClazz(self):
        return self._clazz

    def getEll(self):
        return self.ell

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
            self.cov[clazz] = cov_clazz
            if linalg.cond(cov_clazz) < 1 / sys.float_info.epsilon:
                self.inv_cov[clazz] = linalg.inv(cov_clazz)
                self.det_cov[clazz] = linalg.det(cov_clazz)
            else:  # computing pseudo inverse/determinant to a singular covariance matrix
                self.inv_cov[clazz] = linalg.pinv(cov_clazz)
                eig_values, _ = linalg.eig(cov_clazz)
                self.det_cov[clazz] = np.product(eig_values[(eig_values > 1e-12)])
        return self.cov[clazz], self.inv_cov[clazz], self.det_cov[clazz]

    def fit_max_likelihood(self, query):
        means, probabilities = dict(), dict()
        cov, inv, det = self._cov_group_sample()
        for clazz in self._clazz:
            means[clazz] = self.get_mean_by_clazz(clazz)
            probabilities[clazz] = self.compute_probability(means[clazz], inv, det, query)
        return means, cov, probabilities

    @abc.abstractmethod
    def compute_probability(self, param, inv, det, query):
        pass

    @abc.abstractmethod
    def _cov_group_sample(self):
        """
        :rtype: object
        """
        pass


class LinearDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    """
       Imprecise Linear Discriminant implemented with a imprecise gaussian distribution and
       conjugate exponential family.
    """
    def __init__(self, init_matlab=True, DEBUG=False):
        """
        :param init_matlab:
        """
        super(LinearDiscriminant, self).__init__(init_matlab=init_matlab, DEBUG=DEBUG)
        self.prior = dict()
        self.means, self.inv_cov, self.det_cov = None, None, None
        self._mean_lower, self._mean_upper = None, None
        # self.cov_group_sample = (None, None, None)

    def get_bound_means(self, clazz):
        return self._mean_lower[clazz], self._mean_upper[clazz]

    def _cov_group_sample(self):
        """
        Hint: Improving this method using the SVD method for computing pseudo-inverse matrix
        :return:
        """
        cov = 0
        for clazz in self._clazz:
            covClazz, _, _ = self.get_cov_by_clazz(clazz)
            _nb_instances_by_clazz = self.__nb_by_clazz(clazz)
            cov += covClazz * (_nb_instances_by_clazz - 1)  # biased estimator
        cov /= self._N - self._nb_clazz  # unbiased estimator group
        return cov, linalg.inv(cov), linalg.det(cov)

    def __nb_by_clazz(self, clazz):
        return len(self._data[self._data.y == clazz])

    def learn(self, X, y, ell=20):
        """
        :param X:
        :param y:
        :param ell: be careful with negative ll_upper interval, it needs max(0 , ll_upper)
        :return:
        """
        self.ell = ell
        self.means, self.inv_cov, self.det_cov = dict(), dict(), dict()
        self._mean_lower, self._mean_upper = dict(), dict()

        self._N, self._p = X.shape
        assert self._N == len(y), "Size X and y is not equals."
        y = np.array(y) if type(y) is list else y
        self._data = pd.concat([pd.DataFrame(X), pd.Series(y, dtype="category")], axis=1)
        columns = ["x" + i for i in map(str, range(self._p))]  # create columns names
        columns.extend('y')
        self._data.columns = columns
        self._clazz = np.array(self._data.y.cat.categories.tolist())
        self._nb_clazz = len(self._clazz)

        for clazz in self._clazz:
            mean = self.get_mean_by_clazz(clazz)
            self.get_cov_by_clazz(clazz)
            self.prior[clazz] = self.__nb_by_clazz(clazz) / self._N
            nb_by_clazz = self.__nb_by_clazz(clazz)
            self._mean_lower[clazz] = (-self.ell + nb_by_clazz * mean) / nb_by_clazz
            self._mean_upper[clazz] = (self.ell + nb_by_clazz * mean) / nb_by_clazz

    def evaluate(self, query, method="quadratic"):
        bounds = dict((clazz, dict()) for clazz in self._clazz)
        eng_session = self._eng

        def __get_bounds(clazz, bound="inf"):
            return bounds[clazz][bound] if bound in bounds[clazz] else None

        def __put_bounds(probability, estimator, clazz, bound="inf"):
            bounds[clazz][bound] = (probability, estimator)

        def __probability(_self, query, clazz, inv, det, mean_lower, mean_upper, bound="inf"):

            if __get_bounds(clazz, bound) is None:
                q = query.T @ inv
                if bound == "inf":
                    estimator = _self.__infimum_estimation(inv, q, mean_lower, mean_upper, eng_session)
                else:
                    estimator = _self.__supremum_estimation(inv, q, mean_lower, mean_upper, method)
                prob = _self.compute_probability(np.array(estimator), inv, det, query)
                __put_bounds(prob, estimator, clazz, bound)
            return __get_bounds(clazz, bound)

        lower, upper = [], []
        _, inv, det = self._cov_group_sample()
        for clazz in self._clazz:
            mean_lower = self._mean_lower[clazz]
            mean_upper = self._mean_upper[clazz]
            p_inf, _ = __probability(self, query, clazz, inv, det, mean_lower, mean_upper, bound='inf')
            p_up, _ = __probability(self, query, clazz, inv, det, mean_lower, mean_upper, bound='sup')
            lower.append(p_inf * self.prior[clazz])
            upper.append(p_up * self.prior[clazz])

        from ..representations.voting import Scores

        score = Scores(np.c_[lower, upper])
        return self._clazz[(score.nc_intervaldom_decision() > 0)], bounds

        # inference = np.divide(lower, upper[::-1])
        # answers = self._clazz[(inference > 1)]
        # print("inf/sup:", np.divide(lower, upper[::-1]))
        # return answers, bounds

    def compute_probability(self, mean, inv_cov, det_cov, query):
        _exp = -0.5 * ((query - mean).T @ inv_cov @ (query - mean))
        _const = np.power(det_cov, -0.5) / np.power(2 * np.pi, self._p / 2)
        return _const * np.exp(_exp)

    def __supremum_estimation(self, Q, q, mean_lower, mean_upper, method="quadratic"):

        def __min_convex_qp(A, q, lower, upper, d):
            ell_lower = matrix(lower, (d, 1))
            ell_upper = matrix(upper, (d, 1))
            P = matrix(A)
            q = matrix(-1 * q)
            I = matrix(0.0, (d, d))
            I[::d + 1] = 1
            G = matrix([I, -I])
            h = matrix([ell_upper, -ell_lower])
            return solvers.qp(P=P, q=q, G=G, h=h)

        def __min_convex_cp(Q, q, lower, upper, d):
            i_cov = matrix(Q)
            b = matrix(q)

            def cOptFx(x=None, z=None):
                if x is None: return 0, matrix(0.0, (d, 1))
                f = (0.5 * (x.T * i_cov * x) - b.T * x)
                Df = (i_cov * x - b)
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
            print("[Solution-not-Optimal]", solution)
            raise Exception("Not exist solution optimal!!")

        return [v for v in solution['x']]

    def __infimum_estimation(self, Q, q, mean_lower, mean_upper, eng_session):
        if eng_session is None:
            raise Exception("Environment matlab hadn't been initialized.")
        try:
            __out = io.StringIO()
            __err = io.StringIO()
            Q = matlab.double((-1 * Q).tolist())
            q = matlab.double(q.tolist())
            LB = matlab.double(mean_lower.tolist())
            UB = matlab.double(mean_upper.tolist())
            A = matlab.double([])
            b = matlab.double([])
            Aeq = matlab.double([])
            beq = matlab.double([])
            x, f_val, time, stats = eng_session.quadprogbb(Q, eng_session.transpose(q), A, b, Aeq, beq,
                    eng_session.transpose(LB), eng_session.transpose(UB), nargout=4,
                                        stdout=__out, stderr=__err)
            if self._DEBUG:
                print("[DEBUG_MATHLAB_OUTPUT:", __out.getvalue())
                print("[DEBUG_MATHLAB_OUTPUT:", __err.getvalue())
        except matlab.engine.MatlabExecutionError:
            print("Some error in the QuadProgBB code, inputs:")
            print("Q matrix:", Q)
            print("q vector:", q)
            print("Lower Bound:", mean_lower, "Upper Bound", mean_upper)
            raise Exception("Matlab ERROR execution QuadProgBB")

        return np.asarray(x).reshape((1, self._p))[0]

    def __repr__(self):
        print("lower:", self._mean_lower)
        print("upper:", self._mean_upper)
        print("group cov", self._cov_group_sample())
        print("cov", self.cov)
        print("det cov", self.det_cov)
        print("inv cov",  self.inv_cov)
        print("prior", self.prior)
        print("means", self.means)
        return "QDA!!"

    ## Testing Optimal force brute
    def __brute_force_search(self, clazz, query, lower, upper, inv, det, d):

        def cost_Fx(x, query, inv):
            return 0.5 * (x.T @ inv @ x) + query.T @ inv @ x

        def forRecursive(lowers, uppers, level, L, optimal):
            for current in np.array([lowers[level], uppers[level]]):
                if level < L - 1:
                    forRecursive(lowers, uppers, level + 1, L, np.append(optimal, current))
                else:
                    print("optimal value cost:", clazz, np.append(optimal, current),
                          self.compute_probability(np.append(optimal, current), inv, det, query),
                          cost_Fx(np.append(optimal, current), query, inv))

        forRecursive(lower, upper, 0, d, np.array([]))

    def show_all_brute_optimal(self, query):
        cov, inv, det = self._cov_group_sample()
        for clazz in self._clazz:
            mean_lower = self._mean_lower[clazz]
            mean_upper = self._mean_upper[clazz]
            print("box", mean_lower, mean_upper)
            self.__brute_force_search(clazz, query, mean_lower, mean_upper, inv, det, self._p)

class QuadraticDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    """
       Imprecise Quadratic Discriminant implemented with a imprecise gaussian distribution and
       conjugate exponential family.
    """
    def __init__(self, init_matlab=True):
        super(QuadraticDiscriminant, self).__init__(init_matlab=init_matlab)
        self.ell = None
