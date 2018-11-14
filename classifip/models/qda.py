import abc
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
from numpy import linalg
import sys
import io
import logging


class DiscriminantAnalysis(metaclass=abc.ABCMeta):
    def __init__(self, init_matlab=False, add_path_matlab=None, DEBUG=False):
        """
        :param init_matlab: init engine matlab
        :param DEBUG: print debug log for computation
        """
        # Starting matlab environment
        if init_matlab:
            logging.info("Loading MATHLAB environment.")
            import matlab.engine
            self._eng = matlab.engine.start_matlab()
            self._add_path_matlab = [] if add_path_matlab is None else add_path_matlab
            for _in_path in self._add_path_matlab: self._eng.addpath(_in_path)
        else:
            self._eng = None
        self._DEBUG = DEBUG
        self._N, self._p = None, None
        self._data = None
        self._ell = None
        self._clazz = None
        self._nb_clazz = None
        self._prior = dict()
        # icov: inverse covariance, dcov:  determinant covariance
        self._gp_mean, self._gp_cov, self._gp_icov, self._gp_dcov = dict(), dict(), dict(), dict()
        self._mean_lower, self._mean_upper = None, None
        self.__nb_rebuild_opt = 0

    def get_data(self):
        return self._data

    def get_clazz(self):
        return self._clazz

    def get_ell(self):
        return self._ell

    def get_mean_by_clazz(self, clazz):
        if clazz not in self._gp_mean:
            self._gp_mean[clazz] = self.__mean_by_clazz(clazz)
        return self._gp_mean[clazz]

    def get_cov_by_clazz(self, clazz):
        if clazz not in self._gp_cov:
            cov_clazz = self.__cov_by_clazz(clazz)
            self._gp_cov[clazz] = cov_clazz
            if linalg.cond(cov_clazz) < 1 / sys.float_info.epsilon:
                self._gp_icov[clazz] = linalg.inv(cov_clazz)
                self._gp_dcov[clazz] = linalg.det(cov_clazz)
            else:  # computing pseudo inverse/determinant to a singular covariance matrix
                self._gp_icov[clazz] = linalg.pinv(cov_clazz)
                eig_values, _ = linalg.eig(cov_clazz)
                self._gp_dcov[clazz] = np.product(eig_values[(eig_values > 1e-12)])
        return self._gp_cov[clazz], self._gp_icov[clazz], self._gp_dcov[clazz]

    def probability_density_gaussian(self, mean, inv_cov, det_cov, query):
        _exp = -0.5 * ((query - mean).T @ inv_cov @ (query - mean))
        _const = np.power(det_cov, -0.5) / np.power(2 * np.pi, self._p / 2)
        return _const * np.exp(_exp)

    def fit_max_likelihood(self, query):
        means, probabilities = dict(), dict()
        for clazz in self._clazz:
            cov, inv, det = self.get_cov_by_clazz(clazz)
            means[clazz] = self.get_mean_by_clazz(clazz)
            probabilities[clazz] = self.probability_density_gaussian(means[clazz], inv, det, query)
        return means, probabilities

    def learn(self, learn_data_set=None, ell=2, X=None, y=None):
        """
        :param learn_data_set: (X, y): X matrix of features and y category by number instances
        :param ell: imprecise value for mean parameter
        :param X:
        :param y:
        :return:
        """
        self._ell = ell
        self._gp_mean, self._gp_icov, self._gp_dcov = dict(), dict(), dict()
        self._mean_lower, self._mean_upper = dict(), dict()

        # transformation of Arff data to feature matrix and vector category
        if learn_data_set is not None:
            learn_data_set = np.array(learn_data_set.data)
            X = learn_data_set[:, :-1]
            y = learn_data_set[:, -1]
        elif X is not None and y is not None:
            logging.info("Loading training data set from (X,y) couples.")
        else:
            raise Exception('Not training data set setting.')

        # Create data set in Panda frame structure
        self._N, self._p = X.shape
        assert self._N == len(y), "Size X and y is not equals."
        y = np.array(y) if type(y) is list else y
        self._data = pd.concat([pd.DataFrame(X, dtype="float64"), pd.Series(y, dtype="category")], axis=1)
        columns = ["x" + i for i in map(str, range(self._p))]  # create columns names
        columns.extend('y')
        self._data.columns = columns
        self._clazz = np.array(self._data.y.cat.categories.tolist())
        self._nb_clazz = len(self._clazz)

        # Estimation of imprecise/precise parameters
        for clazz in self._clazz:
            mean = self.get_mean_by_clazz(clazz)
            self.get_cov_by_clazz(clazz)
            nb_by_clazz = self._nb_by_clazz(clazz)
            self._prior[clazz] = nb_by_clazz / self._N
            self._mean_lower[clazz] = (-self._ell + nb_by_clazz * mean) / nb_by_clazz
            self._mean_upper[clazz] = (self._ell + nb_by_clazz * mean) / nb_by_clazz

    def evaluate(self, query, method="quadratic"):
        bounds = dict((clazz, dict()) for clazz in self._clazz)
        eng_session = self._eng

        def __get_bounds(clazz, bound="inf"):
            return bounds[clazz][bound] if bound in bounds[clazz] else None

        def __put_bounds(probability, estimator, clazz, bound="inf"):
            bounds[clazz][bound] = (probability, estimator)

        def __probability(_self, query, clazz, inv, det, mean_lower, mean_upper, bound="inf"):

            if __get_bounds(clazz, bound) is None:
                q = -1 * (query.T @ inv)
                if bound == "inf":
                    estimator = _self.infimum_estimation(inv, q, mean_lower, mean_upper, eng_session, clazz)
                else:
                    estimator = _self.supremum_estimation(inv, q, mean_lower, mean_upper, method)
                prob = _self.probability_density_gaussian(np.array(estimator), inv, det, query)
                __put_bounds(prob, estimator, clazz, bound)
            return __get_bounds(clazz, bound)

        lower, upper = [], []
        for clazz in self._clazz:
            mean_lower = self._mean_lower[clazz]
            mean_upper = self._mean_upper[clazz]
            _, inv, det = self.get_cov_by_clazz(clazz)
            p_inf, _ = __probability(self, query, clazz, inv, det, mean_lower, mean_upper, bound='inf')
            p_up, _ = __probability(self, query, clazz, inv, det, mean_lower, mean_upper, bound='sup')
            lower.append(p_inf * self._prior[clazz])
            upper.append(p_up * self._prior[clazz])

        from classifip.representations.voting import Scores
        score = Scores(np.c_[lower, upper])
        return self._clazz[(score.nc_intervaldom_decision() > 0)], bounds

        # from classifip.representations.intervalsProbability import IntervalsProbability
        # return IntervalsProbability(np.array([upper, lower]))
        # inference = np.divide(lower, upper[::-1])
        # answers = self._clazz[(inference > 1)]
        # print("inf/sup:", np.divide(lower, upper[::-1]))
        # return answers, bounds

    def get_bound_means(self, clazz):
        return self._mean_lower[clazz], self._mean_upper[clazz]

    def _nb_by_clazz(self, clazz):
        assert self._data is not None, "It's necessary to firstly declare a data set."
        return len(self._data[self._data.y == clazz])

    def __mean_by_clazz(self, clazz):
        return self._data[self._data.y == clazz].iloc[:, :-1].mean().as_matrix()

    def __cov_by_clazz(self, clazz):
        return self._data[self._data.y == clazz].iloc[:, :-1].cov().as_matrix()

    def supremum_estimation(self, Q, q, mean_lower, mean_upper, method="quadratic"):

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

        def __min_convex_cp(Q, q, lower, upper, d):
            i_cov = matrix(Q)
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
            logging.debug("[Solution-not-Optimal]", solution)
            raise Exception("Not exist solution optimal!!")

        return [v for v in solution['x']]

    def infimum_estimation(self, Q, q, mean_lower, mean_upper, eng_session, clazz):
        """ This method use a solver implemented in https://github.com/sburer/QuadProgBB
            Authors: Samuel Burer
            QUADPROGBB globally solves the following nonconvex quadratic programming problem
                min  1/2*x'*H*x + f'*x
                s.t.    A * x <= b
                        Aeq * x == beq
                        LB <= x <= UB
            Therefore, I optimize max this non-convex problem, we multiply by -1:
                max  1/2*x'*(-H)*x - f'*x
                s.t.  ....(same)
        """
        import matlab.engine

        if eng_session is None:
            raise Exception("Environment matlab hadn't been initialized.")
        __out = io.StringIO()
        __err = io.StringIO()
        try:
            Q_m = matlab.double((-1 * Q).tolist())
            q_m = matlab.double((-1 * q).tolist())
            LB = matlab.double(mean_lower.tolist())
            UB = matlab.double(mean_upper.tolist())
            A = matlab.double([])
            b = matlab.double([])
            Aeq = matlab.double([])
            beq = matlab.double([])
            x, f_val, time, stats = eng_session.quadprogbb(Q_m, eng_session.transpose(q_m), A, b, Aeq, beq,
                                                           eng_session.transpose(LB), eng_session.transpose(UB),
                                                           nargout=4, stdout=__out, stderr=__err)
            if self._DEBUG: logging.debug("[DEBUG_MATHLAB_OUTPUT:", __out.getvalue())
            self.__nb_rebuild_opt = 0
        except Exception as e:
            """ In case, MATHLAB crash 3 times, optimal point it's mean of category"""
            if self.__nb_rebuild_opt == 3:
                self.__nb_rebuild_opt = 0
                return self.get_mean_by_clazz(clazz)
            self.__nb_rebuild_opt += 1
            logging.debug("[DEBUG_MATHLAB_OUTPUT]", __err.getvalue())
            logging.error("[DEBUG_MATHLAB_EXCEPTION]", e)
            logging.error("[DEBUG_MATHLAB_OUTPUT]: Crash QuadProgBB, inputs:class", clazz)
            logging.error("[DEBUG_MATHLAB_OUTPUT] Q matrix:", (-1 * Q))
            logging.error("[DEBUG_MATHLAB_OUTPUT] q vector:", (-1 * q))
            logging.error("[DEBUG_MATHLAB_OUTPUT] Lower Bound:", mean_lower, "Upper Bound", mean_upper)

            self._eng = matlab.engine.start_matlab()
            for _in_path in self._add_path_matlab: self._eng.addpath(_in_path)
            logging.error("[DEBUG_MATHLAB_OUTPUT] New MATHLAB LOADING")
            return self.infimum_estimation(Q, q, mean_lower, mean_upper, eng_session, clazz)

        return np.asarray(x).reshape((1, self._p))[0]

    def __repr__(self):
        print("lower:", self._mean_lower)
        print("upper:", self._mean_upper)
        print("group cov", self._cov_group_sample())
        print("cov", self._gp_cov)
        print("det cov", self.det_cov)
        print("inv cov", self.inv_cov)
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
                          self.probability_density_gaussian(np.append(optimal, current), inv, det, query),
                          cost_Fx(np.append(optimal, current), query, inv))

        forRecursive(lower, upper, 0, d, np.array([]))

    def show_all_brute_optimal(self, query):
        for clazz in self._clazz:
            cov, inv, det = self.get_cov_by_clazz(clazz)
            mean_lower = self._mean_lower[clazz]
            mean_upper = self._mean_upper[clazz]
            print("box", mean_lower, mean_upper)
            self.__brute_force_search(clazz, query, mean_lower, mean_upper, inv, det, self._p)


class EuclideanDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    """
        Imprecise Euclidean Distance Discriminant implemented with a
        imprecise gaussian distribution and conjugate exponential family.
    """

    def __init__(self, init_matlab=False, add_path_matlab=None, DEBUG=False):
        super(EuclideanDiscriminant, self).__init__(init_matlab=init_matlab,
                                                    add_path_matlab=add_path_matlab,
                                                    DEBUG=DEBUG)

    def get_cov_by_clazz(self, clazz):
        if clazz not in self._gp_cov:
            self._gp_cov[clazz] = np.identity(self._p)
            self._gp_icov[clazz] = np.identity(self._p)
            self._gp_dcov[clazz] = 1
        return self._gp_cov[clazz], self._gp_icov[clazz], self._gp_dcov[clazz]

    def supremum_estimation(self, Q, q, mean_lower, mean_upper, method="quadratic"):
        optimal = np.zeros(self._p)
        x = (-1 * q)  # return true query value
        inside_hypercube = True
        for i in range(self._p):
            if inside_hypercube and (x[i] < mean_lower[i] or x[i] > mean_upper[i]):
                inside_hypercube = False
            if pow(x[i] - mean_lower[i], 2) < pow(x[i] - mean_upper[i], 2):
                optimal[i] = mean_lower[i]
            else:
                optimal[i] = mean_upper[i]

        return x if inside_hypercube else optimal

    def infimum_estimation(self, Q, q, mean_lower, mean_upper, eng_session, clazz):
        optimal = np.zeros(self._p)
        x = (-1 * q)  # return true query value
        for i in range(self._p):
            if pow(x[i] - mean_lower[i], 2) > pow(x[i] - mean_upper[i], 2):
                optimal[i] = mean_lower[i]
            else:
                optimal[i] = mean_upper[i]
        return optimal


class LinearDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    """
       Imprecise Linear Discriminant implemented with a imprecise gaussian distribution and
       conjugate exponential family.
    """

    def __init__(self, init_matlab=True, add_path_matlab=None, DEBUG=False):
        super(LinearDiscriminant, self).__init__(init_matlab=init_matlab,
                                                 add_path_matlab=add_path_matlab,
                                                 DEBUG=DEBUG)
        self._is_compute_total_cov = False

    def get_cov_by_clazz(self, clazz):
        """
        Hint: Improving this method using the SVD method for computing pseudo-inverse matrix
        :return:
        """
        if not self._is_compute_total_cov:
            cov = 0 # estimation of empirical total covariance matrix
            for clazz_gp in self._clazz:
                covClazz, _, _ = super(LinearDiscriminant, self).get_cov_by_clazz(clazz_gp)
                _nb_instances_by_clazz = self._nb_by_clazz(clazz)
                cov += covClazz * (_nb_instances_by_clazz - 1)  # biased estimator
            cov /= self._N - self._nb_clazz  # unbiased estimator group

            for clazz_gp in self._clazz:
                self._gp_cov[clazz_gp] = cov
                self._gp_icov[clazz_gp] = linalg.inv(cov)
                self._gp_dcov[clazz_gp] =linalg.det(cov)

            self._is_compute_total_cov=True

        return self._gp_cov[clazz], self._gp_icov[clazz], self._gp_dcov[clazz]

class QuadraticDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    """
       Imprecise Quadratic Discriminant implemented with a imprecise gaussian distribution and
       conjugate exponential family.
    """

    def __init__(self, init_matlab=True, add_path_matlab=None, DEBUG=False):
        super(QuadraticDiscriminant, self).__init__(init_matlab=init_matlab,
                                                    add_path_matlab=add_path_matlab,
                                                    DEBUG=DEBUG)
