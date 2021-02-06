import abc, sys, io
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
from numpy import linalg
from scipy.stats import multivariate_normal
from ..utils import create_logger, is_level_debug
from classifip.representations.voting import Scores
import cvxpy as cvx
from qcqp import QCQP, RANDOM, COORD_DESCENT

MATLAB_AVAILABLE = True
try:
    import matlab.engine
except Exception as e:
    MATLAB_AVAILABLE = False
    print("MATLAB not installed in host.")


def is_sdp_symmetric(X):
    if X.shape[0] != X.shape[1]:
        return False
    # ToDo: If it is not symmetric, use in whole file: X'=0.5*(X+X.T)
    if not np.allclose(X, X.T, atol=1e-8):
        print("[is_sdp_symmetric] MATRIX IS NOT SYMMETRIC", flush=True)
        return False

    _eigen_values = np.linalg.eigvalsh(X)
    _is_sdp = np.all(_eigen_values > -1e-8)
    if not _is_sdp:
        print("[is_sdp_symmetric] Eigenvalues negatives:", _eigen_values, flush=True)
    return _is_sdp


# First version maximality criterion
def inference_maximal_criterion(lower, upper, clazz):
    pairwise_comparison = []
    pairwise_indifferent = []
    # O(n^2), n: number of classes
    for idl, l in enumerate(lower):
        for idu, u in enumerate(upper):
            if idl != idu:
                if l - u > 0:
                    pairwise_comparison.append([clazz[idl], clazz[idu]])
                elif lower[idu] - upper[idl] <= 0:
                    pairwise_indifferent.append([clazz[idl], clazz[idu]])

    top_max = dict()
    down_min = dict()
    # O(l), l: number of pairwise comparison
    for pairwise in pairwise_comparison:
        if pairwise[0] not in down_min:
            if pairwise[1] in top_max:
                top_max.pop(pairwise[1], None)
            top_max[pairwise[0]] = 1
            down_min[pairwise[1]] = 0
        else:
            down_min[pairwise[1]] = 0

    # O(l), l: adding indifferent elements
    # e.g. (a, b, c) => a > c, b >< a, b >< c => maximality=(a,b)
    for pairwise in pairwise_indifferent:
        if pairwise[0] in top_max or pairwise[1] in top_max:
            if pairwise[0] not in down_min and pairwise[0] not in top_max:
                top_max[pairwise[0]] = 1
            if pairwise[1] not in down_min and pairwise[1] not in top_max:
                top_max[pairwise[1]] = 1

    maximal_elements = list(top_max.keys())
    # If there is no maximals, so maximal elements are all classes.
    return clazz if len(maximal_elements) == 0 else maximal_elements


class DiscriminantAnalysis(metaclass=abc.ABCMeta):
    """
        Classification multi-classes using a set of Gaussian distributions

        TODO:
            - Verify the supremum problem is really a convex problem
            It is not enough to verify symmetric and eigenvalues (is_sdp_symmetric)
    """

    def __init__(self, solver_matlab=False, add_path_matlab=None):
        """
        :param solver_matlab: init engine matlab
        :param DEBUG: logger debug log for computation
        """
        # Starting matlab environment
        self.__solver_matlab = solver_matlab
        if self.__solver_matlab:
            if MATLAB_AVAILABLE:
                self._eng = matlab.engine.start_matlab()
                self._add_path_matlab = [] if add_path_matlab is None else add_path_matlab
                for _in_path in self._add_path_matlab:
                    self._eng.addpath(_in_path)
            else:
                print("MATLAB is not available!, it could use QCQP package !", flush=True)
                self.__solver_matlab = False
        else:
            self._eng = None
        self._N, self._p = None, None
        self._data = None
        self._ell = None
        self._clazz = None
        self._nb_clazz = None
        self._prior = dict()
        # icov: inverse covariance,
        # sdp: bool if it's semi-definite positive symmetric
        self._gp_mean, self._gp_sdp = dict(), dict()
        self._gp_cov, self._gp_icov = dict(), dict()
        self._mean_lower, self._mean_upper = None, None
        self.__nb_rebuild_opt = 0
        self._logger = None
        # save the last computed conditional probabilities X|Y on instance x*
        self._bound_cond_probabilities = None

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
            cov_clazz = self._cov_by_clazz(clazz)
            self._gp_cov[clazz] = cov_clazz
            if linalg.cond(cov_clazz) < 1 / sys.float_info.epsilon:
                self._gp_icov[clazz] = linalg.inv(cov_clazz)
            else:  # computing pseudo-inverse of a singular covariance matrix
                self._gp_icov[clazz] = linalg.pinv(cov_clazz)

            # verify if pseudo-inverse matrix is semi-definite positive symmetric
            # @review: numerical tolerance 1e-7 all matrix are symmetric, is good so much decimal precision?
            self._gp_sdp[clazz] = is_sdp_symmetric(self._gp_icov[clazz])
        return self._gp_cov[clazz], self._gp_icov[clazz]

    def probability_density_gaussian(self, mean, cov, query, log_p=False):
        """
        This method calculate the multivariate Gaussian probability of event (query)
        when the covariance matrix is positive semi-definite matrix.

        This one also does take into consideration if the covariance matrix is singular,
        that means that the multivariate Gaussian is degenerate and the calculation must be
        done differently.
            f(x)= \det^{*}(2\pi*\Sigma)^{-\frac {1}{2}} * exp(-0.5*(x -\mu)^T \Sigma^{+} (x - \mu))
        where:
            det^{*} is the pseudo-determinant of Sigma, and
            Sigma^{+} is the generalized inverse

        version manual in python:
            _exp = -0.5 * ((query - mean).T @ inv_cov @ (query - mean))
            _const = np.power(det_cov, -0.5) / np.power(2 * np.pi, self._p / 2)
            return _const * np.exp(_exp)

        :param mean: mean of
        :param cov: covariance matrix (positive semi-definite or singular)
        :param query: event of multivariate Gaussian probability
        :param log_p: boolean if it is log-probability or just probability
        :return: the probability of 'query' event
        """
        if log_p:
            probability = multivariate_normal.logpdf(query, mean=mean, cov=cov, allow_singular=True)
        else:
            probability = multivariate_normal.pdf(query, mean=mean, cov=cov, allow_singular=True)
        self._logger.debug("Computing the (log-)probability (%s, %s):", log_p, probability)
        return probability

    def fit_max_likelihood(self, query):
        means, probabilities = dict(), dict()
        for clazz in self._clazz:
            cov, inv = self.get_cov_by_clazz(clazz)
            means[clazz] = self.get_mean_by_clazz(clazz)
            probabilities[clazz] = self.probability_density_gaussian(means[clazz], cov, query)
        return means, probabilities

    def learn(self, learn_data_set=None, ell=2, X=None, y=None):
        """
        :param learn_data_set: (X, y): X matrix of features and y category by number instances
        :param ell: imprecise value for mean parameter
        :param X:
        :param y:
        :return:
        """
        assert ell > 10 ^ -6, "Using a positive value ELL, otherwise using precise method LDA/QDA."
        self._ell = ell
        self._gp_mean, self._gp_sdp = dict(), dict()
        self._gp_cov, self._gp_icov = dict(), dict()
        self._mean_lower, self._mean_upper = dict(), dict()

        # transformation of Arff data to feature matrix and vector category
        if learn_data_set is not None:
            learn_data_set = np.array(learn_data_set.data)
            X = learn_data_set[:, :-1]
            y = learn_data_set[:, -1]
        elif X is not None and y is not None:
            self._logger.debug("Loading training data set from (X,y) couples.")
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

    def evaluate(self, query, method="quadratic", criterion="maximality", log_probability=False):
        """
        This method is ..

        :param query: new unlabeled instance
        :param method: optimization method for computing the upper conditional probability
                    either quadratic or nonlinear
        :param criterion: interval_dominance if criterion decision performs with Interval dominance,
                otherwise maximality criterion
        :param log_probability: calculate the maximality decision with log probability
        :return: tuple composed from
                     (1) set-valued predictions  and
                     (2) (lower,upper) bound conditional probabilities  X|Y

        Notations:
                - inf: infimum   and   sup: supremum
        """
        self._bound_cond_probabilities = dict((clazz, dict()) for clazz in self._clazz)
        estimated_bounds = dict((clazz, dict()) for clazz in self._clazz)
        precise_probs = dict()
        eng_session = self._eng

        def __get_bounds(clazz, bound="inf"):
            return estimated_bounds[clazz][bound] if bound in estimated_bounds[clazz] else None

        def __put_bounds(probability, estimator, clazz, bound="inf"):
            estimated_bounds[clazz][bound] = (probability, estimator)

        def __probability(_self, query, clazz, inv, cov, mean_lower, mean_upper, bound="inf", log_p=False):

            if __get_bounds(clazz, bound) is None:
                q = -1 * (query.T @ inv)  # because of transforming supremum to min quadratic program
                _self._logger.debug("[BOUND_ESTIMATION:%s] Q matrix: %s", bound, inv)
                _self._logger.debug("[BOUND_ESTIMATION:%s] q vector: %s", bound, q)
                _self._logger.debug("[BOUND_ESTIMATION:%s] Lower Bound: %s", bound, mean_lower)
                _self._logger.debug("[BOUND_ESTIMATION:%s] Upper Bound %s", bound, mean_upper)
                if bound == "inf":
                    estimator = _self.infimum_estimation(inv, q, mean_lower, mean_upper, eng_session, clazz)
                else:
                    estimator = _self.supremum_estimation(inv, q, mean_lower, mean_upper, clazz, method)
                prob = _self.probability_density_gaussian(np.array(estimator), cov, query, log_p=log_p)
                __put_bounds(prob, estimator, clazz, bound)
            return __get_bounds(clazz, bound)

        if criterion == "maximality":
            for clazz in self._clazz:
                mean_lower = self._mean_lower[clazz]
                mean_upper = self._mean_upper[clazz]
                cov, inv = self.get_cov_by_clazz(clazz)
                __probability(self, query, clazz, inv, cov,
                              mean_lower=mean_lower,
                              mean_upper=mean_upper,
                              bound='sup',
                              log_p=log_probability)
                precise_probs[clazz] = self.probability_density_gaussian(self._gp_mean[clazz],
                                                                         cov, query,
                                                                         log_p=log_probability)

            C = set(self._clazz)
            Z = set([])
            while len(C - Z) > 0:
                max_clazz = max(precise_probs, key=precise_probs.get)
                mean_lower = self._mean_lower[max_clazz]
                mean_upper = self._mean_upper[max_clazz]
                cov, inv = self.get_cov_by_clazz(max_clazz)
                p_inf, _ = __probability(self, query, max_clazz, inv, cov,
                                         mean_lower=mean_lower,
                                         mean_upper=mean_upper,
                                         bound='inf',
                                         log_p=log_probability)
                nopt_clazz = set([])
                for clazz in C - {max_clazz}:
                    p_sup, _ = __get_bounds(clazz=clazz, bound="sup")

                    # if p_inf * self._prior[max_clazz] - p_sup * self._prior[clazz] <= 0: # precise 10e-18 instead 0
                    maximality = ((p_inf * self._prior[max_clazz] - p_sup * self._prior[clazz]) <= 0)
                    if log_probability:
                        maximality = (np.log(self._prior[max_clazz]) + p_inf <= p_sup + np.log(self._prior[clazz]))

                    self._logger.debug(
                        "Query: (%s) preferred/indifferent to (%s) according to maximality decision (%s)",
                        max_clazz, clazz, maximality)
                    # (maximality:true) labels indifferent to maximal-label-choice for assessment maximality
                    if maximality:
                        nopt_clazz.add(clazz)
                    else:
                        precise_probs.pop(clazz, None)
                del precise_probs[max_clazz]
                Z.add(max_clazz)
                nopt_clazz.add(max_clazz)
                C = nopt_clazz.copy()

            self._bound_cond_probabilities = estimated_bounds
            return list(C)

        elif criterion == "interval_dominance" or criterion == "maximality_v1":
            if log_probability:
                raise Exception("Criterion decision does not support with log-probability!!")

            lower, upper = [], []
            for clazz in self._clazz:
                mean_lower = self._mean_lower[clazz]
                mean_upper = self._mean_upper[clazz]
                cov, inv = self.get_cov_by_clazz(clazz)
                p_inf, _ = __probability(self, query, clazz, inv, cov,
                                         mean_lower=mean_lower,
                                         mean_upper=mean_upper,
                                         bound='inf',
                                         log_p=log_probability)
                p_sup, _ = __probability(self, query, clazz, inv, cov,
                                         mean_lower=mean_lower,
                                         mean_upper=mean_upper,
                                         bound='sup',
                                         log_p=log_probability)
                lower.append(p_inf * self._prior[clazz])
                upper.append(p_sup * self._prior[clazz])

            self._bound_cond_probabilities = estimated_bounds
            if criterion == "interval_dominance":
                score = Scores(np.c_[lower, upper])
                return self._clazz[(score.nc_intervaldom_decision() > 0)]
            else:
                return inference_maximal_criterion(lower, upper, self._clazz)
        else:
            raise Exception("Decision criterion not implemented yet or another bug!!")

    def get_bound_cond_probability(self):
        return self._bound_cond_probabilities

    def get_marginal_probabilities(self):
        return self._prior

    def get_bound_means(self, clazz):
        return self._mean_lower[clazz], self._mean_upper[clazz]

    def _nb_by_clazz(self, clazz):
        assert self._data is not None, "It's necessary to firstly declare a data set."
        return len(self._data[self._data.y == clazz])

    def __mean_by_clazz(self, clazz):
        return self._data[self._data.y == clazz].iloc[:, :-1].mean().values  # old call: .as_matrix()

    def _cov_by_clazz(self, clazz):
        _sub_data = self._data[self._data.y == clazz].iloc[:, :-1]
        _n, _p = _sub_data.shape
        if _n > 1:
            return _sub_data.cov().values  # old call: .as_matrix()
        else:
            # Bug: Impossible to compute covariance of just ONE instance
            # assuming an identity covariance matrix
            # e.g. _sub_data.shape = (1, 8)
            raise Exception("it has only 1 sample in class, covariance is ill defined.")

    def supremum_estimation(self, Q, q, mean_lower, mean_upper, clazz, method="quadratic"):
        """
        .. warning::
            * this case does not convergence with quadratic method
                Q = np.array([[19.42941344,  -12.9899322, -5.1907171,   -0.25782677],
                              [-12.9899322,  15.97805787, 1.87087712,   -6.72150886],
                              [-5.1907171,   1.87087712,  36.99333345,  -16.21139038],
                              [-0.25782677,  -6.72150886, -16.21139038, 103.0762929]])
                q = np.array([-45.3553788, 26.52058282, -99.63769322, -61.59361441])
                mean_lower = np.array([4.94791667, 3.36875, 1.41666667, 0.19375])
                mean_upper = np.array([5.04375, 3.46458333, 1.5125, 0.28958333])

        :param Q:
        :param q:
        :param mean_lower:
        :param mean_upper:
        :param clazz:
        :param method:
        :return:
        """
        self._logger.debug("[iS-Inverse-Covariance-SDP] (%s, %s)", clazz, self._gp_sdp[clazz])
        if self._gp_sdp[clazz]:
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
                    if x is None:
                        return 0, matrix(0.0, (d, 1))
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
                self._logger.info("[Solution-not-Optimal] %s", solution)
                # raise Exception("Not exist solution optimal!!")

            return [v for v in solution['x']]
        else:
            self._logger.debug("Eigenvalues of inverse covariance matrix %s",
                               np.round(np.linalg.eigvals(Q), decimals=16))
            if self.__solver_matlab:
                return self.quadprogbb(Q, q, mean_lower, mean_upper, self._eng, clazz)
            else:
                return self.nonconvex_qcqp(Q, q, mean_lower, mean_upper)

    def quadprogbb(self, Q, q, mean_lower, mean_upper, eng_session, clazz):
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
        if eng_session is None:
            raise Exception("Environment matlab hadn't been initialized.")
        _debug = is_level_debug(self._logger)
        __out = None if _debug else io.StringIO()
        __err = io.StringIO()
        try:
            Q_m = matlab.double(Q.tolist())
            q_m = eng_session.transpose(matlab.double(q.tolist()))
            LB = eng_session.transpose(matlab.double(mean_lower.tolist()))
            UB = eng_session.transpose(matlab.double(mean_upper.tolist()))
            A = matlab.double([])
            b = matlab.double([])
            Aeq = matlab.double([])
            beq = matlab.double([])
            x, _, _, _ = eng_session.quadprogbb(Q_m, q_m, A, b, Aeq, beq, LB, UB,
                                                nargout=4, stdout=__out, stderr=__err)
            self.__nb_rebuild_opt = 0
        except Exception as e:
            self._logger.debug("[DEBUG_MATHLAB_EXCEPTION] %s", __err.getvalue())
            self._logger.debug("[DEBUG_MATHLAB_EXCEPTION] %s", e)
            self._logger.debug("[DEBUG_MATHLAB_OUTPUT]: Crash QuadProgBB, inputs:class %s", clazz)
            self._logger.debug("[DEBUG_MATHLAB_OUTPUT] Q matrix: %s", Q)
            self._logger.debug("[DEBUG_MATHLAB_OUTPUT] q vector: %s", q)
            self._logger.debug("[DEBUG_MATHLAB_OUTPUT] Lower Bound: %s", mean_lower)
            self._logger.debug("[DEBUG_MATHLAB_OUTPUT] Upper Bound %s", mean_upper)
            # In the cases matlab crash and session is lost, so we need a new session
            self._eng = matlab.engine.start_matlab()
            for _in_path in self._add_path_matlab: self._eng.addpath(_in_path)
            # In case, MATHLAB crash 3 times, optimal point will take mean of category
            if self.__nb_rebuild_opt > 2:
                self.__nb_rebuild_opt = 0
                return self.get_mean_by_clazz(clazz)
            self.__nb_rebuild_opt += 1
            self._logger.debug("[DEBUG_MATHLAB_OUTPUT] New MATHLAB LOADING")
            return self.quadprogbb(Q, q, mean_lower, mean_upper, self._eng, clazz)

        return np.asarray(x).reshape((1, self._p))[0]

    def infimum_estimation(self, Q, q, mean_lower, mean_upper, eng_session, clazz):
        if self.__solver_matlab:
            return self.quadprogbb((-1 * Q), (-1 * q), mean_lower, mean_upper, eng_session, clazz)
        else:
            return self.nonconvex_qcqp((-1 * Q), (-1 * q), mean_lower, mean_upper)

    def nonconvex_qcqp(self, Q, q, mean_lower, mean_upper):
        x = cvx.Variable(self._p)
        problem = cvx.Problem(
            cvx.Minimize(
                cvx.quad_form(x, 0.5 * Q) + q * x
            ),
            [x >= mean_lower, x <= mean_upper]
        )
        qcqp_solver = QCQP(problem)
        qcqp_solver.suggest(RANDOM)
        f_cd, v_cd = qcqp_solver.improve(COORD_DESCENT, num_iters=500)
        self._logger.debug("Coordinate descent: objective %s, violation %s", f_cd, v_cd)
        solution = np.asarray(x.value).reshape((1, self._p))[0]
        self._logger.debug("Optimal value solution %s", solution)
        return solution


class EuclideanDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    """
        Imprecise Euclidean Distance Discriminant implemented with a
        imprecise gaussian distribution and conjugate exponential family.
    """

    def __init__(self, solver_matlab=False, add_path_matlab=None, DEBUG=False):
        super(EuclideanDiscriminant, self).__init__(solver_matlab=solver_matlab,
                                                    add_path_matlab=add_path_matlab)
        self._logger = create_logger("IEDA", DEBUG)

    def get_cov_by_clazz(self, clazz):
        if clazz not in self._gp_cov:
            self._gp_cov[clazz] = np.identity(self._p)
            self._gp_icov[clazz] = np.identity(self._p)
        return self._gp_cov[clazz], self._gp_icov[clazz]

    def supremum_estimation(self, Q, q, mean_lower, mean_upper, clazz, method="quadratic"):
        optimal = np.zeros(self._p)
        cov, _ = self.get_cov_by_clazz(clazz)
        x = (-1) * (q.T @ cov)  # return true query value
        for i in range(self._p):
            if mean_lower[i] <= x[i] <= mean_upper[i]:
                optimal[i] = x[i]
            elif pow(x[i] - mean_lower[i], 2) < pow(x[i] - mean_upper[i], 2):
                optimal[i] = mean_lower[i]
            else:
                optimal[i] = mean_upper[i]

        return optimal

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

    def __init__(self, solver_matlab=True, add_path_matlab=None, DEBUG=False):
        super(LinearDiscriminant, self).__init__(solver_matlab=solver_matlab,
                                                 add_path_matlab=add_path_matlab)
        self._is_compute_total_cov = False
        self._logger = create_logger("ILDA", DEBUG)
        if DEBUG:
            solvers.options['show_progress'] = True

    def learn(self, learn_data_set=None, ell=None, X=None, y=None):
        self._is_compute_total_cov = False
        super(LinearDiscriminant, self).learn(learn_data_set, ell, X, y)

    def get_cov_by_clazz(self, clazz):
        """
        :return: cov, inv, det of empirical total covariance matrix
        """
        if not self._is_compute_total_cov:
            # estimation of empirical total covariance matrix
            _cov, _inv = np.zeros((self._p, self._p)), np.zeros((self._p, self._p))
            for clazz_gp in self._clazz:
                try:
                    covClazz = super(LinearDiscriminant, self)._cov_by_clazz(clazz_gp)
                except Exception as e:  # if class does not have  1 instance, so matrix-covariance 0
                    self._logger.info("Class %s with one instance, exception: %s", clazz_gp, e)
                    covClazz = 0
                _nb_instances_by_clazz = self._nb_by_clazz(clazz_gp)
                _cov += covClazz * (_nb_instances_by_clazz - 1)  # biased estimator
            _cov = _cov / (self._N - self._nb_clazz)  # unbiased estimator group

            # Hint: Improving this method using the SVD method for computing pseudo-inverse matrix
            if linalg.cond(_cov) < 1 / sys.float_info.epsilon:
                _inv = linalg.inv(_cov)
            else:  # computing pseudo-inverse matrix with a singular-value decomposition (SVD)
                _inv = linalg.pinv(_cov)

            _sdp_sys = is_sdp_symmetric(_inv)
            for clazz_gp in self._clazz:
                self._gp_cov[clazz_gp] = _cov
                self._gp_icov[clazz_gp] = _inv
                self._gp_sdp[clazz_gp] = _sdp_sys

            self._is_compute_total_cov = True
        return self._gp_cov[clazz], self._gp_icov[clazz]


class QuadraticDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    """
       Imprecise Quadratic Discriminant implemented with a imprecise gaussian distribution and
       conjugate exponential family.
    """

    def __init__(self, solver_matlab=True, add_path_matlab=None, DEBUG=False):
        super(QuadraticDiscriminant, self).__init__(solver_matlab=solver_matlab,
                                                    add_path_matlab=add_path_matlab)
        self._logger = create_logger("IQDA", DEBUG)
        if DEBUG:
            solvers.options['show_progress'] = True


class NaiveDiscriminant(EuclideanDiscriminant, metaclass=abc.ABCMeta):
    """
        Imprecise Euclidean Distance Discriminant implemented with a
        imprecise gaussian distribution and conjugate exponential family.
    """

    def __init__(self, solver_matlab=False, add_path_matlab=None, DEBUG=False):
        super(NaiveDiscriminant, self).__init__(solver_matlab=solver_matlab,
                                                add_path_matlab=add_path_matlab)
        self._logger = create_logger("INDA", DEBUG)

    def get_cov_by_clazz(self, clazz):
        if clazz not in self._gp_cov:
            cov_clazz = self._cov_by_clazz(clazz)
            self._gp_cov[clazz], self._gp_icov[clazz] = \
                NaiveDiscriminant.compute_diagonal_cov_and_inv(cov_clazz)
        return self._gp_cov[clazz], self._gp_icov[clazz]

    @staticmethod
    def compute_diagonal_cov_and_inv(cov_clazz):
        diagonal = np.einsum('ii->i', cov_clazz)
        save = diagonal.copy()
        cov_clazz[...] = 0
        diagonal[...] = save
        if linalg.cond(cov_clazz) < 1 / sys.float_info.epsilon:
            inverse_cov = linalg.inv(cov_clazz)
            # self._dcov[clazz] = linalg.det(cov_clazz)
        else:  # computing pseudo inverse to a singular covariance matrix
            inverse_cov = linalg.pinv(cov_clazz)
            # eig_values, _ = linalg.eig(cov_clazz)
            # self._dcov[clazz] = np.product(eig_values[(eig_values > 1e-12)])
        return cov_clazz.copy(), inverse_cov.copy()


__MODEL_TYPES = {'ieda': EuclideanDiscriminant, 'ilda': LinearDiscriminant,
                 'iqda': QuadraticDiscriminant, 'inda': NaiveDiscriminant}


def _factory_igda_model(model_type, **kwargs):
    try:
        return __MODEL_TYPES[model_type.lower()](**kwargs)
    except Exception as _:
        raise Exception("Selected model does not exist")
