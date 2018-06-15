import abc

import matlab.engine
import numpy as np
import pandas as pd
from cvxopt import solvers, matrix
from numpy import linalg
import sys


class DiscriminantAnalysis(metaclass=abc.ABCMeta):
    def __init__(self, init_matlab=False):
        # Starting matlab environment
        if init_matlab:
            self._eng = matlab.engine.start_matlab()
        else:
            self._eng = None
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
            self.cov[clazz] = cov_clazz
            if linalg.cond(cov_clazz) < 1 / sys.float_info.epsilon:
                self.inv_cov[clazz] = linalg.inv(cov_clazz)
                self.det_cov[clazz] = linalg.det(cov_clazz)
            else:  # computing pseudo inverse/determinant to a singular covariance matrix
                self.inv_cov[clazz] = linalg.pinv(cov_clazz)
                eig_values, _ = linalg.eig(cov_clazz)
                self.det_cov[clazz] = np.product(eig_values[(eig_values > 1e-12)])
        return self.cov[clazz], self.inv_cov[clazz], self.det_cov[clazz]


class LinearDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    def __init__(self, init_matlab=True):
        """
        ell : be careful with negative ll_upper interval, it needs max(0 , ll_upper)
        :param ell:
        """
        super(LinearDiscriminant, self).__init__(init_matlab=init_matlab)
        self.ell = None
        self.prior = dict()
        self.means, self.inv_cov, self.det_cov = dict(), dict(), dict()
        self._mean_lower, self._mean_upper = dict(), dict()
        # self.cov_group_sample = (None, None, None)

    def __cov_group_sample(self):
        """
        Hint: Improving this method using the SVD method for computing
              pseudo-inverse matrix
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

        def __get_bounds(clazz, bound="inf"):
            return bounds[clazz][bound] if bound in bounds[clazz] else None

        def __put_bounds(probability, estimator, clazz, bound="inf"):
            bounds[clazz][bound] = (probability, estimator)

        def __probability(_self, query, clazz, inv, det, mean_lower, mean_upper, bound="inf"):

            if __get_bounds(clazz, bound) is None:
                q = query.T @ inv
                if bound == "inf":
                    estimator = _self.__infimum_estimation(inv, q, mean_lower, mean_upper)
                else:
                    estimator = _self.__supremum_estimation(inv, q, mean_lower, mean_upper, method)
                prob_lower = _self.__compute_probability(np.array(estimator), inv, det, query)
                __put_bounds(prob_lower, estimator, clazz, bound)
            return __get_bounds(clazz, bound)

        lower = []
        upper = []
        cov, inv, det = self.__cov_group_sample()
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

    def __compute_probability(self, mean, inv_cov, det_cov, query):
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

    def __infimum_estimation(self, Q, q, mean_lower, mean_upper):
        if self._eng is None:
            raise Exception("Environment matlab hadn't been initialized.")
        try:
            Q = matlab.double((-1 * Q).tolist())
            q = matlab.double(q.tolist())
            LB = matlab.double(mean_lower.tolist())
            UB = matlab.double(mean_upper.tolist())
            A = matlab.double([])
            b = matlab.double([])
            Aeq = matlab.double([])
            beq = matlab.double([])
            x, f_val, time, stats = self._eng.quadprogbb(Q, self._eng.transpose(q), A, b, Aeq, beq,
                                                         self._eng.transpose(LB), self._eng.transpose(UB), nargout=4)
        except matlab.engine.MatlabExecutionError:
            print("Some error in the QuadProgBB code, inputs:")
            print("Q:", Q)
            print("q:", q)
            print("LB", LB, "UP", UB)
            raise Exception("Matlab ERROR execution QuadProgBB")

        return np.asarray(x).reshape((1, self._p))[0]

    def fit_max_likelihood(self, query):
        means, probabilities = dict(), dict()
        cov, inv, det = self.__cov_group_sample()
        for clazz in self._clazz:
            means[clazz] = self.get_mean_by_clazz(clazz)
            probabilities[clazz] = self.__compute_probability(means[clazz], inv, det, query)
        return means, cov, probabilities

    def __repr__(self):
        print("lower:", self._mean_lower)
        print("upper:", self._mean_upper)
        print("group cov", self.__cov_group_sample())
        return "WADA!!"

    ## Plotting for 2D data

    def __check_data_available(self):

        X = self._data.iloc[:, :-1].as_matrix()
        y = self._data.y.tolist()
        if X is None: raise ValueError("It needs to learn one sample training")

        n_row, _ = X.shape
        if not isinstance(y, list): raise ValueError("Y isn't corrected form.")
        if n_row != len(y): raise ValueError("The number of column is not same in (X,y)")
        return X, np.array(y)

    def plot2D_classification(self, query=None, colors=None, markers=['*', 'v', 'o', '+', '-', '.', ',']):

        X, y = self.__check_data_available()
        n_row, n_col = X.shape

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        c_map = plt.cm.get_cmap("hsv", self._nb_clazz + 1)
        colors = dict((self._clazz[idx], c_map(idx)) for idx in range(0, self._nb_clazz)) \
            if colors is None else colors
        markers = dict((self._clazz[idx], markers[idx]) for idx in range(0, self._nb_clazz))

        def plot_constraints(lower, upper, _linestyle="solid"):
            plt.plot([lower[0], lower[0], upper[0], upper[0], lower[0]],
                     [lower[1], upper[1], upper[1], lower[1], lower[1]],
                     linestyle=_linestyle)
            plt.grid()

        def plot2D_scatter(X, y):
            for row in range(0, len(y)):
                plt.scatter(X[row, 0], X[row, 1], marker=markers[y[row]], c=colors[y[row]])

        def plot_ellipse(splot, mean, cov, color):
            from scipy import linalg

            v, w = linalg.eigh(cov)
            u = w[0] / linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi
            ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                                      180 + angle, facecolor="none",
                                      edgecolor=color,
                                      linewidth=2, zorder=2)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.9)
            splot.add_artist(ell)

        if n_col == 2:
            for clazz in self._clazz:
                post_mean_lower = self._mean_lower[clazz]
                post_mean_upper = self._mean_upper[clazz]
                plot_constraints(post_mean_lower, post_mean_upper)
                mean = self.get_mean_by_clazz(clazz)
                prior_mean_lower = mean - self.ell
                prior_mean_upper = mean + self.ell
                plot_constraints(prior_mean_lower, prior_mean_upper, _linestyle="dashed")

            if query is not None:
                ml_mean, ml_cov, ml_prob = self.fit_max_likelihood(query)
                plt.plot([query[0]], [query[1]], marker='h', markersize=5, color="black")
                _, _bounds = self.evaluate(query)
                for clazz in self._clazz:
                    plt.plot([ml_mean[clazz][0]], [ml_mean[clazz][1]], marker='o', markersize=5, color=colors[clazz])
                    _, est_mean_lower = _bounds[clazz]['inf']
                    _, est_mean_upper = _bounds[clazz]['sup']
                    plt.plot([est_mean_lower[0]], [est_mean_lower[1]], marker='x', markersize=4, color="black")
                    plt.plot([est_mean_upper[0]], [est_mean_upper[1]], marker='x', markersize=4, color="black")

            cov, inv, det = self.__cov_group_sample()
            s_plot = plt.subplot()
            for clazz in self._clazz:
                mean = self.get_mean_by_clazz(clazz)
                plot_ellipse(s_plot, mean, cov, colors[clazz])

        elif n_col > 2:
            if query is not None:
                inference, _ = self.evaluate(query)
                X = np.vstack([X, query])
                y = np.append(y, inference[0])

            from sklearn.manifold import Isomap
            iso = Isomap(n_components=2)
            projection = iso.fit_transform(X)
            X = np.c_[projection[:, 0], projection[:, 1]]

            if query is not None:
                color_instance = colors[inference[0]] if len(inference) == 1 else 'black'
                plt.plot([X[n_row, 0]], [X[n_row, 1]], color='red', marker='o', mfc=color_instance)
        else:
            raise Exception("Not implemented for one feature yet.")

        plot2D_scatter(X, y)
        plt.show()

    def plot2D_decision_boundary(self, h=.1, markers=['*', 'v', 'o', '+', '-', '.', ',']):

        X, y = self.__check_data_available()

        _, n_col = X.shape

        if n_col > 2: raise Exception("Not implemented for n-dimension yet.")

        import matplotlib.pyplot as plt

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        z = np.array([])
        clazz_by_index = dict((clazz, idx) for idx, clazz in enumerate(self._clazz, 1))
        NO_CLASS_COMPARABLE = -1
        for query in np.c_[xx.ravel(), yy.ravel()]:
            answer, _ = self.evaluate(query)
            if len(answer) > 1 or len(answer) == 0:
                z = np.append(z, NO_CLASS_COMPARABLE)
            else:
                z = np.append(z, clazz_by_index[answer[0]])

        y_colors = [clazz_by_index[clazz] for clazz in y]
        markers = dict((self._clazz[idx], markers[idx]) for idx in range(0, self._nb_clazz))
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, alpha=0.4)
        for row in range(0, len(y)):
            plt.scatter(X[row, 0], X[row, 1], c=y_colors[row], s=40, marker= markers[y[row]], edgecolor='k')
        plt.show()

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
                          self.__compute_probability(np.append(optimal, current), inv, det, query),
                          cost_Fx(np.append(optimal, current), query, inv))

        forRecursive(lower, upper, 0, d, np.array([]))

    def show_all_brute_optimal(self, query):
        cov, inv, det = self.__cov_group_sample()
        for clazz in self._clazz:
            mean_lower = self._mean_lower[clazz]
            mean_upper = self._mean_upper[clazz]
            print("box", mean_lower, mean_upper)
            self.__brute_force_search(clazz, query, mean_lower, mean_upper, inv, det, self._p)
