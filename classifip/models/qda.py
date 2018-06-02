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
        else: self._eng = None
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
            self.inv_cov[clazz] = inv(cov_clazz)
            self.det_cov[clazz] = det(cov_clazz)
        return self.cov[clazz], self.inv_cov[clazz], self.det_cov[clazz]


class LinearDiscriminant(DiscriminantAnalysis, metaclass=abc.ABCMeta):
    def __init__(self, ell=20, init_matlab=True):
        """
        ell : be careful with negative ll_upper interval, it needs max(0 , ll_upper)
        :param ell:
        """
        super(LinearDiscriminant, self).__init__(init_matlab=init_matlab)
        self.ell = ell
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
        self._clazz = np.array(self._data.y.cat.categories.tolist())
        self._nb_clazz = len(self._clazz)

        for clazz in self._data.y.cat.categories.tolist():
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
            lower.append(p_inf)
            upper.append(p_up)


        inference = np.divide(lower, upper[::-1])
        answers = self._clazz[(inference > 1)]
        print("inf/sup:", np.divide(lower, upper[::-1]))
        #from classifip.representations.intervalsProbability import IntervalsProbability
        #answers.append(IntervalsProbability(np.row_stack((upper, lower))))
        # if len(answers) == 2:
        #     val = np.divide(lower, upper[::-1])
        #     answers = [self._clazz[val.argmax(axis=0)]]
        return answers, bounds, inference[(inference >1)]

    def __compute_probability(self, mean, inv_cov, det_cov, query):
        _exp = -0.5 * ((query - mean).T @ inv_cov @ (query - mean))
        _const = np.power(det_cov, -0.5) / np.power(2 * np.pi, self._p / 2)
        return _const * np.exp(_exp)

    def __supremum_estimation(self, Q, q, mean_lower, mean_upper, method="quadratic"):

        def __min_convex_qp(A, q, lower, upper, d):
            ell_lower = matrix(lower, (d, 1))
            ell_upper = matrix(upper, (d, 1))
            P = matrix(A)
            q = matrix(-1*q)
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
            raise Exception("Not exist solution optimal!!")

        return [v for v in solution['x']]

    def __infimum_estimation(self, Q, q, mean_lower, mean_upper):
        if self._eng is None:
            raise Exception("Environment matlab hadn't been initialized.")

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
        return np.asarray(x).reshape((1, self._p))[0]

    def fit_max_likelihood(self, query):
        means, probabilities = dict(), dict()
        cov, inv, det = self.__cov_group_sample()
        for clazz in self._clazz:
            means[clazz] = self.get_mean_by_clazz(clazz)
            probabilities[clazz] = self.__compute_probability(means[clazz], inv, det, query)
        return means, cov, probabilities

    ## Plotting for 2D data

    def __check_data_available(self):

        X = self._data.iloc[:, :-1].as_matrix()
        y = self._data.y.tolist()
        if X is None: raise ValueError("It needs to learn one sample training")

        n_row, n_col = X.shape
        if n_col != 2: raise ValueError("Dimension in X matrix aren't corrected form.")
        if not isinstance(y, list): raise ValueError("Y isn't corrected form.")
        if n_row != len(y): raise ValueError("The number of column is not same in (X,y)")
        return X, y

    def plot2D_classification(self, query=None, colors={0: 'red', 1: 'blue'}):

        X, y = self.__check_data_available()

        import matplotlib.pyplot as plt

        def plot_constraints(lower, upper):
            plt.plot([lower[0], lower[0], upper[0], upper[0], lower[0]],
                     [lower[1], upper[1], upper[1], lower[1], lower[1]])
            plt.grid()

        def plot2D_scatter(X, y):
            color_by_instance = [colors[c] for c in y]
            plt.scatter(X[:, 0], X[:, 1], c=color_by_instance, marker='+')

        plot2D_scatter(X, y)
        for clazz in self._clazz:
            mean_lower = self._mean_lower[clazz]
            mean_upper = self._mean_upper[clazz]
            plot_constraints(mean_lower, mean_upper)

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

        plt.show()

    def plot2D_decision_boundary(self, colors={0: 'red', 1: 'blue'}, h = .5):

        X, y = self.__check_data_available()

        import matplotlib.pyplot as plt

        def plot2D_scatter(X, y):
            color_by_instance = [colors[c] for c in y]
            plt.scatter(X[:, 0], X[:, 1], c=color_by_instance, marker='+')

        plot2D_scatter(X, y)
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = np.array([])
        NO_CLASS = 999
        for query in np.c_[xx.ravel(), yy.ravel()]:
            answer, _, inf = self.evaluate(query)
            if len(answer) == 0:
                z = np.append(z, NO_CLASS)
            elif len(answer) == 1:
                z = np.append(z, inf[0])
            else:
                raise Exception("There are a error of prediction.")

        z = z.reshape(xx.shape)
        from matplotlib.colors import ListedColormap
        # cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        plt.pcolormesh(xx, yy, z, cmap=cmap_light)
        plt.pcolormesh(xx, yy, z, shading="gouraud")
        plot2D_scatter(X, y)
        # np.savetxt('z_values.txt', z, fmt='%d')
        # np.savetxt('xy_values.txt', np.c_[xx.ravel(), yy.ravel()], fmt='%d')
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

    def supremum_bf(self, query):
        cov, inv, det = self.__cov_group_sample()
        for clazz in self._clazz:
            mean_lower = self._mean_lower[clazz]
            mean_upper = self._mean_upper[clazz]
            print("box", mean_lower, mean_upper)
            self.__brute_force_search(clazz, query, mean_lower, mean_upper, inv, det, self._p)

    def testing_plot(self, h = .05, colors={0: 'red', 1: 'blue'}):

        X, y = self.__check_data_available()

        def plot2D_scatter(X, y):
            color_by_instance = [colors[c] for c in y]
            plt.scatter(X[:, 0], X[:, 1], c=color_by_instance, marker='+')

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = np.loadtxt('z_values.txt')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        print(z.shape, xx.shape)

        # plt.pcolormesh(xx, yy, z, cmap=cmap_light)
        plt.contourf(xx, yy, z)
        plot2D_scatter(X, y)
        # plt.imshow(z, interpolation='nearest', origin='lower', extent=[0,1.5,0,3.78], aspect='auto')

        # print(z, xy[:,1])
        plt.show()


## Testing
current_dir = os.getcwd()
root_path = "/Users/salmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer_.csv"
# root_path = "/Volumes/Data/DBSalmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv"
data = os.path.join(current_dir, root_path)
df_train = pd.read_csv(data)
X = df_train.loc[:, ['x1', 'x2']].values
y = df_train.y.tolist()
lqa = LinearDiscriminant(ell=5, init_matlab=False)
lqa.learn(X, y)
lqa.testing_plot()
# query = np.array([0.830031, 0.108776])
# query = np.array([2, 2])
# answer, _ = lqa.evaluate(query)
# print(answer, _)
# lqa.supremum_bf(query)
# print(lqa.fit_max_likelihood(query))
# lqa.plot2D_classification(query)
# lqa.plot2D_decision_boundary()
# Plots.plot2D_classification(X, y)
# Plots.plot_cov_ellipse(X)
# plt.show()
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
