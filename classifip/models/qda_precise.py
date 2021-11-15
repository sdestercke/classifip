import random, numpy as np, pandas as pd, sys
from numpy import linalg
from .qda import NaiveDiscriminant, LinearDiscriminant
from scipy.stats import multivariate_normal


class BaseEstimator:
    def __init__(self):
        self._data, self._N, self._p = None, 0, 0
        self._clazz, self._nb_clazz = None, None
        self._means, self._prior = dict(), dict()
        self._cov = dict()

    def learn(self, X, y):
        self._N, self._p = X.shape
        self._data = pd.concat([pd.DataFrame(X, dtype="float64"), pd.Series(y, dtype="category")], axis=1)
        columns = ["x" + i for i in map(str, range(self._p))]
        columns.extend('y')
        self._data.columns = columns
        self._clazz = np.array(self._data.y.cat.categories.tolist())
        self._nb_clazz = len(self._clazz)

    def _nb_by_clazz(self, clazz):
        assert self._data is not None, "It's necessary to firstly declare a data set."
        return len(self._data[self._data.y == clazz])

    def _cov_by_clazz(self, clazz):
        _sub_data = self._data[self._data.y == clazz].iloc[:, :-1]
        _n, _p = _sub_data.shape
        if _n > 1:
            return _sub_data.cov().values
        else:
            raise Exception("it has only 1 sample in class, covariance is ill defined.")

    def pdf(self, query, mean, inv_cov, det_cov):
        _exp = -0.5 * ((query - mean).T @ inv_cov @ (query - mean))
        _const = np.power(det_cov, -0.5) / np.power(2 * np.pi, self._p / 2)
        return _const * np.exp(_exp)

    def evaluate(self, queries, with_posterior=False):
        predict_clazz = list()
        probabilities_query = list()
        for query in queries:
            pbs = np.array([
                multivariate_normal.pdf(query, mean=self._means[clazz], cov=self._cov[clazz],
                                        allow_singular=True) * self._prior[clazz]
                for clazz in self._clazz
            ])
            predict_clazz.append(self._clazz[pbs.argmax()])
            sum_pbs = np.sum(pbs)
            probabilities_query.append([pbs[i] / sum_pbs for i in range(len(self._clazz))])
        if with_posterior:
            return predict_clazz, probabilities_query
        else:
            return predict_clazz


class LinearDiscriminantPrecise(BaseEstimator):

    def learn(self, X, y):
        super(LinearDiscriminantPrecise, self).learn(X, y)
        for clazz in self._clazz:
            self._means[clazz] = self._data[self._data.y == clazz].iloc[:, :-1].mean().values
            self._prior[clazz] = len(self._data[self._data.y == clazz]) / self._N
            # @salmuz ToDo: Refactoring this duplicate code
            _cov, _inv = np.zeros((self._p, self._p)), np.zeros((self._p, self._p))
            for clazz_gp in self._clazz:
                try:
                    covClazz = self._cov_by_clazz(clazz_gp)
                except Exception as e:  # if class does not have  1 instance, so matrix-covariance 0
                    self._logger.info("Class %s with one instance, exception: %s", clazz_gp, e)
                    covClazz = 0
                _nb_instances_by_clazz = self._nb_by_clazz(clazz_gp)
                _cov += covClazz * (_nb_instances_by_clazz - 1)  # biased estimator
            _cov = _cov / (self._N - self._nb_clazz)  # unbiased estimator group
            self._cov[clazz] = _cov


class QuadraticDiscriminantPrecise(BaseEstimator):

    def learn(self, X, y):
        super(QuadraticDiscriminantPrecise, self).learn(X, y)
        for clazz in self._clazz:
            self._means[clazz] = self._data[self._data.y == clazz].iloc[:, :-1].mean().values
            self._prior[clazz] = len(self._data[self._data.y == clazz]) / self._N
            self._cov[clazz] = np.cov(self._data[self._data.y == clazz].iloc[:, :-1], rowvar=False)


class EuclideanDiscriminantPrecise(BaseEstimator):

    def learn(self, X, y):
        super(EuclideanDiscriminantPrecise, self).learn(X, y)
        for clazz in self._clazz:
            self._means[clazz] = self._data[self._data.y == clazz].iloc[:, :-1].mean().values
            self._prior[clazz] = len(self._data[self._data.y == clazz]) / self._N
            self._cov[clazz] = np.identity(self._p)


class NaiveDiscriminantPrecise(BaseEstimator):
    """
        Similar to classifier: sklearn.naive_bayes.GaussianNB (verified)
    """

    def learn(self, X, y):
        super(NaiveDiscriminantPrecise, self).learn(X, y)
        # cov_total = np.diag(np.var(self._data.iloc[:, :-1])) # Naive with variance global
        for clazz in self._clazz:
            self._means[clazz] = self._data[self._data.y == clazz].iloc[:, :-1].mean().values
            self._prior[clazz] = len(self._data[self._data.y == clazz]) / self._N
            cov_clazz = np.cov(self._data[self._data.y == clazz].iloc[:, :-1], rowvar=False)
            self._cov[clazz], _ = NaiveDiscriminant.compute_diagonal_cov_and_inv(cov_clazz)


MODEL_TYPES_PRECISE = {'lda': LinearDiscriminantPrecise, 'qda': QuadraticDiscriminantPrecise,
                       'eda': EuclideanDiscriminantPrecise, 'nda': NaiveDiscriminantPrecise}


def _factory_gda_precise(model_type, **kwargs):
    try:
        return MODEL_TYPES_PRECISE[model_type.lower()](**kwargs)
    except Exception as _:
        raise Exception("Selected model does not exist")
