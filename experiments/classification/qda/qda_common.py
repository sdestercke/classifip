from classifip.models.qda import EuclideanDiscriminant, LinearDiscriminant, QuadraticDiscriminant, NaiveDiscriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import random, xxhash, numpy as np, pandas as pd, sys
from sklearn.model_selection import KFold
from collections import Counter
from numpy import linalg

INFIMUM, SUPREMUM = "inf", "sup"


class BaseEstimator:
    def __init__(self, store_covariance=False):
        self._data, self._N, self._p = None, 0, 0
        self._clazz, self._nb_clazz = None, None
        self._means, self._prior = dict(), dict()
        self._icov, self._dcov = dict(), dict()

    def fit(self, X, y):
        self._N, self._p = X.shape
        self._data = pd.concat([pd.DataFrame(X, dtype="float64"), pd.Series(y, dtype="category")], axis=1)
        columns = ["x" + i for i in map(str, range(self._p))]
        columns.extend('y')
        self._data.columns = columns
        self._clazz = np.array(self._data.y.cat.categories.tolist())
        self._nb_clazz = len(self._clazz)

    def pdf(self, query, mean, inv_cov, det_cov):
        _exp = -0.5 * ((query - mean).T @ inv_cov @ (query - mean))
        _const = np.power(det_cov, -0.5) / np.power(2 * np.pi, self._p / 2)
        return _const * np.exp(_exp)

    def predict(self, queries):
        predict_clazz = list()
        for query in queries:
            pbs = np.array(
                [self.pdf(query, self._means[clazz], self._icov[clazz], self._dcov[clazz]) * self._prior[clazz] \
                 for clazz in self._clazz])
            predict_clazz.append(self._clazz[pbs.argmax()])
        return predict_clazz


class EuclideanDiscriminantPrecise(BaseEstimator):

    def fit(self, X, y):
        super(EuclideanDiscriminantPrecise, self).fit(X, y)
        for clazz in self._clazz:
            self._means[clazz] = self._data[self._data.y == clazz].iloc[:, :-1].mean().as_matrix()
            self._prior[clazz] = len(self._data[self._data.y == clazz]) / self._N
            self._icov[clazz] = np.identity(self._p)
            self._dcov[clazz] = 1


class NaiveDiscriminantPrecise(BaseEstimator):

    def fit(self, X, y):
        super(NaiveDiscriminantPrecise, self).fit(X, y)
        # cov_total = np.diag(np.var(self._data.iloc[:, :-1])) # Naive with variance global
        for clazz in self._clazz:
            self._means[clazz] = self._data[self._data.y == clazz].iloc[:, :-1].mean().as_matrix()
            self._prior[clazz] = len(self._data[self._data.y == clazz]) / self._N
            cov_clazz = np.diag(np.var(self._data[self._data.y == clazz].iloc[:, :-1]))
            if linalg.cond(cov_clazz) < 1 / sys.float_info.epsilon:
                self._icov[clazz] = linalg.inv(cov_clazz)
                self._dcov[clazz] = linalg.det(cov_clazz)
            else:  # computing pseudo inverse/determinant to a singular covariance matrix
                self._icov[clazz] = linalg.pinv(cov_clazz)
                eig_values, _ = linalg.eig(cov_clazz)
                self._dcov[clazz] = np.product(eig_values[(eig_values > 1e-12)])


MODEL_TYPES = {'ieda': EuclideanDiscriminant, 'ilda': LinearDiscriminant,
               'iqda': QuadraticDiscriminant, 'inda': NaiveDiscriminant}


def __factory_model(model_type, **kwargs):
    try:
        return MODEL_TYPES[model_type.lower()](**kwargs)
    except:
        raise Exception("Selected model does not exist")


MODEL_TYPES_PRECISE = {'lda': LinearDiscriminantAnalysis, 'qda': QuadraticDiscriminantAnalysis,
                       'eda': EuclideanDiscriminantPrecise, 'nda': NaiveDiscriminantPrecise}


def __factory_model_precise(model_type, **kwargs):
    try:
        if model_type == 'lda': kwargs["solver"] = "svd";
        return MODEL_TYPES_PRECISE[model_type.lower()](**kwargs)
    except:
        raise Exception("Selected model does not exist")


def generate_seeds(nb_seeds):
    return [random.randrange(pow(2, 20)) for _ in range(nb_seeds)]


def generate_sample_cross_validation(data_labels, nb_fold_cv=2, minimum_by_label=1):
    nb_by_label = Counter(data_labels)
    # int(xxx*(1-1/nb_fold_cv)) split minimum 2 training and other testing
    if len(nb_by_label) > 0 and minimum_by_label > int(min(nb_by_label.values()) * (1 - 1 / nb_fold_cv)):
        raise Exception('It is not possible to split a minimum number %s of labels for training '
                        ' and others for testing.' % minimum_by_label)

    while True:
        kf = KFold(n_splits=nb_fold_cv, random_state=None, shuffle=True)
        splits, is_minimum_OK = list([]), True
        for idx_train, idx_test in kf.split(data_labels):
            splits.append((idx_train, idx_test))
            nb_by_label = Counter(data_labels[idx_train])
            if len(nb_by_label) > 0 and minimum_by_label > min(nb_by_label.values()):
                is_minimum_OK = False
                break
        if is_minimum_OK:
            break
    return splits