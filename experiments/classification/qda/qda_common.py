from classifip.models.qda import EuclideanDiscriminant, LinearDiscriminant, QuadraticDiscriminant, NaiveDiscriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import random, xxhash, numpy as np, pandas as pd
from scipy.stats import multivariate_normal as gaussian

INFIMUM, SUPREMUM = "inf", "sup"


class BaseEstimator:
    def __init__(self, store_covariance=False):
        self._data, self._N, self._p = None, 0, 0
        self._clazz, self._nb_clazz = None, None
        self._means, self._prior, self._cov = dict(), dict(), dict()

    def fit(self, X, y):
        self._N, self._p = X.shape
        self._data = pd.concat([pd.DataFrame(X, dtype="float64"), pd.Series(y, dtype="category")], axis=1)
        columns = ["x" + i for i in map(str, range(self._p))]
        columns.extend('y')
        self._data.columns = columns
        self._clazz = np.array(self._data.y.cat.categories.tolist())
        self._nb_clazz = len(self._clazz)

    def predict(self, query):
        pbs = np.array([gaussian.pdf(query, self._means[clazz], self._cov[clazz]) * self._prior[clazz] \
                        for clazz in self._clazz])
        return [self._clazz[pbs.argmax()]]


class EuclideanDiscriminantPrecise(BaseEstimator):

    def fit(self, X, y):
        super(EuclideanDiscriminantPrecise, self).fit(X, y)
        for clazz in self._clazz:
            self._means[clazz] = self._data[self._data.y == clazz].iloc[:, :-1].mean().as_matrix()
            self._prior[clazz] = len(self._data[self._data.y == clazz]) / self._N
            self._cov[clazz] = np.identity(self._p)


class NaiveDiscriminantPrecise(BaseEstimator):

    def fit(self, X, y):
        super(NaiveDiscriminantPrecise, self).fit(X, y)
        #cov_total = np.diag(np.var(self._data.iloc[:, :-1])) # Naive with variance global
        for clazz in self._clazz:
            self._means[clazz] = self._data[self._data.y == clazz].iloc[:, :-1].mean().as_matrix()
            self._prior[clazz] = len(self._data[self._data.y == clazz]) / self._N
            self._cov[clazz] = np.diag(np.var(self._data[self._data.y == clazz].iloc[:, :-1]))


MODEL_TYPES = {'ieda': EuclideanDiscriminant, 'ilda': LinearDiscriminant, 'iqda': QuadraticDiscriminant,
               'inda': NaiveDiscriminant}


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


class StoreQueries:
    def __init__(self, clazz, key_store):
        """
        :param clazz: clazz
        """
        self.key_store = key_store
        self.queries = dict((_c, dict()) for _c in clazz)

    def get_query(self, clazz, query, bound=INFIMUM):
        q_hash = self.__hash(query)
        return self.__get_bounds(clazz, q_hash, bound) if q_hash in self.queries[clazz] else None

    def put_query(self, probability, estimator, clazz, query, bound=INFIMUM):
        q_hash = self.__hash(query)
        if q_hash not in self.queries[clazz]:
            self.queries[clazz][q_hash] = dict()
        self.__put_bounds(probability, estimator, clazz, q_hash, bound)

    def __get_bounds(self, clazz, q_hash, bound):
        return self.queries[clazz][q_hash][bound] if bound in self.queries[clazz][q_hash] else None

    def __put_bounds(self, probability, estimator, clazz, q_hash, bound):
        self.queries[clazz][q_hash][bound] = (probability, estimator)

    def is_same_keystore(self, key_store):
        return self.key_store == key_store

    def __hash(self, query):
        _hash = xxhash.xxh64()
        _hash.update(query)
        res_hash = _hash.hexdigest()
        _hash.reset()
        return res_hash

    def __repr__(self):
        def format(d, tab=0):
            s = ['{\n']
            for k, v in d.items():
                if isinstance(v, dict):
                    v = format(v, tab + 1)
                else:
                    v = repr(v)

                s.append('%s%r: %s,\n' % ('  ' * tab, k, v))
            s.append('%s}' % ('  ' * tab))
            return ''.join(s)

        return format(self.queries)

# If ELL parameter is not same we create a new store_queries
# otherwise we could use the same and improve performance evaluate
# if self.store_queries is None or not self.store_queries.is_same_keystore(self._ell):
#     self.store_queries = StoreQueries(self._clazz, self._ell)
