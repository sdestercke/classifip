from classifip.models.qda import EuclideanDiscriminant, LinearDiscriminant, QuadraticDiscriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import random
import xxhash

MODEL_TYPES = {'ieda': EuclideanDiscriminant, 'ilda': LinearDiscriminant, 'iqda': QuadraticDiscriminant}
MODEL_TYPES_PRECISE = {'lda': LinearDiscriminantAnalysis, 'qda': QuadraticDiscriminantAnalysis}
INFIMUM, SUPREMUM = "inf", "sup"


def __factory_model(model_type, **kwargs):
    try:
        return MODEL_TYPES[model_type.lower()](**kwargs)
    except:
        raise Exception("Selected model does not exist")


def __factory_model_precise(model_type, **kwargs):
    try:
        return MODEL_TYPES_PRECISE[model_type.lower()](**kwargs)
    except:
        raise Exception("Selected model does not exist")


def generate_seeds(nb_seeds):
    return [random.randrange(pow(2, 30)) for i in range(nb_seeds)]


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
            print("--->", q_hash)
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
