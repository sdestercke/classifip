import abc, math, time, random, numpy as np, scipy, pandas as pd
from glmnet_python import *
from classifip.utils import create_logger
from ..representations.intervalsProbability import IntervalsProbability


class ImpreciseLogistic(metaclass=abc.ABCMeta):

    def __init__(self, DEBUG):
        self._DEBUG = DEBUG
        self._logger = None
        self._n = None
        self._p = None
        self._clazz = None
        self._X = None
        self._y = None
        self._data = None

    @abc.abstractmethod
    def learn(self, learn_data_set=None, X=None, y=None):
        # transformation of Arff data to feature matrix and vector category
        if learn_data_set is not None:
            learn_data_set = np.array(learn_data_set.data)
            X = learn_data_set[:, :-1]
            y = learn_data_set[:, -1]
        elif X is not None and y is not None:
            self._logger.debug("Loading training data set from (X,y) couples.")
        else:
            raise Exception('Not training data set setting.')

        self._n, self._p = X.shape
        assert self._n == len(y), "Size X and y is not equals."
        y = np.array(y) if type(y) is list else y
        self._data = pd.concat([pd.DataFrame(X, dtype="float64"), pd.Series(y, dtype="category")], axis=1)
        columns = ["x" + i for i in map(str, range(self._p))]  # create columns names
        columns.extend('y')
        self._data.columns = columns
        self._clazz = np.unique(y)
        self._X, self._y = X, y

    def get_data(self):
        return self._data

    def get_clazz(self):
        return self._clazz

    @abc.abstractmethod
    def evaluate(self,
                 test_dataset,
                 with_precise_probabilities=False):
        pass


class BinaryILogisticLasso(ImpreciseLogistic):
    """
        Package used for computing lasso penalization

        https://github.com/bbalasub1/glmnet_python
            Note that it needs to install from the source code
            (github) and not with pip3.6
    """

    def __init__(self, DEBUG=False):
        super(BinaryILogisticLasso, self).__init__(DEBUG)
        self._logger = create_logger("BinaryILogistic", DEBUG)
        self._lasso_models = None
        self._precise_logit = None
        self._gammas = None

    def learn(self, learn_data_set=None,
              X=None, y=None,
              nb_lasso_models=20,
              min_gamma=0, max_gamma=1):
        super(BinaryILogisticLasso, self).learn(learn_data_set=learn_data_set, X=X, y=y)
        # validation binary classification
        assert len(np.unique(self._y)) == 2, "It is not binary classifier."
        idx_random = np.random.choice(range(self._n), self._n, replace=False)
        self._X = self._X[idx_random]
        self._y = self._y[idx_random]
        # transform the values of y in numeric form
        clazz_numeric = np.zeros(self._n)
        clazz_numeric[self._clazz[1] == self._y] = 1.0
        # ToDo: feature scaling
        # it needs for the package glmnet_python
        cv_ridge_fit = cvglmnet(x=self._X,
                                y=clazz_numeric,
                                family='binomial',
                                ptype='class',
                                alpha=0.0)  # ridge penalty
        beta_ridge_fitted = cvglmnetCoef(cv_ridge_fit, s='lambda_1se')
        # sensibility analyse
        self._lasso_models = [None] * nb_lasso_models
        # self._lasso_models[0] = cv_ridge_fit.copy()
        self._gammas = np.linspace(start=0, stop=max_gamma, num=nb_lasso_models)
        for i, gamma in enumerate(self._gammas):
            w_penalty = 1 / abs(beta_ridge_fitted) ** gamma
            self._lasso_models[i] = cvglmnet(x=self._X,
                                             y=clazz_numeric,
                                             family='binomial',
                                             ptype='class',
                                             alpha=1.0,
                                             keep=False,
                                             penalty_factor=w_penalty)
        self._precise_logit = cv_ridge_fit

    def evaluate(self,
                 test_dataset,
                 with_precise_probabilities=False,
                 **kwargs):
        assert len(self._lasso_models) > 0, "No lasso model is learning."
        set_probabilities = np.ones(shape=(len(test_dataset), 2, len(self._lasso_models))) * -1
        answers, answers_precise = [], []
        for t, test in enumerate(test_dataset):
            resulting_int = np.zeros((2, 2))
            for i, lasso_fitted in enumerate(self._lasso_models):
                _probability = cvglmnetPredict(obj=lasso_fitted,
                                               newx=np.array([test]),
                                               s='lambda_1se',
                                               ptype='response')[0]
                set_probabilities[t, 1, i] = _probability
                set_probabilities[t, 0, i] = 1 - _probability
            self._logger.debug("Set of probabilities of class 0: %s",
                               set_probabilities[t, 0, :])
            self._logger.debug("Set of probabilities of class 1: %s",
                               set_probabilities[t, 1, :])
            # computing the lower and upper probability
            resulting_int[0, :] = [np.max(set_probabilities[t, 0, :]),
                                   np.max(set_probabilities[t, 1, :])]
            resulting_int[1, :] = [np.min(set_probabilities[t, 0, :]),
                                   np.min(set_probabilities[t, 1, :])]
            answers.append(IntervalsProbability(resulting_int))
            # computing the precise probability
            precise_probability = np.zeros(2)
            precise_probability[1] = cvglmnetPredict(obj=self._precise_logit,
                                                     newx=np.array([test]),
                                                     s='lambda_1se',
                                                     ptype='response')[0]
            precise_probability[0] = 1 - precise_probability[1]
            answers_precise.append(precise_probability)
        if with_precise_probabilities:
            return answers, answers_precise
        else:
            return answers

    def get_maximality_from_credal(self, credal_set):
        max_decision = credal_set.getmaximaldecision()
        return self._clazz[max_decision == 1]
