from classifip.dataset.arff import ArffFile
from classifip.representations.voting import Scores
from classifip.models.mlc.mlcncc import MLCNCC
from classifip.models.qda import NaiveDiscriminant, LinearDiscriminant
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from classifip.models.qda_precise import NaiveDiscriminantPrecise
import numpy as np
from math import exp


class NDABR(MLCNCC):
    """
        Naive discriminant analysis for binary relevance

        - It is mandatory that data is normalised (scaling feature)
        for using an imprecise parameter to values c = {0, 0.5, 1, 1.5, 2.0}
    """

    def __init__(self):
        """Build an empty NCCBR structure """
        super(NDABR, self).__init__()
        self.nda_models = None
        self.nb_feature = None
        self._nda_imprecise = LinearDiscriminant(solver_matlab=True, DEBUG=True)

    def learn(self,
              learn_data_set,
              nb_labels,
              ell_imprecision=0.5):
        self.__init__()

        self.nb_labels = nb_labels
        self.training_size = len(learn_data_set.data)
        self.label_names = learn_data_set.attributes[-self.nb_labels:]
        self.feature_names = learn_data_set.attributes[:-self.nb_labels]
        self.nb_feature = len(self.feature_names)
        # create the naive discriminant models
        self.nda_models = dict()
        _np_data = np.array(learn_data_set.data)
        for label_value in self.label_names:
            label_index = learn_data_set.attributes.index(label_value)
            X_learning, y_learning = list(), list()
            for row_index, raw_instance in enumerate(learn_data_set.data):
                if raw_instance[label_index] != '-1':
                    X_learning.append(_np_data[row_index, :self.nb_feature])
                    y_learning.append(_np_data[row_index, label_index])
            X_learning = np.array(X_learning, dtype=np.float)
            y_learning = np.array(y_learning)

            # nda_imprecise = NaiveDiscriminant(solver_matlab=False, DEBUG=False)
            # nda_imprecise.learn(X=X_learning, y=y_learning, ell=ell_imprecision)
            # nda_precise = NaiveDiscriminantPrecise()
            nda_precise = LinearDiscriminantAnalysis()
            nda_precise.fit(X=X_learning, y=y_learning)
            self.nda_models[label_value] = dict({
                # "imprecise": nda_imprecise,
                "imprecise": dict({'X': X_learning, 'y': y_learning, 'ell': ell_imprecision}),
                "precise": nda_precise
            })

    def evaluate(self, test_dataset, **kwargs):
        answers = []
        for instance in test_dataset:
            # validate instance is np-array
            instance = np.array(instance)
            if len(instance) > self.nb_feature:
                instance = np.array(instance[:self.nb_feature], dtype=float)
            else:
                instance = instance.astype(dtype=float)

            skeptic = [None] * self.nb_labels
            precise = [None] * self.nb_labels
            precise_proba = [None] * self.nb_labels
            for i, label_value in enumerate(self.label_names):
                models = self.nda_models[label_value]
                # imprecise inference

                self._nda_imprecise.learn(X=models["imprecise"]["X"],
                                          y=models["imprecise"]["y"],
                                          ell=models["imprecise"]["ell"])

                evaluate = self._nda_imprecise.evaluate(query=instance)
                skeptic[i] = -1 if len(evaluate) > 1 else int(evaluate[0])
                # precise inference
                # evaluate, probabilities = models["precise"].evaluate(queries=[instance],
                #                                                      with_posterior=True)
                # precise[i] = int(evaluate[0])
                # conditional_probXY = probabilities[0]['1']
                # marginal_probX = probabilities[0]['1'] + probabilities[0]['0']
                # precise_proba[i] = conditional_probXY / marginal_probX
                # Precise Sklearn
                precise[i] = models["precise"].predict([instance])[0]
                probabilities = models["precise"].predict_proba([instance])
                precise_proba[i] = probabilities[0][1]
            answers.append((skeptic, precise, precise_proba))
        return answers
