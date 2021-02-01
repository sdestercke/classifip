import numpy as np
from classifip.models.mlc.mlcncc import MLCNCC
from classifip.models.logit import BinaryILogisticLasso
from classifip.representations.voting import Scores
from classifip.representations.probadis import ProbaDis


class Logit_BR(MLCNCC):

    def __init__(self,
                 DEBUG=False):
        super(Logit_BR, self).__init__(DEBUG)
        self.__ibr_models = None

    def learn(self,
              learn_data_set,
              nb_labels):
        """

        :param learn_data_set:
        :param nb_labels:
        :return:
        """
        self.nb_labels = nb_labels
        # Initializing the counts
        self.feature_names = learn_data_set.attributes[:-self.nb_labels]
        self.label_names = learn_data_set.attributes[-self.nb_labels:]
        self.feature_values = learn_data_set.attribute_data.copy()

        _np_data = np.array(learn_data_set.data, dtype=np.float64)
        self.__ibr_models = dict.fromkeys(self.label_names, None)
        for label_value in self.label_names:
            label_index = learn_data_set.attributes.index(label_value)
            not_miss_instances = _np_data[:, label_index] != -1
            X_learning = _np_data[not_miss_instances, :-self.nb_labels]
            y_learning = _np_data[not_miss_instances, label_index]
            self._logger.debug("Learning Imprecise Lasso of label %s.", label_value)
            self.__ibr_models[label_value] = BinaryILogisticLasso(DEBUG=self.DEBUG)
            self.__ibr_models[label_value].learn(X=X_learning, y=y_learning)

    def evaluate(self, test_dataset, **kwargs):
        answers = []
        for test in test_dataset:
            resulting_score = np.zeros((self.nb_labels, 2))
            resulting_mass = [None] * self.nb_labels
            for j, label_value in enumerate(self.label_names):
                evaluate = self.__ibr_models[label_value].evaluate(test_dataset=[test],
                                                                   with_precise_probabilities=True)
                credal_set, precise = evaluate[0][0], evaluate[1][0]
                resulting_score[j, :] = credal_set.lproba[:, 1][::-1]
                resulting_mass[j] = ProbaDis(precise)
            answer = Scores(resulting_score)
            # answers_precises.append(resulting_mass)
            answers.append((answer, resulting_mass))
        return answers
