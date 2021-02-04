import numpy as np, time, sys
from classifip.models.mlc.mlcncc import MLCNCC
from classifip.models.logit import BinaryILogisticLasso
from classifip.representations.voting import Scores
from classifip.representations.probadis import ProbaDis
import multiprocessing
from functools import partial


def _parallel_learn_model(attributes, np_data, nb_labels, nb_lassos_models,
                          min_gamma, max_gamma, DEBUG, label_value):
    label_index = attributes.index(label_value)
    not_miss_instances = np_data[:, label_index] != -1
    X_learning = np_data[not_miss_instances, :-nb_labels]
    y_learning = np_data[not_miss_instances, label_index]
    imprecise_model = BinaryILogisticLasso(DEBUG=DEBUG)
    imprecise_model.learn(X=X_learning, y=y_learning,
                          nb_lasso_models=nb_lassos_models,
                          min_gamma=min_gamma, max_gamma=max_gamma)

    pid = multiprocessing.current_process().name
    print("%s - [Logit_BR:%s] Learning Imprecise Lasso of label %s." %
          (time.strftime('%Y-%m-%d %H:%M:%S'), pid, label_value), flush=True)
    return label_value, imprecise_model


class Logit_BR(MLCNCC):

    def __init__(self,
                 DEBUG=False):
        super(Logit_BR, self).__init__(DEBUG)
        self.__ibr_models = None

    def learn(self,
              learn_data_set,
              nb_labels,
              nb_lassos_models=21,
              min_gamma=0.01,
              max_gamma=1,
              nb_process=1):
        """
            It is preferred to execute this module with python 3.7
            in order to learn the imprecise models in parallel
        :param learn_data_set:
        :param nb_labels:
        :param nb_lassos_models:
        :param min_gamma:
        :param max_gamma:
        :param nb_process:
        :return:
        """
        self.nb_labels = nb_labels
        # Initializing the counts
        self.feature_names = learn_data_set.attributes[:-self.nb_labels]
        self.label_names = learn_data_set.attributes[-self.nb_labels:]
        self.feature_values = learn_data_set.attribute_data.copy()

        _np_data = np.array(learn_data_set.data, dtype=np.float64)
        self.__ibr_models = dict.fromkeys(self.label_names, None)

        if sys.version_info >= (3, 7):
            POOL = multiprocessing.Pool(processes=nb_process)
            target_function = partial(_parallel_learn_model, learn_data_set.attributes,
                                      _np_data, self.nb_labels, nb_lassos_models,
                                      min_gamma, max_gamma, self.DEBUG)
            imprecise_models = POOL.map(target_function, self.label_names)

            for label_value, lasso_model in imprecise_models:
                self.__ibr_models[label_value] = lasso_model
            POOL.close()
            POOL.join()
        else:
            for label_value in self.label_names:
                label_index = learn_data_set.attributes.index(label_value)
                not_miss_instances = _np_data[:, label_index] != -1
                X_learning = _np_data[not_miss_instances, :-self.nb_labels]
                y_learning = _np_data[not_miss_instances, label_index]
                self._logger.debug("Learning Imprecise Lasso of label %s.", label_value)
                self.__ibr_models[label_value] = BinaryILogisticLasso(DEBUG=self.DEBUG)
                self.__ibr_models[label_value].learn(X=X_learning, y=y_learning,
                                                     nb_lasso_models=nb_lassos_models,
                                                     min_gamma=min_gamma, max_gamma=max_gamma)

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
