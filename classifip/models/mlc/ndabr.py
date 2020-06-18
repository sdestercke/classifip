from classifip.dataset.arff import ArffFile
from classifip.representations.voting import Scores
from classifip.models.mlc.mlcncc import MLCNCC
from sklearn.model_selection import KFold
from classifip.models.qda import NaiveDiscriminant
from classifip.evaluation.measures import u65, u80
from classifip.models.qda_precise import NaiveDiscriminantPrecise
import numpy as np
from math import exp


class NDABR(MLCNCC):

    def __init__(self):
        """Build an empty NCCBR structure """
        super(NDABR, self).__init__()
        self.nda_models = None
        self.nb_feature = None

    def learn(self,
              learn_data_set,
              nb_labels,
              n_splits=5,
              learn_data_disc=None):
        self.__init__()

        self.nb_labels = nb_labels
        self.training_size = len(learn_data_set.data)
        self.label_names = learn_data_set.attributes[-self.nb_labels:]
        self.feature_names = learn_data_set.attributes[:-self.nb_labels]
        self.nb_feature = len(self.feature_names)
        self.nda_models = dict()
        _np_data = np.array(learn_data_set.data)
        np.random.shuffle(_np_data)
        # _index_features = np.array(range(self.nb_feature))
        # X_learning = np.array(_np_data[:, _index_features], dtype=np.float)
        from_ell, to_ell, by_ell = 0.1, 15.0, 0.1
        for label_value in self.label_names:
            label_index = learn_data_set.attributes.index(label_value)
            # y_learning = _np_data[:, label_index]
            splits_ell = list([])
            kfSecond = KFold(n_splits=n_splits, random_state=None, shuffle=True)
            ell_u65, ell_u80 = dict(), dict()

            print('--->', len(_np_data))
            X_learning, y_learning = list(), list()
            for row_index, raw_instance in enumerate(learn_data_disc.data):
                if raw_instance[label_index] != '-1':
                    X_learning.append(_np_data[row_index, :self.nb_feature])
                    y_learning.append(_np_data[row_index, label_index])
            X_learning = np.array(X_learning, dtype=np.float)
            y_learning = np.array(y_learning)
            print('--->', len(X_learning))

            nda_model = NaiveDiscriminant()
            for idx_learn_train, idx_learn_test in kfSecond.split(y_learning):
                splits_ell.append((idx_learn_train.copy(), idx_learn_test.copy()))
            for ell_current in np.arange(from_ell, to_ell, by_ell):
                ell_u65[ell_current], ell_u80[ell_current] = 0, 0
                for idx_learn_train, idx_learn_test in splits_ell:
                    X_cv_train, y_cv_train = X_learning[idx_learn_train], y_learning[idx_learn_train]
                    X_cv_test, y_cv_test = X_learning[idx_learn_test], y_learning[idx_learn_test]
                    nda_model.learn(X=X_cv_train, y=y_cv_train, ell=ell_current)
                    n_cv_test = len(y_cv_test)
                    for i, x_test in enumerate(X_cv_test):
                        evaluate, _ = nda_model.evaluate(x_test)
                        if y_cv_test[i] in evaluate:
                            ell_u65[ell_current] += u65(evaluate) / n_cv_test
                            ell_u80[ell_current] += u80(evaluate) / n_cv_test
                ell_u65[ell_current] = ell_u65[ell_current] / n_splits
                ell_u80[ell_current] = ell_u80[ell_current] / n_splits
            acc_ell_u80 = max(ell_u80.values())
            acc_ell_u65 = max(ell_u65.values())
            ell_u80_opt = np.unique([k for k, v in ell_u80.items() if v == acc_ell_u80])
            ell_u65_opt = np.unique([k for k, v in ell_u65.items() if v == acc_ell_u65])
            # if there are many optimal, random choice
            ell_u80_opt = np.random.choice(ell_u80_opt) if len(ell_u80_opt) > 1 else ell_u80_opt[0]
            ell_u65_opt = np.random.choice(ell_u65_opt) if len(ell_u65_opt) > 1 else ell_u65_opt[0]
            nda_model_u80 = NaiveDiscriminant()
            nda_model_u80.learn(X=X_learning, y=y_learning, ell=ell_u80_opt)
            nda_model_u65 = NaiveDiscriminant()
            nda_model_u65.learn(X=X_learning, y=y_learning, ell=ell_u65_opt)
            precise_nda = NaiveDiscriminantPrecise()
            precise_nda.learn(X=X_learning, y=y_learning)
            self.nda_models[label_value] = dict({
                "u80": nda_model_u80,
                "u65": nda_model_u65,
                "acc": precise_nda
            })
            print("---> learning", ell_u65_opt, ell_u80_opt)
            print("80", ell_u80)
            print("65", ell_u65)
        print(self.nda_models)

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
                evaluate, _ = models["u65"].evaluate(query=instance)
                skeptic[i] = -1 if len(evaluate) > 1 else int(evaluate[0])
                evaluate, probabilities = models["acc"].evaluate(queries=[instance],
                                                                 with_posterior=True)
                precise[i] = int(evaluate[0])
                precise_proba[i] = probabilities[0]['1']
            answers.append((skeptic, precise, precise_proba))
        return answers
