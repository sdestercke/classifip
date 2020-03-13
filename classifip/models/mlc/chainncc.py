import abc, math, time
from classifip.representations.voting import Scores
import numpy as np
from .mlcncc import MLCNCC


class MLChaining(MLCNCC, metaclass=abc.ABCMeta):

    def transform_partial_vector(self, chain_prediction):
        partial_vector = []
        for idx in range(self.nb_labels):
            if len(chain_prediction[idx]) > 1:
                partial_vector.append(-1)
            else:
                partial_vector.append(chain_prediction[idx][0])
        return partial_vector

    def evaluate(self,
                 test_dataset,
                 ncc_epsilon=0.001,
                 ncc_s_param=2,
                 has_set_probabilities=False):
        label_prior = [n / float(self.training_size) for n in self.label_counts]
        interval_prob_answers, predict_chain_answers = [], []

        for item in test_dataset:
            # initializing scores
            resulting_score = np.zeros((self.nb_labels, 2))
            chain_predict_labels = []
            # computes lower/upper prob for a chain predict labels
            for j in range(self.nb_labels):
                u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
                    super(MLChaining, self).lower_upper_cond_probability(j, label_prior, item,
                                                                         chain_predict_labels,
                                                                         ncc_s_param,
                                                                         ncc_epsilon)

                # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
                resulting_score[j, 1] = u_numerator_1 / (u_numerator_1 + l_denominator_0)
                resulting_score[j, 0] = l_numerator_1 / (l_numerator_1 + u_denominator_0)

                if resulting_score[j, 0] > 0.5:
                    chain_predict_labels.append([1])
                elif resulting_score[j, 1] < 0.5:
                    chain_predict_labels.append([0])
                else:
                    chain_predict_labels.append([0, 1])

            result = Scores(resulting_score)
            interval_prob_answers.append(result)
            predict_chain_answers.append(self.transform_partial_vector(chain_predict_labels))

        if has_set_probabilities:
            return predict_chain_answers, interval_prob_answers
        else:
            return predict_chain_answers
