import abc, math, time
from classifip.representations.voting import Scores
import numpy as np
from .mlcncc import MLCNCC


class MLChaining(MLCNCC, metaclass=abc.ABCMeta):

    def evaluate(self, test_dataset, ncc_epsilon=0.001, ncc_s_param=2):
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
            predict_chain_answers.append(chain_predict_labels)

        return interval_prob_answers, predict_chain_answers
