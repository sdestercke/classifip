import abc, math, time
import numpy as np
from .mlcncc import MLCNCC
from enum import Enum
from classifip.utils import create_logger
from classifip.representations.voting import Scores


class IMLCStrategy(Enum):
    IMPRECISE_BRANCHING = 1
    TERNARY_IMPRECISE_TREE = 2
    SAFETY_IMPRECISE_CHAINING = 3


class MLChaining(MLCNCC, metaclass=abc.ABCMeta):

    def __init__(self, DEBUG=False):
        super(MLChaining, self).__init__(DEBUG)
        self._logger = create_logger("MLChaining", DEBUG)

    @staticmethod
    def __maximality_decision(interval_probability):
        if interval_probability[0] > 0.5:
            return '1'
        elif interval_probability[1] < 0.5:
            return '0'
        else:
            return str(MLCNCC.LABEL_PARTIAL_VALUE)

    def __compute_optimal_path(self,
                               idx_label_to_infer,
                               idx_imprecise_labels,
                               ncc_s_param):

        optimal_lower_path = ['-1'] * len(idx_imprecise_labels)
        optimal_upper_path = ['-1'] * len(idx_imprecise_labels)
        lower_cum_max, upper_cum_min = 1, 1

        for i, l_index in enumerate(idx_imprecise_labels):  # l_index: label of index
            label_name = self.label_names[l_index]
            label_class_1 = self.label_names[idx_label_to_infer] + '|in|' + label_name
            label_class_0 = self.label_names[idx_label_to_infer] + '|out|' + label_name

            f_val_idx_0 = self.feature_values[label_name].index('0')
            n_fi_c00 = self.feature_count[label_class_0][f_val_idx_0]  # n_{y_i=0}(y_j=0)
            n_fi_c10 = self.feature_count[label_class_1][f_val_idx_0]  # n_{y_i=0}(y_j=1)

            f_val_idx_1 = self.feature_values[label_name].index('1')
            n_fi_c01 = self.feature_count[label_class_0][f_val_idx_1]  # n_{y_i=1}(y_j=0)
            n_fi_c11 = self.feature_count[label_class_1][f_val_idx_1]  # n_{y_i=1}(y_j=1)

            # avoiding ZeroDivision and infinity values
            if n_fi_c11 < 1e-16:
                n_fi_c11 = 1e-16
            if n_fi_c10 < 1e-16:
                n_fi_c10 = 1e-16

            # for computing \underline{P}(Y_j=1)
            lower_path_0 = (n_fi_c00 + ncc_s_param) / n_fi_c10
            lower_path_1 = (n_fi_c01 + ncc_s_param) / n_fi_c11

            # for computing \overline{P}(Y_j=1)
            upper_path_0 = n_fi_c00 / (n_fi_c10 + ncc_s_param)
            upper_path_1 = n_fi_c01 / (n_fi_c11 + ncc_s_param)

            # cumulative binary path 010101...(of lower and upper values)
            lower_path_0 = lower_cum_max * lower_path_0
            lower_path_1 = lower_cum_max * lower_path_1
            upper_path_0 = upper_cum_min * upper_path_0
            upper_path_1 = upper_cum_min * upper_path_1

            # arg_max: lower maximal path
            self._logger.debug("IB (lower_path_0, lower_path_1) (%s, %s)", lower_path_0, lower_path_1)
            if lower_path_0 > lower_path_1:
                optimal_lower_path[i] = '0'
                lower_cum_max = lower_path_0
            elif lower_path_0 < lower_path_1:
                optimal_lower_path[i] = '1'
                lower_cum_max = lower_path_1
            else:
                raise Exception("Not implemented yet __compute_optimal_path")

            # arg_max: upper minimal path
            self._logger.debug("IB (upper_path_0, upper_path_1) (%s, %s)", upper_path_0, upper_path_1)
            if upper_path_0 < upper_path_1:
                optimal_upper_path[i] = '0'
                upper_cum_min = upper_path_0
            elif upper_path_0 > upper_path_1:
                optimal_upper_path[i] = '1'
                upper_cum_min = upper_path_1
            else:
                raise Exception("Not implemented yet __compute_optimal_path")

        return optimal_lower_path, optimal_upper_path

    def __strategy_imprecise_branching(self, new_instance, ncc_s_param, ncc_epsilon):

        resulting_score = np.zeros((self.nb_labels, 2))
        chain_predicted_labels = []
        idx_imprecise_labels = []
        for idx_current_label in range(self.nb_labels):
            optimal_lower_path, optimal_upper_path = None, None
            if len(idx_imprecise_labels) > 0:
                optimal_lower_path, optimal_upper_path = self.__compute_optimal_path(idx_current_label,
                                                                                     idx_imprecise_labels,
                                                                                     ncc_s_param)

            partial_opt_predicted_labels = np.array(chain_predicted_labels)
            if len(idx_imprecise_labels) == 0:
                u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
                    super(MLChaining, self).lower_upper_cond_probability(idx_current_label,
                                                                         new_instance,
                                                                         partial_opt_predicted_labels,
                                                                         ncc_s_param,
                                                                         ncc_epsilon)
                # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
                resulting_score[idx_current_label, 0] = l_numerator_1 / (l_numerator_1 + u_denominator_0)
                resulting_score[idx_current_label, 1] = u_numerator_1 / (u_numerator_1 + l_denominator_0)
            else:
                self._logger.debug("IB (idx_label, chaining, lower_path, upper_path) (%s, %s, %s, %s)",
                                   idx_current_label, chain_predicted_labels, optimal_lower_path, optimal_upper_path)
                # computing lower probability: \underline P(Y_j=1)
                partial_opt_predicted_labels[idx_imprecise_labels] = optimal_lower_path
                _, l_numerator_1, u_denominator_0, _ = \
                    super(MLChaining, self).lower_upper_cond_probability(idx_current_label,
                                                                         new_instance,
                                                                         partial_opt_predicted_labels,
                                                                         ncc_s_param,
                                                                         ncc_epsilon)
                resulting_score[idx_current_label, 0] = l_numerator_1 / (l_numerator_1 + u_denominator_0)
                self._logger.debug("IB (idx_label, opt_upper_path (%s, %s)",
                                   idx_current_label, partial_opt_predicted_labels)

                # computing upper probability: \overline P(Y_j=1)
                partial_opt_predicted_labels[idx_imprecise_labels] = optimal_upper_path
                u_numerator_1, _, _, l_denominator_0 = \
                    super(MLChaining, self).lower_upper_cond_probability(idx_current_label,
                                                                         new_instance,
                                                                         partial_opt_predicted_labels,
                                                                         ncc_s_param,
                                                                         ncc_epsilon)
                resulting_score[idx_current_label, 1] = u_numerator_1 / (u_numerator_1 + l_denominator_0)
                self._logger.debug("IB (idx_label, opt_lower_path (%s, %s)",
                                   idx_current_label, partial_opt_predicted_labels)
                self._logger.debug("IB (idx_label, resulting_score (%s, %s)",
                                   idx_current_label, resulting_score[idx_current_label, :])

            inferred_label = MLChaining.__maximality_decision(resulting_score[idx_current_label, :])
            chain_predicted_labels.append(inferred_label)
            if inferred_label == '-1':
                idx_imprecise_labels.append(idx_current_label)
        return resulting_score, chain_predicted_labels

    def __strategy_ternary_tree(self, new_instance, ncc_s_param, ncc_epsilon):
        resulting_score = np.zeros((self.nb_labels, 2))
        chain_predicted_labels = []
        idx_imprecise_labels = []
        for idx_current_label in range(self.nb_labels):
            idx_precise_inferred_labels = list(set(range(idx_current_label)) - set(idx_imprecise_labels))
            u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
                super(MLChaining, self).lower_upper_cond_probability(idx_current_label,
                                                                     new_instance,
                                                                     chain_predicted_labels,
                                                                     ncc_s_param,
                                                                     ncc_epsilon,
                                                                     idx_precise_inferred_labels)
            # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
            resulting_score[idx_current_label, 0] = l_numerator_1 / (l_numerator_1 + u_denominator_0)
            resulting_score[idx_current_label, 1] = u_numerator_1 / (u_numerator_1 + l_denominator_0)
            inferred_label = MLChaining.__maximality_decision(resulting_score[idx_current_label, :])
            chain_predicted_labels.append(inferred_label)
            if inferred_label == '-1':
                idx_imprecise_labels.append(idx_current_label)

        return resulting_score, chain_predicted_labels

    def evaluate(self,
                 test_dataset,
                 ncc_epsilon=0.001,
                 ncc_s_param=2,
                 precision=None,
                 has_set_probabilities=False,
                 type_strategy=IMLCStrategy.IMPRECISE_BRANCHING):
        interval_prob_answers, predict_chain_answers = [], []

        for item in test_dataset:
            if IMLCStrategy.IMPRECISE_BRANCHING == type_strategy:
                rs_score, prediction = self.__strategy_imprecise_branching(item,
                                                                           ncc_s_param,
                                                                           ncc_epsilon)
            elif IMLCStrategy.TERNARY_IMPRECISE_TREE == type_strategy:
                rs_score, prediction = self.__strategy_ternary_tree(item,
                                                                    ncc_s_param,
                                                                    ncc_epsilon)
            elif IMLCStrategy.SAFETY_IMPRECISE_CHAINING == type_strategy:
                raise Exception("Not STRATEGY implemented yet")
            else:
                raise Exception("Not STRATEGY implemented yet")
            interval_prob_answers.append(Scores(rs_score))
            predict_chain_answers.append(list(map(int, prediction)))

        if has_set_probabilities:
            return predict_chain_answers, interval_prob_answers
        else:
            return predict_chain_answers

    #
    # def transform_partial_vector(self, chain_prediction):
    #     partial_vector = []
    #     for idx in range(self.nb_labels):
    #         if len(chain_prediction[idx]) > 1:
    #             partial_vector.append(-1)
    #         else:
    #             partial_vector.append(chain_prediction[idx][0])
    #     return partial_vector
    #
    # @staticmethod
    # def outer_maximality_decision(interval_probability):
    #     if interval_probability[0] > 0.5:
    #         return [1]
    #     elif interval_probability[1] < 0.5:
    #         return [0]
    #     else:
    #         return [0, 1]
    #
    # def naive_chaining(self,
    #                    label_prior,
    #                    ncc_s_param,
    #                    ncc_epsilon,
    #                    new_instance):
    #     resulting_score = np.zeros((self.nb_labels, 2))
    #     chain_predict_labels = []
    #     for idx_current_label in range(self.nb_labels):
    #         # computes lower/upper prob for a chain predict labels
    #         u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
    #             super(MLChaining, self).lower_upper_cond_probability(idx_current_label,
    #                                                                  label_prior,
    #                                                                  new_instance,
    #                                                                  chain_predict_labels,
    #                                                                  ncc_s_param,
    #                                                                  ncc_epsilon)
    #
    #         # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
    #         resulting_score[j, 1] = u_numerator_1 / (u_numerator_1 + l_denominator_0)
    #         resulting_score[j, 0] = l_numerator_1 / (l_numerator_1 + u_denominator_0)
    #
    #         if resulting_score[j, 0] > 0.5:
    #             chain_predict_labels.append([1])
    #         elif resulting_score[j, 1] < 0.5:
    #             chain_predict_labels.append([0])
    #         else:
    #             chain_predict_labels.append([0, 1])
    #
    #     return Scores(resulting_score), self.transform_partial_vector(chain_predict_labels)
    #
    # def interval_tight_chain(self,
    #                          label_prior,
    #                          ncc_s_param,
    #                          ncc_epsilon,
    #                          new_instance):
    #     resulting_score = np.zeros((self.nb_labels, 2))
    #     idx_chain_predicted_labels = []
    #     chain_predicted_labels = []
    #     prediction = np.ones(self.nb_labels, dtype=np.int)
    #     for j in range(self.nb_labels):
    #         # initializing scores
    #         r_score = self.get_set_probabilities_chain(idx_chain_predicted_labels,
    #                                                    chain_predicted_labels,
    #                                                    label_prior,
    #                                                    ncc_s_param, ncc_epsilon,
    #                                                    new_instance)
    #         len_intvl = r_score[:, 1] - r_score[:, 0]
    #         idx_tightly, label_value = None, None
    #         imprecise_labels = dict()
    #         is_found = True
    #         # len_intvl = np.around(len_intvl, 16)
    #         print("--->", np.around(r_score, 100), len_intvl)
    #         for _ in range(self.nb_labels - j):
    #             idx_tightly = np.argmin(len_intvl)
    #             print("--->", idx_tightly)
    #             if r_score[idx_tightly, 0] < 0.5 < r_score[idx_tightly, 1]:
    #                 print("---->", r_score[idx_tightly, 1])
    #                 label_value = [np.random.binomial(1, r_score[idx_tightly, 1])]
    #                 resulting_score[idx_tightly, :] = r_score[idx_tightly, :]
    #                 imprecise_labels[idx_tightly] = dict({
    #                     "index": idx_tightly,
    #                     "value": label_value
    #                 })
    #                 len_intvl[idx_tightly] = 1e16
    #             else:
    #                 label_value = MLChaining.outer_maximality_decision(r_score[idx_tightly, :])
    #                 resulting_score[idx_tightly, :] = r_score[idx_tightly, :]
    #                 is_found = True
    #                 break
    #         print("----->", idx_tightly, label_value, idx_chain_predicted_labels,
    #               chain_predicted_labels)
    #
    #         if not is_found:
    #             raise Exception("Il faut improve the strategy")
    #
    #         idx_chain_predicted_labels.append(idx_tightly)
    #         chain_predicted_labels.append(label_value)
    #         prediction[idx_tightly] = -1 if len(label_value) > 1 else label_value[0]
    #
    #     return resulting_score, prediction

    # def get_set_probabilities_chain(self,
    #                                 idx_chain_predicted_labels,
    #                                 chain_predict_labels,
    #                                 ncc_s_param,
    #                                 ncc_epsilon,
    #                                 new_instance):
    #     resulting_score = np.zeros((self.nb_labels, 2))
    #     resulting_score[idx_chain_predicted_labels, 0] = 0
    #     resulting_score[idx_chain_predicted_labels, 1] = 1e16
    #     for idx_current_label in set(range(self.nb_labels)) - set(idx_chain_predicted_labels):
    #         u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
    #             super(MLChaining, self).lower_upper_cond_probability(idx_current_label,
    #                                                                  new_instance,
    #                                                                  chain_predict_labels,
    #                                                                  ncc_s_param,
    #                                                                  ncc_epsilon,
    #                                                                  idx_chain_predicted_labels)
    #         # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
    #         resulting_score[idx_current_label, 1] = u_numerator_1 / (u_numerator_1 + l_denominator_0)
    #         resulting_score[idx_current_label, 0] = l_numerator_1 / (l_numerator_1 + u_denominator_0)
    #     return resulting_score
