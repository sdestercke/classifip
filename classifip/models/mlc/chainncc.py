import abc, math, time
import numpy as np
from .mlcncc import MLCNCC
from enum import Enum
from classifip.utils import create_logger


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

    def __compute_opt_path_branching(self,
                                     idx_label_to_infer,
                                     idx_imprecise_labels,
                                     ncc_s_param):
        """
        ...Todo:
            upper_path_0 == upper_path_1 and lower_path_0 == lower_path_1
            To fix, normally it happens when ncc_s_param is so close to 0
        :param idx_label_to_infer:
        :param idx_imprecise_labels:
        :param ncc_s_param:
        :return:
        """

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
            # and uppers equals to 0.0
            if n_fi_c11 < 1e-19:
                n_fi_c11 = 1e-19
            if n_fi_c10 < 1e-19:
                n_fi_c10 = 1e-19
            if n_fi_c00 < 1e-19:
                n_fi_c00 = 1e-19
            if n_fi_c01 < 1e-19:
                n_fi_c01 = 1e-19

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
                self._logger.debug("Random-IB (lower_path_0, lower_path_1) (%s, %s)", lower_path_0, lower_path_1)
                optimal_lower_path[i] = np.random.choice(['0', '1'], 1)[0]
                lower_cum_max = lower_path_1
                # raise Exception("Not implemented yet __compute_opt_path_branching") @salmuz

            # arg_max: upper minimal path
            self._logger.debug("IB (upper_path_0, upper_path_1) (%s, %s)", upper_path_0, upper_path_1)
            if upper_path_0 < upper_path_1:
                optimal_upper_path[i] = '0'
                upper_cum_min = upper_path_0
            elif upper_path_0 > upper_path_1:
                optimal_upper_path[i] = '1'
                upper_cum_min = upper_path_1
            else:
                self._logger.debug("Random-IB (upper_path_0, upper_path_1) (%s, %s)", upper_path_0, upper_path_1)
                optimal_upper_path[i] = np.random.choice(['0', '1'], 1)[0]
                upper_cum_min = upper_path_1

        return optimal_lower_path, optimal_upper_path

    def __get_set_probabilities_branching(self,
                                          idx_current_label,
                                          new_instance,
                                          idx_predicted_labels,
                                          idx_imprecise_labels,
                                          chain_predicted_labels,
                                          ncc_s_param,
                                          ncc_epsilon):
        self._logger.debug("BR (idx_imprecise_labels, idx_predicted_labels) (%s,%s)",
                           idx_imprecise_labels, idx_predicted_labels)

        optimal_lower_path, optimal_upper_path = None, None
        if len(idx_imprecise_labels) > 0:
            optimal_lower_path, optimal_upper_path = self.__compute_opt_path_branching(idx_current_label,
                                                                                       idx_imprecise_labels,
                                                                                       ncc_s_param)
        partial_opt_predicted_labels = np.array(chain_predicted_labels)
        self._logger.debug("IB (chaining, lower_path, upper_path) (%s, %s, %s)",
                           partial_opt_predicted_labels, optimal_lower_path, optimal_upper_path)
        if len(idx_imprecise_labels) == 0:
            u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
                super(MLChaining, self).lower_upper_cond_probability(idx_current_label,
                                                                     new_instance,
                                                                     partial_opt_predicted_labels,
                                                                     ncc_s_param,
                                                                     ncc_epsilon,
                                                                     idx_predicted_labels)
            # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
            try:
                lower_cond_prob_1 = l_numerator_1 / (l_numerator_1 + u_denominator_0)
            except ZeroDivisionError:
                lower_cond_prob_1 = 0
            try:
                upper_cond_prob_1 = u_numerator_1 / (u_numerator_1 + l_denominator_0)
            except ZeroDivisionError:
                upper_cond_prob_1 = 0

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
                                                                     ncc_epsilon,
                                                                     idx_predicted_labels)

            try:
                lower_cond_prob_1 = l_numerator_1 / (l_numerator_1 + u_denominator_0)
            except ZeroDivisionError:
                lower_cond_prob_1 = 0
            self._logger.debug("IB (idx_label, opt_upper_path (%s, %s)",
                               idx_current_label, partial_opt_predicted_labels)

            # computing upper probability: \overline P(Y_j=1)
            partial_opt_predicted_labels[idx_imprecise_labels] = optimal_upper_path
            u_numerator_1, _, _, l_denominator_0 = \
                super(MLChaining, self).lower_upper_cond_probability(idx_current_label,
                                                                     new_instance,
                                                                     partial_opt_predicted_labels,
                                                                     ncc_s_param,
                                                                     ncc_epsilon,
                                                                     idx_predicted_labels)
            try:
                upper_cond_prob_1 = u_numerator_1 / (u_numerator_1 + l_denominator_0)
            except ZeroDivisionError:
                upper_cond_prob_1 = 0
            self._logger.debug("IB (idx_label, opt_lower_path (%s, %s)",
                               idx_current_label, partial_opt_predicted_labels)
            self._logger.debug("IB (idx_label, resulting_score (%s, %s)",
                               idx_current_label, [lower_cond_prob_1, upper_cond_prob_1])
        return lower_cond_prob_1, upper_cond_prob_1

    def __strategy_imprecise_branching(self, new_instance, ncc_s_param, ncc_epsilon, is_dynamic_context):
        resulting_score = np.zeros((self.nb_labels, 2))
        chain_predicted_labels = [None] * self.nb_labels
        idx_imprecise_labels = []
        idx_predicted_labels = []
        for idx_current_label in range(self.nb_labels):

            # dynamic selection of context-dependence label
            if is_dynamic_context:
                idx_current_label = self.__dynamic_context_dependence_label(chain_predicted_labels,
                                                                            idx_predicted_labels,
                                                                            idx_imprecise_labels,
                                                                            new_instance,
                                                                            ncc_s_param,
                                                                            ncc_epsilon,
                                                                            self.__get_set_probabilities_branching)
                idx_precise_inferred_labels = list(idx_predicted_labels)
            else:
                idx_precise_inferred_labels = list(set(range(idx_current_label)) - set(idx_imprecise_labels))

            lw_cond_prob_1, up_cond_prob_1 = self.__get_set_probabilities_branching(idx_current_label,
                                                                                    new_instance,
                                                                                    idx_precise_inferred_labels,
                                                                                    idx_imprecise_labels,
                                                                                    chain_predicted_labels,
                                                                                    ncc_s_param,
                                                                                    ncc_epsilon)

            resulting_score[idx_current_label, 0] = lw_cond_prob_1
            resulting_score[idx_current_label, 1] = up_cond_prob_1
            inferred_label = MLChaining.__maximality_decision(resulting_score[idx_current_label, :])
            self._logger.debug("BR (chain_predicted_labels, idx_current_label, inferred_label) (%s, %s, %s)",
                               chain_predicted_labels, idx_current_label, inferred_label)
            chain_predicted_labels[idx_current_label] = inferred_label
            if inferred_label == '-1':
                idx_imprecise_labels.append(idx_current_label)
            else:
                idx_predicted_labels.append(idx_current_label)
        return resulting_score, chain_predicted_labels

    def __get_set_probabilities_ternary_tree(self, idx_current_label,
                                             new_instance,
                                             idx_predicted_labels,
                                             idx_imprecise_labels,
                                             chain_predicted_labels,
                                             ncc_s_param,
                                             ncc_epsilon):
        self._logger.debug("TE (idx_imprecise_labels, idx_predicted_labels) (%s,%s)",
                           idx_imprecise_labels, idx_predicted_labels)
        u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
            super(MLChaining, self).lower_upper_cond_probability(idx_current_label,
                                                                 new_instance,
                                                                 chain_predicted_labels,
                                                                 ncc_s_param,
                                                                 ncc_epsilon,
                                                                 idx_predicted_labels)
        # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
        try:
            upper_cond_prob_1 = u_numerator_1 / (u_numerator_1 + l_denominator_0)
        except ZeroDivisionError:
            upper_cond_prob_1 = 0.0
        try:
            lower_cond_prob_1 = l_numerator_1 / (l_numerator_1 + u_denominator_0)
        except ZeroDivisionError:
            lower_cond_prob_1 = 0.0

        return lower_cond_prob_1, upper_cond_prob_1

    def __strategy_ternary_tree(self, new_instance, ncc_s_param, ncc_epsilon, is_dynamic_context):
        resulting_score = np.zeros((self.nb_labels, 2))
        chain_predicted_labels = [None] * self.nb_labels
        idx_imprecise_labels = []
        idx_predicted_labels = []
        for idx_current_label in range(self.nb_labels):

            # dynamic selection of context-dependence label
            if is_dynamic_context:
                idx_current_label = self.__dynamic_context_dependence_label(chain_predicted_labels,
                                                                            idx_predicted_labels,
                                                                            idx_imprecise_labels,
                                                                            new_instance,
                                                                            ncc_s_param,
                                                                            ncc_epsilon,
                                                                            self.__get_set_probabilities_ternary_tree)
                idx_precise_inferred_labels = list(idx_predicted_labels)
            else:
                idx_precise_inferred_labels = list(set(range(idx_current_label)) - set(idx_imprecise_labels))

            # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
            lw_cond_prob_1, up_cond_prob_1 = self.__get_set_probabilities_ternary_tree(idx_current_label,
                                                                                       new_instance,
                                                                                       idx_precise_inferred_labels,
                                                                                       idx_imprecise_labels,
                                                                                       chain_predicted_labels,
                                                                                       ncc_s_param,
                                                                                       ncc_epsilon)
            resulting_score[idx_current_label, 1] = up_cond_prob_1
            resulting_score[idx_current_label, 0] = lw_cond_prob_1
            inferred_label = MLChaining.__maximality_decision(resulting_score[idx_current_label, :])
            self._logger.debug("TE (chain_predicted_labels, idx_current_label, inferred_label) (%s, %s, %s)",
                               chain_predicted_labels, idx_current_label, inferred_label)
            chain_predicted_labels[idx_current_label] = inferred_label
            if inferred_label == '-1':
                idx_imprecise_labels.append(idx_current_label)
            else:
                idx_predicted_labels.append(idx_current_label)

        return resulting_score, chain_predicted_labels

    def __dynamic_context_dependence_label(self,
                                           chain_predicted_labels,
                                           idx_predicted_labels,
                                           idx_imprecise_labels,
                                           new_instance,
                                           ncc_s_param,
                                           ncc_epsilon,
                                           compute_set_probabilities):
        def __varphi(lower_prob, upper_prob):
            return upper_prob - lower_prob

        def __get_set_probabilities(idx_predicted_labels,
                                    chain_predicted_labels,
                                    ncc_s_param,
                                    ncc_epsilon,
                                    new_instance,
                                    set_idx_remaining_labels):
            resulting_score = np.zeros((len(set_idx_remaining_labels), 2))
            for idx, idx_current_label in enumerate(set_idx_remaining_labels):
                # calculating lower and upper probability [\underline P(Y_j=1), \overline P(Y_j=1)]
                lw_cond_prob_1, up_cond_prob_1 = compute_set_probabilities(idx_current_label,
                                                                           new_instance,
                                                                           idx_predicted_labels,
                                                                           idx_imprecise_labels,
                                                                           chain_predicted_labels,
                                                                           ncc_s_param,
                                                                           ncc_epsilon)
                resulting_score[idx, 1] = up_cond_prob_1
                resulting_score[idx, 0] = lw_cond_prob_1
            return resulting_score

        set_idx_remaining_labels = list(set(range(self.nb_labels)) -
                                        set(idx_predicted_labels) -
                                        set(idx_imprecise_labels))
        r_score = __get_set_probabilities(idx_predicted_labels,
                                          chain_predicted_labels,
                                          ncc_s_param,
                                          ncc_epsilon,
                                          new_instance,
                                          set_idx_remaining_labels)

        idx_uncertainty_labels = np.where((r_score[:, 0] < 0.5) & (0.5 < r_score[:, 1]))[0]
        self._logger.debug("CDL (r_score, idx_uncertainty_labels) (%s, %s)", r_score, idx_uncertainty_labels)
        # verify that non all interval contains 0.5
        if len(idx_uncertainty_labels) == len(set_idx_remaining_labels):
            length_intvl = r_score[:, 1] - r_score[:, 0]
            idx_label_tightly = np.argmin(length_intvl)
            idx_dynamic_selected = set_idx_remaining_labels[idx_label_tightly]
        else:
            idx_precises = list(set(range(len(r_score))) - set(idx_uncertainty_labels))
            self._logger.debug("CDL (idx_uncertainty_labels, idx_precises) (%s, %s, %s)",
                               idx_uncertainty_labels, idx_precises, len(set_idx_remaining_labels))
            lower_prob_j = r_score[:, 0].tolist().index(np.max(r_score[idx_precises, 0]))
            upper_prob_j = r_score[:, 1].tolist().index(np.min(r_score[idx_precises, 1]))
            phi_lw_j = __varphi(r_score[lower_prob_j, 0], r_score[lower_prob_j, 1])
            phi_up_j = __varphi(r_score[upper_prob_j, 0], r_score[upper_prob_j, 1])

            self._logger.debug("CDL (r_score[lower_j], r_score[upper_j]) (%s, %s)",
                               r_score[lower_prob_j, :], r_score[upper_prob_j, :])
            self._logger.debug("CDL (lower_j, upper_j) (%s, %s)", lower_prob_j, upper_prob_j)
            self._logger.debug("CDL (phi_lw_j, phi_up_j) (%s, %s)", phi_lw_j, phi_up_j)
            # select lower uncertainty label
            if phi_up_j > phi_lw_j:
                idx_dynamic_selected = set_idx_remaining_labels[lower_prob_j]
            else:
                idx_dynamic_selected = set_idx_remaining_labels[upper_prob_j]

        return idx_dynamic_selected

    def evaluate(self,
                 test_dataset,
                 ncc_epsilon=0.001,
                 ncc_s_param=2,
                 with_imprecise_marginal=False,
                 type_strategy=IMLCStrategy.IMPRECISE_BRANCHING,
                 is_dynamic_context=False,
                 has_set_probabilities=False):
        # setting the global class-scope variable
        self.has_imprecise_marginal = with_imprecise_marginal
        interval_prob_answers, predict_chain_answers = [], []

        for item in test_dataset:
            if IMLCStrategy.IMPRECISE_BRANCHING == type_strategy:
                rs_score, prediction = self.__strategy_imprecise_branching(item,
                                                                           ncc_s_param,
                                                                           ncc_epsilon,
                                                                           is_dynamic_context)
            elif IMLCStrategy.TERNARY_IMPRECISE_TREE == type_strategy:
                rs_score, prediction = self.__strategy_ternary_tree(item,
                                                                    ncc_s_param,
                                                                    ncc_epsilon,
                                                                    is_dynamic_context)
            else:
                raise Exception("Not STRATEGY implemented yet")

            interval_prob_answers.append(rs_score)
            predict_chain_answers.append(list(map(int, prediction)))

        self.has_imprecise_marginal = False  # reboot the global class-scope variable
        if has_set_probabilities:
            return predict_chain_answers, interval_prob_answers
        else:
            return predict_chain_answers
