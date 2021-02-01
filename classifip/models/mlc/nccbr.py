from classifip.dataset.arff import ArffFile
from classifip.representations.voting import Scores
from classifip.models.mlc.mlcncc import MLCNCC
from classifip.models.ncc import NCC
import numpy as np
from math import exp


class NCCBR(MLCNCC):

    def __init__(self,
                 DEBUG=False,
                 has_imprecise_marginal=False):
        """Build an empty NCCBR structure """
        super(NCCBR, self).__init__(DEBUG)
        self.marginal_props = None  # precise distribution Y
        self.has_imprecise_marginal = has_imprecise_marginal

    def learn(self,
              learn_data_set,
              nb_labels):
        """

        :param learn_data_set:
        :param nb_labels:
        :return:

        TODO:
            Using the ncc version improved with laplace smoothing
        """
        self.nb_labels = nb_labels
        # Initializing the counts
        self.feature_names = learn_data_set.attributes[:-self.nb_labels]
        self.label_names = learn_data_set.attributes[-self.nb_labels:]
        self.feature_values = learn_data_set.attribute_data.copy()
        self.marginal_props = dict({i: dict() for i in range(self.nb_labels)})

        for label_index, label_value in enumerate(self.label_names):
            label_set_one = learn_data_set.select_col_vals(label_value, ['1'])
            label_set_zero = learn_data_set.select_col_vals(label_value, ['0'])
            nb_count_one, nb_count_zero = len(label_set_one.data), len(label_set_zero.data)
            # The missing label is identified in the data set when the value of label is -1,
            # so it does not take into account.
            try:
                self.marginal_props[label_index][0] = nb_count_zero
                self.marginal_props[label_index][1] = nb_count_one
                self.marginal_props[label_index]['all'] = nb_count_one + nb_count_zero
            except ZeroDivisionError:
                self.marginal_props[label_index][0] = 0
                self.marginal_props[label_index][1] = 0
                self.marginal_props[label_index]['all'] = 0

            for feature in self.feature_names:
                count_vector_one = []
                count_vector_zero = []
                feature_index = learn_data_set.attributes.index(feature)
                for feature_value in learn_data_set.attribute_data[feature]:
                    nb_items_one = [row[feature_index] for row in label_set_one.data].count(feature_value)
                    count_vector_one.append(nb_items_one)
                    nb_items_zero = [row[feature_index] for row in label_set_zero.data].count(feature_value)
                    count_vector_zero.append(nb_items_zero)
                self.feature_count[label_value + '|in|' + feature] = count_vector_one
                self.feature_count[label_value + '|out|' + feature] = count_vector_zero

    def lower_upper_marginal(self,
                             idx_label_to_infer,
                             value_label_to_infer,
                             label_dimension,
                             ncc_s_param):
        """
            ToDo: Code refactoring of NCC.__computing_laplace_smoothing method with this one.
        :param marginal_props:
        :param idx_label_to_infer:
        :param value_label_to_infer:
        :param label_dimension:
        :param ncc_s_param:
        :param with_imprecise_marginal:
        :return:
        """

        def __laplace_smoothing(bits, all_bits, dimension, alpha=1):
            try:
                add_smooth = bits / all_bits
            except ZeroDivisionError:
                add_smooth = (bits + alpha) / (all_bits + dimension)
            return add_smooth

        if not self.has_imprecise_marginal:
            all_bits_label = self.marginal_props[idx_label_to_infer]["all"]
            prop_marginal_label_1 = __laplace_smoothing(self.marginal_props[idx_label_to_infer][1],
                                                        all_bits_label,
                                                        label_dimension)
            if value_label_to_infer == 1:
                return prop_marginal_label_1, prop_marginal_label_1
            else:
                return 1 - prop_marginal_label_1, 1 - prop_marginal_label_1
        else:
            bits_label_Y = self.marginal_props[idx_label_to_infer][value_label_to_infer]
            n_label_data = self.marginal_props[idx_label_to_infer]["all"]
            p_lower = __laplace_smoothing(bits_label_Y,
                                          n_label_data + ncc_s_param,
                                          label_dimension)
            p_upper = __laplace_smoothing(bits_label_Y + ncc_s_param,
                                          n_label_data + ncc_s_param,
                                          label_dimension)
        return p_lower, p_upper

    def evaluate(self, test_dataset, ncc_epsilon=0.001, ncc_s_param=2.0, precision=None, **kargs):
        """evaluate the instances and return a list of probability intervals.
        
        :param test_dataset: list of input features of instances to evaluate
        :type test_dataset: list
        :param ncc_epsilon: espilon issued from [#corani2010]_ (should be > 0)
            to avoid zero count issues
        :type ncc_epsilon: float
        :param ncc_s_param: s parameter used in the IDM learning (settle imprecision level)
        :type ncc_s_param: float
        :param precision Number of digits of precision for floating, if necessary
        :returns: for each value of ncc_s_param, a set of scores for each label
        :rtype: lists of :class:`~classifip.representations.voting.Scores`
 
        .. note::
    
            * Precise prior
                prior class probabilities are assumed to be precise to speed up
                computations. The impact on the result is small, unless
                the number of class example in the training set is close to s or lower.
        
        .. warning::
    
            * zero float division can happen if too many input features
            * fixed: To avoid probability zero, we use the Laplace Smoothing
                https://en.wikipedia.org/wiki/Additive_smoothing

        .. TODO:

            * Using the NCC classifier already implemented !!
            
        """

        answers = []
        for item in test_dataset:
            # initializing scores
            resulting_score = np.zeros((self.nb_labels, 2))
            # computes product of lower/upper prob for each class
            for j in range(self.nb_labels):
                lower_cond_prob_0, upper_cond_prob_0 = self.lower_upper_marginal(idx_label_to_infer=j,
                                                                                 value_label_to_infer=0,
                                                                                 label_dimension=self.nb_labels,
                                                                                 ncc_s_param=ncc_s_param)
                lower_cond_prob_1, upper_cond_prob_1 = self.lower_upper_marginal(idx_label_to_infer=j,
                                                                                 value_label_to_infer=1,
                                                                                 label_dimension=self.nb_labels,
                                                                                 ncc_s_param=ncc_s_param)

                for f_index, feature in enumerate(self.feature_names):
                    # computation of denominator (label=1)
                    f_val_index = self.feature_values[feature].index(item[f_index])
                    count_string = self.label_names[j] + '|in|' + feature
                    all_count_of_feature_by_clazz = float(sum(self.feature_count[count_string]))
                    feature_value_count = self.feature_count[count_string][f_val_index]
                    feature_dimension = len(self.feature_count[count_string])
                    lower, upper = NCC._computing_lower_and_upper(feature_value_count,
                                                                  all_count_of_feature_by_clazz,
                                                                  feature_dimension,
                                                                  ncc_s_param,
                                                                  laplace_smoothing=False)
                    lower_cond_prob_1 = lower_cond_prob_1 * ((1 - ncc_epsilon) * lower + ncc_epsilon / feature_dimension)
                    upper_cond_prob_1 = upper_cond_prob_1 * ((1 - ncc_epsilon) * upper + ncc_epsilon / feature_dimension)
                    # computation of numerator (label=0)
                    count_string = self.label_names[j] + '|out|' + feature
                    all_count_of_feature_by_clazz = float(sum(self.feature_count[count_string]))
                    feature_value_count = self.feature_count[count_string][f_val_index]
                    feature_dimension = len(self.feature_count[count_string])
                    lower, upper = NCC._computing_lower_and_upper(feature_value_count,
                                                                  all_count_of_feature_by_clazz,
                                                                  feature_dimension,
                                                                  ncc_s_param,
                                                                  laplace_smoothing=False)
                    upper_cond_prob_0 = upper_cond_prob_0 * ((1 - ncc_epsilon) * upper + ncc_epsilon / feature_dimension)
                    lower_cond_prob_0 = lower_cond_prob_0 * ((1 - ncc_epsilon) * lower + ncc_epsilon / feature_dimension)

                resulting_score[j, 1] = upper_cond_prob_1 / (upper_cond_prob_1 + lower_cond_prob_0)
                resulting_score[j, 0] = lower_cond_prob_1 / (lower_cond_prob_1 + upper_cond_prob_0)
            # ToDo: change representation to IntervalsProbability
            result = Scores(resulting_score, precision=precision)
            answers.append(result)

        return answers
