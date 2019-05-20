from ..representations.voting import Scores
import numpy as np


class MLCNCC(object):

    def __init__(self):
        self.feature_names = []
        self.label_names = []
        self.feature_values = dict()
        self.feature_count = dict()
        self.label_counts = []
        self.nb_labels = 0
        self.training_size = 0

    def learn(self, learn_data_set, nb_labels=1, seed_random_label=None):
        self.__init__()
        self.nb_labels = nb_labels
        self.training_size = len(learn_data_set.data)

        # Initializing the counts
        self.feature_names = learn_data_set.attributes[:-self.nb_labels]
        self.label_names = learn_data_set.attributes[-self.nb_labels:]

        # Generation random position for chain label
        if seed_random_label is not None:
            np.random.seed(seed_random_label)
            np.random.shuffle(self.label_names)
        self.feature_values = learn_data_set.attribute_data.copy()
        for label_value in self.label_names:
            label_set_one = learn_data_set.select_col_vals(label_value, ['1'])
            self.label_counts.append(len(label_set_one.data))
            label_set_zero = learn_data_set.select_col_vals(label_value, ['0'])

            for feature in self.feature_names:
                count_vector_one, count_vector_zero = [], []
                feature_index = learn_data_set.attributes.index(feature)
                for feature_value in learn_data_set.attribute_data[feature]:
                    nb_items_one = [row[feature_index] for row in label_set_one.data].count(feature_value)
                    count_vector_one.append(nb_items_one)
                    nb_items_zero = [row[feature_index] for row in label_set_zero.data].count(feature_value)
                    count_vector_zero.append(nb_items_zero)
                self.feature_count[label_value + '|in|' + feature] = count_vector_one
                self.feature_count[label_value + '|out|' + feature] = count_vector_zero

            for label_feature in self.label_names:
                if label_feature != label_value:
                    label_feature_index = learn_data_set.attributes.index(label_feature)
                    count_vector_one, count_vector_zero = [], []
                    for label_feature_value in learn_data_set.attribute_data[label_feature]:
                        nb_items_one = [row[label_feature_index] for row in label_set_one.data].count(label_feature_value)
                        count_vector_one.append(nb_items_one)
                        nb_items_zero = [row[label_feature_index] for row in label_set_zero.data].count(label_feature_value)
                        count_vector_zero.append(nb_items_zero)
                    self.feature_count[label_value + '|in|' + label_feature] = count_vector_one
                    self.feature_count[label_value + '|out|' + label_feature] = count_vector_zero


    def __lower_upper_probability(self, feature, feature_value, ncc_s_param, feature_class_name, ncc_epsilon):

        def __restricting_idm(probability, ncc_epsilon, len_features):
            return (1 - ncc_epsilon) * probability + ncc_epsilon / len_features

        f_val_index = self.feature_values[feature].index(feature_value)  #
        num_items = float(sum(self.feature_count[feature_class_name]))
        n_fi_c = self.feature_count[feature_class_name][f_val_index]  # n(f_i|c)
        # n(f_i|c)/(n(c)+s), lower probability: t(f_1|c)->0, t(c)->1
        p_lower = (n_fi_c / (num_items + ncc_s_param))
        # (n(f_i|c)+s)/(n(c)+s), upper probability: t(f_1|c)->1, t(c)->1
        p_upper = ((n_fi_c + ncc_s_param) / (num_items + ncc_s_param))
        len_fi = len(self.feature_count[feature_class_name])  # |F_i|
        # some regularization with epsilon
        p_lower = __restricting_idm(p_lower, ncc_epsilon, len_fi)
        p_upper = __restricting_idm(p_upper, ncc_epsilon, len_fi)
        return p_lower, p_upper

    def evaluate(self, testdataset, ncc_epsilon=0.001, ncc_s_param=2):
        label_prop = [n / float(self.training_size) for n in self.label_counts]
        interval_prob_answers, predict_chain_answers = [], []

        for item in testdataset:
            # initializing scores
            resulting_score = np.zeros((self.nb_labels, 2))
            chain_predict_labels = []
            # computes lower/upper prob for a chain predict labels
            for j in range(self.nb_labels):
                # (n(c)+st(c))/(N+s), with s=0 (i.e. prior probabilities precise, P(Y))
                u_numerator_0 = 1 - label_prop[j]
                l_numerator_0 = 1 - label_prop[j]
                u_denominator_1 = label_prop[j]
                l_denominator_1 = label_prop[j]
                for f_index, feature in enumerate(self.feature_names):
                    # computation of denominator (label=1)
                    feature_class_name = self.label_names[j] + '|in|' + feature  # (f_i, c=1)
                    p_lower, p_upper = self.__lower_upper_probability(feature, item[f_index], ncc_s_param,
                                                                      feature_class_name, ncc_epsilon)
                    l_denominator_1 = l_denominator_1 * p_lower  # prod \underline{P}(f_i|c=1)
                    u_denominator_1 = u_denominator_1 * p_upper  # prod \overline{P}(f_i|c=1)

                    # computation of numerator (label=0)
                    feature_class_name = self.label_names[j] + '|out|' + feature
                    p_lower, p_upper = self.__lower_upper_probability(feature, item[f_index], ncc_s_param,
                                                                      feature_class_name, ncc_epsilon)
                    l_numerator_0 = l_numerator_0 * p_lower  # prod \underline{P}(f_i|c=0)
                    u_numerator_0 = u_numerator_0 * p_upper  # prod \underline{P}(f_i|c=0)

                for l_index, label in enumerate(self.label_names[:len(chain_predict_labels)]):
                    for label_predicted_value in chain_predict_labels[l_index]:
                        # computation of denominator (label=1)
                        label_class_name = self.label_names[j] + '|in|' + label  # (l_i=1, c=1)
                        p_lower, p_upper = self.__lower_upper_probability(label, label_predicted_value, ncc_s_param,
                                                                           label_class_name, ncc_epsilon)
                        l_denominator_1 = l_denominator_1 * p_lower  # prod \underline{P}(f_i|c=1)
                        u_denominator_1 = u_denominator_1 * p_upper  # prod \overline{P}(f_i|c=1)

                        # computation of numerator (label=0)
                        label_class_name = self.label_names[j] + '|out|' + label # (l_i=0, c=0)
                        p_lower, p_upper = self.__lower_upper_probability(label, label_predicted_value, ncc_s_param,
                                                                          label_class_name, ncc_epsilon)
                        l_numerator_0 = l_numerator_0 * p_lower  # prod \underline{P}(f_i|c=0)
                        u_numerator_0 = u_numerator_0 * p_upper  # prod \underline{P}(f_i|c=0)

                resulting_score[j, 1] = u_denominator_1 / (u_denominator_1 + l_numerator_0)
                resulting_score[j, 0] = l_denominator_1 / (l_denominator_1 + u_numerator_0)

                if resulting_score[j, 0] > 0.5:
                    chain_predict_labels.append(["1"])
                elif resulting_score[j, 1] < 0.5:
                    chain_predict_labels.append(["0"])
                else:
                    chain_predict_labels.append(["0", "1"])

            result = Scores(resulting_score)
            interval_prob_answers.append(result)
            predict_chain_answers.append(chain_predict_labels)

        return interval_prob_answers, predict_chain_answers
