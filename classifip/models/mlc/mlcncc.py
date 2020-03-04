import abc, math, time
import numpy as np


class MLCNCC(metaclass=abc.ABCMeta):

    def __init__(self):
        self.feature_names = []
        self.label_names = []
        self.feature_values = dict()
        self.feature_count = dict()
        self.label_counts = []
        self.nb_labels = 0
        self.training_size = 0
        self.LABEL_PARTIAL_VALUE = 3

    def learn(self, learn_data_set, nb_labels=1, missing_pct=0.0, seed_random_label=None):
        self.__init__()
        self.nb_labels = nb_labels
        self.training_size = int(len(learn_data_set.data) * (1 - missing_pct)) if missing_pct > 0.0 \
            else len(learn_data_set.data)

        # Initializing the counts
        self.feature_names = learn_data_set.attributes[:-self.nb_labels]
        self.label_names = learn_data_set.attributes[-self.nb_labels:]

        # Generation random position for chain label
        if seed_random_label is not None:
            np.random.seed(seed_random_label)
            np.random.shuffle(self.label_names)
        self.feature_values = learn_data_set.attribute_data.copy()
        for label_value in self.label_names:
            missing_label_index = None
            if missing_pct > 0:
                missing_label_index = np.random.choice(len(learn_data_set.data),
                                                       int(len(learn_data_set.data) * missing_pct),
                                                       replace=False)
            # recovery count of class 1 and 0
            label_set_one = learn_data_set.select_col_vals(label_value, ['1'], missing_label_index)
            self.label_counts.append(len(label_set_one.data))
            label_set_zero = learn_data_set.select_col_vals(label_value, ['0'], missing_label_index)
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
                        nb_items_one = [row[label_feature_index] for row in label_set_one.data].count(
                            label_feature_value)
                        count_vector_one.append(nb_items_one)
                        nb_items_zero = [row[label_feature_index] for row in label_set_zero.data].count(
                            label_feature_value)
                        count_vector_zero.append(nb_items_zero)
                    self.feature_count[label_value + '|in|' + label_feature] = count_vector_one
                    self.feature_count[label_value + '|out|' + label_feature] = count_vector_zero

    def lower_upper_probability(self, feature, feature_value, ncc_s_param, feature_class_name, ncc_epsilon):

        def __restricting_idm(probability, ncc_epsilon_ip, len_features):
            return (1 - ncc_epsilon_ip) * probability + ncc_epsilon_ip / len_features

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

    def lower_upper_probability_feature(self, j, label_prior, item, ncc_s_param, ncc_epsilon):
        # (n(c)+st(c))/(N+s), with s=0 (i.e. prior probabilities precise, P(Y))
        u_denominator_0 = 1 - label_prior[j]  # \overline P(Yj=0)
        l_denominator_0 = 1 - label_prior[j]  # \underline P(Yj=0)
        u_numerator_1 = label_prior[j]  # \overline P(Yj=1)
        l_numerator_1 = label_prior[j]  # \underline P(Yj=1)
        for f_index, feature in enumerate(self.feature_names):
            # computation of denominator (label=1)
            feature_class_name = self.label_names[j] + '|in|' + feature  # (f_i, c=1)
            p_lower, p_upper = self.lower_upper_probability(feature, item[f_index], ncc_s_param,
                                                            feature_class_name, ncc_epsilon)
            l_numerator_1 = l_numerator_1 * p_lower  # prod \underline{P}(f_i|c=1)
            u_numerator_1 = u_numerator_1 * p_upper  # prod \overline{P}(f_i|c=1)

            # computation of numerator (label=0)
            feature_class_name = self.label_names[j] + '|out|' + feature
            p_lower, p_upper = self.lower_upper_probability(feature, item[f_index], ncc_s_param,
                                                            feature_class_name, ncc_epsilon)
            l_denominator_0 = l_denominator_0 * p_lower  # prod \underline{P}(f_i|c=0)
            u_denominator_0 = u_denominator_0 * p_upper  # prod \overline{P}(f_i|c=0)

        return u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0

    def lower_upper_probability_labels(self, j, augmented_labels, ncc_s_param, ncc_epsilon):
        """
        :param j: name of label selected
        :param augmented_labels: list of characters values '0' or '1'
        :param ncc_s_param:
        :param ncc_epsilon:
        :return:
        """
        u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = 1, 1, 1, 1
        for l_index, label in enumerate(self.label_names[:len(augmented_labels)]):
            for label_predicted_value in augmented_labels[l_index]:
                label_predicted_value = str(label_predicted_value)
                # computation of denominator (label=1)
                label_class_name = self.label_names[j] + '|in|' + label  # (l_i=1, c=1)
                p_lower, p_upper = self.lower_upper_probability(label, label_predicted_value, ncc_s_param,
                                                                label_class_name, ncc_epsilon)
                l_numerator_1 = l_numerator_1 * p_lower  # prod \underline{P}(f_i|c=1)
                u_numerator_1 = u_numerator_1 * p_upper  # prod \overline{P}(f_i|c=1)

                # computation of numerator (label=0)
                label_class_name = self.label_names[j] + '|out|' + label  # (l_i=0, c=0)
                p_lower, p_upper = self.lower_upper_probability(label, label_predicted_value, ncc_s_param,
                                                                label_class_name, ncc_epsilon)
                l_denominator_0 = l_denominator_0 * p_lower  # prod \underline{P}(f_i|c=0)
                u_denominator_0 = u_denominator_0 * p_upper  # prod \overline{P}(f_i|c=0)
        return u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0

    def lower_upper_cond_probability(self, j, label_prior, item, chain_predict_labels, ncc_s_param, ncc_epsilon):
        u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0 = \
            self.lower_upper_probability_feature(j, label_prior, item, ncc_s_param, ncc_epsilon)

        u_numerator_label_1, l_numerator_label_1, u_denominator_label_0, l_denominator_label_0 = \
            self.lower_upper_probability_labels(j, chain_predict_labels, ncc_s_param, ncc_epsilon)

        u_numerator_1 = u_numerator_1 * u_numerator_label_1
        l_numerator_1 = l_numerator_1 * l_numerator_label_1
        u_denominator_0 = u_denominator_0 * u_denominator_label_0
        l_denominator_0 = l_denominator_0 * l_denominator_label_0
        return u_numerator_1, l_numerator_1, u_denominator_0, l_denominator_0
