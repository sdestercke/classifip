from ..dataset.arff import ArffFile
from ..representations.intervalsProbability import IntervalsProbability
from ..representations.probadis import ProbaDis
from ..representations.voting import Scores
import numpy as np
from math import exp


class NCC(object):
    """NCC implements the naive credal classification method using the IDM. 
    
    It returns a class
    :class:`~classifip.representations.intervalsProbability.IntervalsProbability`
    that corresponds to probability intervals on each class. The method is based
    on [#zaffalon2002]_ and on the improvement proposed by [#corani2010]_
    
    :param feature_count: store counts of couples class/feature
    :type feature_count: dictionnary with keys class/feature
    :param label_counts: store counts of class labels (to instanciate prior)
    :type label_counts: list
    :param feature_names: store the names of features, including the "class"
    :type feature_names: list
    :param feature_values: store modalities of features, including the classes
    :type feature_values: dictionnary associating each feature name to a list
    
    .. note::
    
        Assumes that the class attribute is the last one in samples
        in the learning method
    
    .. todo::
    
        * Make it possible for the class to be in any column (retrieve index)
    """

    def __init__(self):
        """Build an empty NCC structure
        """

        # both feature names and feature values contains the class (assumed to be the last)
        self.feature_names = []
        self.feature_values = dict()
        self.feature_count = dict()
        self.label_count = []

    def learn(self, learndataset):
        """learn the NCC, mainly storing counts of feature/class pairs
        
        :param learndataset: learning instances
        :type learndataset: :class:`~classifip.dataset.arff.ArffFile`
        """
        self.__init__()
        # Initializing the counts
        self.feature_names = learndataset.attributes[:]
        self.feature_values = learndataset.attribute_data.copy()
        for class_value in learndataset.attribute_data['class']:
            subset = [row for row in learndataset.data if row[-1] == class_value]
            self.label_count.append(len(subset))
            for feature in learndataset.attributes[:-1]:
                count_vector = []
                feature_index = learndataset.attributes.index(feature)
                for feature_value in learndataset.attribute_data[feature]:
                    nb_items = [row[feature_index] for row in subset].count(feature_value)
                    count_vector.append(nb_items)
                self.feature_count[class_value + '|' + feature] = count_vector

    def evaluate(self,
                 test_dataset,
                 ncc_epsilon=0.001,
                 ncc_s_param=2,
                 maxi=False,
                 precision=None,
                 laplace_smoothing=False):
        """evaluate the instances and return a list of probability intervals.
        
        :param test_dataset: list of input features of instances to evaluate
        :type test_dataset: list
        :param ncc_epsilon: espilon issued from [#corani2010]_ (should be > 0)
            to avoid zero count issues
        :type ncc_epsilon: float
        :param ncc_s_param: s parameter used in the IDM learning (settle
            imprecision level)
        :type ncc_s_param: float
        :param maxi: specify whether the decision is maximality (default=false)
        :type maxi: boolean
        :param precision: minimum decimal numbers to round the probability
        :param laplace_smoothing: (True) to use regularized Laplace smoothing or not (False)
        :returns: for each value of ncc_s_param, a set of probability intervals and or of scores
        :rtype: lists of :class:`~classifip.representations.intervalsProbability.IntervalsProbability` or
            lists of :class:`~classifip.representations.voting.Scores`
        .. note::
            
            * Precise prior
                prior class probabilities are assumed to be precise to speed up
                computations. The impact of the result is small, unless
                the number of class example in the training set is close to s or lower.
            * To avoid probability zero, we adopt the Laplace Smoothing
                https://en.wikipedia.org/wiki/Additive_smoothing

        .. Example: Computing the lower and upper conditional probabilities of an instance
            Classes: Y = {a, b, c}

                                                            P(Y=a) * lower(P)(X=x|Y=a)
            lower(P)(Y=a|X=x) = ------------------------------------------------------------------------------------
                                P(Y=a) * lower(P)(X=x|Y=a) + P(Y=b) * upper(P)(X=x|Y=b) + P(Y=c) * upper(P)(X=x|Y=c)

                                                            P(Y=a) * upper(P)(X=x|Y=a)
            upper(P)(Y=a|X=x) = ------------------------------------------------------------------------------------
                                P(Y=a) * upper(P)(X=x|Y=a) + P(Y=b) * lower(P)(X=x|Y=b) + P(Y=c) * lower(P)(X=x|Y=c)

            where
                lower(P)(X=x|Y=a) = prod_i lower(P)(X_i = x_i | Y=a)
                upper(P)(X=x|Y=a) = prod_i upper(P)(X_i = x_i | Y=a)
        """

        # computing class proportions with smooth laplace regularization
        class_ct = [float(n) for n in self.label_count]
        class_prop = np.ones(len(class_ct))
        for i, n in enumerate(class_ct):
            try:
                class_prop[i] = n / sum(class_ct)
            except ZeroDivisionError:
                class_prop[i] = (n + 1) / (sum(class_ct) + len(class_ct))
        answers = []
        for item in test_dataset:

            # initializing probability interval argument
            resulting_int = np.zeros((2, len(self.feature_values['class'])))
            resulting_sc = np.zeros((len(self.feature_values['class']), 2))
            resulting_sc[:, 0] = 0.9
            resulting_sc[:, 1] = 1.0

            # computes product of lower/upper prob for each class
            cl_index = 0
            for clazz in self.feature_values['class']:
                upper_sum_cond_prob = 0  # sum_{y<>clazz} P(Y=y) * upper(P)(X=x|Y=y)
                lower_sum_cond_prob = 0  # sum_{y<>clazz} P(Y=y) * lower(P)(X=x|Y=y)
                upper_cond_prob = class_prop[cl_index]  # upper(P)(X=x|Y=clazz)
                lower_cond_prob = class_prop[cl_index]  # lower(P)(X=x|Y=clazz)
                # computing:  P(Y=clazz) * lower(P)(X=x|Y=clazz) or P(Y=a) * upper(P)(X=x|Y=clazz)
                for f_index, feature in enumerate(self.feature_names):
                    if feature != 'class':
                        f_val_index = self.feature_values[feature].index(item[f_index])
                        count_string = clazz + '|' + feature
                        all_count_of_feature_by_clazz = float(sum(self.feature_count[count_string]))
                        feature_value_count = self.feature_count[count_string][f_val_index]
                        feature_dimension = len(self.feature_count[count_string])
                        lower, upper = self.__computing_lower_and_upper(feature_value_count,
                                                                        all_count_of_feature_by_clazz,
                                                                        feature_dimension,
                                                                        ncc_s_param,
                                                                        laplace_smoothing)

                        lower_cond_prob = lower_cond_prob * ((1 - ncc_epsilon) * lower + ncc_epsilon / feature_dimension)
                        upper_cond_prob = upper_cond_prob * ((1 - ncc_epsilon) * upper + ncc_epsilon / feature_dimension)

                # computing: sum_{y<>clazz} P(Y=y) * upper(P)(X=x|Y=y) (or lower(P)(X=x|Y=y))
                for other_cl in set(self.feature_values['class']) - set([clazz]):
                    # sum_{y<>clazz} P(Y=y) * upper(P)(X=x|Y=y)
                    upper_others_cond_prob = class_prop[self.feature_values['class'].index(other_cl)]
                    # sum_{y<>clazz} P(Y=y) * lower(P)(X=x|Y=y)
                    lower_others_cond_prob = class_prop[self.feature_values['class'].index(other_cl)]
                    for feature in self.feature_names:
                        if feature != 'class':
                            f_index = self.feature_names.index(feature)
                            f_val_index = self.feature_values[feature].index(item[f_index])
                            count_string = other_cl + '|' + feature
                            all_count_of_feature_by_clazz = float(sum(self.feature_count[count_string]))
                            feature_value_count = self.feature_count[count_string][f_val_index]
                            feature_dimension = len(self.feature_count[count_string])
                            lower, upper = self.__computing_lower_and_upper(feature_value_count,
                                                                            all_count_of_feature_by_clazz,
                                                                            feature_dimension,
                                                                            ncc_s_param,
                                                                            laplace_smoothing)
                            restricting_idm = lambda bound: (1 - ncc_epsilon) * bound + ncc_epsilon / feature_dimension
                            lower_others_cond_prob = lower_others_cond_prob * restricting_idm(lower)
                            upper_others_cond_prob = upper_others_cond_prob * restricting_idm(upper)

                    # check for maximality
                    if lower_cond_prob - upper_others_cond_prob > 0.:
                        resulting_sc[self.feature_values['class'].index(other_cl), 0] = 0.
                        resulting_sc[self.feature_values['class'].index(other_cl), 1] = 0.1
                    # update numerator/denominator for probability interval computation
                    upper_sum_cond_prob += upper_others_cond_prob
                    lower_sum_cond_prob += lower_others_cond_prob
                    # computing: upper(P)(Y=clazz|X=x) and lower(P)(Y=clazz|X=x)
                    resulting_int[0, cl_index] = upper_cond_prob / (upper_cond_prob + lower_sum_cond_prob)
                    resulting_int[1, cl_index] = lower_cond_prob / (lower_cond_prob + upper_sum_cond_prob)
                cl_index += 1
            if not maxi and ncc_s_param != 0:
                result = IntervalsProbability(resulting_int, precision)
            elif not maxi and ncc_s_param == 0:
                result = ProbaDis(resulting_int[0, :] / resulting_int[0, :].sum())
            else:
                result = Scores(resulting_sc, precision)
            answers.append(result)

        return answers

    def __computing_lower_and_upper(self,
                                    count_feature_by_value_and_clazz,
                                    all_count_of_feature_by_clazz,
                                    size_dimension_feature,
                                    ncc_s_param,
                                    laplace_smoothing):
        """
            By default it compute the laplace smoothing if
            the zero division error is thrown.
        :return:
        """
        lower_smooth, upper_smooth = self.__computing_laplace_smoothing(count_feature_by_value_and_clazz,
                                                                        all_count_of_feature_by_clazz,
                                                                        size_dimension_feature,
                                                                        ncc_s_param)
        if laplace_smoothing:
            return lower_smooth, upper_smooth
        else:
            denominator = (all_count_of_feature_by_clazz + ncc_s_param)
            try:
                lower = count_feature_by_value_and_clazz / denominator
            except ZeroDivisionError:
                lower = lower_smooth

            try:
                upper = (count_feature_by_value_and_clazz + ncc_s_param) / denominator
            except ZeroDivisionError:
                upper = upper_smooth

            return lower, upper

    @staticmethod
    def __computing_laplace_smoothing(count_feature_by_value_and_clazz,
                                      all_count_feature_by_clazz,
                                      size_dimension_feature,
                                      ncc_s_param):
        """
            Example:
            After discretization of the feature X1 with 4 value levels
                0 = {str} '<=0.04084'
                1 = {str} '(0.04084;0.05804]'
                2 = {str} '(0.05804;0.07241]'
                3 = {str} '(0.07241;0.09205]'
                4 = {str} '>0.09205'

            Counts obtained of the feature X1 by classes {0, 1} after training data set
                  class : 0              class: 1
                0 = {int} 27 			0 = {int} 1
                1 = {int} 14 			1 = {int} 2
                2 = {int} 13 			2 = {int} 0
                3 = {int} 15 			3 = {int} 6
                4 = {int} 8 			4 = {int} 6

            count_feature_by_value_and_clazz: (X1=1|class=0) = 14
            all_count_feature_by_clazz: (X1\in{0,1,2,3,4}|class=0) = 27+14+13+15+8 = 77
            size_dimension_feature: |X1| = 5

        :return:
        """
        denominator = (all_count_feature_by_clazz + ncc_s_param + size_dimension_feature)
        lower = (count_feature_by_value_and_clazz + 1) / denominator
        upper = (count_feature_by_value_and_clazz + ncc_s_param + 1) / denominator
        return lower, upper
