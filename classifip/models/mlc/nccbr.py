from classifip.dataset.arff import ArffFile
from classifip.representations.voting import Scores
import numpy as np
from math import exp


class NCCBR(object):
    """NCCBR implements the naive credal classification method using the IDM for
    multilabel classification with binary relevance.
    
    If data are all precise, it returns
    :class:`~classifip.representations.voting.Scores`. The base classifier method
    is based on [#zaffalon2002]_ and on the improvement proposed by [#corani2010]_
    
    :param feature_count: store counts of couples label/feature
    :type feature_count: dictionnary with keys label/feature
    :param label_counts: store counts of class labels (to instanciate prior)
    :type label_counts: list
    :param feature_names: store the names of features
    :type feature_names: list
    :param feature_values: store modalities of features
    :type feature_values: dictionnary associating each feature name to a list
    
    """

    def __init__(self):
        """Build an empty NCCBR structure
        """
        self.feature_names = []
        self.label_names = []
        self.feature_values = dict()
        self.feature_count = dict()
        self.nblabels = 0
        self.trainingsize = 0
        self.labelcounts = []

    def learn(self, learn_data_set, nb_labels, missing_pct=0.0):
        """learn the NCC for each label, mainly storing counts of feature/label pairs
        
        :param learn_data_set: learning instances
        :type learn_data_set: :class:`~classifip.dataset.arff.ArffFile`
        :param nb_labels: number of labels
        :type nb_labels: integer
        :param missing_pct: percentage of missing labels
        :type missing_pct: float
        """
        self.__init__()
        if missing_pct < 0.0 or missing_pct > 1.0:
            raise Exception('Negative percentage or Percentage higher than one.')
        self.nblabels = nb_labels
        self.trainingsize = int(len(learn_data_set.data) * (1 - missing_pct)) if missing_pct > 0.0 \
            else len(learn_data_set.data)
        # Initializing the counts
        self.feature_names = learn_data_set.attributes[:-self.nblabels]
        self.label_names = learn_data_set.attributes[-self.nblabels:]
        self.feature_values = learn_data_set.attribute_data.copy()
        for label_value in self.label_names:
            missing_label_index = None
            if missing_pct > 0:
                missing_label_index = np.random.choice(len(learn_data_set.data),
                                                       int(len(learn_data_set.data) * missing_pct),
                                                       replace=False)

            label_set_one = learn_data_set.select_col_vals(label_value, ['1'], missing_label_index)
            self.labelcounts.append(len(label_set_one.data))
            label_set_zero = learn_data_set.select_col_vals(label_value, ['0'], missing_label_index)
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

    def evaluate(self, test_dataset, ncc_epsilon=0.001, ncc_s_param=2.0):
        """evaluate the instances and return a list of probability intervals.
        
        :param test_dataset: list of input features of instances to evaluate
        :type dataset: list
        :param ncc_epsilon: espilon issued from [#corani2010]_ (should be > 0)
            to avoid zero count issues
        :type ncc_espilon: float
        :param ncc_s_param: s parameter used in the IDM learning (settle
        imprecision level)
        :type ncc_s_param: float
        :returns: for each value of ncc_s_param, a set of scores for each label
        :rtype: lists of :class:`~classifip.representations.voting.Scores`
 
        .. note::
    
            * Precise prior
                prior class probabilities are assumed to be precise to speed up
                computations. The impact on the result is small, unless
                the number of class example in the training set is close to s or lower.
        
        .. warning::
    
            * zero float division can happen if too many input features
            
        .. todo::
        
            * solve the zero division problem
            
        """
        # computing label proportions
        label_prop = [n / float(self.trainingsize) for n in self.labelcounts]
        answers = []
        for item in test_dataset:

            # initializing scores
            resulting_score = np.zeros((self.nblabels, 2))
            # computes product of lower/upper prob for each class
            for j in range(self.nblabels):
                u_numerator = 1 - label_prop[j]
                l_numerator = 1 - label_prop[j]
                u_denom = label_prop[j]
                l_denom = label_prop[j]
                for f_index, feature in enumerate(self.feature_names):
                    # computation of denominator (label=1)
                    f_val_index = self.feature_values[feature].index(item[f_index])
                    count_string = self.label_names[j] + '|in|' + feature
                    num_items = float(sum(self.feature_count[count_string]))
                    lower = (self.feature_count[count_string][f_val_index] / (num_items + ncc_s_param))
                    l_denom = l_denom * (
                            (1 - ncc_epsilon) * lower + ncc_epsilon / len(self.feature_count[count_string]))
                    upper = ((self.feature_count[count_string][f_val_index] + ncc_s_param) / (num_items + ncc_s_param))
                    u_denom = u_denom * (
                            (1 - ncc_epsilon) * upper + ncc_epsilon / len(self.feature_count[count_string]))

                    # computation of numerator (label=0)
                    count_string = self.label_names[j] + '|out|' + feature
                    num_items = float(sum(self.feature_count[count_string]))
                    upper = ((self.feature_count[count_string][f_val_index] + ncc_s_param) / (num_items + ncc_s_param))
                    l_numerator = l_numerator * (
                            (1 - ncc_epsilon) * upper + ncc_epsilon / len(self.feature_count[count_string]))
                    lower = (self.feature_count[count_string][f_val_index]) / (num_items + ncc_s_param)
                    u_numerator = u_numerator * (
                            (1 - ncc_epsilon) * lower + ncc_epsilon / len(self.feature_count[count_string]))
                resulting_score[j, 1] = u_denom / (u_denom + u_numerator)
                resulting_score[j, 0] = l_denom / (l_denom + l_numerator)
            result = Scores(resulting_score)
            answers.append(result)

        return answers
