from . import nccof
from ..dataset.arff import ArffFile
import numpy as np
import copy
from constraint import *

def inference_ranking_csp(inferences):
    ranking_utility = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    for inference in inferences:
        problem = Problem()
        for idx, classifier in enumerate(inference):
            maxDecision = classifier.getmaximaldecision(ranking_utility)
            problem.addVariable("R" + str(idx), np.where(maxDecision > 0)[0])
        problem.addConstraint(AllDifferentConstraint())
        print(problem.getSolution())

class NCCLR(object):
    """NCCLR implements the naive credal classification method using the IDM for
       Label ranking problem with label-wise decomposition.

    """

    def __init__(self):
        self.nb_labels = 0
        self.setnccof = []

    def learn(self, learn_data_set):
        try:
            classes = learn_data_set.attribute_data['L'][:]
            self.nb_labels = len(classes)
            for class_value in classes:
                model = nccof.NCCOF()
                datarep = ArffFile()
                datarep.attribute_data = learn_data_set.attribute_data.copy()
                datarep.attribute_types = learn_data_set.attribute_types.copy()
                datarep.data = copy.deepcopy(learn_data_set.data)
                datarep.relation = learn_data_set.relation
                datarep.attributes = copy.copy(learn_data_set.attributes)
                datarep.comment = copy.copy(learn_data_set.comment)
                datarep.attribute_data['class'] = [str(i) for i in range(self.nb_labels)]
                datarep.attribute_types['class'] = 'nominal'
                datarep.attributes.append('class')
                for number, instance in enumerate(datarep.data):
                    label_ranking = instance[-1].split(">")
                    if len(label_ranking) == 0 or len(label_ranking) < self.nb_labels:
                        raise Exception("Error: Number labels for ranking is not correct in sample " + str(number))
                    instance.append(str(label_ranking.index(class_value)))
                datarep.remove_col('L')
                model.learn(datarep)
                self.setnccof.append(model)
        except KeyError:
            raise Exception("Error: The name of ranking attribute should be called 'L'.")

    def evaluate(self, test_data_set, ncc_s_param=2):
        answers = []
        for item in test_data_set:
            ans_lw_ranking = []
            for j in range(self.nb_labels):
                ans_lw_ranking.extend(self.setnccof[j].evaluate([item], ncc_epsilon=0.001, ncc_s_param=ncc_s_param))
            answers.append(ans_lw_ranking)
        return answers
