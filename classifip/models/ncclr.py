from . import nccof
from ..dataset.arff import ArffFile
import numpy as np
import copy
from constraint import *


def _create_utility_matrix(size):
    utility = np.zeros([size, size])
    for i in range(0, size):
        for j in range(i, size):
            utility[i, j] = size + i - j - 1
            utility[j, i] = size + i - j - 1
    return utility

class NCCLR(object):
    """NCCLR implements the naive credal classification method using the IDM for
       Label ranking problem with label-wise decomposition.

    """

    def __init__(self):
        self.nb_clazz = 0
        self.set_nccof = dict()
        self.clazz = []
        self.ranking_utility = None;

    def learn(self, learn_data_set):
        try:
            classes = learn_data_set.attribute_data['L'][:]
            self.nb_clazz = len(classes)
            self.clazz = classes
            self.ranking_utility = _create_utility_matrix(self.nb_clazz)
            rankings = [str(i + 1) for i in range(self.nb_clazz)]
            for class_value in classes:
                # print("Building ranking classifier %s" % class_value)
                model = nccof.NCCOF()
                datarep = ArffFile()
                datarep.attribute_data = learn_data_set.attribute_data.copy()
                datarep.attribute_types = learn_data_set.attribute_types.copy()
                datarep.data = copy.deepcopy(learn_data_set.data)
                datarep.relation = learn_data_set.relation
                datarep.attributes = copy.copy(learn_data_set.attributes)
                datarep.comment = copy.copy(learn_data_set.comment)
                datarep.define_attribute(name="class", atype="nominal", data=rankings)
                for number, instance in enumerate(datarep.data):
                    label_ranking = instance[-1].split(">")
                    if len(label_ranking) == 0 or len(label_ranking) < self.nb_clazz:
                        raise Exception("Error: Number labels for ranking is not correct in sample " + str(number))
                    instance.append(str(label_ranking.index(class_value) + 1))
                datarep.remove_col('L')
                model.learn(datarep)
                self.set_nccof[class_value] = model
        except KeyError:
            raise Exception("Error: The name of ranking attribute should be called 'L'.")

    def evaluate(self, test_data_set, ncc_s_param=2):
        answers = []
        for item in test_data_set:
            ans_lw_ranking = dict()
            for clazz in self.clazz:
                ans_lw_ranking[clazz] = self.set_nccof[clazz].evaluate([item], ncc_epsilon=0.001, ncc_s_param=ncc_s_param)[0]
            answers.append(ans_lw_ranking)
        return answers

    def predict_CSP(self, evaluates):
        solutions = []
        for evaluate in evaluates:
            problem = Problem()
            for clazz, classifier in evaluate.items():
                maxDecision = classifier.getmaximaldecision(self.ranking_utility)
                problem.addVariable(clazz, np.where(maxDecision > 0)[0])
            problem.addConstraint(AllDifferentConstraint())
            solution = problem.getSolution()
            solutions.append(None if solution is None else solution)
        return solutions
