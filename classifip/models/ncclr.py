from . import nccof
from ..dataset.arff import ArffFile
import numpy as np
import copy


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
                datarep.dump()
                model.learn(datarep)
                self.setnccof.append(model)
        except KeyError:
            raise Exception("Error: The name of ranking attribute should be called 'L'.")

    def evaluate(self, test_dataset, ncc_epsilon=0.001, ncc_s_param=2):

        answers = []
        for j in range(self.nb_labels):
            answer = self.setnccof[j].evaluate(test_dataset, ncc_epsilon, ncc_s_param)
            answers.append(answer[0])

        return answers
