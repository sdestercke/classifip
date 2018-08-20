from . import nccof
from ..dataset.arff import ArffFile
import copy

class NCCLR(object):
    """NCCLR implements the naive credal classification method using the IDM for
       Label ranking problem with label-wise decomposition.

    """

    def __init__(self):
        self.labels = []
        self.radius = 0.0
        self.normal = []
        self.setnccof = []

    def learn(self, learn_data_set):
        classes = learn_data_set.attribute_data['L'][:]
        self.nb_labels = len(classes)
        for class_value in classes[0:-1]:
            model = nccof.NCCOF()
            datarep = ArffFile()
            datarep.attribute_data = learn_data_set.attribute_data.copy()
            datarep.attribute_types = learn_data_set.attribute_types.copy()
            datarep.data = copy.deepcopy(learn_data_set.data)
            datarep.relation = learn_data_set.relation
            datarep.attributes = copy.copy(learn_data_set.attributes)
            datarep.comment = copy.copy(learn_data_set.comment)
            datarep.attribute_data['class'] = ['0', '1', '2']
            datarep.attribute_types['class'] = 'nominal'
            datarep.attributes.append('class')
            for number,instance in enumerate(datarep.data):
                label_ranking = instance[-1].split(">")
                if len(label_ranking) == 0 or len(label_ranking) < self.nb_labels:
                    raise Exception("Error: Number labels for ranking is not correct in sample "+ str(number))
                instance.append(str(label_ranking.index(class_value)))
            datarep.remove_col('L')
            print(datarep.dump())
            model.learn(datarep)
            self.setnccof.append(model)


