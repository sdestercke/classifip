from classifip.dataset.arff import ArffFile
from classifip.representations.voting import Scores
from classifip.models.mlc.mlcncc import MLCNCC
from classifip.models.ncc import NCC
from scipy.spatial import kdtree, distance
import numpy as np
import copy
from math import exp


class KNN_NCC_BR(MLCNCC):

    def __init__(self):
        super(KNN_NCC_BR, self).__init__()
        self.kd_tree = None
        self.nb_labels = None
        self.nda_models = None
        self.nb_feature = None
        self.x_learning = None
        self.y_learning = None
        self.learn_disc_set = None
        self.radius = None
        self.skeleton_learn_knn = None

    def learn(self,
              learn_data_set,
              nb_labels,
              learn_disc_set=None):
        """
            Warning: it should be normalized since it do an Euclidean distance
        """
        self.__init__()

        self.nb_labels = nb_labels
        self.nda_models = dict()
        self.nb_feature = len(learn_data_set.attributes[:-self.nb_labels])

        _np_data = np.array(learn_data_set.data)
        _index_features = np.array(range(self.nb_feature))
        _index_labels = np.array(np.arange(self.nb_feature,
                                           self.nb_feature + self.nb_labels))
        self.x_learning = np.array(_np_data[:, _index_features], dtype=np.float)
        self.y_learning = _np_data[:, _index_labels].copy()

        # procedure create kd_tree by classifier with missing instances
        self.kd_tree = dict()
        self.radius = np.ones(nb_labels)
        for label_index in range(nb_labels):
            missing_index = []
            # assuming the index learn_disc_set and learn_data_set are same (salmuz)
            for row_index, row_instance in enumerate(learn_disc_set.data):
                if row_instance[self.nb_feature + label_index] == '-1':
                    missing_index.append(row_index)
            x_marginal = np.delete(self.x_learning, missing_index, axis=0)
            _distances = distance.cdist(x_marginal, self.x_learning)
            self.radius[label_index] = np.mean(_distances[np.tril_indices(len(x_marginal), k=-1)])
            self.kd_tree[label_index] = kdtree.KDTree(x_marginal)

        self.learn_disc_set = learn_disc_set.make_clone()

        # vacuous skeleton arff file
        self.skeleton_learn_knn = ArffFile()
        self.skeleton_learn_knn.attribute_data = self.learn_disc_set.attribute_data.copy()
        self.skeleton_learn_knn.attribute_types = self.learn_disc_set.attribute_types.copy()
        self.skeleton_learn_knn.attributes = self.learn_disc_set.attributes.copy()
        for label_name in learn_data_set.attributes[-self.nb_labels:]:
            del self.skeleton_learn_knn.attribute_data[label_name]
            del self.skeleton_learn_knn.attribute_types[label_name]
            self.skeleton_learn_knn.attributes.pop()
        self.skeleton_learn_knn.data = list()
        self.skeleton_learn_knn.define_attribute(name="class", atype="nominal", data=['0', '1'])

    def evaluate(self, test_dataset, ncc_epsilon=0.001, ncc_s_param=2.0, k=1, type_knn=2, precision=16):
        """
        :param test_dataset:
        :param k: k*size_avg_radius pairwise all instances to get neighbors (ball)
        :param ncc_epsilon:
        :param ncc_s_param:
        :param type_knn: (1) k nearest neighbors,
                         (2) the nearest neighbors relative to ball Euclidean distance,
                             where the k*radius is equals to average on all instances
        :param precision:

        ...note::
            if row_instance[.] == '-1', thus it is a missing label,
            not considering as training instance.

        :return:
        """
        if type_knn not in [1, 2]:
            raise Exception('Setting k-nearest neighbors is not implemented yet.')

        answers = []
        model_ncc = NCC()
        for raw_instance, disc_instance in test_dataset:

            # validate instance is np-array
            instance = np.array(raw_instance)
            if len(instance) > self.nb_feature:
                instance = np.array(instance[:self.nb_feature], dtype=float)
            else:
                instance = instance.astype(dtype=float)

            resulting_score_ncc = np.zeros((self.nb_labels, 2))
            resulting_score_prec = np.zeros((self.nb_labels, 2))
            for label_index in range(self.nb_labels):
                if type_knn == 1:
                    _, index_disk_knn = self.kd_tree[label_index].query(instance, k=k)
                else:
                    index_disk_knn = self.kd_tree[label_index].query_ball_point(instance, k * self.radius[label_index])
                # learning and predicting in local model by with respect to unlabelled instance
                if len(index_disk_knn) == 0:
                    resulting_score_ncc[label_index, 1] = 1
                    resulting_score_prec[label_index, 1] = np.random.uniform(size=1)
                    resulting_score_prec[label_index, 0] = resulting_score_prec[label_index, 1]
                else:
                    data_knn = list()
                    for row_index in index_disk_knn:
                        row_instance = self.learn_disc_set.data[row_index].copy()
                        if row_instance[self.nb_feature + label_index] != '-1':
                            clazz = row_instance[self.nb_feature + label_index]
                            data_knn.append(row_instance[:self.nb_feature] + [clazz])
                    data_learn_knn = self.skeleton_learn_knn.make_clone()
                    data_learn_knn.data = data_knn
                    model_ncc.learn(data_learn_knn)
                    ans_credal = model_ncc.evaluate(test_dataset=[disc_instance],
                                                    ncc_s_param=ncc_s_param,
                                                    precision=precision)[0]
                    ans_precise = model_ncc.evaluate(test_dataset=[disc_instance],
                                                     ncc_s_param=0,
                                                     ncc_epsilon=0,
                                                     precision=precision)[0]
                    resulting_score_ncc[label_index, :] = list(reversed(ans_credal.lproba[:, 1]))
                    resulting_score_prec[label_index, :] = ans_precise.proba[1]
            ans_credal = Scores(resulting_score_ncc, precision=precision)
            ans_precise = Scores(resulting_score_prec, precision=precision)
            answers.append((ans_credal, ans_precise))
        return answers
