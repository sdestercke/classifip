import math
import numpy as np
from itertools import product
from classifip.dataset import arff

CONST_PARTIAL_VALUE = -1


def init_scores(param_imprecision, ich_skep, cph_skep, acc_prec, ich_reject, cph_reject, epsilon_rejects):
    ich_skep[param_imprecision], cph_skep[param_imprecision], acc_prec[param_imprecision] = 0, 0, 0
    if epsilon_rejects is not None:
        ich_reject[param_imprecision] = dict.fromkeys(np.array(epsilon_rejects, dtype='<U'), 0)
        cph_reject[param_imprecision] = dict.fromkeys(np.array(epsilon_rejects, dtype='<U'), 0)
    else:
        ich_reject[param_imprecision], cph_reject[param_imprecision] = 0, 0


def init_dataset(in_path, remove_features, scaling):
    data_learning = arff.ArffFile()
    data_learning.load(in_path)
    if remove_features is not None:
        for r_feature in remove_features:
            try:
                data_learning.remove_col(r_feature)
            except Exception as err:
                print("Remove feature error: {0}".format(err))
    nb_labels = get_nb_labels_class(data_learning)
    if scaling:
        normalize(data_learning, n_labels=nb_labels)
    return data_learning, nb_labels


def expansion_partial_to_full_set_binary_vector(partial_binary_vector):
    new_set_binary_vector = list()
    for label in partial_binary_vector:
        if label == CONST_PARTIAL_VALUE:
            new_set_binary_vector.append([1, 0])
        else:
            new_set_binary_vector.append([label])

    set_binary_vector = list(product(*new_set_binary_vector))
    return set_binary_vector


def transform_semi_partial_vector(full_binary_vector, nb_labels):
    """
    Semi-partial binary vector is like:
            Y = [(0, 1, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0), (0, 1, 1, 1, 1, 0)]
    Partial binary vector is like:
            Y = [(0, 1, 1, 0, 1, 0), (0, 1, 1, 1, 1, 0)] = [(0, 1, 1, *, 1, 0)]
    :param full_binary_vector:
    :return:
    """
    _full_binary_vector = np.array(full_binary_vector)
    result = np.zeros(nb_labels, dtype=np.int)
    for idx_label in range(nb_labels):
        label_value = np.unique(_full_binary_vector[:, idx_label])
        if len(label_value) > 1:
            result[idx_label] = CONST_PARTIAL_VALUE
        else:
            result[idx_label] = label_value[0]
    return result


def distance_cardinal_set_inferences(inference_outer, inference_exact, nb_labels):
    """
        This method aims to check if the improved exact inference (3^m-1 comparisons)
        has same number of solutions than the exact inference (all comparisons)

    :param inference_outer:
    :param inference_exact:
    :param nb_labels:
    :return:
    """
    power_outer = 0
    for j in range(nb_labels):
        if inference_outer[j] == CONST_PARTIAL_VALUE:
            power_outer += 1
    return math.pow(2, power_outer) - len(inference_exact)


def normalize(dataArff, n_labels, method='minimax'):
    from classifip.utils import normalize_minmax
    np_data = np.array(dataArff.data, dtype=float)
    np_data = np_data[..., :-n_labels]
    if method == "minimax":
        np_data = normalize_minmax(np_data)
        dataArff.data = [np_data[i].tolist() + dataArff.data[i][-n_labels:] for i in range(len(np_data))]
    else:
        raise Exception("Not found method implemented yet.")


def get_nb_labels_class(dataArff, type_class='nominal'):
    nominal_class = [item for item in dataArff.attribute_types.values() if item == type_class]
    return len(nominal_class)


def incorrectness_completeness_measure(y_true, y_prediction):
    Q, m, hamming = [], len(y_true), 0
    for i, y in enumerate(y_true):
        if y_prediction[i] != CONST_PARTIAL_VALUE:
            Q.append(y_prediction[i])
            if y_prediction[i] != y:
                hamming += 1
    lenQ = len(Q)
    if lenQ != 0:
        return hamming / lenQ, lenQ / m
    else:
        return 0, 0


def setaccuracy_completeness_measure(y_true, y_prediction):
    Q, m, is_set_accuracy = [], len(y_true), True
    for i, y in enumerate(y_true):
        if y_prediction[i] != CONST_PARTIAL_VALUE:
            Q.append(y_prediction[i])
            if y_prediction[i] != y:
                is_set_accuracy = False
    lenQ = len(Q)
    if lenQ != 0:
        return 1 if is_set_accuracy else 0, lenQ / m
    else:
        return 1, 0


def compute_jaccard_similarity_score(x, y):
    """
    x, y two set of set-valued predictions
    Jaccard Similarity J (A,B) = | Intersection (A,B) | /
                                    | Union (A,B) |
    e.g.
        >> x = [(0, 1, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0), (0, 1, 0, 0, 0, 0)]
        >> y = [(0, 1, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0), (0, 1, 1, 1, 1, 0)]
        >> jaccard_similarity(x, y)
        Out >> 0.5

    source: https://gist.github.com/Renien/9672f174e31b6f96f356da09eb481d2c
    """
    intersection_cardinality = len(set(x).intersection(set(y)))
    union_cardinality = len(set(x).union(set(y)))
    return intersection_cardinality / float(union_cardinality)
