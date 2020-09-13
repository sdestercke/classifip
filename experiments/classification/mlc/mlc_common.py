import math
import numpy as np

CONST_PARTIAL_VALUE = -1


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
