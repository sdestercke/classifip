import math
import numpy as np


def distance_cardinal_set_inferences(inference_outer, inference_exact, nb_labels):
    power_outer = 0
    for j in range(nb_labels):
        if inference_outer[j] == -1:
            power_outer += 1
    return math.pow(2, power_outer) - len(inference_exact)


def normalize(dataArff, n_labels, method='minimax'):
    from classifip.utils import normalize_minmax
    np_data = np.array(dataArff.data, dtype=float)
    np_data = np_data[..., :-n_labels]
    if method == "minimax":
        np_data = normalize_minmax(np_data)
        dataArff.data = [np_data[i].tolist() + dataArff.data[i][-n_labels:] for i, d in enumerate(np_data)]
    else:
        raise Exception("Not found method implemented yet.")


def get_nb_labels_class(dataArff, type_class='nominal'):
    nominal_class = [item for item in dataArff.attribute_types.values() if item == type_class]
    return len(nominal_class)


def incorrectness_completeness_measure(y_true, y_prediction):
    Q, m, hamming = [], len(y_true), 0
    for i, y in enumerate(y_true):
        if y_prediction[i] != -1:
            Q.append(y_prediction[i])
            if y_prediction[i] != y:
                hamming += 1
    lenQ = len(Q)
    if lenQ != 0:
        return hamming / lenQ, lenQ / m
    else:
        return 0, 0


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
