import math
import numpy as np
from itertools import product
from operator import itemgetter, add
from classifip.dataset import arff

CONST_PARTIAL_VALUE = -1


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
    """
        Hamming loss when y_prediction is not partial or cautious
    :param y_true:
    :param y_prediction:
    :return:
    """
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


def reject_partial_hamming_measure(epsilon_rejects, y_eq_1_precise_probs, nb_labels):
    precise_rejects = dict()
    if epsilon_rejects is not None and len(epsilon_rejects) > 0:
        for epsilon_reject in epsilon_rejects:
            precise_reject = -2 * np.ones(nb_labels, dtype=int)
            all_idx = set(range(nb_labels))
            probabilities_yi_eq_1 = y_eq_1_precise_probs.copy()
            ones = set(np.where(probabilities_yi_eq_1 >= 0.5 + epsilon_reject)[0])
            zeros = set(np.where(probabilities_yi_eq_1 <= 0.5 - epsilon_reject)[0])
            stars = all_idx - ones - zeros
            precise_reject[list(stars)] = CONST_PARTIAL_VALUE
            precise_reject[list(zeros)] = 0
            precise_reject[list(ones)] = 1
            precise_rejects[str(epsilon_reject)] = precise_reject
    return precise_rejects


def abstention_partial_hamming_measure(y_true, y_eq_1_probabilities, c_spe, c_par):
    """
    :param y_true:  ground truth observed
    :param y_eq_1_probabilities: marginal probability P(Y_i = 1)
    :param c_par:
    :param c_spe:
    :return:
    """

    nb_labels = len(y_true)

    def SEP(probabilities, c):
        y_sep_prediction = list()
        for idx_label in range(nb_labels):
            if 1 - probabilities[idx_label] == \
                    min(probabilities[idx_label], 1 - probabilities[idx_label], c):
                y_sep_prediction.append(1)
            elif probabilities[idx_label] == \
                    min(probabilities[idx_label], 1 - probabilities[idx_label], c):
                y_sep_prediction.append(0)
            else:
                y_sep_prediction.append(CONST_PARTIAL_VALUE)
        return y_sep_prediction

    def SEP_score(y_pred_abstention, c):
        nb_abstentions = 0
        sep_score = 0
        for idx_label in range(nb_labels):
            if y_pred_abstention[idx_label] == CONST_PARTIAL_VALUE:
                nb_abstentions += 1
            elif y_pred_abstention[idx_label] != y_true[idx_label]:
                sep_score += 1
        pct_nb_abstentions = nb_abstentions / nb_labels  # percentage of number of abstentions
        sep_score = (nb_abstentions * c) / nb_labels + sep_score / nb_labels
        return sep_score, pct_nb_abstentions

    def PAR(probabilities, c):
        score = [min(probabilities[i], 1 - probabilities[i]) for i in range(nb_labels)]
        y_par_prediction = [CONST_PARTIAL_VALUE] * nb_labels
        indices, score_sorted = zip(*sorted(enumerate(score), key=itemgetter(1)))
        e_mins = [c * 0.5]
        tempVal = 0
        for idx_label in range(nb_labels):
            tempVal += score_sorted[idx_label]
            _e_add = (c * (nb_labels - idx_label - 1)) / (2 * nb_labels - idx_label - 1)
            e_mins.append(tempVal / nb_labels + _e_add)
        k_opt = e_mins.index(min(e_mins))
        for lab in range(k_opt):
            if score[indices[lab]] == probabilities[indices[lab]]:
                y_par_prediction[indices[lab]] = 0
            else:
                y_par_prediction[indices[lab]] = 1
        return y_par_prediction

    def PAR_score(y_pred_abstention, c):
        nb_abstentions = 0
        par_score = 0
        for idx_label in range(nb_labels):
            if y_pred_abstention[idx_label] == CONST_PARTIAL_VALUE:
                nb_abstentions += 1
            elif y_pred_abstention[idx_label] != y_true[idx_label]:
                par_score += 1
        pct_nb_abstentions = nb_abstentions / nb_labels  # percentage of number of abstentions
        par_score = (nb_abstentions * c) / (nb_labels + nb_abstentions) + par_score / nb_labels
        return par_score, pct_nb_abstentions

    y_sep_prediction = dict()
    y_sep_score = dict()
    for c in c_spe:
        y_pred_abstention = SEP(y_eq_1_probabilities, c)
        y_sep_prediction[str(c)] = y_pred_abstention
        y_sep_score[str(c)] = SEP_score(y_pred_abstention, c)

    y_par_prediction = dict()
    y_par_score = dict()
    for c in c_par:
        y_pred_abstention = PAR(y_eq_1_probabilities, c)
        y_par_prediction[str(c)] = y_pred_abstention
        y_par_score[str(c)] = PAR_score(y_pred_abstention, c)

    return y_sep_prediction, y_sep_score, y_par_prediction, y_par_score




