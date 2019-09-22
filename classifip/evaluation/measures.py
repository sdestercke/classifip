import math

"""
Zaffalon et al in [1] proposes an utility-discounted accuracy measure
with u65 with a gain of 0.65 and u80 with a gain of 0.8.

[1] Marco ZAFFALON, Giorgio CORANI et Denis MAUÁ.
   “Evaluating credal classifiers by utility-discounted predictive accuracy”.
    In : International Journal of Approximate Reasoning 53.8 (2012), p. 1282-1301.
"""


def u65(Y):
    """
    :param Y: set of predictions
    :return:
    """
    mod_Y = len(Y)
    return 1.6 / mod_Y - 0.6 / mod_Y ** 2


def u80(Y):
    """
    :param Y: set of predictions
    :return:
    """
    mod_Y = len(Y)
    return 2.2 / mod_Y - 1.2 / mod_Y ** 2


"""
    Sebastien et al in [1] proposes an correctness and completeness accuracy measure
    for cautious label-wise ranking problem  
        [1]  DESTERCKE, Sébastien et ALARCON, Yonatan Carlos Carranza. 
        Cautious label-wise ranking with constraint satisfactions.
"""


def correctness_measure(y_true, y_predicts):
    """
        Spearman Footrule for the correctnees, given the predictions \hat{R}_i$, $i=1,\dots,k,

            CR(\widehat{R}) = 1-\frac{\sum_{i=1}^k \min_{\widehat{r}_i \in \widehat{R}_i} |\widehat{r}_i - r_i|}{0.5k^2}

        Attributes:

        :param y_true: set of ground-truth ranking by label
        :param y_predicts: set of ranking predictions
        :return:

        Example:

            y_true = ['L5', 'L7', 'L6', 'L9', 'L11', 'L1', 'L2', 'L3', 'L4', 'L8', 'L10']
            y_predicts =
            [{'L2': 9, 'L3': 10, 'L1': 8, 'L10': 7, 'L11': 4, 'L9': 6, 'L4': 5, 'L5': 3, 'L6': 2, 'L7': 0, 'L8': 1}
             {'L2': 9, 'L3': 10, 'L1': 8, 'L10': 7, 'L11': 4, 'L9': 6, 'L4': 5, 'L5': 3, 'L6': 1, 'L7': 2, 'L8': 0}
             {'L2': 9, 'L3': 10, 'L1': 8, 'L10': 7, 'L11': 4, 'L9': 6, 'L4': 5, 'L5': 3, 'L6': 1, 'L7': 0, 'L8': 2},...]
    """
    if y_predicts is None: return 0.0;
    k = len(y_true)
    sum_dist = 0
    for idx, label in enumerate(y_true):
        min_dist_label = math.inf
        for y_predict in y_predicts:
            new_dist = abs(idx - y_predict[label])
            if min_dist_label > new_dist:
                min_dist_label = new_dist
        sum_dist += min_dist_label

    return 1 - sum_dist / (0.5 * k * k)


def completeness_measure(y_true, y_predicts):
    """
       Completeness which is null if all \hat{R}_i contains the k possible ranks,

            CP(\hat{R}) = \frac{k^2 - \sum_{i=1}^k |\hat{R}_i|}{k^2 - k}

        Attributes:

        :param y_true: set of ground-truth ranking by label
        :param y_predicts: set of ranking predictions
        :return:

        Example:

            y_true = ['L5', 'L7', 'L6', 'L9', 'L11', 'L1', 'L2', 'L3', 'L4', 'L8', 'L10']
            y_predicts =
            [{'L2': 9, 'L3': 10, 'L1': 8, 'L10': 7, 'L11': 4, 'L9': 6, 'L4': 5, 'L5': 3, 'L6': 2, 'L7': 0, 'L8': 1}
             {'L2': 9, 'L3': 10, 'L1': 8, 'L10': 7, 'L11': 4, 'L9': 6, 'L4': 5, 'L5': 3, 'L6': 1, 'L7': 2, 'L8': 0}
             {'L2': 9, 'L3': 10, 'L1': 8, 'L10': 7, 'L11': 4, 'L9': 6, 'L4': 5, 'L5': 3, 'L6': 1, 'L7': 0, 'L8': 2},...]
    """

    k = len(y_true)
    R = 0
    if y_predicts is not None:
        learn_ranks = dict()
        for label in y_true:
            learn_ranks[label] = set()

        for y_predict in y_predicts:
            for idx, label in enumerate(y_true):
                learn_ranks[label].add(y_predict[label])

        # all possible ranks by labels
        for ranks in learn_ranks.values():
            R += len(ranks)
    return (k * k - R) / (k * k - k)
