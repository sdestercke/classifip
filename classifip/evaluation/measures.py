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
