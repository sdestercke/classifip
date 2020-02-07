import math


def distance_cardinal_set_inferences(inference_outer, inference_exact, nb_labels):
    power_outer = 0
    for j in range(nb_labels):
        if inference_outer[j] == -1:
            power_outer += 1
    return math.pow(2, power_outer) - len(inference_exact)
