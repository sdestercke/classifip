import random, numpy as np
import queue, os, multiprocessing, csv
from functools import partial
from classifip.utils import create_logger
from itertools import permutations, combinations, product
from classifip.representations.intervalsProbability import IntervalsProbability
from classifip.representations.voting import Scores
from classifip.evaluation.measures import hamming_distance
from mlc_common import distance_cardinal_set_inferences
from classifip.utils import timeit


def print_binary_tree(root):
    def __print(node, _p=0):
        bound_probabilities = node["prob"]

        str_buf1 = "[%.3f, %.3f]" % (bound_probabilities.lproba[1, 0], bound_probabilities.lproba[0, 0])
        str_buf2 = "[%.3f, %.3f]" % (bound_probabilities.lproba[1, 1], bound_probabilities.lproba[0, 1])
        tab_len = 14
        if _p == 0:
            print(node["0"]["label"], " " * (tab_len - 2), node["1"]["label"])
            print(str_buf1, str_buf2)
        else:
            print("    " * _p + node["0"]["label"] + " " * (tab_len - len(node["0"]["label"]) + 1) + node["1"]["label"])
            print("    " * _p + str_buf1, str_buf2)
        _p += 1

        if "prob" in node["0"]:
            __print(node["0"], _p)

        if "prob" in node["1"]:
            __print(node["1"], _p)

    return __print(root)


def binaryToDecimal(n):
    return int(n, 2)


def generation_interval_probabilities(epsilon=0.5):
    theta = np.random.uniform(0.0, 1.0, 1)[0]
    lower_prob_y1 = max(0, theta - epsilon)
    upper_prob_y1 = min(theta + epsilon, 1)
    interval_probability = np.array([[1.0 - lower_prob_y1, upper_prob_y1], [1.0 - upper_prob_y1, lower_prob_y1]])
    return IntervalsProbability(interval_probability)


def generation_independent_labels(nb_labels, epsilon=None):
    root = dict({"label": ''})
    tbinary = queue.LifoQueue()
    tbinary.put(root)
    while not tbinary.empty():
        node = tbinary.get()
        epsilon = epsilon if epsilon is not None else np.random.uniform(size=1)[0]
        node["prob"] = generation_interval_probabilities(epsilon=epsilon)
        node["0"] = dict({'label': node['label'] + "0"})
        node["1"] = dict({'label': node['label'] + "1"})
        if len(node['label']) < nb_labels - 1:
            tbinary.put(node["1"])
            tbinary.put(node["0"])
    return root


def getlowerexpectation(cost_vector, root):
    def lowerExp(node):
        bound_probabilities = node["prob"]
        if "prob" not in node["0"] or "prob" not in node["1"]:
            expInf = bound_probabilities.getlowerexpectation(
                function=np.array([cost_vector[binaryToDecimal(node["0"]["label"])],
                                   cost_vector[binaryToDecimal(node["1"]["label"])]]))
        else:
            expInf = bound_probabilities.getlowerexpectation(
                function=np.array([lowerExp(node["0"]),
                                   lowerExp(node["1"])]))
        return expInf

    return lowerExp(root)


def getlowerexpectationGeneration(root):
    import xxhash
    leafs = dict()
    leafs_var = dict()

    def __hash(query):
        _hash = xxhash.xxh64()
        _hash.update(query)
        res_hash = _hash.hexdigest()
        _hash.reset()
        return res_hash

    def getlowerexpectation(prob, is_leaf=False, idx0=None, idx1=None):
        if is_leaf:
            expDichStr = "min(c[%s]*%s+c[%s]*%s,c[%s]*%s+c[%s]*%s)" % \
                         (idx0, prob[1, 0], idx1, prob[0, 1], idx0, prob[0, 0], idx1, prob[1, 1])
            name_var = "a" + str(idx0) + str(idx1)
            leafs_var[__hash(expDichStr)] = name_var + "=" + expDichStr
            leafs[__hash(expDichStr)] = name_var
        else:
            keyx0 = __hash(idx0)
            new_idx0 = leafs[keyx0] if keyx0 in leafs else idx0
            keyx1 = __hash(idx1)
            new_idx1 = leafs[keyx1] if keyx1 in leafs else idx1
            expDichStr = "min(%s*%s+%s*%s,%s*%s+%s*%s)" \
                         % (new_idx0, prob[1, 0], new_idx1, prob[0, 1], new_idx0, prob[0, 0], new_idx1, prob[1, 1])
            name_var = str(new_idx0) + str(new_idx1)
            leafs_var[__hash(expDichStr)] = name_var + "=" + expDichStr
            leafs[__hash(expDichStr)] = name_var

        return expDichStr

    def lowerExp(node):
        bound_probabilities = node["prob"]
        if "prob" not in node["0"] or "prob" not in node["1"]:
            expInfStr = getlowerexpectation(bound_probabilities.lproba,
                                            is_leaf=True,
                                            idx0=binaryToDecimal(node["0"]["label"]),
                                            idx1=binaryToDecimal(node["1"]["label"]))
        else:
            expInfStr = getlowerexpectation(bound_probabilities.lproba,
                                            idx0=lowerExp(node["0"]),
                                            idx1=lowerExp(node["1"]))
        return expInfStr

    lexp = lowerExp(root)
    lexp_cost_var = "\n\t".join(list(leafs_var.values())[:-1])
    return lexp, lexp_cost_var


def marginal_probabilities_v1(root, nb_labels):
    all_output_space = np.array(list(product([0, 1], repeat=nb_labels)))
    resulting_score = np.zeros((nb_labels, 2))
    for idx_label in range(nb_labels):
        cost_vector = all_output_space[:, idx_label]
        lower_probability_y1 = getlowerexpectation(cost_vector, root)
        lower_probability_y0 = getlowerexpectation(1 - cost_vector, root)
        resulting_score[idx_label, 1] = 1 - lower_probability_y0
        resulting_score[idx_label, 0] = lower_probability_y1
    return Scores(resulting_score)


def inference_exact_inference(root, nb_labels):
    indices_labels = list(range(nb_labels))
    all_output_space = list(product([0, 1], repeat=nb_labels))
    maximality_sets = dict.fromkeys(all_output_space, True)
    cardinal_all_output = len(all_output_space)
    all_output_space = np.array(all_output_space)

    PARTIAL_VALUE = 3
    expectation_infimum_str, exp_cost_leaf_var = getlowerexpectationGeneration(root)
    # print("global expectation\ndef expectation(c):\n\t" + exp_cost_leaf_var + "\n\treturn " + expectation_infimum_str)
    exec("global exec_expectation_inf\n"
         "def exec_expectation_inf(c):\n\t" + exp_cost_leaf_var + "\n\treturn " + expectation_infimum_str)

    def calculation_cost(a_sub_vector, neq_idx_labels=None):
        """
            Applying by column with a binary singleton Hamming loss function:
                        L(y1, y2) = 1_{y1 \neq y2}
                        L(y1, y2) = 1_{y1 ==0 and y2==1} + 1_{y1 ==1 and y2==0}
                        L(y1, y2) = (1-y1)*y2 + (1-y2)*y1
                        L(y1, y2) = y1 + y2 - 2*y2*y1
                        L(y1, y2) = (y2 - y1)^2 = | y2 - y1 |
            This operation can be computed by column to get vector cost hamming multilabel
        :param a_sub_vector:
        :param neq_idx_labels:
        :return:
        """
        a_fully = np.array([a_sub_vector, ] * cardinal_all_output)
        if neq_idx_labels is not None:
            return np.sum(abs(all_output_space[:, neq_idx_labels] - a_fully), axis=1)
        else:
            return np.sum(abs(all_output_space - a_fully), axis=1)

    def expansion_partial_binary_vector(binary_vector):
        new_partial_vector = list()
        new_net_partial_vector = list()
        for label in binary_vector:
            if label == PARTIAL_VALUE:
                new_partial_vector.append([1, 0])
                new_net_partial_vector.append([1, 0])
            else:
                new_partial_vector.append([label])
                new_net_partial_vector.append([1 - label])

        set_binary_vector = list(product(*new_partial_vector))
        set_dominated_vector = list(product(*new_net_partial_vector))
        return set_binary_vector, set_dominated_vector

    def is_solution_not_dominated(partial_prediction):
        if PARTIAL_VALUE in partial_prediction:
            set_dominant, set_dominated = expansion_partial_binary_vector(partial_prediction)
        else:
            set_dominant, set_dominated = [tuple(partial_prediction)], [tuple(1 - partial_prediction)]
        is_not_dominated = False
        for idx in range(len(set_dominant)):
            if maximality_sets[set_dominant[idx]] and maximality_sets[set_dominated[idx]]:
                is_not_dominated = True
                break
        return is_not_dominated, set_dominated

    def mark_solution_dominated(set_dominated_preds):
        for dominated in set_dominated_preds:
            maximality_sets[dominated] = False

    def inf_not_equal_labels(nb_labels, p_n_not_equal_indices, p_neq_idx=None):
        for a_vector in product([0, 1], repeat=p_n_not_equal_indices):
            partial_prediction = PARTIAL_VALUE * np.ones(nb_labels, dtype=int)
            partial_prediction[p_neq_idx] = np.array(a_vector)
            is_not_dominated, set_dominated_preds = is_solution_not_dominated(partial_prediction)
            # print("%s ==> is_not_dominated %s" % (partial_prediction, is_not_dominated), flush=True)
            if is_not_dominated:
                not_a_vector = 1 - np.array(a_vector)
                cost_vector = calculation_cost(not_a_vector, p_neq_idx)
                # inf_expectation = getlowerexpectation(cost_vector, root)
                inf_expectation = exec_expectation_inf(cost_vector)
                # print("%s >_M %s ==> %s <? %s" % (
                #     partial_prediction, not_a_vector, (p_n_not_equal_indices * 0.5),
                #     inf_expectation), flush=True)
                if (p_n_not_equal_indices * 0.5) < inf_expectation:
                    mark_solution_dominated(set_dominated_preds)

    # some equal labels in comparison (m1 > m2)
    for i in reversed(range(nb_labels - 1)):
        n_not_equal_indices = (nb_labels - 1) - i
        a_set_indices = list(combinations(indices_labels, n_not_equal_indices))
        for neq_idx in a_set_indices:
            inf_not_equal_labels(nb_labels, n_not_equal_indices, list(neq_idx))
    # none equal labels (all different)
    inf_not_equal_labels(nb_labels, nb_labels)

    solution_exact = list(filter(lambda k: maximality_sets[k], maximality_sets.keys()))
    # _logger.debug("set solutions improved exact inference %s", solution_exact)
    return solution_exact


def parallel_inferences(pid_tree, nb_labels, epsilon):
    pid = multiprocessing.current_process().name
    root = generation_independent_labels(nb_labels, epsilon=epsilon)
    inference_exact = inference_exact_inference(root, nb_labels)
    set_marginal_probabilities = marginal_probabilities_v1(root, nb_labels)
    inference_outer = set_marginal_probabilities.multilab_dom()
    distance_cardinal = distance_cardinal_set_inferences(inference_outer, inference_exact, nb_labels)
    # if len(inference_exact) > 1 and len(inference_exact) % 2 == 1:
    #     print('Example Lemme 5 (%s, %s)' % (inference_exact, inference_outer))
    #     print_binary_tree(root)

    print("Pid (%s, %s) Exact versus Outer (%s, %s, %s)" %
          (pid, pid_tree, len(inference_exact), inference_outer, distance_cardinal), flush=True)
    if distance_cardinal < 0:
        raise Exception("Not possible %s, %s" % (inference_exact, inference_outer))
    return distance_cardinal


def computing_outer_vs_exact_inference_random_tree(out_path,
                                                   nb_labels=3,
                                                   nb_repeats=100,
                                                   nb_process=1,
                                                   seed=None,
                                                   min_epsilon_param=0.1,
                                                   max_epsilon_param=0.5,
                                                   step_epsilon_param=0.1):
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_outer_vs_exact_inference_random_tree", True)
    logger.info('Results file (%s)', out_path)
    logger.info("(nb_repeats, nb_process, nb_labels) (%s, %s, %s)",
                nb_repeats, nb_process, nb_labels)
    logger.info("(min_epsilon_param, max_epsilon_param, step_epsilon_param) (%s, %s, %s)",
                min_epsilon_param, max_epsilon_param, step_epsilon_param)
    if seed is None:
        seed = random.randrange(pow(2, 20))
    random.seed(seed)
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)

    POOL = multiprocessing.Pool(processes=nb_process)
    for epsilon in np.arange(min_epsilon_param, max_epsilon_param, step_epsilon_param):
        target_function = partial(parallel_inferences, nb_labels=nb_labels, epsilon=epsilon)
        set_distance_cardinal = POOL.map(target_function, range(nb_repeats))
        writer.writerow(np.hstack((epsilon, set_distance_cardinal)))
        file_csv.flush()
        logger.info("Partial-s-k_step (%s, %s)", str(epsilon), sum(set_distance_cardinal) / nb_repeats)
    file_csv.close()
    logger.info("Results Final")


sys_nb_labels = 4
sys_out_path = "results_labels" + str(sys_nb_labels) + ".csv"
computing_outer_vs_exact_inference_random_tree(out_path=sys_out_path,
                                               nb_labels=sys_nb_labels,
                                               nb_repeats=10,
                                               nb_process=1)
