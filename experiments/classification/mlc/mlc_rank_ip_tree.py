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


def inference_exact_inference(root, nb_labels):
    all_output_space = list(product([0, 1], repeat=nb_labels))
    all_output_space = np.array(all_output_space)
    expectation_infimum_str, exp_cost_leaf_var = getlowerexpectationGeneration(root)

    # print("global expectation\ndef expectation(c):\n\t" + exp_cost_leaf_var + "\n\treturn " + expectation_infimum_str)
    exec("global exec_expectation_inf\n"
         "def exec_expectation_inf(c):\n\t" + exp_cost_leaf_var +
         "\n\treturn " + expectation_infimum_str)

    w_pair_labels = list(permutations(list(range(nb_labels)), 2))
    maximality_sets = []

    def diff_hamming_costs(yi_idx, yj_idx):
        print("->", abs(all_output_space[:, list(yj_idx)] - 1), flush=True)
        print("->", abs(all_output_space[:, list(yi_idx)] - 1), flush=True)
        print("-->", abs(all_output_space[:, list(yj_idx)] - 1) -
                      abs(all_output_space[:, list(yi_idx)] - 1))
        print("->", np.sum(abs(all_output_space[:, list(yj_idx)] - 1) -
                      abs(all_output_space[:, list(yi_idx)] - 1), axis=1))
        return np.sum(abs(all_output_space[:, list(yj_idx)] - 1) -
                      abs(all_output_space[:, list(yi_idx)] - 1), axis=1)

    def maximality(set_disagreements):
        yi_idx, yj_idx = set(), set()
        for lambda_i, lambda_j in set_disagreements:
            yi_idx.add(lambda_i)
            yj_idx.add(lambda_j)
        cost_vector = diff_hamming_costs(yi_idx, yj_idx)
        inf_expectation = exec_expectation_inf(cost_vector)
        if inf_expectation > 0:
            maximality_sets.append(set_disagreements)

    def check_disagreement(l, W, I):
        if l == 1:
            for w_pair in W:
                I_temp = I.copy()
                I_temp.append(w_pair)
                print(I_temp)
                maximality(I_temp)
        else:
            W_copy = W.copy()
            for w_pair in W:
                I_temp = I.copy()
                I_temp.append(w_pair)
                print(I_temp)
                maximality(I_temp)
                setwp = W_copy.copy()
                setwp.remove(w_pair)
                opposite = w_pair[::-1]
                if opposite in setwp:
                    setwp.remove(opposite)
                check_disagreement(l - 1, setwp, I_temp)
                W_copy.remove(w_pair)

    check_disagreement(nb_labels - 1, w_pair_labels, [])
    return maximality_sets


def parallel_inferences(pid_tree, nb_labels, epsilon):
    pid = multiprocessing.current_process().name
    root = generation_independent_labels(nb_labels, epsilon=epsilon)
    inference_exact = inference_exact_inference(root, nb_labels)
    print_binary_tree(root)
    print(inference_exact)
    # set_marginal_probabilities = marginal_probabilities_v1(root, nb_labels)
    # inference_outer = set_marginal_probabilities.multilab_dom()
    # distance_cardinal = distance_cardinal_set_inferences(inference_outer, inference_exact, nb_labels)
    # if len(inference_exact) > 1 and len(inference_exact) % 2 == 1:
    #     print('Example Lemme 5 (%s, %s)' % (inference_exact, inference_outer))
    #     print_binary_tree(root)

    # print("Pid (%s, %s) Exact versus Outer (%s, %s, %s)" %
    #      (pid, pid_tree, len(inference_exact), inference_outer, distance_cardinal), flush=True)
    # if distance_cardinal < 0:
    #    raise Exception("Not possible %s, %s" % (inference_exact, inference_outer))
    return 1


def computing_outer_vs_exact_ranking_random_tree(out_path,
                                                 nb_labels=3,
                                                 nb_repeats=100,
                                                 nb_process=1,
                                                 seed=None,
                                                 min_epsilon_param=0.05,
                                                 max_epsilon_param=0.50,
                                                 step_epsilon_param=0.05):
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


sys_nb_labels = 3
sys_out_path = "results_labels" + str(sys_nb_labels) + ".csv"
computing_outer_vs_exact_ranking_random_tree(out_path=sys_out_path,
                                             nb_labels=sys_nb_labels,
                                             nb_repeats=10,
                                             nb_process=1)
