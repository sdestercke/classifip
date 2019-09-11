import classifip
import random
import sys
import numpy as np
from constraint import *
import matplotlib.pyplot as plt
import json
import math


def testing_CSP():
    problem = Problem()
    problem.addVariable("a", [2])
    problem.addVariable("c", [1])
    problem.addVariable("b", [2])
    problem.addConstraint(AllDifferentConstraint())
    print(problem.getSolutions())


def kendall_tau(y_idx_true, y_idx_predict, M):
    C, D = 0, 0
    for idx in range(M):
        for idy in range(idx + 1, M):
            if (y_idx_true[idx] < y_idx_true[idy] and y_idx_predict[idx] < y_idx_predict[idy]) or \
                    (y_idx_true[idx] > y_idx_true[idy] and y_idx_predict[idx] > y_idx_predict[idy]):
                C += 1
            if (y_idx_true[idx] < y_idx_true[idy] and y_idx_predict[idx] > y_idx_predict[idy]) or \
                    (y_idx_true[idx] > y_idx_true[idy] and y_idx_predict[idx] < y_idx_predict[idy]):
                D += 1
    return (C-D)/(M*(M-1)/2)


def spearman_distance(y_idx_true, y_idx_predict):
    return np.power(np.array(y_idx_true) - np.array(y_idx_predict), 2)


def correctness_measure(y_true, y_predicts):
    # print("Inference [true,predict]:", y_true, y_predicts)
    y_true = y_true.split(">")
    if y_predicts is None: return 0.0;
    k = len(y_true)
    sum_dist = 0
    for idx, label in enumerate(y_true):
        min_dist = np.array([], dtype=int)
        for y_predict in y_predicts:
            min_dist = np.append(min_dist, abs(idx-y_predict[label]))
        sum_dist += np.min(min_dist)
    return 1 - sum_dist/(0.5*k*k)


def completeness_measure(y_true, y_predicts):
    y_true = y_true.split(">")
    k = len(y_true)
    R = 0
    if y_predicts is not None:
        learn_ranks = dict()
        for y_predict in y_predicts:
            for idx, label in enumerate(y_true):
                if label not in learn_ranks:
                    learn_ranks[label] = np.array([], dtype=int)
                learn_ranks[label] = np.append(learn_ranks[label], y_predict[label])
        # learning ranking models: learn_ranks
        for key, _ in learn_ranks.items():
            R += len(np.unique(learn_ranks[key]))
    return (k*k-R)/(k*k-k)


def plot_save_results(results_dict, texte):

    # save results into json file
    with open('results_%s.json' % texte, 'w') as outfile:
        json.dump(results_dict, outfile, indent=4)

    x = list()
    y = list()
    impression = list()
    for discre in results_dict.keys():
        list_s = sorted(list(results_dict[discre].keys()))
        for impre in list_s:
            x.append(results_dict[discre][impre]['comp'])
            y.append(results_dict[discre][impre]['corr'])
            impression.append(str(round(impre, 2)))

    plt.figure()
    plt.plot(x, y)
    plt.scatter(x, y)

    for text in impression:
        plt.annotate(text, (x[impression.index(text)], y[impression.index(text)]))

    plt.xlabel('Completeness')
    plt.ylabel('Correctness')
    plt.title('Discretisation with s values')
    plt.grid(True)
    plt.savefig("results_plot_%s.png"%texte)

    plt.close()


def random_floats(low, high, size):
    floats = [round(random.uniform(low, high), 2) for _ in range(size)]
    return sorted(floats)


def euclidean_distance(xa, xb, ya, yb):
    return math.sqrt(pow(xa - xb, 2) + pow(ya - yb, 2))


def get_s_values(results_dict):
    list_s = sorted(list(results_dict.keys()))
    x = [results_dict[s]['comp'] for s in list_s]
    y = [results_dict[s]['corr'] for s in list_s]
    distances = []

    for i in range(len(list_s)-1):
        distances.append(euclidean_distance(x[i], x[i+1], y[i], y[i+1]))
    indx = distances.index(max(distances))

    new_s = (list_s[indx]+list_s[indx+1])/2

    return new_s


def get_cr_cp(s, model, train, num_kFold, seed_kFold):
    cv_kFold = classifip.evaluation.k_fold_cross_validation(train, num_kFold, randomise=True,
                                                            random_seed=seed_kFold)
    avg_cv_correctness = 0
    avg_cv_completeness = 0
    for set_train, set_test in cv_kFold:
        model.learn(set_train)
        evaluate_pBox = model.evaluate(set_test.data, ncc_s_param=s)
        avg_correctness = 0
        avg_completeness = 0
        for idx, pBox in enumerate(evaluate_pBox):
            predicts = model.predict_CSP([pBox])
            avg_correctness += correctness_measure(set_test.data[idx][-1], predicts[0])
            avg_completeness += completeness_measure(set_test.data[idx][-1], predicts[0])
        avg_cv_correctness += avg_correctness / len(set_test.data)
        avg_cv_completeness += avg_completeness / len(set_test.data)
    avg_cv_correctness = avg_cv_correctness / num_kFold
    avg_cv_completeness = avg_cv_completeness / num_kFold
    return(avg_cv_correctness, avg_cv_completeness)


def find_max_dichotomy(l, r, model, training, num_kFold, seed_kFold):
    if l != r:
        mid = round((r+l)/2, 2)
    else:
        print("No mid, l value saved %s"%l, flush=True)
        l_corr, l_comp = get_cr_cp(l, model, training, num_kFold, seed_kFold)
        return l, l_corr, l_comp
    # Calculate mid corr and comp scores
    mid_corr, mid_comp = get_cr_cp(mid, model, training, num_kFold, seed_kFold)

    print(l, mid, r, mid_corr, flush=True)
    if 0.95 <= mid_corr < 1:
        print("Found", flush=True)
        print(mid, mid_corr, flush=True)
        return mid, mid_corr, mid_comp
    else:
        l_corr, l_comp = get_cr_cp(l, model, training, num_kFold, seed_kFold)
        r_corr, r_comp = get_cr_cp(r, model, training, num_kFold, seed_kFold)
        if l_corr < r_corr:
            if mid_corr < 1:
                return find_max_dichotomy(mid, r, model, training, num_kFold, seed_kFold)
            if mid_corr == 1:
                return find_max_dichotomy(l, mid, model, training, num_kFold, seed_kFold)
        else:
            if mid_corr < 1:
                return find_max_dichotomy(l, mid, model, training, num_kFold, seed_kFold)
            if mid_corr == 1:
                return find_max_dichotomy(mid, r, model, training, num_kFold, seed_kFold)


def find_min_dichotomy(l, r, model, training, num_kFold, seed_kFold):
    if l != r:
        mid = round((r+l)/2, 2)
    else:
        print("No mid, l value saved %s"%l, flush=True)
        l_corr, l_comp = get_cr_cp(l, model, training, num_kFold, seed_kFold)
        return l, l_corr, l_comp
    # Calculate mid corr and comp scores
    mid_corr, mid_comp = get_cr_cp(mid, model, training, num_kFold, seed_kFold)
    print(l, mid, r, mid_comp, flush=True)
    if 0.95 <= mid_comp < 1:
        print("Found", flush=True)
        print(mid, mid_comp, flush=True)
        return mid, mid_corr, mid_comp
    else:
        l_corr, l_comp = get_cr_cp(l, model, training, num_kFold, seed_kFold)
        r_corr, r_comp = get_cr_cp(r, model, training, num_kFold, seed_kFold)
        if l_comp < r_comp:
            if mid_comp < 1:
                return find_min_dichotomy(mid, r, model, training, num_kFold, seed_kFold)
            if mid_comp >= 1:
                return find_min_dichotomy(l, mid, model, training, num_kFold, seed_kFold)
        else :
            if mid_comp < 1:
                return find_min_dichotomy(l, mid, model, training, num_kFold, seed_kFold)
            if mid_comp >= 1:
                return find_min_dichotomy(mid, r, model, training, num_kFold, seed_kFold)


def find_min(model, training, num_kFold, seed_kFold):
    print("Searching for min S value", flush=True)
    s = 2
    s_prev = 0
    l = 0
    r = 0
    while r == 0:
        s_corr, s_comp = get_cr_cp(s, model, training, num_kFold, seed_kFold)
        print(s, s_prev, s_comp, flush=True)
        if 0.95 <= s_comp < 1:
            print("Found without dichotomy")
            print(s, s_comp, flush=True)
            return s, s_corr, s_comp
        else:
            if s_comp >= 1:
                if s_prev == s*2:
                    r = s*2
                    l = s
                else:
                    s_prev = s
                    s = s*2
            else:
                if s_prev == s/2:
                    l = s/2
                    r = s
                else:
                    s_prev = s
                    s = s/2
    print("Now dichotomy", flush=True)
    return find_min_dichotomy(l, r, model, training, num_kFold, seed_kFold)


def find_max(s, model, training, num_kFold, seed_kFold):
    print("Searching for max S value", flush=True)
    s_orig = s
    s_prev = 0
    l = 0
    r = 0
    while r == 0:
        s_corr, s_comp = get_cr_cp(s, model, training, num_kFold, seed_kFold)
        print(s, s_prev, s_corr, flush=True)
        if 0.95 <= s_corr < 1 and s_orig != s:
            print("Found without dichotomy")
            print(s, s_corr, flush=True)
            return s, s_corr, s_comp
        else:
            if s_corr < 1:
                if s_prev == s*2:
                    r = s*2
                    l = s
                else:
                    s_prev = s
                    s = s*2
            else:
                if s_prev == s/2:
                    l = s/2
                    r = s
                else:
                    s_prev = s
                    s = s/2
    print("Now dichotomy", flush=True)
    return find_max_dichotomy(round(l, 2), round(r, 2), model, training, num_kFold, seed_kFold)


def experiment_03():
    random.seed(1234)
    model = classifip.models.ncclr.NCCLR()
    seed = random.randrange(sys.maxsize)
    seed_kFold = random.randrange(sys.maxsize)
    max_ncc_s_param, num_kFold, pct_testing = 5, 20, 0.2
    print("Seed generated for system is (%5d, %5d)." % (seed, seed_kFold), flush=True)
    min_disc, max_disc = 5,  6
    avg_accuracy = dict()

    for nb_int in range(min_disc, max_disc):
        print("Number interval for discreteness %5d." % nb_int, flush=True)
        dataArff = classifip.dataset.arff.ArffFile()
        dataArff.load("/home/lab/smessoud/classifip/datasets_rang/wisconsin_dense.xarff")
        dataset_name = "euclidean_wisconsin"
        # dataArff.load("D:/Users/smessoud/Downloads/datasets_rang/fried_dense.xarff")
        dataArff.discretize(discmet="eqfreq", numint=nb_int)
        training, testing = classifip.evaluation.train_test_split(dataArff, test_pct=pct_testing, random_seed=seed)

        avg_accuracy[str(nb_int)] = dict()

        # find min and max
        min_s, min_s_corr, min_s_comp = find_min(model, training, num_kFold, seed_kFold)
        max_s, max_s_corr, max_s_comp = find_max(min_s, model, training, num_kFold, seed_kFold)
        moy_s = round((min_s + max_s)/2, 2)

        # init values into accuracy dict
        avg_accuracy[str(nb_int)][(min_s)] = \
            dict({'corr': min_s_corr, 'comp': min_s_comp})

        avg_accuracy[str(nb_int)][(max_s)] = \
            dict({'corr': max_s_corr, 'comp': max_s_comp})

        moy_s_corr, moy_s_comp = get_cr_cp(moy_s, model, training, num_kFold, seed_kFold)
        avg_accuracy[str(nb_int)][(moy_s)] = \
            dict({'corr': moy_s_corr, 'comp': moy_s_comp})

        n = 3
        for i in range(n):
            new_s = get_s_values(avg_accuracy[str(nb_int)])
            avg_cv_correctness, avg_cv_completeness = get_cr_cp(new_s, model, training, num_kFold, seed_kFold)
            avg_accuracy[str(nb_int)][new_s] = \
                dict({'corr': avg_cv_correctness, 'comp': avg_cv_completeness})

    print("Results:", avg_accuracy, flush=True)

    # plot results
    plot_save_results(avg_accuracy, dataset_name + "_" + str(n+3))


experiment_03()
