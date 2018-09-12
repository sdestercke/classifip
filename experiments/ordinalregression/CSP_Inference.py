import classifip
import random
import sys
import numpy as np
from scipy import stats
from constraint import *

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
    print("Inferece[true,predict]:", y_true, y_predicts)
    y_true = y_true.split(">")
    if y_predicts is None: return 0.0;
    distances  = []
    for y_predict in y_predicts:
        y_idx_true = np.array([], dtype=int)
        y_idx_predict = np.array([], dtype=int)
        for idx, value in enumerate(y_true):
            y_idx_true = np.append(y_idx_true, idx)
            y_idx_predict = np.append(y_idx_predict, y_predict[value])
        distances.append(sum(spearman_distance(y_idx_true, y_idx_predict)))
    k = len(y_true)
    return 1 - 6*min(distances)/(k*(k*k -1))

def completeness_measure(y_true, y_predicts):
    y_true = y_true.split(">")
    k = len(y_true)
    R = 0
    if y_predicts is not None:
        set_ranks = dict()
        for y_predict in y_predicts:
            for idx, label in enumerate(y_true):
                if label not in set_ranks:
                    set_ranks[label] = np.array([], dtype=int)
                set_ranks[label] = np.append(set_ranks[label], y_predict[label])
        for key, _ in set_ranks.items():
            R += len(np.unique(set_ranks[key]))
    return 1 - R/(k*k)

def experiment_01():
    model = classifip.models.ncclr.NCCLR()
    seed = random.randrange(sys.maxsize)
    seed_kFold = random.randrange(sys.maxsize)
    max_ncc_s_param, num_kFold, pct_testing = 10, 10, 0.2
    print("Seed generated for system is (%5d, %5d)." % (seed, seed_kFold))
    avg_accuracy = dict()
    for nb_int in range(6, 7):
        print("Number interval for discreteness %5d." % nb_int)
        dataArff = classifip.dataset.arff.ArffFile()
        dataArff.load("/Users/salmuz/Downloads/datasets/iris_dense.xarff")
        dataArff.discretize(discmet="eqfreq", numint=nb_int)
        training, testing = classifip.evaluation.train_test_split(dataArff, test_pct=pct_testing, random_seed=seed)
        avg_accuracy[str(nb_int)] = dict()
        for ncc_imprecise in range(2, max_ncc_s_param + 1):
            print("Level imprecision %5d." % ncc_imprecise)
            cv_kFold = classifip.evaluation.k_fold_cross_validation(training, num_kFold, randomise=True, random_seed=seed_kFold)
            avg_cv_correctness = 0
            avg_cv_completeness = 0
            for set_train, set_test in cv_kFold:
                model.learn(set_train)
                evaluate_pBox = model.evaluate(set_test.data, ncc_s_param=ncc_imprecise)
                avg_correctness = 0
                avg_completeness = 0
                for idx, pBox in enumerate(evaluate_pBox):
                    predicts = model.predict_CSP([pBox])
                    avg_correctness += correctness_measure(set_test.data[idx][-1], predicts[0])
                    avg_completeness += completeness_measure(set_test.data[idx][-1], predicts[0])
                avg_cv_correctness += avg_correctness/len(set_test.data)
                avg_cv_completeness += avg_completeness/len(set_test.data)
            avg_accuracy[str(nb_int)][str(ncc_imprecise)] = \
                dict({'corr': avg_cv_correctness/num_kFold, 'comp': avg_cv_completeness/num_kFold})
            print("avg imprecision", ncc_imprecise, avg_cv_correctness)
    print("Results:", avg_accuracy)

experiment_01()