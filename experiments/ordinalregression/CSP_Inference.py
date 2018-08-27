import classifip
import random
import sys
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

def measure_classifier_preference(y_true, y_predict):
    y_true = y_true.split(">")
    if y_predict is None: return 0.0;
    y_idx_true = []
    y_idx_predict = []
    for idx, value in enumerate(y_true):
        y_idx_true.append(idx)
        y_idx_predict.append(y_predict[value])
    tau, _ = stats.kendalltau(y_idx_predict, y_idx_true)
    # print(tau, y_idx_true, y_idx_predict)
    return tau

def experiment_01():
    model = classifip.models.ncclr.NCCLR()
    dataArff= classifip.dataset.arff.ArffFile()
    dataArff.load("/Users/salmuz/Downloads/datasets/iris_dense.xarff")
    seed = random.randrange(sys.maxsize)
    seed_kFold = random.randrange(sys.maxsize)
    max_ncc_s_param, num_kFold, pct_testing = 10, 10, 0.4
    print("Seed generated for system is (%5d, %5d)." % (seed, seed_kFold))
    avg_accuracy = dict()
    for nb_int in range(5, 11):
        print("Number interval for discreteness %5d." % nb_int)
        dataArff.discretize(discmet="eqfreq", numint=nb_int)
        training, testing = classifip.evaluation.train_test_split(dataArff, test_pct=pct_testing, random_seed=seed)
        avg_accuracy[str(nb_int)] = dict()
        for ncc_imprecise in range(2, max_ncc_s_param + 1):
            print("Level imprecision %5d." % ncc_imprecise)
            cv_kFold = classifip.evaluation.k_fold_cross_validation(training, num_kFold, randomise=True, random_seed=seed_kFold)
            avg_cv = 0
            for set_train, set_test in cv_kFold:
                model.learn(set_train)
                evaluate_pBox = model.evaluate(set_test.data, ncc_s_param=ncc_imprecise)
                avg_test = 0
                for idx, pBox in enumerate(evaluate_pBox):
                    solution = model.predict_CSP([pBox])
                    avg_test += measure_classifier_preference(set_test.data[idx][-1], solution[0])
                avg_cv += avg_test/len(evaluate_pBox)
            avg_accuracy[str(nb_int)][str(ncc_imprecise)] = avg_cv
            print("avg imprecision", ncc_imprecise, avg_cv)
    print("Results:", avg_accuracy)

experiment_01()