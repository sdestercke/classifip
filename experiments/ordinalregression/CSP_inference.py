import classifip
import random
import sys
import numpy as np
from classifip.evaluation.measures import correctness_measure, completeness_measure


def experiment_01():
    model = classifip.models.ncclr.NCCLR()
    seed = random.randrange(sys.maxsize)
    seed_kFold = random.randrange(sys.maxsize)
    max_ncc_s_param, num_kFold, pct_testing = 10, 10, 0.1
    print("Seed generated for system is (%5d, %5d)." % (seed, seed_kFold), flush=True)
    min_disc, max_disc = 5, 9
    avg_accuracy = dict()
    for nb_int in range(min_disc, max_disc):
        print("Number interval for discreteness %5d." % nb_int, flush=True)
        dataArff = classifip.dataset.arff.ArffFile()
        dataArff.load("/Users/salmuz/Downloads/datasets_rang/iris_dense.xarff")
        dataArff.discretize(discmet="eqfreq", numint=nb_int)
        training, testing = classifip.evaluation.train_test_split(dataArff, test_pct=pct_testing, random_seed=seed)
        avg_accuracy[str(nb_int)] = dict()
        for ncc_imprecise in np.arange(0.1, max_ncc_s_param + 1, 1):
            print("Level imprecision %5d." % ncc_imprecise, flush=True)
            cv_kFold = classifip.evaluation.k_fold_cross_validation(training, num_kFold, randomise=True,
                                                                    random_seed=seed_kFold)
            avg_cv_correctness = 0
            avg_cv_completeness = 0
            for set_train, set_test in cv_kFold:
                model.learn(set_train)
                evaluate_pBox = model.evaluate(set_test.data, ncc_s_param=ncc_imprecise)
                avg_correctness = 0
                avg_completeness = 0
                for idx, pBox in enumerate(evaluate_pBox):
                    predicts = model.inference_CSP([pBox])
                    y_ground_truth = set_test.data[idx][-1].split(">")
                    avg_correctness += correctness_measure(y_ground_truth, predicts[0])
                    avg_completeness += completeness_measure(y_ground_truth, predicts[0])
                avg_cv_correctness += avg_correctness / len(set_test.data)
                avg_cv_completeness += avg_completeness / len(set_test.data)
            avg_accuracy[str(nb_int)][str(ncc_imprecise)] = \
                dict({'corr': avg_cv_correctness / num_kFold, 'comp': avg_cv_completeness / num_kFold})
            print("avg imprecision", ncc_imprecise, avg_cv_correctness, flush=True)
        print("Partial results:", avg_accuracy, flush=True)
    print("Results:", avg_accuracy)


experiment_01()
