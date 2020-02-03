from classifip.evaluation import k_fold_cross_validation
from classifip.utils import create_logger
from classifip.dataset import arff
from classifip.models import nccbr
from classifip.models.mlcncc import MLCNCCExact
import math, os, random, sys, csv, numpy as np


def get_nb_labels_class(dataArff, type_class='nominal'):
    nominal_class = [item for item in dataArff.attribute_types.values() if item == type_class]
    return len(nominal_class)


def distance_cardinal_set_inferences(inference_outer, inference_exact, nb_labels):
    power_outer, power_exact = 0, 0
    for j in range(nb_labels):
        # if isinstance(inference_outer[j], list):
        if inference_outer[j] == -1:
            power_outer += 1
        if isinstance(inference_exact[j], list):
            power_exact += 1
    return abs(math.pow(2, power_exact) - math.pow(2, power_outer))


def computing_outer_vs_exact_inference(in_path=None, out_path=None, seed=None, nb_kFold=10,
                                       intvl_ncc_s_param=None, step_ncc_s=1.0):
    intvl_ncc_s_param = list([2, 5]) if intvl_ncc_s_param is None else intvl_ncc_s_param
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean_cv", True)
    logger.info('Training dataset (%s, %s)', in_path, out_path)

    # Seeding a random value for k-fold top learning-testing data
    if seed is not None:
        random.seed(seed)
    seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)

    # Create the models
    model_br = nccbr.NCCBR()
    model_exact = MLCNCCExact()

    diff_inferences = dict()
    min_discretize, max_discretize = 5, 6
    for nb_disc in range(min_discretize, max_discretize):
        data_learning = arff.ArffFile()
        data_learning.load(in_path)
        nb_labels = get_nb_labels_class(data_learning)
        data_learning.discretize(discmet="eqfreq", numint=nb_disc)

        for time in range(nb_kFold):  # 10-10 times cross-validation
            logger.info("Number interval for discreteness and labels (%1d, %1d)." % (nb_disc, nb_labels))
            cv_kfold = k_fold_cross_validation(data_learning, nb_kFold, randomise=True, random_seed=seed[time])

            splits_s = list([])
            for training, testing in cv_kfold:
                splits_s.append((training, testing))

            disc = str(nb_disc) + "-" + str(time)
            diff_inferences[disc] = dict()
            for s_ncc in np.arange(intvl_ncc_s_param[0], intvl_ncc_s_param[1], step_ncc_s):
                diff_inferences[disc][str(s_ncc)] = 0
                for idx_fold, (training, testing) in enumerate(splits_s):
                    model_br.learn(training, nb_labels)
                    model_exact.learn(training, nb_labels)
                    nb_testing = len(testing.data)
                    for test in testing.data:
                        set_prob_marginal = model_br.evaluate([test[:-nb_labels]], ncc_s_param=s_ncc)
                        inference_outer = set_prob_marginal[0].multilab_dom()
                        inference_exact = model_exact.evaluate([test[:-nb_labels]], ncc_s_param=s_ncc)
                        # inference_all_exact = model_exact.evaluate_exact([test[:-nb_labels]], ncc_s_param=s_ncc)
                        dist_measure = distance_cardinal_set_inferences(inference_outer, inference_exact, nb_labels)
                        # print("--->", inference_outer, inference_exact, inference_all_exact, dist_measure, flush=True)
                        diff_inferences[disc][str(s_ncc)] += dist_measure / nb_testing
                diff_inferences[disc][str(s_ncc)] = diff_inferences[disc][str(s_ncc)] / nb_kFold
                writer.writerow([str(nb_disc), s_ncc, time, diff_inferences[disc][str(s_ncc)] / nb_kFold])
                file_csv.flush()
                logger.debug("Partial-s-k_step (%s, %s, %s, %s)", disc, s_ncc, time, diff_inferences[disc][str(s_ncc)])

    logger.debug("Results Final: %s", diff_inferences)


_name = "labels3"
sys_in_path = _name + ".arff"
sys_out_path = "results_" + _name + ".csv"
# QPBB_PATH_SERVER = []  # executed in host
computing_outer_vs_exact_inference(in_path=sys_in_path, out_path=sys_out_path,
                                   intvl_ncc_s_param=[1, 5], step_ncc_s=0.5)
