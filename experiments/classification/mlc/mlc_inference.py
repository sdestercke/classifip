from classifip.evaluation import train_test_split, k_fold_cross_validation
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
from classifip.dataset import arff
from multitask import ManagerWorkers
import sys, os, random, csv, numpy as np


def normalize(dataArff, n_labels, method='minimax'):
    from classifip.utils import normalize_minmax
    np_data = np.array(dataArff.data, dtype=float)
    np_data = np_data[..., :-n_labels]
    if method == "minimax":
        np_data = normalize_minmax(np_data)
        dataArff.data = [np_data[i].tolist() + dataArff.data[i][-n_labels:] for i, d in enumerate(np_data)]
    else:
        raise Exception("Not found method implemented yet.")


def get_nb_labels_class(dataArff, type_class='nominal'):
    nominal_class = [item for item in dataArff.attribute_types.values() if item == type_class]
    return len(nominal_class)


def incorrectness_completeness_measure(y_true, y_prediction):
    Q, m, hamming = [], len(y_true), 0
    for i, y in enumerate(y_true):
        if len(y_prediction[i]) == 1:
            Q.append(y_prediction[i])
            if y_prediction[i][0] != y:
                hamming += 1
    lenQ = len(Q)
    if lenQ != 0:
        return hamming / lenQ, lenQ / m
    else:
        return 0, 0


def computing_training_testing_step(learn_data_set, test_data_set, nb_labels, ncc_imprecise, manager, ich, cph):
    # Send training data model to every parallel process
    manager.addNewTraining(learn_data_set=learn_data_set, nb_labels=nb_labels)

    # Send testing data to every parallel process
    for test in test_data_set.data:
        manager.addTask(
            {'kwargs': {'testdataset': [test[:-nb_labels]], 'ncc_epsilon': 0.001, 'ncc_s_param': ncc_imprecise},
             'y_test': test[-nb_labels:]})
    manager.poisonPillWorkers()
    manager.joinTraining()  # wait all process for computing results
    nb_tests = len(test_data_set.data)
    # Recovery all inference data of all parallel process
    shared_results = manager.getResults()
    shared_results.put('STOP')  ## stop loop queue
    for utility in iter(shared_results.get, 'STOP'):
        _, y_prediction = utility['prediction']
        y_true = utility['ground_truth']
        inc_ich, inc_cph = incorrectness_completeness_measure(y_true, y_prediction[0])
        ich = ich + inc_ich / nb_tests
        cph = cph + inc_cph / nb_tests
    return ich, cph


def computing_best_imprecise_mean(in_path=None, out_path=None, seed=None, nb_kFold=10,
                                  nb_process=1, scaling=True, max_ncc_s_param=5, remove_features=None):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean_cv", True)
    logger.info('Training dataset (%s, %s)', in_path, out_path)

    # Seeding a random value for k-fold top learning-testing data
    if seed is not None: random.seed(seed)
    seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process)
    manager.executeAsync(class_model="classifip.models.mlcncc.MLCNCC")

    ich, cph = dict(), dict()
    min_discretize, max_discretize = 5, 9
    for nb_disc in range(min_discretize, max_discretize):
        data_learning = arff.ArffFile()
        data_learning.load(in_path)
        if remove_features is not None:
            for r_feature in remove_features: data_learning.remove_col(r_feature)
        nb_labels = get_nb_labels_class(data_learning)
        if scaling: normalize(data_learning, n_labels=nb_labels)
        data_learning.discretize(discmet="eqfreq", numint=nb_disc)

        for time in range(nb_kFold):  # 10-10 times cross-validation
            logger.info("Number interval for discreteness and labels (%1d, %1d)." % (nb_disc, nb_labels))
            cv_kfold = k_fold_cross_validation(data_learning, nb_kFold, randomise=True, random_seed=seed[time])

            splits_s = list([])
            for training, testing in cv_kfold:
                splits_s.append((training, testing))
                logger.info("Splits %s train %s", len(training.data), training.data[0])
                logger.info("Splits %s test %s", len(testing.data), testing.data[0])

            disc = str(nb_disc) + "-" + str(time)
            ich[disc], cph[disc] = dict(), dict()
            for s_ncc in np.arange(0.1, max_ncc_s_param + 1, 1):
                ks_ncc = str(s_ncc)
                ich[disc][ks_ncc], cph[disc][ks_ncc] = 0, 0
                for idx_fold, (training, testing) in enumerate(splits_s):
                    ich[disc][ks_ncc], cph[disc][ks_ncc] = computing_training_testing_step(training, testing,
                                                                                           nb_labels, s_ncc, manager,
                                                                                           ich[disc][ks_ncc],
                                                                                           cph[disc][ks_ncc])

                writer.writerow([str(nb_disc), s_ncc, time, ich[disc][ks_ncc] / nb_kFold, cph[disc][ks_ncc] / nb_kFold])
                file_csv.flush()
                logger.debug("Partial-s-k_step (%s, %s, %s, %s, %s)", disc, s_ncc, time, ich[disc][ks_ncc] / nb_kFold,
                             cph[disc][ks_ncc] / nb_kFold)
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s, %s", ich, cph)


in_path = "/Users/salmuz/Downloads/datasets_mlc/nuswide-cVLADplus.arff"
out_path = "/Users/salmuz/Downloads/results_iris.csv"
# QPBB_PATH_SERVER = []  # executed in host
computing_best_imprecise_mean(in_path=in_path, out_path=out_path, nb_process=1)
