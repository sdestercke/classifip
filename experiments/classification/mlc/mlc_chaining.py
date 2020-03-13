from classifip.evaluation import train_test_split, k_fold_cross_validation
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
from classifip.dataset import arff
from mlc_manager import ManagerWorkers
from mlc_common import incorrectness_completeness_measure
import sys, os, random, csv, numpy as np


def normalize(dataArff, n_labels, method='minimax'):
    from classifip.utils import normalize_minmax
    np_data = np.array(dataArff.data, dtype=float)
    np_data = np_data[..., :-n_labels]
    if method == "minimax":
        np_data = normalize_minmax(np_data)
        dataArff.data = [np_data[i].tolist() + dataArff.data[i][-n_labels:]
                         for i, d in enumerate(np_data)]
    else:
        raise Exception("Not found method implemented yet.")


def get_nb_labels_class(dataArff, type_class='nominal'):
    nominal_class = [item for item in dataArff.attribute_types.values() if item == type_class]
    return len(nominal_class)


def computing_training_testing_step(learn_data_set,
                                    test_data_set,
                                    nb_labels,
                                    ncc_imprecise,
                                    manager,
                                    ich, cph,
                                    acc):
    # Send training data model to every parallel process
    manager.addNewTraining(learn_data_set=learn_data_set, nb_labels=nb_labels)

    # Send testing data to every parallel process
    for test in test_data_set.data:
        manager.addTask(
            {'kwargs': {'test_dataset': [test[:-nb_labels]],
                        'ncc_epsilon': 0.001,
                        'ncc_s_param': ncc_imprecise},
             'y_test': test[-nb_labels:]})
    manager.poisonPillWorkers()
    manager.joinTraining()  # wait all process for computing results
    nb_tests = len(test_data_set.data)
    # Recovery all inference data of all parallel process
    shared_results = manager.getResults()
    for _ in range(manager.NUMBER_OF_PROCESSES):
        job_predictions = shared_results.pop()
        for prediction in job_predictions:
            y_prediction = prediction['prediction']
            y_precise = prediction['precise']
            y_true = np.array(prediction['ground_truth'], dtype=np.int)
            inc_ich, inc_cph = incorrectness_completeness_measure(y_true, y_prediction[0])
            acc_ich, _ = incorrectness_completeness_measure(y_true, y_precise[0])
            ich += inc_ich / nb_tests
            cph += inc_cph / nb_tests
            acc += acc_ich / nb_tests
    return ich, cph, acc


def computing_best_imprecise_mean(in_path=None,
                                  out_path=None,
                                  seed=None,
                                  nb_kFold=10,
                                  nb_process=1,
                                  min_ncc_s_param=0.5,
                                  max_ncc_s_param=6.0,
                                  step_ncc_s_param=1.0,
                                  remove_features=None,
                                  scaling=True, ):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean", True)
    logger.info('Training dataset (%s, %s)', in_path, out_path)

    # Seeding a random value for k-fold top learning-testing data
    if seed is None:
        seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process)
    manager.executeAsync(class_model="classifip.models.mlc.chainncc.MLChaining",
                         class_model_challenger="classifip.models.mlc.chainncc.MLChaining")

    ich, cph, acc = dict(), dict(), dict()
    min_discretize, max_discretize = 5, 9
    for nb_disc in range(min_discretize, max_discretize):
        data_learning = arff.ArffFile()
        data_learning.load(in_path)
        if remove_features is not None:
            for r_feature in remove_features:
                try:
                    data_learning.remove_col(r_feature)
                except Exception as err:
                    print("Remove feature error: {0}".format(err))
        nb_labels = get_nb_labels_class(data_learning)
        if scaling:
            normalize(data_learning, n_labels=nb_labels)
        data_learning.discretize(discmet="eqfreq", numint=nb_disc)

        for time in range(nb_kFold):  # 10-10 times cross-validation
            logger.info("Number interval for discreteness and labels (%1d, %1d)." % (nb_disc, nb_labels))
            cv_kfold = k_fold_cross_validation(data_learning,
                                               nb_kFold,
                                               randomise=True,
                                               random_seed=seed[time])

            splits_s = list([])
            for training, testing in cv_kfold:
                splits_s.append((training.make_clone(), testing.make_clone()))
                logger.info("Splits %s train %s", len(training.data), training.data[0])
                logger.info("Splits %s test %s", len(testing.data), testing.data[0])

            disc = str(nb_disc) + "-" + str(time)
            ich[disc], cph[disc], acc[disc] = dict(), dict(), dict()
            for s_ncc in np.arange(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
                ks_ncc = str(s_ncc)
                ich[disc][ks_ncc], cph[disc][ks_ncc], acc[disc][ks_ncc] = 0, 0, 0
                for idx_fold, (training, testing) in enumerate(splits_s):
                    res = computing_training_testing_step(training,
                                                          testing,
                                                          nb_labels,
                                                          s_ncc,
                                                          manager,
                                                          ich[disc][ks_ncc],
                                                          cph[disc][ks_ncc],
                                                          acc[disc][ks_ncc])
                    ich[disc][ks_ncc], cph[disc][ks_ncc], acc[disc][ks_ncc] = res
                    logger.debug("Partial-step-cumulative (acc, ich) (%s, %s)",
                                 acc[disc][ks_ncc], ich[disc][ks_ncc])
                writer.writerow([str(nb_disc), s_ncc, time,
                                 ich[disc][ks_ncc] / nb_kFold,
                                 cph[disc][ks_ncc] / nb_kFold,
                                 acc[disc][ks_ncc] / nb_kFold])
                file_csv.flush()
                logger.debug("Partial-s-k_step (%s, %s, %s, %s, %s)", disc, s_ncc, time,
                             ich[disc][ks_ncc] / nb_kFold,
                             cph[disc][ks_ncc] / nb_kFold)
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s, %s", ich, cph)


in_path = "/Users/salmuz/Downloads/datasets_mlc/emotions.arff"
out_path = "/Users/salmuz/Downloads/results_iris.csv"
# QPBB_PATH_SERVER = []  # executed in host
computing_best_imprecise_mean(in_path=in_path,
                              out_path=out_path,
                              nb_process=1,
                              remove_features=["image_name"])
