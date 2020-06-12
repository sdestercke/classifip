from classifip.evaluation import train_test_split, k_fold_cross_validation
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
from classifip.dataset import arff
from mlc_manager import ManagerWorkers
from mlc_common import incorrectness_completeness_measure, get_nb_labels_class, normalize, CONST_PARTIAL_VALUE
import sys, os, random, csv, numpy as np
from classifip.models.mlc.chainncc import IMLCStrategy


def transform_maximin_imprecise_to_precise(y_partial_binary, set_probabilities):
    y_maximin_precise = list()
    for y_index, y_binary in enumerate(y_partial_binary):
        if y_binary == CONST_PARTIAL_VALUE:
            p_lower, p_upper = set_probabilities.scores[y_index, :]
            if p_lower > (1 - p_upper):
                y_maximin_precise.append(1)
            else:
                y_maximin_precise.append(0)
        else:
            y_maximin_precise.append(y_binary)
    return y_maximin_precise


def computing_training_testing_step(learn_data_set,
                                    test_data_set,
                                    nb_labels,
                                    ncc_imprecise,
                                    manager,
                                    strategy_chaining,
                                    ich, cph, acc, acc_trans):
    # Send training data model to every parallel process
    manager.addNewTraining(learn_data_set=learn_data_set, nb_labels=nb_labels)

    # Send testing data to every parallel process
    for test in test_data_set.data:
        manager.addTask(
            {'kwargs': {'test_dataset': [test[:-nb_labels]],
                        'ncc_epsilon': 0.001,
                        'ncc_s_param': ncc_imprecise,
                        'type_strategy': strategy_chaining,
                        'has_set_probabilities': True},
             'y_test': test[-nb_labels:]})
    manager.poisonPillWorkers()
    manager.joinTraining()  # wait all process for computing results
    nb_tests = len(test_data_set.data)
    # Recovery all inference data of all parallel process
    shared_results = manager.getResults()
    for prediction in shared_results:
        y_prediction, set_probabilities = prediction['prediction']
        y_challenger, _ = prediction['challenger']  # precise prediction (in this case)
        y_true = np.array(prediction['ground_truth'], dtype=np.int)
        y_trans_precise = transform_maximin_imprecise_to_precise(y_prediction[0], set_probabilities[0])
        inc_ich, inc_cph = incorrectness_completeness_measure(y_true, y_prediction[0])
        acc_ich_trans, _ = incorrectness_completeness_measure(y_true, y_trans_precise)
        acc_ich, _ = incorrectness_completeness_measure(y_true, y_challenger[0])
        # print("---> ich", inc_ich, acc_ich, acc_ich_trans)
        ich += inc_ich / nb_tests
        cph += inc_cph / nb_tests
        acc += acc_ich / nb_tests
        acc_trans += acc_ich_trans / nb_tests
    manager.restartResults()
    return ich, cph, acc, acc_trans


def experiments_chaining_imprecise(in_path=None,
                                   out_path=None,
                                   seed=None,
                                   nb_kFold=10,
                                   nb_process=1,
                                   min_ncc_s_param=0.5,
                                   max_ncc_s_param=6.0,
                                   step_ncc_s_param=1.0,
                                   remove_features=None,
                                   scaling=True,
                                   strategy_chaining=IMLCStrategy.IMPRECISE_BRANCHING):
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
    manager.executeAsync(class_model="classifip.models.mlc.chainncc.MLChaining")

    ich, cph, acc, acc_trans = dict(), dict(), dict(), dict()
    min_discretize, max_discretize = 5, 7
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
            ich[disc], cph[disc] = dict(), dict()
            acc_trans[disc], acc[disc] = dict(), dict()
            for s_ncc in np.arange(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
                ks_ncc = str(s_ncc)
                ich[disc][ks_ncc], cph[disc][ks_ncc] = 0, 0
                acc[disc][ks_ncc], acc_trans[disc][ks_ncc] = 0, 0
                for idx_fold, (training, testing) in enumerate(splits_s):
                    res = computing_training_testing_step(training,
                                                          testing,
                                                          nb_labels,
                                                          s_ncc,
                                                          manager,
                                                          strategy_chaining,
                                                          ich[disc][ks_ncc],
                                                          cph[disc][ks_ncc],
                                                          acc[disc][ks_ncc],
                                                          acc_trans[disc][ks_ncc])
                    ich[disc][ks_ncc], cph[disc][ks_ncc] = res[0], res[1]
                    acc[disc][ks_ncc], acc_trans[disc][ks_ncc] = res[2], res[3]
                    logger.debug("Partial-step-cumulative (acc, ich, acc_trans) (%s, %s, %s)",
                                 acc[disc][ks_ncc], ich[disc][ks_ncc], acc_trans[disc][ks_ncc])
                writer.writerow([str(nb_disc), s_ncc, time,
                                 ich[disc][ks_ncc] / nb_kFold,
                                 cph[disc][ks_ncc] / nb_kFold,
                                 acc[disc][ks_ncc] / nb_kFold,
                                 acc_trans[disc][ks_ncc] / nb_kFold])
                file_csv.flush()
                logger.debug("Partial-s-k_step (%s, %s, %s, %s, %s, %s)",
                             disc, s_ncc, time,
                             ich[disc][ks_ncc] / nb_kFold,
                             cph[disc][ks_ncc] / nb_kFold,
                             acc_trans[disc][ks_ncc] / nb_kFold)
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s, %s", ich, cph)


# np.set_printoptions(suppress=True)
in_path = ".../datasets_mlc/emotions.arff"
out_path = ".../results_iris.csv"
# QPBB_PATH_SERVER = []  # executed in host
experiments_chaining_imprecise(in_path=in_path,
                               out_path=out_path,
                               nb_process=1,
                               min_ncc_s_param=0.1, max_ncc_s_param=5.7, step_ncc_s_param=0.5,
                               strategy_chaining=IMLCStrategy.TERNARY_IMPRECISE_TREE,
                               remove_features=["image_name"], )
