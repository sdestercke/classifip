from classifip.evaluation import k_fold_cross_validation
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
from classifip.dataset import arff
from mlc_manager import ManagerWorkers
from mlc_common import setaccuracy_completeness_measure, get_nb_labels_class, normalize, CONST_PARTIAL_VALUE
import sys, os, random, csv, numpy as np
from classifip.models.mlc.chainncc import IMLCStrategy
from classifip.models.mlc.mlcncc import MLCNCC


def transform_maximin_imprecise_to_precise(y_partial_binary, set_probabilities):
    y_maximin_precise = list()
    for y_index, y_binary in enumerate(y_partial_binary):
        if y_binary == CONST_PARTIAL_VALUE:
            p_lower, p_upper = set_probabilities[y_index, :]
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
                                    safety_chaining,
                                    missing_pct,
                                    noise_label_pct,
                                    noise_label_type,
                                    noise_label_prob,
                                    ich, cph, acc, acc_trans,
                                    avg_solutions):
    # Send training data model to every parallel process
    manager.addNewTraining(learn_data_set=learn_data_set,
                           nb_labels=nb_labels,
                           missing_pct=missing_pct,
                           noise_label_pct=noise_label_pct,
                           noise_label_type=noise_label_type,
                           noise_label_prob=noise_label_prob)

    # Send testing data to every parallel process
    for test in test_data_set.data:
        manager.addTask(
            {'kwargs': {'test_dataset': [test[:-nb_labels]],
                        'ncc_epsilon': 0.001,
                        'ncc_s_param': ncc_imprecise,
                        'type_strategy': strategy_chaining,
                        'has_set_probabilities': True,
                        'is_dynamic_context': safety_chaining,
                        'with_imprecise_marginal': False},
             'y_test': test[-nb_labels:]})
    manager.poisonPillWorkers()
    manager.joinTraining()  # wait all process for computing results
    nb_tests = len(test_data_set.data)
    # Recovery all inference data of all parallel process
    shared_results = manager.getResults()
    for prediction in shared_results:
        y_prediction, set_probabilities = prediction['prediction']
        y_challenger, set_challenger = prediction['challenger']  # precise prediction (in this case)
        y_true = np.array(prediction['ground_truth'], dtype=np.int)
        y_trans_precise = transform_maximin_imprecise_to_precise(y_prediction[0], set_probabilities[0])
        nb_set_valued_solutions = 2 ** y_prediction[0].count(-1)
        inc_ich, inc_cph = setaccuracy_completeness_measure(y_true, y_prediction[0])
        acc_ich_trans, _ = setaccuracy_completeness_measure(y_true, y_trans_precise)
        acc_ich, _ = setaccuracy_completeness_measure(y_true, y_challenger[0])
        ich += inc_ich / nb_tests
        cph += inc_cph / nb_tests
        acc += acc_ich / nb_tests
        acc_trans += acc_ich_trans / nb_tests
        avg_solutions += (nb_set_valued_solutions / (2 ** nb_labels)) / nb_tests
    manager.restartResults()
    return ich, cph, acc, acc_trans, avg_solutions


def experiments_chaining_imprecise(in_path=None,
                                   out_path=None,
                                   seed=None,
                                   nb_kFold=10,
                                   nb_process=1,
                                   min_ncc_s_param=0.5,
                                   max_ncc_s_param=6.0,
                                   step_ncc_s_param=1.0,
                                   missing_pct=0.0,
                                   noise_label_pct=0.0,
                                   noise_label_type=-1,
                                   noise_label_prob=0.5,
                                   remove_features=None,
                                   scaling=False,
                                   strategy_chaining=IMLCStrategy.IMPRECISE_BRANCHING,
                                   safety_chaining=False):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean", True)
    logger.info('Training dataset (%s, %s)', in_path, out_path)
    logger.info("(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param) (%s, %s, %s)",
                min_ncc_s_param, max_ncc_s_param, step_ncc_s_param)
    logger.info("(scaling, remove_features, process) (%s, %s, %s)",
                scaling, remove_features, nb_process)
    logger.info("(missing_pct, noise_label_pct, noise_label_type, noise_label_prob) (%s, %s, %s, %s)",
                missing_pct, noise_label_pct, noise_label_type, noise_label_prob)
    logger.info("(strategy_chaining, safety_chaining) (%s, %s)",
                strategy_chaining, safety_chaining)

    # Seeding a random value for k-fold top learning-testing data
    if seed is None:
        seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process)
    manager.executeAsync(class_model="classifip.models.mlc.chainncc.MLChaining")

    ich, cph, acc, acc_trans, avg_sols = dict(), dict(), dict(), dict(), dict()
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
                train_clone_data = training.make_clone()
                test_clone_data = testing.make_clone()
                MLCNCC.shuffle_labels_train_testing(train_clone_data,
                                                    test_clone_data,
                                                    nb_labels=nb_labels)
                logger.info("Splits %s train %s", len(training.data), training.data[0])
                logger.info("Splits %s test %s", len(testing.data), testing.data[0])
                splits_s.append((train_clone_data, test_clone_data))

            disc = str(nb_disc) + "-" + str(time)
            ich[disc], cph[disc] = dict(), dict()
            acc_trans[disc], acc[disc] = dict(), dict()
            avg_sols[disc] = dict()
            for s_ncc in np.arange(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
                ks_ncc = str(s_ncc)
                ich[disc][ks_ncc], cph[disc][ks_ncc] = 0, 0
                acc[disc][ks_ncc], acc_trans[disc][ks_ncc] = 0, 0
                avg_sols[disc][ks_ncc] = 0
                for idx_fold, (training, testing) in enumerate(splits_s):
                    res = computing_training_testing_step(training,
                                                          testing,
                                                          nb_labels,
                                                          s_ncc,
                                                          manager,
                                                          strategy_chaining,
                                                          safety_chaining,
                                                          missing_pct,
                                                          noise_label_pct,
                                                          noise_label_type,
                                                          noise_label_prob,
                                                          ich[disc][ks_ncc],
                                                          cph[disc][ks_ncc],
                                                          acc[disc][ks_ncc],
                                                          acc_trans[disc][ks_ncc],
                                                          avg_sols[disc][ks_ncc])
                    ich[disc][ks_ncc], cph[disc][ks_ncc] = res[0], res[1]
                    acc[disc][ks_ncc], acc_trans[disc][ks_ncc] = res[2], res[3]
                    avg_sols[disc][ks_ncc] = res[4]
                    logger.debug("Partial-step-cumulative (acc, ich, acc_trans, avg_sols) (%s, %s, %s, %s)",
                                 acc[disc][ks_ncc],
                                 ich[disc][ks_ncc],
                                 acc_trans[disc][ks_ncc],
                                 avg_sols[disc][ks_ncc])
                writer.writerow([str(nb_disc), s_ncc, time,
                                 ich[disc][ks_ncc] / nb_kFold,
                                 cph[disc][ks_ncc] / nb_kFold,
                                 acc[disc][ks_ncc] / nb_kFold,
                                 acc_trans[disc][ks_ncc] / nb_kFold,
                                 avg_sols[disc][ks_ncc] / nb_kFold])
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
out_path = ".../results_emotions.csv"
experiments_chaining_imprecise(in_path=in_path,
                               out_path=out_path,
                               scaling=False,
                               nb_process=1,
                               min_ncc_s_param=0.5, max_ncc_s_param=5.6, step_ncc_s_param=0.5,
                               missing_pct=0.0,
                               noise_label_pct=0.0, noise_label_type=-1, noise_label_prob=0.8,
                               strategy_chaining=IMLCStrategy.IMPRECISE_BRANCHING,
                               safety_chaining=False,
                               remove_features=["image_name"])
