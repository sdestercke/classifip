from classifip.evaluation import train_test_split, k_fold_cross_validation
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
from classifip.dataset import arff
from classifip.models.mlc import nccbr
from classifip.models.mlc.mlcncc import MLCNCC
from mlc_manager import ManagerWorkers, __create_dynamic_class
import sys, os, random, csv, numpy as np
from mlc_common import *
from itertools import product


def transform_semi_partial_vector(full_binary_vector):
    """
    Semi-partial binary vector is like:
            Y = [(0, 1, 1, 0, 0, 0), (0, 1, 1, 0, 1, 0), (0, 1, 1, 1, 1, 0)]
    Partial binary vector is like:
            Y = [(0, 1, 1, 0, 1, 0), (0, 1, 1, 1, 1, 0)] = [(0, 1, 1, *, 1, 0)]
    :param full_binary_vector:
    :return:
    """
    full_binary_vector = np.array(full_binary_vector)
    _, nb_labels = full_binary_vector.shape
    result = np.zeros(nb_labels, dtype=np.int)
    for idx_label in range(nb_labels):
        label_value = np.unique(full_binary_vector[:, idx_label])
        if len(label_value) > 1:
            result[idx_label] = CONST_PARTIAL_VALUE
        else:
            result[idx_label] = label_value[0]
    return result


def expansion_partial_to_full_set_binary_vector(partial_binary_vector):
    new_set_binary_vector = list()
    for label in partial_binary_vector:
        if label == CONST_PARTIAL_VALUE:
            new_set_binary_vector.append([1, 0])
        else:
            new_set_binary_vector.append([label])

    set_binary_vector = list(product(*new_set_binary_vector))
    return set_binary_vector


def skeptical_prediction(pid, tasks, queue, results, class_model, class_model_challenger=None):
    try:
        model_br = nccbr.NCCBR()
        model_exact = __create_dynamic_class(class_model)
        while True:
            training = queue.get()
            if training is None:
                break
            MLCNCC.missing_labels_learn_data_set(learn_data_set=training["learn_data_set"],
                                                 nb_labels=training["nb_labels"],
                                                 missing_pct=training["missing_pct"])
            MLCNCC.noise_labels_learn_data_set(learn_data_set=training["learn_data_set"],
                                               nb_labels=training["nb_labels"],
                                               noise_label_pct=training["noise_label_pct"],
                                               noise_label_type=training["noise_label_type"],
                                               noise_label_prob=training["noise_label_prob"])
            # remove some keys of dict unused for learn method
            del training['noise_label_pct']
            del training['noise_label_type']
            del training['noise_label_prob']
            model_br.learn(**training)
            model_exact.learn(**training)
            while True:
                task = tasks.get()
                if task is None:
                    break
                set_prob_marginal = model_br.evaluate(**task['kwargs'])
                outer_inference = set_prob_marginal[0].multilab_dom()
                if task['do_inference_exact']:
                    skeptical_inference = model_exact.evaluate(**task['kwargs'])[0]
                    # skeptical_inference_exact = model_exact.evaluate_exact(**task['kwargs'])[0]
                else:
                    skeptical_inference = [-1]*len(task['y_test'])
                task['kwargs']['ncc_s_param'] = 0.0
                set_prob_marginal = model_br.evaluate(**task['kwargs'])
                precise_inference = set_prob_marginal[0].multilab_dom()
                print("(pid, skeptical, outer, precise, ground-truth) ",
                      pid, len(skeptical_inference), outer_inference, precise_inference,
                      task['y_test'], flush=True)
                results.append(dict({'skeptical': skeptical_inference,
                                     'outer': outer_inference,
                                     'precise': precise_inference,
                                     'ground_truth': task['y_test']}))
            queue.task_done()
    except Exception as e:
        raise Exception(e, "Error in job of PID " + str(pid))
    finally:
        print("Worker PID finished", pid, flush=True)


def computing_training_testing_step(learn_data_set,
                                    test_data_set,
                                    missing_pct,
                                    noise_label_pct,
                                    noise_label_type,
                                    noise_label_prob,
                                    nb_labels,
                                    ncc_imprecise,
                                    manager,
                                    do_inference_exact,
                                    ich_skep, cph_skep,
                                    ich_out, cph_out,
                                    acc_prec, jacc_skep):
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
                        'ncc_s_param': ncc_imprecise},
             'y_test': test[-nb_labels:],
             'do_inference_exact': do_inference_exact})
    manager.poisonPillWorkers()
    manager.joinTraining()  # wait all process for computing results
    # Recovery all inference data of all parallel process
    nb_tests = len(test_data_set.data)
    shared_results = manager.getResults()
    for prediction in shared_results:
        y_true = np.array(prediction['ground_truth'], dtype=np.int)
        y_skeptical_exact = prediction['skeptical']
        y_outer = prediction['outer']
        y_precise = prediction['precise']
        # if enable to do the exact skeptical inference
        if do_inference_exact:
            y_skeptical_exact_partial = transform_semi_partial_vector(y_skeptical_exact)
            inc_ich_skep, inc_cph_skep = incorrectness_completeness_measure(y_true, y_skeptical_exact_partial)
            y_outer_full_set = expansion_partial_to_full_set_binary_vector(y_outer)
            inc_jacc = compute_jaccard_similarity_score(y_outer_full_set, y_skeptical_exact)
        else:
            inc_jacc = -1
            inc_ich_skep, inc_cph_skep = -1, -1
        inc_ich_out, inc_cph_out = incorrectness_completeness_measure(y_true, y_outer)
        inc_acc_prec, _ = incorrectness_completeness_measure(y_true, y_precise)
        # print("(outer, precise, y_outer, y_precise, y_true)  (%s, %s, %s, %s, %s)" %
        #       (round(inc_ich_out, 3), round(inc_acc_prec, 3), y_outer, y_precise, y_true), flush=True)
        ich_skep += inc_ich_skep / nb_tests
        cph_skep += inc_cph_skep / nb_tests
        ich_out += inc_ich_out / nb_tests
        cph_out += inc_cph_out / nb_tests
        acc_prec += inc_acc_prec / nb_tests
        jacc_skep += inc_jacc / nb_tests
    manager.restartResults()
    return ich_skep, cph_skep, ich_out, cph_out, acc_prec, jacc_skep


def computing_best_imprecise_mean(in_path=None,
                                  out_path=None,
                                  seed=None,
                                  missing_pct=0.0,
                                  noise_label_pct=0.0,
                                  noise_label_type=-1,
                                  noise_label_prob=0.5,
                                  nb_kFold=10,
                                  nb_process=1,
                                  scaling=True,
                                  min_ncc_s_param=0.5,
                                  max_ncc_s_param=6.0,
                                  step_ncc_s_param=1.0,
                                  remove_features=None,
                                  do_inference_exact=False):
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

    # Seeding a random value for k-fold top learning-testing data
    if seed is None:
        seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process, fun_prediction=skeptical_prediction)
    manager.executeAsync(class_model="classifip.models.mlc.exactncc.MLCNCCExact")

    ich_skep, cph_skep, jacc_skep, ich_out, cph_out, acc_prec = dict(), dict(), dict(), dict(), dict(), dict()
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
            cv_kfold = k_fold_cross_validation(data_learning, nb_kFold, randomise=True, random_seed=seed[time])

            splits_s = list([])
            for training, testing in cv_kfold:
                # making a clone because it send the same address memory
                splits_s.append((training.make_clone(), testing.make_clone()))
                logger.info("Splits %s train %s", len(training.data), training.data[0][1:4])
                logger.info("Splits %s test %s", len(testing.data), testing.data[0][1:4])

            disc = str(nb_disc) + "-" + str(time)
            ich_skep[disc], cph_skep[disc], jacc_skep[disc] = dict(), dict(), dict()
            ich_out[disc], cph_out[disc], acc_prec[disc] = dict(), dict(), dict()
            for s_ncc in np.arange(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
                ks_ncc = str(s_ncc)
                ich_skep[disc][ks_ncc], cph_skep[disc][ks_ncc], jacc_skep[disc][ks_ncc] = 0, 0, 0
                ich_out[disc][ks_ncc], cph_out[disc][ks_ncc], acc_prec[disc][ks_ncc] = 0, 0, 0
                for idx_fold, (training, testing) in enumerate(splits_s):
                    logger.info("Splits %s train %s", len(training.data), training.data[0][1:4])
                    logger.info("Splits %s test %s", len(testing.data), testing.data[0][1:4])
                    rs = computing_training_testing_step(training,
                                                         testing,
                                                         missing_pct,
                                                         noise_label_pct,
                                                         noise_label_type,
                                                         noise_label_prob,
                                                         nb_labels,
                                                         s_ncc,
                                                         manager,
                                                         do_inference_exact,
                                                         ich_skep[disc][ks_ncc], cph_skep[disc][ks_ncc],
                                                         ich_out[disc][ks_ncc], cph_out[disc][ks_ncc],
                                                         acc_prec[disc][ks_ncc], jacc_skep[disc][ks_ncc])
                    ich_skep[disc][ks_ncc], cph_skep[disc][ks_ncc] = rs[0], rs[1]
                    ich_out[disc][ks_ncc], cph_out[disc][ks_ncc] = rs[2], rs[3]
                    acc_prec[disc][ks_ncc], jacc_skep[disc][ks_ncc] = rs[4], rs[5]
                    logger.debug("Partial-s-k_step (acc, ich_out) (%s, %s)",
                                 acc_prec[disc][ks_ncc], ich_out[disc][ks_ncc])
                writer.writerow([str(nb_disc), s_ncc, time,
                                 ich_skep[disc][ks_ncc] / nb_kFold,
                                 cph_skep[disc][ks_ncc] / nb_kFold,
                                 ich_out[disc][ks_ncc] / nb_kFold,
                                 cph_out[disc][ks_ncc] / nb_kFold,
                                 acc_prec[disc][ks_ncc] / nb_kFold,
                                 jacc_skep[disc][ks_ncc] / nb_kFold])
                file_csv.flush()
                logger.debug("Partial-s-k_step (disc, s, time, ich_skep, cph_skep, ich_out, cph_out, acc, jacc) "
                             "(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                             disc, s_ncc, time,
                             ich_skep[disc][ks_ncc] / nb_kFold,
                             cph_skep[disc][ks_ncc] / nb_kFold,
                             ich_out[disc][ks_ncc] / nb_kFold,
                             cph_out[disc][ks_ncc] / nb_kFold,
                             acc_prec[disc][ks_ncc] / nb_kFold,
                             jacc_skep[disc][ks_ncc] / nb_kFold)
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s, %s, %s, %s, %s, %s",
                 ich_skep, cph_skep, ich_out, cph_out, acc_prec, jacc_skep)


in_path = "/Users/salmuz/Downloads/datasets_mlc/emotions.arff"
out_path = "/Users/salmuz/Downloads/results_iris.csv"
computing_best_imprecise_mean(in_path=in_path,
                              out_path=out_path,
                              nb_process=1,
                              missing_pct=0.0,
                              noise_label_pct=0.0, noise_label_type=-1, noise_label_prob=0.2,
                              min_ncc_s_param=0.5, max_ncc_s_param=6, step_ncc_s_param=1,
                              remove_features=["image_name"])
