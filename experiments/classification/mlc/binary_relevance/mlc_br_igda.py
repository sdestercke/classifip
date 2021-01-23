from classifip.evaluation import train_test_split, k_fold_cross_validation
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
from classifip.dataset import arff
from classifip.models.mlc.mlcncc import MLCNCC
import sys, os, random, csv, numpy as np
from itertools import product

sys.path.append("..")
from mlc_manager import ManagerWorkers, __create_dynamic_class
from mlc_common import *


def skeptical_prediction(pid, tasks, queue, results, class_model, class_model_challenger=None):
    # export LD_PRELOAD=/usr/local/MATLAB/R2018b/sys/os/glnxa64/libstdc++.so.6.0.22
    # QPBB_PATH_SERVER = ['/home/lab/ycarranz/QuadProgBB', '/opt/cplex128/cplex/matlab/x86-64_linux']
    # QPBB_PATH_SERVER = ['/volper/users/ycarranz/QuadProgBB', '/volper/users/ycarranz/cplex128/cplex/matlab/x86-64_linux']
    QPBB_PATH_SERVER = []  # executed in host

    try:
        model_skeptic = __create_dynamic_class(class_model,
                                               solver_matlab=False,
                                               gda_method="nda",
                                               add_path_matlab=QPBB_PATH_SERVER,
                                               DEBUG=False)
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
            del training["missing_pct"]

            model_skeptic.learn(learn_data_set=training["learn_data_set"],
                                nb_labels=training["nb_labels"],
                                ell_imprecision=training["ell_imprecision"])

            nb_labels = training["nb_labels"]
            while True:
                task = tasks.get()
                if task is None:
                    break

                skeptical_inference, precise_inference, prec_prob_marginal = \
                    model_skeptic.evaluate(**task['kwargs'])[0]

                # procedure to reject option
                epsilon_rejects = task["epsilon_rejects"]
                precise_rejects = dict()
                if epsilon_rejects is not None and len(epsilon_rejects) > 0:
                    for epsilon_reject in epsilon_rejects:
                        precise_reject = -2 * np.ones(nb_labels, dtype=int)
                        all_idx = set(range(nb_labels))
                        probabilities_yi_eq_1 = np.array(prec_prob_marginal.copy())
                        ones = set(np.where(probabilities_yi_eq_1 >= 0.5 + epsilon_reject)[0])
                        zeros = set(np.where(probabilities_yi_eq_1 <= 0.5 - epsilon_reject)[0])
                        stars = all_idx - ones - zeros
                        precise_reject[list(stars)] = -1
                        precise_reject[list(zeros)] = 0
                        precise_reject[list(ones)] = 1
                        precise_rejects[str(epsilon_reject)] = precise_reject

                # print partial prediction results
                print("(pid, skeptical, precise, precise_reject ground-truth) ",
                      pid, skeptical_inference, precise_inference, precise_rejects,
                      task['y_test'], flush=True)
                results.append(dict({'skeptical': skeptical_inference,
                                     'precise': precise_inference,
                                     'reject': precise_rejects,
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
                                    ell_imprecision,
                                    manager,
                                    epsilon_rejects,
                                    ich_skep, cph_skep,
                                    acc_prec,
                                    ich_reject, cph_reject):
    # Send training data model to every parallel process
    manager.addNewTraining(learn_data_set=learn_data_set,
                           nb_labels=nb_labels,
                           ell_imprecision=ell_imprecision,
                           missing_pct=missing_pct,
                           noise_label_pct=noise_label_pct,
                           noise_label_type=noise_label_type,
                           noise_label_prob=noise_label_prob)

    # Send testing data to every parallel process
    for test in test_data_set.data:
        manager.addTask({
            'kwargs': {'test_dataset': [test[:-nb_labels]]},
            'y_test': test[-nb_labels:],
            'epsilon_rejects': epsilon_rejects
        })
    manager.poisonPillWorkers()
    manager.joinTraining()  # wait all process for computing results
    # Recovery all inference data of all parallel process
    nb_tests = len(test_data_set.data)
    shared_results = manager.getResults()
    for prediction in shared_results:
        y_true = np.array(prediction['ground_truth'], dtype=np.int)
        y_skeptical = prediction['skeptical']
        y_precise = prediction['precise']
        y_rejects = prediction['reject']

        inc_ich_skep, inc_cph_skep = incorrectness_completeness_measure(y_true, y_skeptical)
        inc_acc_prec, _ = incorrectness_completeness_measure(y_true, y_precise)

        # why incorrectness < accuracy
        if inc_ich_skep > inc_acc_prec:
            print("[inc < acc](outer, precise, ground-truth) ",
                  y_skeptical, y_precise, y_true, inc_ich_skep, inc_acc_prec, flush=True)

        # skeptical measures
        ich_skep += inc_ich_skep / nb_tests
        cph_skep += inc_cph_skep / nb_tests
        acc_prec += inc_acc_prec / nb_tests
        # reject measures: if enable rejection option
        for epsilon, y_reject in y_rejects.items():
            inc_ich_reject, inc_cph_reject = incorrectness_completeness_measure(y_true, y_reject)
            ich_reject[epsilon] += inc_ich_reject / nb_tests
            cph_reject[epsilon] += inc_cph_reject / nb_tests

    manager.restartResults()
    return ich_skep, cph_skep, acc_prec, ich_reject, cph_reject


def experiments_binr_vs_imprecise(in_path=None,
                                  out_path=None,
                                  seed=None,
                                  missing_pct=0.0,
                                  noise_label_pct=0.0,
                                  noise_label_type=-1,
                                  noise_label_prob=0.5,
                                  nb_kFold=10,
                                  nb_process=1,
                                  scaling=False,
                                  epsilon_rejects=None,
                                  min_ell_param=0.5,
                                  max_ell_param=6.0,
                                  step_ell_param=1.0,
                                  remove_features=None):
    """
    Experiments with binary relevant imprecise and missing/noise data.

    :param in_path:
    :param out_path:
    :param seed:
    :param missing_pct: percentage of missing labels
    :param noise_label_pct: percentage of noise labels
    :param noise_label_type: type of perturbation noise
    :param noise_label_prob: probaiblity of noise labesl
    :param nb_kFold:
    :param nb_process: number of process in parallel
    :param scaling: scaling X input space (used for kkn-nccbr classifier)
    :param epsilon_rejects: epsilon of reject option (for comparing with imprecise version)
    :param min_ell_param: minimum value of imprecise parameter s
    :param max_ell_param: maximum value of imprecise parameter s
    :param step_ell_param: discretization step of parameter s
    :param remove_features: features not to take into account

    ...note::
        TODO: Bug when the missing percentage is higher (90%) to fix.

    """
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean", True)
    logger.info('Training dataset (%s, %s)', in_path, out_path)
    logger.info("(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param) (%s, %s, %s)",
                min_ell_param, max_ell_param, step_ell_param)
    logger.info("(scaling, remove_features, process, epsilon_rejects) (%s, %s, %s, %s)",
                scaling, remove_features, nb_process, epsilon_rejects)
    logger.info("(missing_pct, noise_label_pct, noise_label_type, noise_label_prob) (%s, %s, %s, %s)",
                missing_pct, noise_label_pct, noise_label_type, noise_label_prob)

    # Seeding a random value for k-fold top learning-testing data
    if seed is None:
        seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)

    # instance class classifier
    manager = ManagerWorkers(nb_process=nb_process, fun_prediction=skeptical_prediction)
    manager.executeAsync(class_model="classifip.models.mlc.igdabr.IGDA_BR")

    ich_skep, cph_skep, acc_prec = dict(), dict(), dict()
    ich_reject, cph_reject = dict(), dict()
    data_learning, nb_labels = init_dataset(in_path, remove_features, scaling)
    for time in range(nb_kFold):  # 10-10 times cross-validation
        logger.info("Number labels %s", nb_labels)
        cv_kfold = k_fold_cross_validation(data_learning,
                                           K=nb_kFold,
                                           randomise=True,
                                           random_seed=seed[time])

        splits_s = list([])
        for training, testing in cv_kfold:
            # making a clone because it send the same address memory
            splits_s.append((training.make_clone(), testing.make_clone()))
            logger.info("Splits %s train %s", len(training.data), training.data[0][1:4])
            logger.info("Splits %s test %s", len(testing.data), testing.data[0][1:4])

        for ell_imprecision in np.arange(min_ell_param, max_ell_param, step_ell_param):
            ks_ell = str(ell_imprecision)
            init_scores(ks_ell, ich_skep, cph_skep, acc_prec, ich_reject, cph_reject, epsilon_rejects)
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
                                                     ell_imprecision,
                                                     manager,
                                                     epsilon_rejects,
                                                     ich_skep[ks_ell],
                                                     cph_skep[ks_ell],
                                                     acc_prec[ks_ell],
                                                     ich_reject[ks_ell],
                                                     cph_reject[ks_ell])
                ich_skep[ks_ell], cph_skep[ks_ell] = rs[0], rs[1]
                acc_prec[ks_ell] = rs[2]
                ich_reject[ks_ell], cph_reject[ks_ell] = rs[3], rs[4]
                logger.debug("Partial-ell_step (acc, ich_skep) (%s, %s)",
                             acc_prec[ks_ell], ich_skep[ks_ell])
            ich_skep[ks_ell] = ich_skep[ks_ell] / nb_kFold
            cph_skep[ks_ell] = cph_skep[ks_ell] / nb_kFold
            acc_prec[ks_ell] = acc_prec[ks_ell] / nb_kFold
            _partial_saving = [ell_imprecision, time,
                               ich_skep[ks_ell],
                               cph_skep[ks_ell],
                               acc_prec[ks_ell]]

            if epsilon_rejects is not None:
                _reject_ich = [e / nb_kFold for e in ich_reject[ks_ell].values()]
                _reject_cph = [e / nb_kFold for e in cph_reject[ks_ell].values()]
                _partial_saving = _partial_saving + _reject_ich + _reject_cph
            else:
                _reject_ich, _reject_cph = [], []

            logger.debug("Partial-s-k_step reject values (%s)", ich_reject[ks_ell])
            writer.writerow(_partial_saving)
            file_csv.flush()
            logger.debug("Partial-s-k_step (ell, time, ich_skep, cph_skep, acc, ich_reject, cph_reject) "
                         "(%s, %s, %s, %s, %s, %s, %s)", ell_imprecision, time,
                         ich_skep[ks_ell], cph_skep[ks_ell],
                         acc_prec[ks_ell], _reject_ich, _reject_cph)

    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s, %s, %s", ich_skep, cph_skep, acc_prec)


in_path = "/Users/salmuz/Downloads/datasets_mlc/emotions.arff"
out_path = "/Users/salmuz/Downloads/results_emotions.csv"
experiments_binr_vs_imprecise(in_path=in_path,
                              out_path=out_path,
                              nb_process=1,
                              scaling=True,
                              missing_pct=0.0,
                              noise_label_pct=0.0, noise_label_type=-1, noise_label_prob=0.2,
                              min_ell_param=0.05, max_ell_param=0.06, step_ell_param=0.05,
                              epsilon_rejects=[0.05, 0.15, 0.25, 0.35, 0.45],
                              remove_features=["image_name"])
