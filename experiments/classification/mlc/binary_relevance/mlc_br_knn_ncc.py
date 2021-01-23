from classifip.evaluation import train_test_split, k_fold_cross_validation
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
from classifip.models.mlc import nccbr
from classifip.models.mlc import knnnccbr
from classifip.models.mlc.mlcncc import MLCNCC
import sys, os, random, csv, numpy as np
from itertools import product

sys.path.append("..")
from mlc_manager import ManagerWorkers, __create_dynamic_class
from mlc_common import *


def skeptical_prediction(pid, tasks, queue, results, class_model, class_model_challenger=None):
    try:
        model_skeptic = __create_dynamic_class(class_model)
        while True:
            training = queue.get()
            if training is None:
                break

            nb_labels = training["nb_labels"]
            p_dimension_all = training.pop('p_dimension') + nb_labels
            global_data_continuous = training.pop('data_continuous')
            new_continuous_data = global_data_continuous.make_clone()
            new_continuous_data.data = list()
            for row_instance in training["learn_data_set"].data:
                row_index = row_instance.pop(p_dimension_all)  # delete index raw instance by reference
                new_continuous_data.data.append(global_data_continuous.data[int(row_index)].copy())

            # Missing and Noise labels if the percentage is greater than 0
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

            model_skeptic.learn(learn_data_set=new_continuous_data,
                                nb_labels=training["nb_labels"],
                                learn_disc_set=training["learn_data_set"])

            while True:
                task = tasks.get()
                if task is None:
                    break

                # procedure to skeptic inference
                instance_test = task['kwargs']['test_dataset'][0]
                index_test = instance_test.pop(p_dimension_all)
                raw_instance_test = global_data_continuous.data[int(index_test)]
                inferences = model_skeptic.evaluate(test_dataset=[(raw_instance_test, instance_test)],
                                                    ncc_s_param=task['kwargs']['ncc_s_param'],
                                                    k=task['k_nearest'],
                                                    laplace_smoothing=task['kwargs']['laplace_smoothing'])[0]
                set_prob_marginal = inferences[0]
                prec_prob_marginal = inferences[1]

                skeptical_inference = set_prob_marginal.multilab_dom()
                precise_inference = prec_prob_marginal.multilab_dom()
                # procedure to reject option
                epsilon_rejects = task["epsilon_rejects"]
                precise_rejects = dict()
                if epsilon_rejects is not None and len(epsilon_rejects) > 0:
                    for epsilon_reject in epsilon_rejects:
                        precise_reject = -2 * np.ones(nb_labels, dtype=int)
                        all_idx = set(range(nb_labels))
                        probabilities_yi_eq_1 = prec_prob_marginal.scores[:, 0].copy()
                        ones = set(np.where(probabilities_yi_eq_1 >= 0.5 + epsilon_reject)[0])
                        zeros = set(np.where(probabilities_yi_eq_1 <= 0.5 - epsilon_reject)[0])
                        stars = all_idx - ones - zeros
                        precise_reject[list(stars)] = -1
                        precise_reject[list(zeros)] = 0
                        precise_reject[list(ones)] = 1
                        precise_rejects[str(epsilon_reject)] = precise_reject

                # print partial prediction results
                print("(pid, skeptical, precise, precise_reject ground-truth) ",
                      pid, skeptical_inference, precise_inference,
                      precise_rejects, task['y_test'], flush=True)
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
                                    p_dimension,
                                    ncc_imprecise,
                                    manager,
                                    epsilon_rejects,
                                    ich_skep, cph_skep,
                                    acc_prec,
                                    ich_reject, cph_reject,
                                    data_continuous,
                                    k_nearest_neighbors):
    # Send training data model to every parallel process
    manager.addNewTraining(learn_data_set=learn_data_set,
                           nb_labels=nb_labels,
                           missing_pct=missing_pct,
                           noise_label_pct=noise_label_pct,
                           noise_label_type=noise_label_type,
                           noise_label_prob=noise_label_prob,
                           data_continuous=data_continuous,
                           p_dimension=p_dimension)

    # Send testing data to every parallel process
    for test in test_data_set.data:
        manager.addTask(
            {'kwargs': {'test_dataset': [test],
                        'ncc_epsilon': 0.001,
                        'ncc_s_param': ncc_imprecise,
                        'laplace_smoothing': True},
             'y_test': test[p_dimension:(p_dimension + nb_labels)],
             'epsilon_rejects': epsilon_rejects,
             'k_nearest': k_nearest_neighbors})
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
                                  min_ncc_s_param=0.5,
                                  max_ncc_s_param=6.0,
                                  step_ncc_s_param=1.0,
                                  remove_features=None,
                                  k_nearest_neighbors=None):
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
    :param min_ncc_s_param: minimum value of imprecise parameter s
    :param max_ncc_s_param: maximum value of imprecise parameter s
    :param step_ncc_s_param: discretization step of parameter s
    :param remove_features: features not to take into account
    :param k_nearest_neighbors: k*radius_distance_pairwise_all_instance,
            how big is ball containing neighbors.

    ...note::
        TODO: Bug when the missing percentage is higher (90%) to fix.

    """
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"
    assert k_nearest_neighbors is not None, "None value, it needs a value for the knn algorithm"
    assert k_nearest_neighbors > 0, "Need a value for the knn algorithm"

    logger = create_logger("computing_best_imprecise_mean", True)
    logger.info('Training dataset (%s, %s)', in_path, out_path)
    logger.info("(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param) (%s, %s, %s)",
                min_ncc_s_param, max_ncc_s_param, step_ncc_s_param)
    logger.info("(scaling, remove_features, process, epsilon_rejects) (%s, %s, %s, %s)",
                scaling, remove_features, nb_process, epsilon_rejects)
    logger.info("(missing_pct, noise_label_pct, noise_label_type, noise_label_prob) (%s, %s, %s, %s)",
                missing_pct, noise_label_pct, noise_label_type, noise_label_prob)
    logger.info("( k_nearest_neighbors)  (%s)", k_nearest_neighbors)

    # Seeding a random value for k-fold top learning-testing data
    if seed is None:
        seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)

    manager = ManagerWorkers(nb_process=nb_process, fun_prediction=skeptical_prediction)
    manager.executeAsync(class_model="classifip.models.mlc.knnnccbr.KNN_NCC_BR")

    ich_skep, cph_skep, acc_prec = dict(), dict(), dict()
    ich_reject, cph_reject = dict(), dict()
    min_discretize, max_discretize = 5, 7
    for nb_disc in range(min_discretize, max_discretize):
        data_learning, nb_labels = init_dataset(in_path, remove_features, scaling)
        p_dimension = len(data_learning.data[0]) - nb_labels

        # saving continuous data and index instances for KNN-NCC-BR classification
        data_continuous = data_learning.make_clone()
        # adding raw-index to each instance if we use knn-ncc
        for idx, row_instance in enumerate(data_learning.data):
            row_instance.insert(p_dimension + nb_labels, idx)
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
            ich_skep[disc], cph_skep[disc], acc_prec[disc] = dict(), dict(), dict()
            ich_reject[disc], cph_reject[disc] = dict(), dict()
            for s_ncc in np.arange(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
                ks_ncc = str(s_ncc)
                init_scores(ks_ncc, ich_skep[disc], cph_skep[disc], acc_prec[disc],
                            ich_reject[disc], cph_reject[disc], epsilon_rejects)
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
                                                         p_dimension,
                                                         s_ncc,
                                                         manager,
                                                         epsilon_rejects,
                                                         ich_skep[disc][ks_ncc],
                                                         cph_skep[disc][ks_ncc],
                                                         acc_prec[disc][ks_ncc],
                                                         ich_reject[disc][ks_ncc],
                                                         cph_reject[disc][ks_ncc],
                                                         data_continuous,
                                                         k_nearest_neighbors)
                    ich_skep[disc][ks_ncc], cph_skep[disc][ks_ncc] = rs[0], rs[1]
                    acc_prec[disc][ks_ncc] = rs[2]
                    ich_reject[disc][ks_ncc], cph_reject[disc][ks_ncc] = rs[3], rs[4]
                    logger.debug("Partial-s-k_step (acc, ich_skep) (%s, %s)",
                                 acc_prec[disc][ks_ncc], ich_skep[disc][ks_ncc])
                ich_skep[disc][ks_ncc] = ich_skep[disc][ks_ncc] / nb_kFold
                cph_skep[disc][ks_ncc] = cph_skep[disc][ks_ncc] / nb_kFold
                acc_prec[disc][ks_ncc] = acc_prec[disc][ks_ncc] / nb_kFold
                _partial_saving = [str(nb_disc), s_ncc, time,
                                   ich_skep[disc][ks_ncc],
                                   cph_skep[disc][ks_ncc],
                                   acc_prec[disc][ks_ncc]]
                if epsilon_rejects is not None:
                    _reject_ich = [e / nb_kFold for e in ich_reject[disc][ks_ncc].values()]
                    _reject_cph = [e / nb_kFold for e in cph_reject[disc][ks_ncc].values()]
                    _partial_saving = _partial_saving + _reject_ich + _reject_cph
                else:
                    _reject_ich, _reject_cph = [], []
                logger.debug("Partial-s-k_step reject values (%s)", ich_reject[disc][ks_ncc])
                writer.writerow(_partial_saving)
                file_csv.flush()
                logger.debug("Partial-s-k_step (disc, s, time, ich_skep, cph_skep, acc, ich_reject, cph_reject)"
                             "(%s, %s, %s, %s, %s, %s, %s, %s)", disc, s_ncc, time,
                             ich_skep[disc][ks_ncc],
                             cph_skep[disc][ks_ncc],
                             acc_prec[disc][ks_ncc],
                             _reject_ich,
                             _reject_cph)
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s, %s, %s",
                 ich_skep, cph_skep, acc_prec)


in_path = "..../datasets_mlc/emotions.arff"
out_path = ".../results_emotions.csv"
experiments_binr_vs_imprecise(in_path=in_path,
                              out_path=out_path,
                              nb_process=1,
                              missing_pct=0.0,
                              noise_label_pct=0.0, noise_label_type=-1, noise_label_prob=0.2,
                              min_ncc_s_param=0.1, max_ncc_s_param=6, step_ncc_s_param=1,
                              epsilon_rejects=[0.05, 0.15, 0.25, 0.35, 0.45],
                              k_nearest_neighbors=0.5,
                              remove_features=["image_name"])
