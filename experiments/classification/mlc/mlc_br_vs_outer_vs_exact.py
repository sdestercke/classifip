from classifip.evaluation import train_test_split, k_fold_cross_validation
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
from classifip.dataset import arff
from classifip.models.mlc import nccbr
from classifip.models.mlc import knnnccbr
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
    _full_binary_vector = np.array(full_binary_vector)
    _, nb_labels = full_binary_vector.shape
    result = np.zeros(nb_labels, dtype=np.int)
    for idx_label in range(nb_labels):
        label_value = np.unique(_full_binary_vector[:, idx_label])
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
        model_outer = __create_dynamic_class(class_model_challenger)
        model_exact = __create_dynamic_class(class_model)
        while True:
            training = queue.get()
            if training is None:
                break

            nb_labels = training["nb_labels"]
            p_dimension_all = training.pop('p_dimension') + nb_labels
            global_data_continuous = training.pop('data_continuous')
            new_continuous_data = None
            if global_data_continuous is not None:
                new_continuous_data = global_data_continuous.make_clone()
                new_continuous_data.data = list()
                for row_instance in training["learn_data_set"].data:
                    row_index = row_instance.pop(p_dimension_all)  # delete index raw instance by reference
                    new_continuous_data.data.append(global_data_continuous.data[int(row_index)].copy())

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

            if new_continuous_data is not None:
                model_outer.learn(learn_data_set=new_continuous_data,
                                  nb_labels=training["nb_labels"],
                                  learn_disc_set=training["learn_data_set"])
            else:
                model_outer.learn(**training)

            model_exact.learn(**training)
            while True:
                task = tasks.get()
                if task is None:
                    break

                # naive and improvement exact maximality inference
                # skeptical_inference_exact = model_exact.evaluate_exact(**task['kwargs'])[0]
                skeptical_inference = model_exact.evaluate(**task['kwargs'])[0] if task['do_inference_exact'] \
                    else [-1] * len(task['y_test'])

                # procedure to skeptic inference
                k_nearest = task['k_nearest']
                if k_nearest is None or k_nearest == 0:
                    # outer-approximation binary relevance
                    set_prob_marginal = model_outer.evaluate(**task['kwargs'])[0]
                    # precise and e-precise inference with s equal 0, epsilon > 0.0
                    task['kwargs']['ncc_s_param'] = 0.0
                    prec_prob_marginal = model_outer.evaluate(**task['kwargs'])[0]
                else:
                    instance_test = task['kwargs']['test_dataset'][0]
                    index_test = instance_test.pop(p_dimension_all)
                    raw_instance_test = global_data_continuous.data[int(index_test)]
                    inferences = model_outer.evaluate(test_dataset=[(raw_instance_test, instance_test)],
                                                      ncc_s_param=task['kwargs']['ncc_s_param'],
                                                      k=k_nearest)[0]
                    set_prob_marginal = inferences[0]
                    prec_prob_marginal = inferences[1]

                outer_inference = set_prob_marginal.multilab_dom()
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
                print("(pid, skeptical, outer, precise, precise_reject ground-truth) ",
                      pid, len(skeptical_inference), outer_inference, precise_inference,
                      precise_rejects, task['y_test'], flush=True)
                results.append(dict({'skeptical': skeptical_inference,
                                     'outer': outer_inference,
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
                                    do_inference_exact,
                                    epsilon_rejects,
                                    ich_skep, cph_skep,
                                    ich_out, cph_out,
                                    acc_prec, jacc_skep,
                                    ich_reject, cph_reject,
                                    jacc_reject,
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
                        'ncc_s_param': ncc_imprecise},
             'y_test': test[p_dimension:(p_dimension + nb_labels)],
             'do_inference_exact': do_inference_exact,
             'epsilon_rejects': epsilon_rejects,
             'k_nearest': k_nearest_neighbors})
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
        y_rejects = prediction['reject']
        # decompose the partial to full prediction
        y_outer_full_set = expansion_partial_to_full_set_binary_vector(y_outer)

        # if enable to do the exact skeptical inference
        inc_jacc = -1
        inc_ich_skep, inc_cph_skep = -1, -1
        if do_inference_exact:
            y_skeptical_exact_partial = transform_semi_partial_vector(y_skeptical_exact)
            inc_ich_skep, inc_cph_skep = incorrectness_completeness_measure(y_true, y_skeptical_exact_partial)
            inc_jacc = compute_jaccard_similarity_score(y_outer_full_set, y_skeptical_exact)

        inc_ich_out, inc_cph_out = incorrectness_completeness_measure(y_true, y_outer)
        inc_acc_prec, _ = incorrectness_completeness_measure(y_true, y_precise)

        # why incorrectness < accuracy
        if inc_ich_out > inc_acc_prec:
            print("[inc < acc](outer, precise, ground-truth) ",
                  y_outer, y_precise, y_true, inc_ich_out, inc_acc_prec, flush=True)

        # skeptical measures
        ich_skep += inc_ich_skep / nb_tests
        cph_skep += inc_cph_skep / nb_tests
        jacc_skep += inc_jacc / nb_tests
        # outer measures
        ich_out += inc_ich_out / nb_tests
        cph_out += inc_cph_out / nb_tests
        acc_prec += inc_acc_prec / nb_tests
        # reject measures: if enable rejection option
        for epsilon, y_reject in y_rejects.items():
            y_reject_full_set = expansion_partial_to_full_set_binary_vector(y_reject)
            inc_ich_reject, inc_cph_reject = incorrectness_completeness_measure(y_true, y_reject)
            inc_jacc_reject = compute_jaccard_similarity_score(y_outer_full_set, y_reject_full_set)

            ich_reject[epsilon] += inc_ich_reject / nb_tests
            cph_reject[epsilon] += inc_cph_reject / nb_tests
            jacc_reject[epsilon] += inc_jacc_reject / nb_tests

    manager.restartResults()
    return (ich_skep, cph_skep, ich_out, cph_out,
            acc_prec, jacc_skep, ich_reject, cph_reject, jacc_reject)


def experiments_binr_vs_imprecise(in_path=None,
                                  out_path=None,
                                  seed=None,
                                  missing_pct=0.0,
                                  noise_label_pct=0.0,
                                  noise_label_type=-1,
                                  noise_label_prob=0.5,
                                  nb_kFold=10,
                                  nb_process=1,
                                  scaling=True,
                                  epsilon_rejects=None,
                                  min_ncc_s_param=0.5,
                                  max_ncc_s_param=6.0,
                                  step_ncc_s_param=1.0,
                                  remove_features=None,
                                  do_inference_exact=False,
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
    :param do_inference_exact: exact inferences (exploring the probabilistic tree)
    :param k_nearest_neighbors: k*radius_distance_pairwise_all_instance,
            how big is ball containing neighbors.

    ...note::
        TODO: Bug when the missing percentage is higher (90%) to fix.

    """
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean", True)
    logger.info('Training dataset (%s, %s)', in_path, out_path)
    logger.info("(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param) (%s, %s, %s)",
                min_ncc_s_param, max_ncc_s_param, step_ncc_s_param)
    logger.info("(scaling, remove_features, process, epsilon_rejects) (%s, %s, %s, %s)",
                scaling, remove_features, nb_process, epsilon_rejects)
    logger.info("(missing_pct, noise_label_pct, noise_label_type, noise_label_prob) (%s, %s, %s, %s)",
                missing_pct, noise_label_pct, noise_label_type, noise_label_prob)
    logger.info("(do_inference_exact, k_nearest_neighbors)  (%s, %s)",
                do_inference_exact, k_nearest_neighbors)

    # Seeding a random value for k-fold top learning-testing data
    if seed is None:
        seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process, fun_prediction=skeptical_prediction)
    _class_model_challenger = "classifip.models.mlc.nccbr.NCCBR"
    if k_nearest_neighbors is not None and k_nearest_neighbors > 0:
        _class_model_challenger = "classifip.models.mlc.knnnccbr.KNN_NCC_BR"
    manager.executeAsync(class_model="classifip.models.mlc.exactncc.MLCNCCExact",
                         class_model_challenger=_class_model_challenger)
    logger.info('Classifier binary relevant (%s)', _class_model_challenger)

    ich_skep, cph_skep, jacc_skep = dict(), dict(), dict()
    ich_out, cph_out, acc_prec = dict(), dict(), dict()
    ich_reject, cph_reject, jacc_reject = dict(), dict(), dict()
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
        p_dimension = len(data_learning.data[0]) - nb_labels
        if scaling:
            normalize(data_learning, n_labels=nb_labels)

        # procedure to model the knn-and-ncc classification
        data_continuous = None
        if k_nearest_neighbors is not None and k_nearest_neighbors > 0:
            # saving continuous data and index instances for KNN-NCC-BR
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
            ich_skep[disc], cph_skep[disc], jacc_skep[disc] = dict(), dict(), dict()
            ich_out[disc], cph_out[disc], acc_prec[disc] = dict(), dict(), dict()
            ich_reject[disc], cph_reject[disc], jacc_reject[disc] = dict(), dict(), dict()
            for s_ncc in np.arange(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
                ks_ncc = str(s_ncc)
                ich_skep[disc][ks_ncc], cph_skep[disc][ks_ncc], jacc_skep[disc][ks_ncc] = 0, 0, 0
                ich_out[disc][ks_ncc], cph_out[disc][ks_ncc], acc_prec[disc][ks_ncc] = 0, 0, 0
                ich_reject[disc][ks_ncc] = dict.fromkeys(np.array(epsilon_rejects, dtype='<U'), 0)
                cph_reject[disc][ks_ncc] = dict.fromkeys(np.array(epsilon_rejects, dtype='<U'), 0)
                jacc_reject[disc][ks_ncc] = dict.fromkeys(np.array(epsilon_rejects, dtype='<U'), 0)
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
                                                         do_inference_exact,
                                                         epsilon_rejects,
                                                         ich_skep[disc][ks_ncc], cph_skep[disc][ks_ncc],
                                                         ich_out[disc][ks_ncc], cph_out[disc][ks_ncc],
                                                         acc_prec[disc][ks_ncc], jacc_skep[disc][ks_ncc],
                                                         ich_reject[disc][ks_ncc], cph_reject[disc][ks_ncc],
                                                         jacc_reject[disc][ks_ncc],
                                                         data_continuous,
                                                         k_nearest_neighbors)
                    ich_skep[disc][ks_ncc], cph_skep[disc][ks_ncc] = rs[0], rs[1]
                    ich_out[disc][ks_ncc], cph_out[disc][ks_ncc] = rs[2], rs[3]
                    acc_prec[disc][ks_ncc], jacc_skep[disc][ks_ncc] = rs[4], rs[5]
                    ich_reject[disc][ks_ncc], cph_reject[disc][ks_ncc] = rs[6], rs[7]
                    jacc_reject[disc][ks_ncc] = rs[8]
                    logger.debug("Partial-s-k_step (acc, ich_out) (%s, %s)",
                                 acc_prec[disc][ks_ncc], ich_out[disc][ks_ncc])
                ich_skep[disc][ks_ncc] = ich_skep[disc][ks_ncc] / nb_kFold
                cph_skep[disc][ks_ncc] = cph_skep[disc][ks_ncc] / nb_kFold
                ich_out[disc][ks_ncc] = ich_out[disc][ks_ncc] / nb_kFold
                cph_out[disc][ks_ncc] = cph_out[disc][ks_ncc] / nb_kFold
                acc_prec[disc][ks_ncc] = acc_prec[disc][ks_ncc] / nb_kFold
                jacc_skep[disc][ks_ncc] = jacc_skep[disc][ks_ncc] / nb_kFold
                _partial_saving = [str(nb_disc), s_ncc, time,
                                   ich_skep[disc][ks_ncc],
                                   cph_skep[disc][ks_ncc],
                                   ich_out[disc][ks_ncc],
                                   cph_out[disc][ks_ncc],
                                   acc_prec[disc][ks_ncc],
                                   jacc_skep[disc][ks_ncc]]
                _reject_ich = [e / nb_kFold for e in ich_reject[disc][ks_ncc].values()]
                _reject_cph = [e / nb_kFold for e in cph_reject[disc][ks_ncc].values()]
                _reject_jacc = [e / nb_kFold for e in jacc_reject[disc][ks_ncc].values()]
                _partial_saving = _partial_saving + _reject_ich + _reject_cph + _reject_jacc
                logger.debug("Partial-s-k_step reject values (%s)", ich_reject[disc][ks_ncc])
                writer.writerow(_partial_saving)
                file_csv.flush()
                logger.debug("Partial-s-k_step (disc, s, time, ich_skep, cph_skep, ich_out, "
                             "cph_out, acc, jacc, ich_reject, cph_reject, jacc_reject) "
                             "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                             disc, s_ncc, time,
                             ich_skep[disc][ks_ncc] / nb_kFold,
                             cph_skep[disc][ks_ncc] / nb_kFold,
                             ich_out[disc][ks_ncc] / nb_kFold,
                             cph_out[disc][ks_ncc] / nb_kFold,
                             acc_prec[disc][ks_ncc] / nb_kFold,
                             jacc_skep[disc][ks_ncc] / nb_kFold,
                             _reject_ich,
                             _reject_cph,
                             _reject_jacc)
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s, %s, %s, %s, %s, %s",
                 ich_skep, cph_skep, ich_out, cph_out, acc_prec, jacc_skep)


in_path = ".../datasets_mlc/emotions.arff"
out_path = ".../Downloads/results_emotions.csv"
experiments_binr_vs_imprecise(in_path=in_path,
                              out_path=out_path,
                              nb_process=1,
                              scaling=True,
                              missing_pct=0.0,
                              noise_label_pct=0.0, noise_label_type=-1, noise_label_prob=0.2,
                              min_ncc_s_param=0.5, max_ncc_s_param=6, step_ncc_s_param=1,
                              epsilon_rejects=[0.05, 0.15, 0.25, 0.35, 0.45],
                              k_nearest_neighbors=0,
                              do_inference_exact=False,
                              remove_features=["image_name"])
