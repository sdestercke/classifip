from classifip.evaluation import k_fold_cross_validation
from classifip.utils import create_logger
from classifip.dataset import arff
from classifip.models.mlc import nccbr
from classifip.models.mlc.exactncc import MLCNCCExact
from mlc_manager import ManagerWorkers, __create_dynamic_class
import math, os, random, sys, csv, numpy as np


def get_nb_labels_class(dataArff, type_class='nominal'):
    nominal_class = [item for item in dataArff.attribute_types.values() if item == type_class]
    return len(nominal_class)


def distance_cardinal_exact_inference(inference_exact, inference_exact_improved):
    """
        This method aims to check if the improved exact inference (3^m-1 comparisons)
        has same number of solutions than the exact inference (all comparisons)
    :param inference_exact:
    :param inference_exact_improved:
    :return:
    """
    return abs(len(inference_exact) - len(inference_exact_improved))


def distance_cardinal_set_inferences(inference_outer, inference_exact, nb_labels):
    power_outer = 0
    for j in range(nb_labels):
        if inference_outer[j] == -1:
            power_outer += 1
    return abs(math.pow(2, power_outer) - len(inference_exact))


def prediction_dist(pid, tasks, queue, results, class_model):
    try:
        model_br = nccbr.NCCBR()
        model_exact = __create_dynamic_class(class_model)
        while True:
            training = queue.get()
            if training is None:
                break
            model_exact.learn(**training)
            model_br.learn(**training)
            dist_measure = 0
            while True:
                task = tasks.get()
                if task is None:
                    break
                nb_labels = len(task['y_test'])
                set_prob_marginal = model_br.evaluate(**task['kwargs'])
                inference_outer = set_prob_marginal[0].multilab_dom()
                inference_exact = model_exact.evaluate(**task['kwargs'])[0]
                distance_cardinal = distance_cardinal_set_inferences(inference_outer, inference_exact, nb_labels)
                dist_measure += distance_cardinal
                print("(pid, exact, outer, ground-truth, distance) ", pid, inference_exact, inference_outer,
                      task['y_test'], distance_cardinal, flush=True)
            results.append(dict({'dist_measure': dist_measure}))
            queue.task_done()
    except Exception as e:
        raise Exception(e, "Error in job of PID " + pid)
    finally:
        print("Worker PID finished", pid, flush=True)


def computing_training_testing_step(learn_data_set,
                                    test_data_set,
                                    nb_labels,
                                    ncc_imprecise,
                                    manager,
                                    dist_measure):
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
    nb_testing = len(test_data_set.data)
    # Recovery all inference data of all parallel process
    shared_results = manager.getResults()
    for _ in range(manager.NUMBER_OF_PROCESSES):
        utility = shared_results.pop()
        dist_measure += utility["dist_measure"] / nb_testing
    return dist_measure


def computing_outer_vs_exact_inference(in_path=None, out_path=None, seed=None, nb_kFold=10,
                                       intvl_ncc_s_param=None, step_ncc_s=1.0, nb_process=1):
    intvl_ncc_s_param = list([2, 5]) if intvl_ncc_s_param is None else intvl_ncc_s_param
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean_cv", False)
    logger.info('Training dataset (%s, %s)', in_path, out_path)

    # Seeding a random value for k-fold top learning-testing data
    if seed is not None:
        random.seed(seed)
    seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)

    # Create manager
    manager = ManagerWorkers(nb_process=nb_process, fun_prediction=prediction_dist)
    manager.executeAsync(class_model="classifip.models.mlc.exactncc.MLCNCCExact")

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
                ks_ncc = str(s_ncc)
                diff_inferences[disc][ks_ncc] = 0
                for idx_fold, (training, testing) in enumerate(splits_s):
                    diff_inferences[disc][ks_ncc] = computing_training_testing_step(training, testing,
                                                                                    nb_labels, s_ncc, manager,
                                                                                    diff_inferences[disc][ks_ncc])
                diff_inferences[disc][ks_ncc] = diff_inferences[disc][ks_ncc] / nb_kFold
                writer.writerow([str(nb_disc), s_ncc, time, diff_inferences[disc][ks_ncc] / nb_kFold])
                file_csv.flush()
                logger.info("Partial-s-k_step (%s, %s, %s, %s)", disc, s_ncc, time, diff_inferences[disc][ks_ncc])
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s", diff_inferences)


_name = "labels2"
sys_in_path = _name + ".arff"
sys_out_path = "results_" + _name + ".csv"
# QPBB_PATH_SERVER = []  # executed in host
computing_outer_vs_exact_inference(in_path=sys_in_path, out_path=sys_out_path,
                                   intvl_ncc_s_param=[1, 5], step_ncc_s=0.5, nb_process=1)
