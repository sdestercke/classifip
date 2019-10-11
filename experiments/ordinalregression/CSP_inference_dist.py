import numpy as np, random, os, time, sys
from classifip.utils import create_logger
from CSP_common import *
from classifip.models.ncclr import NCCLR
from classifip.evaluation.measures import correctness_measure, completeness_measure
from classifip.dataset.arff import ArffFile
from classifip.evaluation import k_fold_cross_validation
import multiprocessing
from functools import partial

logger = create_logger("computing_best_min_s_cross_validation", True)


def parallel_prediction_csp(model, test_data, dataset, evaluatePBOX):
    idx, pBox = evaluatePBOX
    predicts = model.inference_CSP([pBox])
    y_ground_truth = test_data[idx][-1].split(">")
    correctness = correctness_measure(y_ground_truth, predicts[0])
    completeness = completeness_measure(y_ground_truth, predicts[0])
    is_coherent = False
    # verify if the prediction is coherent
    pid = multiprocessing.current_process().name

    def _pinfo(message, kwargs):
        print("[" + pid + "][" + time.strftime('%x %X %Z') + "]", "-", message % kwargs, flush=True)

    if predicts[0] is not None:
        is_coherent = True
        _desc_features = ",".join([
            '{0:.18f}'.format(feature) if str(feature).upper().find("E-") > 0 else str(feature)
            for feature in dataset[test_data[idx][0]][:-1]
        ])
        _pinfo("(ground_truth, nb_predictions) [%s] (%s, %s) ", (test_data[idx][0], y_ground_truth, len(predicts[0])))
        _pinfo("INSTANCE-COHERENT [%s] ( %s ) %s", (test_data[idx][0], _desc_features, correctness))
    else:
        incoherent_prediction = []
        for clazz, classifier in pBox.items():
            maxDecision = classifier.getmaximaldecision(model.ranking_utility)
            incoherent_prediction.append({clazz: np.where(maxDecision > 0)[0]})
        _pinfo("Solution incoherent (ground-truth, prediction) [%s] (%s, %s)",
               (test_data[idx][0], y_ground_truth, incoherent_prediction))

    return correctness, completeness, test_data[idx][0], is_coherent


def computing_training_testing_step(training, testing, s_current, all_data_set):
    # getting training and testing data without raw-index
    p_size = len(all_data_set[0])  # p_size: size of features + label ranking
    set_train_wridx, set_test_wridx = create_train_test_data_without_raw_index(training, testing, p_size)

    # learning model
    model = NCCLR()
    model.learn(set_train_wridx)
    # testing model
    evaluate_pBox = model.evaluate(set_test_wridx, ncc_s_param=s_current)
    # inference constraint satisfaction problem
    target_function = partial(parallel_prediction_csp, model, testing.data, all_data_set)
    acc_comp_var = POOL.map(target_function, enumerate(evaluate_pBox))
    # computing correctness and completeness on testing data set
    avg_cv_correctness, avg_cv_completeness, inc_coherent, raw_idx_coherent_instances = 0, 0, 0, []
    for correctness, completeness, idx_coherent, is_coherent in acc_comp_var:
        avg_cv_correctness += correctness
        avg_cv_completeness += completeness
        inc_coherent += is_coherent
        if is_coherent:
            raw_idx_coherent_instances.append(idx_coherent)

    nb_testing = len(testing.data)
    logger.info(
        "Computing train/test (s_current, correctness, completeness, nb_testing, nb_coherent) (%s, %s, %s, %s, %s)",
        s_current, avg_cv_correctness, avg_cv_completeness, nb_testing, inc_coherent
    )
    return avg_cv_correctness / nb_testing, avg_cv_completeness / nb_testing, raw_idx_coherent_instances


def recovery_instances_raw_index(idx_coherent_insts, set_testing):
    instances_retains = []
    for test_instance in set_testing.data:
        if test_instance[0] in idx_coherent_insts:
            instances_retains.append(test_instance)
    set_testing.data = instances_retains


def get_cr_cp(s_current, training, all_data_set, k_splits_cv, instances_coherent=None):
    cv_kFold = k_fold_cross_validation(training, k_splits_cv, randomise=False)
    # checking if we retain the coherent instances for calculations
    has_inst_pre_selected = (instances_coherent is not None)
    instances_coherent = [] if instances_coherent is None else instances_coherent
    avg_cv_correctness, avg_cv_completeness = 0, 0
    for set_train, set_test in cv_kFold:
        if has_inst_pre_selected:
            recovery_instances_raw_index(instances_coherent, set_test)
        cr, cp, raw_idx_instances = computing_training_testing_step(set_train, set_test, s_current, all_data_set)
        avg_cv_correctness += cr
        avg_cv_completeness += cp
        if not has_inst_pre_selected:
            instances_coherent.extend(raw_idx_instances)

    avg_cv_correctness = avg_cv_correctness / k_splits_cv
    avg_cv_completeness = avg_cv_completeness / k_splits_cv
    logger.info(
        "Get_cr_cp avg-k-fold cross-validation (s_current, correctness, completeness, k_splits) (%s, %s, %s, %s)",
        s_current, avg_cv_correctness, avg_cv_completeness, k_splits_cv
    )
    return avg_cv_correctness, avg_cv_completeness, instances_coherent


def create_train_test_data_without_raw_index(training, testing, p_size):
    # remove the raw-index of training data
    train_without_raw_index = training.make_clone()
    for data in train_without_raw_index.data:
        if len(data) > p_size:
            data.pop(0)

    # testing data without raw-index
    test_without_raw_index = []
    for data in testing.data:
        if len(data) > p_size:
            test_without_raw_index.append(data[1:])
        else:
            test_without_raw_index.append(data)
    return train_without_raw_index, test_without_raw_index


def computing_best_min_s_cross_validation(in_path,
                                          out_path=".",
                                          type_distance_s=TypeDistance.CORRECTNESS,
                                          k_splits_cv=10,
                                          s_theoretical=2,
                                          min_discretization=5,
                                          max_discretization=7,
                                          nb_points_into_curve=3,
                                          iplot_evolution_s=True,
                                          SEED_SCIENTIFIC=1234,
                                          is_retain_instances_coherent=False):
    assert os.path.exists(in_path), "Without dataset, it not possible to computing"
    assert os.path.exists(out_path), "Directory for putting results does not exist"

    logger.info('Training dataset (%s, %s, %s)', in_path, SEED_SCIENTIFIC, type_distance_s.name)
    logger.info('Parameters (min_disc, max_disc, k_split_cv, nb_points_into_curve) (%s, %s, %s, %s)',
                min_discretization, max_discretization, k_splits_cv, nb_points_into_curve)

    random.seed(SEED_SCIENTIFIC)
    overall_accuracy = dict()
    fnt_distance = type_distance_s.value[0]
    name_type_distance = type_distance_s.name.lower()
    dataset_name = os.path.basename(in_path).replace("_dense.xarff", "")

    for nb_intervals in range(min_discretization, max_discretization):
        logger.info("Starting discretization %s", nb_intervals)

        seed = generate_seeds(1)[0]
        seed_2ndkfcv = generate_seeds(k_splits_cv)
        logger.info("Seed generated for system is (%s, %s).", seed, seed_2ndkfcv)
        logger.info("Number interval for discreteness %s", nb_intervals)

        # load dataset
        dataArff = ArffFile()
        dataArff.load(in_path)
        # recovery real data set (without discretization)
        dataset = dataArff.make_clone().data
        # discretization data set
        dataArff.discretize(discmet="eqfreq", numint=nb_intervals)
        learning_kfcv = k_fold_cross_validation(dataArff, k_splits_cv, randomise=True, random_seed=seed)

        # adding raw-index to each instance
        for idx, data in enumerate(dataArff.data):
            data.insert(0, idx)

        # save partial performance correctness/completeness
        avg_kfold_correctness, avg_kfold_completeness = np.array([]), np.array([])
        avg_accuracy = dict()
        key_interval = str(nb_intervals)
        for training, testing in learning_kfcv:
            kfold = len(avg_kfold_correctness)

            # shuffle data, just one time for cross-validation (randomness)
            random.seed(seed_2ndkfcv[kfold])
            random.shuffle(training.data)

            # args for calculate s-min and s-max from cross-validation
            args_get_cr_cp = {"training": training,
                              "all_data_set": dataset,
                              "k_splits_cv": k_splits_cv}

            # find s-min and s-max methods
            min_s, min_s_corr, min_s_comp = find_min_or_max(s_theoretical,
                                                            TypeMeasure.COMPLETENESS,
                                                            get_cr_cp,
                                                            args_get_cr_cp)
            # coherent instances of s-minimum form k-fold
            if is_retain_instances_coherent:
                _, _, inst_coherent_s_min_kfcv = get_cr_cp(min_s, **args_get_cr_cp)
                logger.info("[RAW-INDEX-COHERENT-INSTANCES] %s", inst_coherent_s_min_kfcv)
                args_get_cr_cp["instances_coherent"] = inst_coherent_s_min_kfcv

            # recovery s-max with instances coherent and the others 's'
            max_s, max_s_corr, max_s_comp = find_min_or_max(min_s, TypeMeasure.CORRECTNESS, get_cr_cp, args_get_cr_cp)
            middle_s = round((min_s + max_s) / 2, 2)
            avg_accuracy[key_interval] = dict()

            # init values into accuracy dict
            avg_accuracy[key_interval][min_s] = dict({'corr': min_s_corr, 'comp': min_s_comp})
            avg_accuracy[key_interval][max_s] = dict({'corr': max_s_corr, 'comp': max_s_comp})
            middle_s_corr, middle_s_comp, _ = get_cr_cp(middle_s, **args_get_cr_cp)
            avg_accuracy[key_interval][middle_s] = dict({'corr': middle_s_corr, 'comp': middle_s_comp})

            for _ in range(nb_points_into_curve):
                new_s = get_s_values(avg_accuracy[key_interval], fnt_distance)
                avg_cv_correctness, avg_cv_completeness, _ = get_cr_cp(new_s, **args_get_cr_cp)
                avg_accuracy[key_interval][new_s] = dict({'corr': avg_cv_correctness, 'comp': avg_cv_completeness})

            # plot results k-fold learning
            if iplot_evolution_s:
                plot_save_results(results_dict=avg_accuracy,
                                  dataset_name=dataset_name,
                                  file_name=dataset_name + "_fold_" + str(kfold) + "_disc_" + str(nb_intervals),
                                  criterion=name_type_distance,
                                  out_root=out_path)

            # computing correctness with s-optimal
            logger.info("[STARTING] avg. cross-validation correctness/completeness s-optimal")
            s_optimal = min_s
            avg_validation_cr, avg_validation_cp, _ = computing_training_testing_step(training=training,
                                                                                      testing=testing,
                                                                                      s_current=s_optimal,
                                                                                      all_data_set=dataset)
            logger.info("[ENDING] avg. cross-validation correctness/completeness s-optimal (%s, %s, %s, %s)",
                        kfold, s_optimal, avg_validation_cr, avg_validation_cp)

            # save average of k-fold for std and mean correctness
            avg_kfold_completeness = np.append(avg_kfold_completeness, avg_validation_cp)
            avg_kfold_correctness = np.append(avg_kfold_correctness, avg_validation_cr)

        # print partial results
        logger.info("Discretization %s and partial results  %s", key_interval, avg_accuracy[key_interval])

        # save partial results in a list
        overall_accuracy[key_interval] = dict({'correctness': list(avg_kfold_correctness),
                                               'mean_corr': np.mean(avg_kfold_correctness),
                                               'std_corr': np.std(avg_kfold_correctness),
                                               'completeness': list(avg_kfold_completeness),
                                               'mean_comp': np.mean(avg_kfold_completeness),
                                               'std_comp': np.std(avg_kfold_completeness)})

        logger.info("Ending discretization %s", nb_intervals)

    # save summary results into json file
    out_dir_results = os.path.join(out_path, name_type_distance + '_distance_results')
    with open(out_dir_results + '/summary_%s.json' % dataset_name, 'w') as outfile:
        json.dump(overall_accuracy, outfile, indent=5)


# args. parameters
in_path_dataset = sys.argv[1]
out_path_results = sys.argv[2]
nb_process = int(sys.argv[3])

# creation pools process
POOL = multiprocessing.Pool(processes=nb_process)
seed_random_experiment = generate_seeds(1)[0]
computing_best_min_s_cross_validation(in_path=in_path_dataset,
                                      out_path=out_path_results,
                                      SEED_SCIENTIFIC=seed_random_experiment,
                                      is_retain_instances_coherent=False)
