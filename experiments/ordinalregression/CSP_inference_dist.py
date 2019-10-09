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


def parallel_prediction_csp(model, test_data, evaluatePBOX):
    idx, pBox = evaluatePBOX
    predicts = model.inference_CSP([pBox])
    y_ground_truth = test_data[idx][-1].split(">")
    correctness = correctness_measure(y_ground_truth, predicts[0])
    completeness = completeness_measure(y_ground_truth, predicts[0])
    is_coherent = False
    # verify if the prediction is coherent
    pid = multiprocessing.current_process().name

    def _pinfo(message, kwargs):
        print("[" + pid + "]" + time.strftime('%x %X %Z'), "-", message % kwargs, flush=True)

    if predicts[0] is not None:
        is_coherent = True
        _desc_features = ",".join([
            '{0:.18f}'.format(feature) if str(feature).upper().find("E-") > 0 else str(feature)
            for feature in test_data[idx][:-1]
        ])
        _pinfo("(ground_truth, nb_predictions) (%s, %s) ", (y_ground_truth, len(predicts[0])))
        _pinfo("INSTANCE-COHERENT ( %s ) %s", (_desc_features, correctness))
    else:
        incoherent_prediction = []
        for clazz, classifier in pBox.items():
            maxDecision = classifier.getmaximaldecision(model.ranking_utility)
            incoherent_prediction.append({clazz: np.where(maxDecision > 0)[0]})
        _pinfo("Solution incoherent (ground-truth, prediction) (%s, %s)", (y_ground_truth, incoherent_prediction))

    return correctness, completeness, is_coherent


def computing_training_testing_step(training, testing, s_current):
    # learning model
    model = NCCLR()
    model.learn(training)
    # testing model
    evaluate_pBox = model.evaluate(testing.data, ncc_s_param=s_current)
    # inference constraint satisfaction problem
    target_function = partial(parallel_prediction_csp, model, testing.data)
    acc_comp_var = POOL.map(target_function, enumerate(evaluate_pBox))
    # computing correctness and completeness on testing data set
    avg_cv_correctness, avg_cv_completeness, inc_coherent = 0, 0, 0
    for correctness, completeness, is_coherent in acc_comp_var:
        avg_cv_correctness += correctness
        avg_cv_completeness += completeness
        inc_coherent += is_coherent

    nb_testing = len(testing.data)
    logger.info(
        "Computing train/test (s_current, correctness, completeness, nb_testing, nb_coherent) (%s, %s, %s, %s, %s)",
        s_current, avg_cv_correctness, avg_cv_completeness, nb_testing, inc_coherent
    )
    return avg_cv_correctness / nb_testing, avg_cv_completeness / nb_testing


def get_cr_cp(s_current, training, k_splits_cv, seed_kfold):
    cv_kFold = k_fold_cross_validation(training, k_splits_cv, randomise=True, random_seed=seed_kfold)

    avg_cv_correctness, avg_cv_completeness = 0, 0
    for set_train, set_test in cv_kFold:
        cr, cp = computing_training_testing_step(set_train, set_test, s_current)
        avg_cv_correctness += cr
        avg_cv_completeness += cp
    avg_cv_correctness = avg_cv_correctness / k_splits_cv
    avg_cv_completeness = avg_cv_completeness / k_splits_cv
    return avg_cv_correctness, avg_cv_completeness


def computing_best_min_s_cross_validation(in_path,
                                          out_path=".",
                                          type_distance_s=TypeDistance.CORRECTNESS,
                                          k_splits_cv=10,
                                          min_discretization=5,
                                          max_discretization=7,
                                          nb_points_into_curve=3,
                                          iplot_evolution_s=True,
                                          SEED_SCIENTIFIC=1234):
    assert os.path.exists(in_path), "Without dataset, it not possible to computing"
    assert os.path.exists(out_path), "Directory for putting results does not exist"

    logger.info('Training dataset (%s, %s, %s)', in_path, SEED_SCIENTIFIC, type_distance_s.name)
    logger.info('Parameters (min_disc, max_disc, k_split_cv, nb_points_into_curve) (%s, %s, %s, %s)',
                min_discretization, max_discretization, k_splits_cv, nb_points_into_curve)

    random.seed(SEED_SCIENTIFIC)
    overall_accuracy = dict()
    fnt_distance = type_distance_s.value[0]
    name_type_distance = type_distance_s.name.lower()
    dataset_name = os.path.basename(in_path)

    for nb_intervals in range(min_discretization, max_discretization):
        logger.info("Starting discretization %s", nb_intervals)

        seed = generate_seeds(1)[0]
        seed_2ndkfcv = generate_seeds(k_splits_cv)
        logger.info("Seed generated for system is (%s, %s).", seed, seed_2ndkfcv)
        logger.info("Number interval for discreteness %s", nb_intervals)

        # Load dataset and discretization
        dataArff = ArffFile()
        dataArff.load(in_path)
        dataArff.discretize(discmet="eqfreq", numint=nb_intervals)
        learning_kfcv = k_fold_cross_validation(dataArff, k_splits_cv, randomise=True, random_seed=seed)

        # save partial performance correctness/completeness
        avg_kfold_correctness, avg_kfold_completeness = np.array([]), np.array([])
        avg_accuracy = dict()
        key_interval = str(nb_intervals)
        for training, testing in learning_kfcv:
            kfold = len(avg_kfold_correctness)
            # args for calculate s-min and s-max from cross-validation
            args_get_cr_cp = {"training": training,
                              "k_splits_cv": k_splits_cv,
                              "seed_kfold": seed_2ndkfcv[kfold]}

            # find s-min and s-max methods
            min_s, min_s_corr, min_s_comp = find_min_or_max(2, TypeMeasure.COMPLETENESS, get_cr_cp, args_get_cr_cp)
            max_s, max_s_corr, max_s_comp = find_min_or_max(min_s, TypeMeasure.CORRECTNESS, get_cr_cp, args_get_cr_cp)
            middle_s = round((min_s + max_s) / 2, 2)
            avg_accuracy[key_interval] = dict()

            # init values into accuracy dict
            avg_accuracy[key_interval][min_s] = dict({'corr': min_s_corr, 'comp': min_s_comp})
            avg_accuracy[key_interval][max_s] = dict({'corr': max_s_corr, 'comp': max_s_comp})
            middle_s_corr, middle_s_comp = get_cr_cp(middle_s, **args_get_cr_cp)
            avg_accuracy[key_interval][middle_s] = dict({'corr': middle_s_corr, 'comp': middle_s_comp})

            for _ in range(nb_points_into_curve):
                new_s = get_s_values(avg_accuracy[key_interval], fnt_distance)
                avg_cv_correctness, avg_cv_completeness = get_cr_cp(new_s, **args_get_cr_cp)
                avg_accuracy[key_interval][new_s] = dict({'corr': avg_cv_correctness, 'comp': avg_cv_completeness})

            # plot results k-fold learning
            if iplot_evolution_s:
                plot_save_results(results_dict=avg_accuracy,
                                  dataset_name=dataset_name,
                                  file_name=dataset_name + "_fold_" + str(kfold) + "_disc_" + str(nb_intervals),
                                  criterion=name_type_distance,
                                  out_root=out_path)

            # computing correctness with s-optimal
            s_optimal = min_s
            avg_validation_cr, avg_validation_cp = computing_training_testing_step(training, testing, s_optimal)
            logger.info("avg. cross-validation correctness/completeness s-optimal (%s, %s, %s, %s)",
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
                                      SEED_SCIENTIFIC=seed_random_experiment)
