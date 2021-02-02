from classifip.evaluation import k_fold_cross_validation
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger
from classifip.dataset.arff import ArffFile
from classifip.models.mlc import nccbr
from classifip.models.mlc import knnnccbr
from classifip.models.mlc.mlcncc import MLCNCC
import sys, os, random, csv, numpy as np, copy

sys.path.append("..")
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
from mlc_manager import ManagerWorkers, __create_dynamic_class
from mlc_common import init_dataset
from mlc_metrics_perf import MetricsPerformances


def skeptical_prediction(pid, tasks, queue, results, class_model, class_model_challenger=None):
    try:
        model_outer = __create_dynamic_class(class_model_challenger, has_imprecise_marginal=True)
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
            del training['missing_pct']

            model_outer.learn(**training)
            model_exact.learn(**training)
            while True:
                task = tasks.get()
                if task is None:
                    break

                # naive and improvement exact maximality inference
                skeptical_inference = model_exact.evaluate(**task['kwargs'])[0] \
                    if task['do_inference_exact'] \
                    else [-1] * len(task['y_test'])

                # outer-approximation binary relevance
                set_prob_marginal = model_outer.evaluate(**task['kwargs'])[0]
                # precise and e-precise inference with s equal 0, epsilon > 0.0
                task['kwargs']['ncc_s_param'] = 0.0
                prec_prob_marginal = model_outer.evaluate(**task['kwargs'])[0]

                outer_inference = set_prob_marginal.multilab_dom()
                precise_inference = prec_prob_marginal.multilab_dom()
                probabilities_yi_eq_1 = prec_prob_marginal.scores[:, 0].copy()

                # print partial prediction results
                print("(pid, skeptical, outer, precise, ground-truth, probabilities_yi_eq_1) ",
                      pid, len(skeptical_inference), outer_inference, precise_inference,
                      task['y_test'], probabilities_yi_eq_1, flush=True)
                results.append(dict({'skeptical': skeptical_inference,
                                     'outer': outer_inference,
                                     'precise': precise_inference,
                                     'y_eq_1_probabilities': probabilities_yi_eq_1,
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
                                    do_inference_exact,
                                    ncc_imprecise,
                                    disc_key_level,
                                    manager,
                                    metrics):
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
        y_br_skeptical = prediction['outer']
        y_precise = prediction['precise']
        y_eq_1_precise_probs = prediction['y_eq_1_probabilities']

        metrics.compute_metrics_performance(y_true, y_skeptical_exact,
                                            y_br_skeptical, y_precise,
                                            y_eq_1_precise_probs, nb_tests,
                                            str(ncc_imprecise), disc_key_level)

    manager.restartResults()


def cv10fold_br_vs_ibr(splits, s_ncc, disc_time, manager, metrics, logger, nb_labels, do_inference_exact,
                       missing_pct, noise_label_pct, noise_label_type, noise_label_prob):
    for idx_fold, (training, testing) in enumerate(splits):
        logger.info("Splits %s train %s", len(training.data), training.data[0][1:4])
        logger.info("Splits %s test %s", len(testing.data), testing.data[0][1:4])
        computing_training_testing_step(training,
                                        testing,
                                        missing_pct,
                                        noise_label_pct,
                                        noise_label_type,
                                        noise_label_prob,
                                        nb_labels,
                                        do_inference_exact,
                                        ncc_imprecise=s_ncc,
                                        disc_key_level=disc_time,
                                        manager=manager,
                                        metrics=metrics)
        logger.debug("Partial-fold_step (acc, ich_out) (%s, %s)",
                     metrics.score_hamming[disc_time][str(s_ncc)],
                     metrics.ich_iid_skeptic[disc_time][str(s_ncc)])


def cv10x10fold_br_vs_ibr(in_path, remove_features, scaling,
                          logger, nb_kFold, discretization, do_inference_exact,
                          manager, metrics, writer, seed, file_csv, missing_pct,
                          noise_label_pct, noise_label_type, noise_label_prob,
                          min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
    data_learning, nb_labels = init_dataset(in_path, remove_features, scaling)
    data_learning.discretize(discmet="eqfreq", numint=discretization)

    for time in range(nb_kFold):  # 10-10 times cross-validation
        logger.info("Number interval for discreteness and labels (%1d, %1d)." % (discretization, nb_labels))
        cv_kfold = k_fold_cross_validation(data_learning, nb_kFold, randomise=True, random_seed=seed[time])

        splits_s = list([])
        for training, testing in cv_kfold:
            # making a clone because it send the same address memory
            splits_s.append((training.make_clone(), testing.make_clone()))
            logger.info("Splits %s train %s", len(training.data), training.data[0][1:4])
            logger.info("Splits %s test %s", len(testing.data), testing.data[0][1:4])

        disc_time = str(discretization) + "-" + str(time)
        metrics.init_sub_level(disc_time)
        for s_ncc in np.arange(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
            str_s_ncc = str(s_ncc)
            metrics.init_level_imprecision(str_s_ncc, disc_time)
            cv10fold_br_vs_ibr(splits_s, s_ncc, disc_time, manager, metrics, logger, nb_labels, do_inference_exact,
                               missing_pct, noise_label_pct, noise_label_type, noise_label_prob)

            _partial_saving = metrics.generate_row_line(str_s_ncc, time, nb_kFold, sub_level=disc_time)
            _partial_saving.insert(0, str(discretization))
            writer.writerow(_partial_saving)
            file_csv.flush()
            logger.debug("Partial-ncc_step (disc, s, time, ich_skep, cph_skep, ich_out, "
                         "cph_out, acc, jacc, ich_reject, cph_reject, jacc_reject) (%s, %s, %s, %s)",
                         disc_time, s_ncc, time, metrics)


def re_sampling_with_pct_train(in_path, logger, nb_resampling, discretization, do_inference_exact,
                               manager, metrics, writer, file_csv, missing_pct,
                               noise_label_pct, noise_label_type, noise_label_prob,
                               min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
    logger.info("Number interval for discreteness (%1d)." % discretization)
    for pct_training in np.arange(10, 100, 10):
        logger.info("Percentage of training set: %s.", pct_training)

        for resampling in range(nb_resampling):
            # Loading data set
            in_path_train = in_path % ("train", resampling + 1, int(pct_training))
            in_path_test = in_path % ("test", resampling + 1, int(pct_training))
            logger.info("Evaluate training/test data set: (%s, %s).", in_path_train, in_path_test)
            data_training, nb_labels = init_dataset(in_path_train, None, False)
            data_test, _ = init_dataset(in_path_test, None, False)
            # discretization global data set
            data_training.data = [row + ["T"] for row in data_training.data]
            data_test.data = [row + ["V"] for row in data_test.data]
            data_global = ArffFile()
            data_global.attribute_data = data_training.attribute_data.copy()
            data_global.attribute_types = data_training.attribute_types.copy()
            data_global.data = copy.deepcopy(data_training.data + data_test.data)
            data_global.attributes = copy.copy(data_training.attributes)
            data_global.discretize(discmet="eqfreq", numint=discretization)
            # recopy the training discretized data set
            data_training.attribute_data = data_global.attribute_data.copy()
            data_training.attribute_types = data_global.attribute_types.copy()
            data_training.attributes = copy.copy(data_global.attributes)
            data_training.data = [row[:-1] for row in data_global.data if row[-1] == "T"]
            # recopy the test discretized data set
            data_test.attribute_data = data_global.attribute_data.copy()
            data_test.attribute_types = data_global.attribute_types.copy()
            data_test.attributes = copy.copy(data_global.attributes)
            data_test.data = [row[:-1] for row in data_global.data if row[-1] == "V"]

            # putting in 1 split training and test data set
            splits_s = [(data_training, data_test)]
            disc_re_pct = str(discretization) + "-" + str(resampling) + "-" + str(pct_training)
            metrics.init_sub_level(disc_re_pct)
            for s_ncc in np.arange(min_ncc_s_param, max_ncc_s_param, step_ncc_s_param):
                metrics.init_level_imprecision(str(s_ncc), disc_re_pct)
                # training and test a partition of sampling
                cv10fold_br_vs_ibr(splits_s, s_ncc, disc_re_pct, manager, metrics, logger,
                                   nb_labels, do_inference_exact,
                                   missing_pct, noise_label_pct, noise_label_type, noise_label_prob)

                _partial_saving = metrics.generate_row_line(str(s_ncc), resampling, 1, sub_level=disc_re_pct)
                _partial_saving.insert(0, str(discretization))
                _partial_saving.insert(0, str(pct_training))
                writer.writerow(_partial_saving)
                file_csv.flush()
                logger.debug("Partial-ncc_step (disc, s, time, ich_skep, cph_skep, ich_out, "
                             "cph_out, acc, jacc, ich_reject, cph_reject, jacc_reject) (%s, %s, %s, %s)",
                             disc_re_pct, s_ncc, resampling, metrics)


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
                                  do_inference_exact=False,
                                  is_resampling=False):
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
    :param is_resampling: if re-sampling of test and training data sets generated beforehand

    ...note::
        TODO: Bug when the missing percentage is higher (90%) to fix.

    """
    if not is_resampling:
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
    logger.info("(do_inference_exact)  (%s)", do_inference_exact)

    # Seeding a random value for k-fold top learning-testing data
    if seed is None:
        seed = [random.randrange(sys.maxsize) for _ in range(nb_kFold)]
    logger.debug("[FIRST-STEP-SEED] SEED: %s", seed)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process, fun_prediction=skeptical_prediction)
    _class_model_challenger = "classifip.models.mlc.nccbr.NCCBR"
    manager.executeAsync(class_model="classifip.models.mlc.exactncc.MLCNCCExact",
                         class_model_challenger=_class_model_challenger)
    logger.info('Classifier binary relevant (%s)', _class_model_challenger)

    # c constant for abstained multilabel
    list_c_spe = [round((num + 1) * .05, 2) for num in range(10)]
    list_c_par = [round((num + 1) * .1, 2) for num in range(10)]
    # metrics performances
    metrics = MetricsPerformances(do_inference_exact=do_inference_exact,
                                  epsilon_rejects=epsilon_rejects,
                                  list_constants_spe=list_c_spe,
                                  list_constants_par=list_c_par)

    min_discretize, max_discretize = 5, 7
    for nb_disc in range(min_discretize, max_discretize):
        if not is_resampling:  # it is cross-validation
            cv10x10fold_br_vs_ibr(in_path, remove_features, scaling,
                                  logger, nb_kFold, nb_disc, do_inference_exact,
                                  manager, metrics, writer, seed, file_csv, missing_pct,
                                  noise_label_pct, noise_label_type, noise_label_prob,
                                  min_ncc_s_param, max_ncc_s_param, step_ncc_s_param)
        else:
            re_sampling_with_pct_train(in_path, logger, nb_kFold, nb_disc, do_inference_exact,
                                       manager, metrics, writer, file_csv, missing_pct,
                                       noise_label_pct, noise_label_type, noise_label_prob,
                                       min_ncc_s_param, max_ncc_s_param, step_ncc_s_param)

    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s",
                 metrics.ich_iid_skeptic, metrics.cph_iid_skeptic,
                 metrics.score_hamming,
                 metrics.ich_spe_partial, metrics.cph_spe_partial,
                 metrics.ich_par_partial, metrics.cph_par_partial,
                 metrics.spe_partial_score, metrics.par_partial_score,
                 metrics.ich_reject, metrics.cph_reject)


# in_path = ".../Downloads/datasets_mlc/emotions.arff"
in_path = ".../Downloads/emotions_%s_%i_%i.arff"
out_path = ".../Downloads/results_emotions.csv"
experiments_binr_vs_imprecise(in_path=in_path,
                              out_path=out_path,
                              nb_process=1,
                              missing_pct=0.4,
                              noise_label_pct=0.0, noise_label_type=-1, noise_label_prob=0.2,
                              min_ncc_s_param=0.5, max_ncc_s_param=6, step_ncc_s_param=1,
                              epsilon_rejects=[0.05, 0.15, 0.25, 0.35, 0.45],
                              do_inference_exact=False,
                              remove_features=["image_name"],
                              is_resampling=True,
                              nb_kFold=100)
