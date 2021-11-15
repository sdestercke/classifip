from classifip.evaluation import k_fold_cross_validation
from classifip.utils import create_logger
from classifip.dataset import arff
from classifip.models.mlc.mlcncc import MLCNCC
import sys, os, random, csv, numpy as np
from itertools import product

sys.path.append("..")
from mlc_manager import ManagerWorkers, __create_dynamic_class
from mlc_common import init_dataset
from mlc_metrics_perf import MetricsPerformances


def skeptical_prediction(pid, tasks, queue, results, class_model, class_model_challenger=None):
    try:
        model_skeptic = __create_dynamic_class(class_model, DEBUG=True)
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
                                nb_lassos_models=41,
                                min_gamma=0.01, max_gamma=1,
                                nb_process=int(training["nb_labels"]/2))
            while True:
                task = tasks.get()
                if task is None:
                    break

                # skeptical inference with binary relevance
                credal_set, precise_probability = model_skeptic.evaluate(**task['kwargs'])[0]
                skeptical_inference = credal_set.multilab_dom()

                # precise inference by label
                precise_inference = []
                probabilities_yi_eq_1 = []
                for prob_label in precise_probability:
                    precise_inference.append(prob_label.getmaximaldecision())
                    probabilities_yi_eq_1.append(prob_label.proba[1])

                # print("==================================================", flush=True)
                # print("----->", skeptical_inference, precise_inference, flush=True)
                # print("----->", credal_set, flush=True)
                # print("----->", probabilities_yi_eq_1, flush=True)
                # print("==================================================", flush=True)

                # print partial prediction results
                print("(pid, skeptical, precise, ground-truth, probabilities_yi_eq_1) ",
                      pid, skeptical_inference, precise_inference, task['y_test'],
                      probabilities_yi_eq_1, flush=True)
                results.append(dict({'skeptical': skeptical_inference,
                                     'precise': precise_inference,
                                     'y_eq_1_probabilities': np.array(probabilities_yi_eq_1),
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
                                    time,
                                    sub_level,
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
        manager.addTask({
            'kwargs': {'test_dataset': [test[:-nb_labels]]},
            'y_test': test[-nb_labels:]
        })
    manager.poisonPillWorkers()
    manager.joinTraining()  # wait all process for computing results
    # Recovery all inference data of all parallel process
    nb_tests = len(test_data_set.data)
    shared_results = manager.getResults()
    for prediction in shared_results:
        y_true = np.array(prediction['ground_truth'], dtype=np.int)
        y_br_skeptical = prediction['skeptical']
        y_precise = prediction['precise']
        y_eq_1_precise_probs = prediction['y_eq_1_probabilities']

        metrics.compute_metrics_performance(y_true, None,
                                            y_br_skeptical, y_precise,
                                            y_eq_1_precise_probs, nb_tests,
                                            param_imprecision=str(time),
                                            sub_level=sub_level)

    manager.restartResults()


def cv10fold_br_vs_ibr(logger, splits_s, time, nb_labels, manager, metrics, sub_level,
                       missing_pct, noise_label_pct, noise_label_type, noise_label_prob):
    for idx_fold, (training, testing) in enumerate(splits_s):
        logger.info("[cv10fold] Splits %s train %s", len(training.data), training.data[0][1:4])
        logger.info("[cv10fold] Splits %s test %s", len(testing.data), testing.data[0][1:4])
        computing_training_testing_step(training,
                                        testing,
                                        missing_pct,
                                        noise_label_pct,
                                        noise_label_type,
                                        noise_label_prob,
                                        nb_labels,
                                        time=time,
                                        sub_level=sub_level,
                                        manager=manager,
                                        metrics=metrics)
        logger.debug("Partial-fold_step (acc, ich_skep) (%s, %s)",
                     metrics.score_hamming, metrics.ich_iid_skeptic)


def cv10x10fold_br_vs_ibr(logger, in_path, remove_features, scaling, nb_kFold,
                          seed, file_csv, writer, metrics, manager,
                          missing_pct, noise_label_pct, noise_label_type, noise_label_prob):
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

        metrics.init_level_imprecision(str(time))
        cv10fold_br_vs_ibr(logger, splits_s, time, nb_labels, manager, metrics, None,
                           missing_pct, noise_label_pct, noise_label_type, noise_label_prob)

        _partial_saving = metrics.generate_row_line(str(time), None, nb_kFold)
        del _partial_saving[1]  # remove time replicate
        writer.writerow(_partial_saving)
        file_csv.flush()
        logger.debug("Partial-s-k_step (time, ich_skep, cph_skep, acc, "
                     "ich_reject, cph_reject) (%s, %s, %s)", time,
                     metrics.score_hamming, metrics.ich_iid_skeptic)


def re_sampling_with_pct_train(logger, in_path, nb_resampling, file_csv, writer, metrics, manager,
                               missing_pct, noise_label_pct, noise_label_type, noise_label_prob):
    for pct_training in np.arange(10, 100, 10):
        logger.info("Percentage of training set: %s.", pct_training)
        re_pct = str(pct_training)
        metrics.init_sub_level(re_pct)
        for resampling in range(nb_resampling):
            # Loading data set
            in_path_train = in_path % ("train", resampling + 1, int(pct_training))
            in_path_test = in_path % ("test", resampling + 1, int(pct_training))
            logger.info("Evaluate training/test data set: (%s, %s).", in_path_train, in_path_test)
            data_training, nb_labels = init_dataset(in_path_train, None, False)
            data_test, _ = init_dataset(in_path_test, None, False)

            # putting in 1 split training and test data set
            splits_s = [(data_training, data_test)]
            # level gamma hyper-parameter
            metrics.init_level_imprecision(str(resampling), sub_level=re_pct)
            cv10fold_br_vs_ibr(logger, splits_s, resampling, nb_labels, manager, metrics, re_pct,
                               missing_pct, noise_label_pct, noise_label_type, noise_label_prob)

            _partial_saving = metrics.generate_row_line(str(resampling), resampling, 1, sub_level=re_pct)
            del _partial_saving[1]  # remove resampling replicate
            _partial_saving.insert(0, re_pct)
            writer.writerow(_partial_saving)
            file_csv.flush()
            logger.debug("Partial-s-k_step (time, hamming, ich_skep) (%s, %s, %s)", resampling,
                         metrics.score_hamming, metrics.ich_iid_skeptic)


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
                                  remove_features=None,
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
    :param remove_features: features not to take into account
    :param is_resampling: if re-sampling of test and training data sets generated beforehand

    """
    if not is_resampling:
        assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean", True)
    logger.info('Training dataset (%s, %s)', in_path, out_path)
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
    manager.executeAsync(class_model="classifip.models.mlc.logitbr.Logit_BR")

    # c constant for abstained multilabel
    list_c_spe = [(num + 1) * .05 for num in range(10)]
    list_c_par = [(num + 1) * .1 for num in range(10)]

    # metrics performances
    metrics = MetricsPerformances(do_inference_exact=False,
                                  epsilon_rejects=epsilon_rejects,
                                  list_constants_spe=list_c_spe,
                                  list_constants_par=list_c_par)

    if not is_resampling:
        cv10x10fold_br_vs_ibr(logger, in_path, remove_features, scaling, nb_kFold,
                              seed, file_csv, writer, metrics, manager,
                              missing_pct, noise_label_pct, noise_label_type, noise_label_prob)
    else:
        re_sampling_with_pct_train(logger, in_path, nb_kFold, file_csv, writer, metrics, manager,
                                   missing_pct, noise_label_pct, noise_label_type, noise_label_prob)

    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Results Final: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s",
                 metrics.ich_iid_skeptic, metrics.cph_iid_skeptic,
                 metrics.score_hamming,
                 metrics.ich_spe_partial, metrics.cph_spe_partial,
                 metrics.ich_par_partial, metrics.cph_par_partial,
                 metrics.spe_partial_score, metrics.par_partial_score,
                 metrics.ich_reject, metrics.cph_reject)


# in_path = "/Users/salmuz/Downloads/labels2.arff"
in_path = ".../emotions_%s_%i_%i.arff"
out_path = ".../results_emotions.csv"
experiments_binr_vs_imprecise(in_path=in_path,
                              out_path=out_path,
                              nb_process=1,
                              scaling=False,
                              missing_pct=0.0,
                              noise_label_pct=0.0, noise_label_type=-1, noise_label_prob=0.2,
                              epsilon_rejects=[0.05, 0.15, 0.25, 0.35, 0.45],
                              remove_features=["image_name"],
                              is_resampling=True,
                              nb_kFold=50)
