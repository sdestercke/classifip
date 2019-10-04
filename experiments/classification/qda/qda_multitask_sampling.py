from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from classifip.evaluation.measures import u65, u80
from classifip.utils import create_logger, normalize_minmax
import sys, random, os, csv, numpy as np, pandas as pd
from qda_common import __factory_model, generate_seeds, generate_sample_cross_validation
from qda_manager import computing_training_testing_step, ManagerWorkers

## Server env:
# export LD_PRELOAD=/usr/local/MATLAB/R2018b/sys/os/glnxa64/libstdc++.so.6.0.22
QPBB_PATH_SERVER = ['/home/lab/ycarranz/QuadProgBB', '/opt/cplex128/cplex/matlab/x86-64_linux']


def performance_cv_accuracy_imprecise(in_path=None, model_type="ilda", ell_optimal=0.1, nb_process=2,
                                      lib_path_server=None, cv_n_fold=10, seeds=None, criterion="maximality"):
    assert os.path.exists(in_path), "Without training data, not testing"
    data = pd.read_csv(in_path)
    logger = create_logger("performance_cv_accuracy_imprecise", True)
    logger.info('Training dataset (%s, %s, %s, %s)', in_path, model_type, ell_optimal, criterion)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())
    avg_u65, avg_u80 = 0, 0
    seeds = generate_seeds(cv_n_fold) if seeds is None else seeds
    logger.info('Seeds used for accuracy %s', seeds)
    manager = ManagerWorkers(nb_process=nb_process, criterion=criterion)
    manager.executeAsync(model_type, lib_path_server)
    for time in range(cv_n_fold):
        kf = KFold(n_splits=cv_n_fold, random_state=seeds[time], shuffle=True)
        mean_u65, mean_u80 = 0, 0
        for idx_train, idx_test in kf.split(y):
            logger.info("Splits train %s", idx_train)
            logger.info("Splits test %s", idx_test)
            X_cv_train, y_cv_train = X[idx_train], y[idx_train]
            X_cv_test, y_cv_test = X[idx_test], y[idx_test]
            mean_u65, mean_u80 = computing_training_testing_step(X_cv_train, y_cv_train, X_cv_test, y_cv_test,
                                                                 ell_optimal, manager, mean_u65, mean_u80)
            logger.debug("Partial-kfold (%s, %s, %s, %s)", ell_optimal, time, mean_u65, mean_u80)
        logger.info("Time, seed, u65, u80 (%s, %s, %s, %s)", time, seeds[time],
                    mean_u65 / cv_n_fold, mean_u80 / cv_n_fold)
        avg_u65 += mean_u65 / cv_n_fold
        avg_u80 += mean_u80 / cv_n_fold
    manager.poisonPillTraining()
    logger.debug("total-ell (%s, %s, %s, %s)", in_path, ell_optimal, avg_u65 / cv_n_fold, avg_u80 / cv_n_fold)


def computing_best_imprecise_mean(in_path=None, out_path=None, cv_nfold=10, model_type="ilda", test_size=0.4,
                                  from_ell=0.1, to_ell=1.0, by_ell=0.1, seeds=None, lib_path_server=None,
                                  nb_process=2, n_sampling=10, skip_n_sample=0, criterion="maximality", scaling=False):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean_sampling", True)
    logger.info('Training dataset (%s, %s, %s)', in_path, model_type, criterion)
    logger.info('Parameters (size, ells, nbProcess, sampling, nSkip) (%s, %s, %s, %s, %s, %s, %s)', test_size, from_ell,
                to_ell, by_ell, nb_process, n_sampling, skip_n_sample)
    data = pd.read_csv(in_path, header=None)
    X = data.iloc[:, :-1].values
    if scaling: X = normalize_minmax(X)
    y = np.array(data.iloc[:, -1].tolist())

    # Seed for get back up if process is killed
    seeds = generate_seeds(n_sampling) if seeds is None else seeds
    logger.debug("MODEL: %s, SEED: %s", model_type, seeds)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)
    manager = ManagerWorkers(nb_process=nb_process, criterion=criterion)
    manager.executeAsync(model_type, lib_path_server)
    acc_u80, acc_u65 = dict(), dict()
    for sampling in range(min(n_sampling, len(seeds))):
        X_learning, X_testing, y_learning, y_testing = \
            train_test_split(X, y, test_size=test_size, random_state=seeds[sampling])
        logger.info("Splits %s learning %s", sampling, y_learning)
        logger.info("Splits %s testing %s", sampling, y_testing)

        # n-Skipping sampling and reboot parameter from_ell to 0.01 next sampling
        if skip_n_sample != 0 and sampling > skip_n_sample: from_ell = 0.01
        # n-Skipping sampling testing (purpose for parallel computing)
        if sampling >= skip_n_sample:
            ell_u65, ell_u80 = dict(), dict()
            splits = generate_sample_cross_validation(y_learning, cv_nfold, 2)

            for index, value in enumerate(splits):
                idx_train, idx_test = value
                logger.info("Sampling %s Splits %s train %s", sampling, index, idx_train)
                logger.info("Sampling %s Splits %s test %s", sampling, index, idx_test)

            for ell_current in np.arange(from_ell, to_ell, by_ell):
                ell_u65[ell_current], ell_u80[ell_current] = 0, 0
                logger.info("ELL_CURRENT %s", ell_current)
                for idx_train, idx_test in splits:
                    logger.info("Splits class train %s", y_learning[idx_train])
                    logger.info("Splits class test %s", y_learning[idx_test])
                    X_cv_train, y_cv_train = X_learning[idx_train], y_learning[idx_train]
                    X_cv_test, y_cv_test = X_learning[idx_test], y_learning[idx_test]
                    # Computing accuracy testing for cross-validation step
                    ell_u65[ell_current], ell_u80[ell_current] = \
                        computing_training_testing_step(X_cv_train, y_cv_train, X_cv_test, y_cv_test, ell_current,
                                                        manager, ell_u65[ell_current], ell_u80[ell_current])
                    logger.info("Partial-kfold (%s, %s, %s)", ell_current, ell_u65[ell_current], ell_u80[ell_current])

                ell_u65[ell_current] = ell_u65[ell_current] / cv_nfold
                ell_u80[ell_current] = ell_u80[ell_current] / cv_nfold
                writer.writerow([ell_current, sampling, ell_u65[ell_current], ell_u80[ell_current]])
                file_csv.flush()
                logger.debug("Partial-ell-sampling (%s, %s, %s, %s)", ell_current, sampling, ell_u65, ell_u80)
            logger.debug("Total-ell-sampling (%s, %s, %s, %s)", in_path, sampling, ell_u65, ell_u80)

            # Computing optimal ells for using in testing step
            acc_ellu80 = max(ell_u80.values())
            acc_ellu65 = max(ell_u65.values())
            ellu80_opts = [k for k, v in ell_u80.items() if v == acc_ellu80]
            ellu65_opts = [k for k, v in ell_u65.items() if v == acc_ellu65]
            acc_u65[sampling], acc_u80[sampling] = 0, 0
            n_ell80_opts, n_ell65_opts = len(ellu80_opts), len(ellu65_opts)

            for ellu80_opt in ellu80_opts:
                logger.info("ELL_OPTIMAL_SAMPLING_U80 %s", ellu80_opt)
                _, acc_u80[sampling] = \
                    computing_training_testing_step(X_learning, y_learning, X_testing, y_testing, ellu80_opt,
                                                    manager, 0, acc_u80[sampling])

            for ellu65_opt in ellu65_opts:
                logger.info("ELL_OPTIMAL_SAMPLING_U65 %s", ellu65_opt)
                acc_u65[sampling], _ = \
                    computing_training_testing_step(X_learning, y_learning, X_testing, y_testing, ellu65_opt,
                                                    manager, acc_u65[sampling], 0)

            acc_u65[sampling] = acc_u65[sampling] / n_ell65_opts
            acc_u80[sampling] = acc_u80[sampling] / n_ell80_opts
            writer.writerow([-999, sampling, acc_u65[sampling], acc_u80[sampling]])
            file_csv.flush()
            logger.debug("Partial-ell-2step (%s, %s, %s, %s)", -999, ellu80_opts, acc_u65[sampling], acc_u80[sampling])

    writer.writerow([-9999, -9, np.mean(list(acc_u65.values())), np.mean(list(acc_u80.values()))])
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Total-accuracy (%s, %s, %s)", in_path, acc_u65, acc_u80)
    logger.debug("Total-avg-accuracy (%s, %s, %s)", in_path, np.mean(list(acc_u65.values())),
                 np.mean(list(acc_u80.values())))


in_path = sys.argv[1]
out_path = sys.argv[2]
# QPBB_PATH_SERVER = []  # executed in host
computing_best_imprecise_mean(in_path=in_path, out_path=out_path, model_type="ilda",
                              from_ell=0.01, to_ell=5.5, by_ell=0.01,  # seeds=XXX, skip_n_sample=X,
                              lib_path_server=QPBB_PATH_SERVER, nb_process=1)  # , n_sampling=1)

# in_path = sys.argv[1]
# ell_optimal = float(sys.argv[2])
# QPBB_PATH_SERVER = []  # executed in host
# performance_cv_accuracy_imprecise(in_path=in_path, ell_optimal=ell_optimal, model_type="ilda",
#                                   lib_path_server=QPBB_PATH_SERVER, nb_process=1)
