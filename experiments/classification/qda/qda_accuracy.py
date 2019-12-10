from classifip.evaluation.measures import u65, u80
from classifip.dataset.uci_data_set import export_data_set
from qda_common import __factory_model, generate_seeds, __factory_model_precise
from sklearn.model_selection import KFold
from classifip.utils import create_logger, normalize_minmax
import numpy as np, pandas as pd, sys, os, csv, ntpath
from sklearn.model_selection import train_test_split
from qda_manager import computing_training_testing_step, ManagerWorkers

# Server env:
# export LD_PRELOAD=/usr/local/MATLAB/R2018b/sys/os/glnxa64/libstdc++.so.6.0.22
# export OPENBLAS_MAIN_FREE=1
# export OPENBLAS_NUM_THREADS=1
QPBB_PATH_SERVER = ['/home/lab/ycarranz/QuadProgBB', '/opt/cplex128/cplex/matlab/x86-64_linux']


# QPBB_PATH_SERVER = ['/volper/users/ycarranz/QuadProgBB', '/volper/users/ycarranz/cplex128/cplex/matlab/x86-64_linux']


def dataset_to_Xy(in_data_path, scaling=False, idx_label=-1):
    data = pd.read_csv(in_data_path, header=None)
    X = data.iloc[:, :idx_label].values
    if scaling: X = normalize_minmax(X)
    y = np.array(data.iloc[:, idx_label].tolist())
    return X, y


def performance_accuracy_hold_out(in_path=None, model_type="ilda", ell_optimal=0.1, lib_path_server=None,
                                  seeds=None, DEBUG=False, scaling=False):
    assert os.path.exists(in_path), "Without training data, cannot performing cross hold-out accuracy"
    logger = create_logger("performance_accuracy_hold_out", True)
    logger.info('Training dataset (%s, %s, %s)', in_path, model_type, ell_optimal)
    X, y = dataset_to_Xy(in_path, scaling=scaling)

    seeds = generate_seeds(cv_n_fold) if seeds is None else seeds
    logger.info('Seeds used for accuracy %s', seeds)
    n_time = len(seeds)
    mean_u65, mean_u80 = 0, 0
    model = __factory_model(model_type, init_matlab=True, add_path_matlab=lib_path_server, DEBUG=DEBUG)
    for k in range(0, n_time):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seeds[k])
        model.learn(X=X_cv_train, y=y_cv_train, ell=ell_optimal)
        sum_u65, sum_u80 = 0, 0
        n_test, _ = X_test.shape
        for i, test in enumerate(X_test):
            evaluate, _ = lqa.evaluate(test)
            logger.debug("(testing, ell_current, prediction, ground-truth) (%s, %s, %s, %s)",
                         i, ell_optimal, evaluate, y_test[i])
            if y_test[i] in evaluate:
                sum_u65 += u65(evaluate)
                sum_u80 += u80(evaluate)
        logger.debug("Partial-kfold (%s, %s, %s, %s)", ell_current, k, sum_u65 / n_test, sum_u80 / n_test)
        mean_u65 += sum_u65 / n_test
        mean_u80 += sum_u80 / n_test
    mean_u65 = mean_u65 / n_time
    mean_u80 = mean_u80 / n_time
    logger.debug("Total-ell (%s, %s, %s, %s)", in_path, ell_optimal, mean_u65, mean_u80)


def performance_cv_accuracy_imprecise(in_path=None, model_type="ilda", ell_optimal=0.1, scaling=False,
                                      lib_path_server=None, cv_n_fold=10, seeds=None, nb_process=10):
    assert os.path.exists(in_path), "Without training data, cannot performing cross validation accuracy"
    logger = create_logger("performance_cv_accuracy_imprecise", True)
    logger.info('Training dataset (%s, %s, %s)', in_path, model_type, ell_optimal)
    X, y = dataset_to_Xy(in_path, scaling=scaling)

    avg_u65, avg_u80 = 0, 0
    seeds = generate_seeds(cv_n_fold) if seeds is None else seeds
    logger.info('Seeds used for accuracy %s', seeds)

    manager = ManagerWorkers(nb_process=nb_process)
    manager.executeAsync(model_type, lib_path_server)
    for time in range(cv_n_fold):
        kf = KFold(n_splits=cv_n_fold, random_state=seeds[time], shuffle=True)
        mean_u65, mean_u80 = 0, 0
        for idx_train, idx_test in kf.split(y):
            mean_u65, mean_u80, _ = computing_training_testing_step(X[idx_train], y[idx_train], X[idx_test],
                                                                    y[idx_test], ell_optimal, manager,
                                                                    mean_u65, mean_u80)
            logger.debug("Partial-kfold (%s, %s, %s, %s)", ell_optimal, time, mean_u65, mean_u80)
        logger.info("Time, seed, u65, u80 (%s, %s, %s, %s)", time, seeds[time], mean_u65 / cv_n_fold,
                    mean_u80 / cv_n_fold)
        avg_u65 += mean_u65 / cv_n_fold
        avg_u80 += mean_u80 / cv_n_fold
    manager.poisonPillTraining()
    logger.debug("Total-ell (%s, %s, %s, %s)", in_path, ell_optimal, avg_u65 / cv_n_fold, avg_u80 / cv_n_fold)


def performance_accuracy_of_n_noise_corrupted_test_data(in_train_path=None, in_tests_path=None,
                                                        model_type_precise='lda', model_type_imprecise='ilda',
                                                        ell_optimal=0.1, scaling=False, lib_path_server=None,
                                                        nb_process=10):
    assert os.path.exists(in_train_path), "Without training data, cannot create to model"
    assert isinstance(in_tests_path, list), "Without training data, cannot performing accuracy"

    logger = create_logger("performance_accuracy_noise_corrupted_test_data", True)
    logger.info('Training dataset (%s, %s, %s)', in_train_path, model_type_imprecise, ell_optimal)
    X_train, y_train = dataset_to_Xy(in_train_path, scaling=scaling)

    manager = ManagerWorkers(nb_process=nb_process)
    manager.executeAsync(model_type_imprecise, lib_path_server)
    model_precise = __factory_model_precise(model_type_precise, store_covariance=True)
    model_precise.fit(X_train, y_train)
    versus = model_type_imprecise + "_vs_" + model_type_precise
    file_csv = open("results_" + versus + "_noise_accuracy.csv", 'w')
    writer = csv.writer(file_csv)
    accuracies = dict({})
    for in_test_path in in_tests_path:
        X_test, y_test = dataset_to_Xy(in_test_path, scaling=scaling)
        _u65, _u80, _ = computing_training_testing_step(X_train, y_train, X_test, y_test, ell_optimal, manager, 0, 0)
        evaluate = model_precise.predict(X_test)
        acc = sum(1 for k, j in zip(evaluate, y_test) if k == j) / len(y_test)
        logger.debug("accuracy-in_test_path (%s, %s, %s, %s, %s)", in_test_path, ell_optimal, _u65, _u80, acc)
        accuracies[ntpath.basename(in_test_path)] = [ell_optimal, _u65, _u80, acc]
        writer.writerow([ntpath.basename(in_test_path), ell_optimal, _u65, _u80, acc])
        file_csv.flush()
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("finish-accuracy-noise-corrupted_test %s", accuracies)


def performance_accuracy_noise_corrupted_test_data(in_train_paths=None, in_tests_paths=None,
                                                   model_type_precise='lda', model_type_imprecise='ilda',
                                                   ell_optimal=0.1, scaling=False, lib_path_server=None,
                                                   nb_process=10):
    assert isinstance(in_train_paths, list), "Without training data, cannot create to model"
    assert isinstance(in_tests_paths, list), "Without training data, cannot performing accuracy"

    logger = create_logger("performance_accuracy_noise_corrupted_test_data", True)
    logger.info('Training dataset (%s, %s, %s)', in_train_paths, model_type_imprecise, ell_optimal)

    manager = ManagerWorkers(nb_process=nb_process)
    manager.executeAsync(model_type_imprecise, lib_path_server)
    versus = model_type_imprecise + "_vs_" + model_type_precise
    file_csv = open("results_" + versus + "_noise_accuracy.csv", 'w')
    writer = csv.writer(file_csv)
    model_precise = __factory_model_precise(model_type_precise, store_covariance=True)
    for in_train_path in in_train_paths:
        X_train, y_train = dataset_to_Xy(in_train_path, scaling=scaling)
        model_precise.fit(X_train, y_train)
        accuracies = dict({})
        for in_test_path in in_tests_paths:
            X_test, y_test = dataset_to_Xy(in_test_path, scaling=scaling)
            _u65, _u80, _set = computing_training_testing_step(X_train, y_train, X_test, y_test,
                                                               ell_optimal, manager, 0, 0, 0)
            evaluate = model_precise.predict(X_test)
            _acc = sum(1 for k, j in zip(evaluate, y_test) if k == j) / len(y_test)
            logger.debug("accuracy-in_test_path (%s, %s, %s, %s, %s, %s)", ntpath.basename(in_train_path),
                         ntpath.basename(in_test_path), ell_optimal, _u65, _u80, _acc)
            accuracies[ntpath.basename(in_test_path)] = [ell_optimal, _u65, _u80, _set, _acc]
            writer.writerow([ntpath.basename(in_train_path), ntpath.basename(in_test_path),
                             ell_optimal, _u65, _u80, _set, _acc])
            file_csv.flush()
        logger.debug("Partial-finish-accuracy-noise-corrupted_test %s: %s", ntpath.basename(in_train_path), accuracies)
    manager.poisonPillTraining()
    file_csv.close()
    logger.debug("Finish-accuracy-noise-corrupted_test")


ii_path = sys.argv[1]
ii_ell_optimal = float(sys.argv[2])
# QPBB_PATH_SERVER = []
performance_cv_accuracy_imprecise(in_path=ii_path, ell_optimal=ii_ell_optimal, model_type="ilda",
                                  lib_path_server=QPBB_PATH_SERVER, nb_process=2)

# experiments with noise testing data and noise distribution testing data
root = sys.argv[1]
method = sys.argv[2]
i_method = "i" + method
ii_ell_optimal = float(sys.argv[3])
ii_train_path = root + "dsyn01_train.csv"
ii_test_paths = []
for i in range(20):
    ii_test_paths.append(root + "noise_gamma/dsyn01_test_gamma_" + str((i + 1)) + ".csv")

for i in range(20):
    ii_test_paths.append(root + "noise_tau/dsyn01_test_tau_" + str((i + 1)) + ".csv")

for idx_epsilon in range(101):
    for repetition in range(10):
        ii_test_paths.append(root + "noise_distribution/dsyn01_test_mean_" +
                             str((idx_epsilon + 1)) + "_" + str(repetition) + ".csv")

performance_accuracy_noise_corrupted_test_data(ii_train_path, ii_test_paths, ell_optimal=ii_ell_optimal,
                                               model_type_precise=method, model_type_imprecise=i_method,
                                               lib_path_server=QPBB_PATH_SERVER, nb_process=2)

# experiments with 1-noise testing data and n-different training data
root = sys.argv[1]
method = sys.argv[2]
i_method = "i" + method
ii_ell_optimal = float(sys.argv[3])
ii_train_paths = []
ii_test_paths = []
sample_size = [10, 50, 100]

for _size in sample_size:
    for i in range(10):
        ii_train_paths.append(
            root + "dsyn01_trains_reptitions/dsyn01_all_data_" + str(_size) + "_" + str((i + 1)) + ".csv"
        )

for i in range(51):
    ii_test_paths.append(root + "noise_means/dsyn01_test_mean_" + str((i + 1)) + ".csv")

performance_accuracy_noise_corrupted_test_data(in_train_paths=ii_train_paths, in_tests_paths=ii_test_paths,
                                               model_type_precise=method, model_type_imprecise=i_method,
                                               ell_optimal=ii_ell_optimal, lib_path_server=QPBB_PATH_SERVER,
                                               nb_process=2)
