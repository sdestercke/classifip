from classifip.evaluation.measures import u65, u80
from qda_common import __factory_model_precise, __factory_model, generate_seeds
import numpy as np, pandas as pd, sys, time, os
from classifip.utils import create_logger, normalize_minmax
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

## Server env:
# export LD_PRELOAD=/usr/local/MATLAB/R2018b/sys/os/glnxa64/libstdc++.so.6.0.22
QPBB_PATH_SERVER = ['/home/lab/ycarranz/QuadProgBB', '/opt/cplex128/cplex/matlab/x86-64_linux']


def computing_precise_vs_imprecise(in_path=None, ell_optimal=0.1, cv_n_fold=10, seeds=None, lib_path_server=None,
                                   model_type_precise='lda', model_type_imprecise='ilda', scaling=True):
    data = export_data_set('iris.data') if in_path is None else pd.read_csv(in_path)
    logger = create_logger("computing_precise_vs_imprecise", True)
    logger.info('Training dataset and models (%s, %s, %s, %s)', in_path, model_type_precise,
                model_type_imprecise, ell_optimal)
    X = data.iloc[:, :-1].values
    if scaling: X = normalize_minmax(X)
    y = np.array(data.iloc[:, -1].tolist())
    seeds = generate_seeds(cv_n_fold) if seeds is None else seeds
    model_impr = __factory_model(model_type_imprecise, init_matlab=True, add_path_matlab=lib_path_server, DEBUG=False)
    model_prec = __factory_model_precise(model_type_precise, store_covariance=True)
    avg_imprecise, avg_precise, n_real_times = 0, 0, 0
    for time in range(cv_n_fold):
        kf = KFold(n_splits=cv_n_fold, random_state=seeds[time], shuffle=True)
        imprecise_mean, precise_mean, n_real_fold = 0, 0, 0
        for idx_train, idx_test in kf.split(y):
            X_cv_train, y_cv_train = X[idx_train], y[idx_train]
            X_cv_test, y_cv_test = X[idx_test], y[idx_test]
            model_impr.learn(X=X_cv_train, y=y_cv_train, ell=ell_optimal)
            model_prec.fit(X_cv_train, y_cv_train)
            n_real_tests, time_precise, time_imprecise = 0, 0, 0
            n_test, _ = X_cv_test.shape
            for i, test in enumerate(X_cv_test):
                evaluate_imp, _ = model_impr.evaluate(test)
                evaluate = model_prec.predict([test])
                if len(evaluate_imp) > 1:
                    n_real_tests += 1
                    if y_cv_test[i] in evaluate_imp: time_imprecise += 1
                    if y_cv_test[i] in evaluate: time_precise += 1
                logger.debug("(time, iTest, ellOptimal, cautious, prediction, ground-truth)(%s, %s, %s, %s, %s, %s)",
                             time, i, ell_optimal, evaluate_imp, evaluate, y_cv_test[i])
            logger.debug("(time, ellOptimal, nRealTests, timeImprecise, timePrecise) (%s, %s, %s, %s, %s)", time,
                         ell_optimal, n_real_tests, time_imprecise, time_precise)
            if n_real_tests > 0:
                n_real_fold += 1
                imprecise_mean += time_imprecise / n_real_tests
                precise_mean += time_precise / n_real_tests
        logger.debug("(time, nRealFold, imprecise, precise) (%s, %s, %s, %s)", time, n_real_fold, imprecise_mean,
                     precise_mean)
        if n_real_fold > 0:
            n_real_times += 1
            avg_imprecise += imprecise_mean / n_real_fold
            avg_precise += precise_mean / n_real_fold
    logger.debug("(dataset, models, imprec, prec) (%s, %s, %s, %s, %s)", in_path, model_type_imprecise,
                 model_type_precise, avg_imprecise / n_real_times, avg_precise / n_real_times)


def computing_time_prediction(in_path=None, ell_optimal=0.1, lib_path_server=None, model_type="ilda",
                              criterion="maximality", k_repetition=10):
    assert os.path.exists(in_path), "Without training data, not testing"
    data = pd.read_csv(in_path, header=None)
    logger = create_logger("computing_time_prediction", True)
    logger.info('Training dataset %s', in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = __factory_model(model_type, init_matlab=True, add_path_matlab=lib_path_server, DEBUG=False)
    model.learn(X=X_train, y=y_train, ell=ell_optimal)
    sum_time = 0
    n, _ = X_test.shape
    avg = np.array([])
    for k in range(k_repetition):
        logger.info("%s-fold repetition randomly", k)
        for i, test in enumerate(X_test):
            start = time.time()
            evaluate, _ = model.evaluate(test, criterion=criterion)
            end = time.time()
            logger.info("Evaluate %s, Ground-truth %s, Time %s ", evaluate, y_test[i], (end - start))
            sum_time += (end - start)
        avg = np.append(avg, sum_time/n)
    logger.info("Total time (%s, %s) and average %s and sd %s of %s testing", in_path, avg,
                np.mean(avg), np.std(avg), n)


in_path = sys.argv[1]
ell_optimal = float(sys.argv[2])
# QPBB_PATH_SERVER = []  # executed in host
computing_precise_vs_imprecise(in_path=in_path, ell_optimal=ell_optimal,
                               model_type_precise='lda', model_type_imprecise='ilda',
                               lib_path_server=QPBB_PATH_SERVER)

# computing_time_prediction(in_path=in_path, ell_optimal=ell_optimal, model_type="ilda",
#                           lib_path_server=QPBB_PATH_SERVER)
