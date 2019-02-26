from classifip.utils import create_logger
from sklearn.model_selection import train_test_split
from qda_common import __factory_model
from sklearn.model_selection import KFold
from classifip.evaluation.measures import u65, u80
import random, os, csv, sys, numpy as np, pandas as pd


def computing_best_imprecise_mean(in_path=None, out_path=None, cv_nfold=10, model_type="ieda", test_size=0.4,
                                  from_ell=0.1, to_ell=1.0, by_ell=0.1, seed=None, lib_path_server=None):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean", True)
    logger.info('Training dataset %s', in_path)
    data = pd.read_csv(in_path)  # , header=None)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())

    ell_u65, ell_u80 = dict(), dict()
    seed = random.randrange(pow(2, 30)) if seed is None else seed
    logger.debug("MODEL: %s, SEED: %s", model_type, seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    kf = KFold(n_splits=cv_nfold, random_state=None, shuffle=True)
    splits = list([])
    for idx_train, idx_test in kf.split(y_train):
        splits.append((idx_train, idx_test))
        logger.info("Splits %s train %s", len(splits), idx_train)
        logger.info("Splits %s test %s", len(splits), idx_test)

    # Create a CSV file for saving results
    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)

    model = __factory_model(model_type, init_matlab=True, add_path_matlab=lib_path_server, DEBUG=True)
    for ell_current in np.arange(from_ell, to_ell, by_ell):
        ell_u65[ell_current], ell_u80[ell_current] = 0, 0
        logger.info("ELL_CURRENT %s", ell_current)
        for idx_train, idx_test in splits:
            logger.info("Splits train %s", idx_train)
            logger.info("Splits test %s", idx_test)
            X_cv_train, y_cv_train = X_train[idx_train], y_train[idx_train]
            X_cv_test, y_cv_test = X_train[idx_test], y_train[idx_test]
            model.learn(X=X_cv_train, y=y_cv_train, ell=ell_current)
            sum_u65, sum_u80 = 0, 0
            n_test = len(idx_test)
            for i, test in enumerate(X_cv_test):
                evaluate, _ = model.evaluate(test)
                logger.debug("(testing, ell_current, prediction, ground-truth) (%s, %s, %s, %s)",
                             i, ell_current, evaluate, y_cv_test[i])
                if y_cv_test[i] in evaluate:
                    sum_u65 += u65(evaluate)
                    sum_u80 += u80(evaluate)
            ell_u65[ell_current] += sum_u65 / n_test
            ell_u80[ell_current] += sum_u80 / n_test
            logger.debug("Partial-kfold (%s, %s, %s)", ell_current, ell_u65[ell_current], ell_u80[ell_current])
        ell_u65[ell_current] = ell_u65[ell_current] / cv_nfold
        ell_u80[ell_current] = ell_u80[ell_current] / cv_nfold
        writer.writerow([ell_current, ell_u65[ell_current], ell_u80[ell_current]])
        file_csv.flush()
        logger.debug("Partial-ell (%s, %s, %s)", ell_current, ell_u65, ell_u80)
    file_csv.close()
    logger.debug("Total-ell %s %s %s", in_path, ell_u65, ell_u80)


## Server env:
# export LD_PRELOAD=/usr/local/MATLAB/R2018b/sys/os/glnxa64/libstdc++.so.6.0.22
# QPBB_PATH_SERVER = ['/home/lab/ycarranz/QuadProgBB', '/opt/cplex128/cplex/matlab/x86-64_linux']

# Experiments with several datasets
# Windows
# QPBB_PATH_SERVER = []
# in_path = "C:\\datasets\\glass.csv"
# out_path = "C:\\Users\\hds\\Dropbox\\PhD\\results_glass.csv"

# Linux
# QPBB_PATH_SERVER = ['~/Dropbox/PhD/code/QuadProgBB', ' /opt/ibm/ILOG/CPLEX_Studio128/cplex/matlab/x86-64_linux']
# in_path = "/home/cnrs/Downloads/datasets/libras.csv"
# out_path = "/home/cnrs/Dropbox/PhD/results_libras.csv"

# Macbook
# QPBB_PATH_SERVER = []  # executed in host
# in_path = "/Users/salmuz/Downloads/datasets/iris.csv"
# out_path = "/Users/salmuz/Downloads/results_iris_inda.csv"

QPBB_PATH_SERVER = []
in_path = sys.argv[1]
out_path = sys.argv[2]

computing_best_imprecise_mean(in_path=in_path, out_path=out_path, model_type="inda",
                              from_ell=0.01, to_ell=5.5, by_ell=0.01, # seed=XXX,
                              lib_path_server=QPBB_PATH_SERVER)
