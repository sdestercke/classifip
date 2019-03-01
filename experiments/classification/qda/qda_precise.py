from classifip.evaluation.measures import u65, u80
from qda_common import __factory_model_precise, generate_seeds
from classifip.dataset.uci_data_set import export_data_set
from sklearn.model_selection import KFold
import numpy as np, pandas as pd, sys, os, csv
from classifip.utils import create_logger
from sklearn.model_selection import train_test_split


def performance_hold_out(in_path=None, model_type='lda', test_pct=0.4, n_times=10, seeds=None):
    assert os.path.exists(in_path), "Without training data, not testing"
    data = pd.read_csv(in_path, header=None)
    logger = create_logger("performance_hold_out", True)
    logger.info('Training data set %s, test percentage %s, model_type %s', in_path, test_pct, model_type)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    model = __factory_model_precise(model_type, solver="svd", store_covariance=True)
    mean_u65, mean_u80 = 0, 0
    seeds = generate_seeds(n_times) if seeds is None else seeds
    logger.info('Seeds generated %s', seeds)
    for i in range(0, n_times):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=seeds[i])
        sum_u65, sum_u80 = 0, 0
        model.fit(X_train, y_train)
        n, _ = X_test.shape
        for j, test in enumerate(X_test):
            evaluate = model.predict([test])
            if y_test[j] in evaluate:
                sum_u65 += u65(evaluate)
                sum_u80 += u80(evaluate)
        logger.info("Time, u65, u80 (%s, %s, %s)", i, sum_u65 / n, sum_u80 / n)
        mean_u65 += sum_u65 / n
        mean_u80 += sum_u80 / n
    logger.info("[Total:data-set:avgResults] (%s, %s)", mean_u65 / n_times, mean_u80 / n_times)


def performance_cv_accuracy(in_path=None, model_type='lda', cv_n_fold=10, seeds=None):
    assert os.path.exists(in_path), "Without training data, not testing"
    data = pd.read_csv(in_path, header=None)
    logger = create_logger("performance_cv_accuracy", True)
    logger.info('Training data set %s, cv_n_fold %s, model_type %s', in_path, cv_n_fold, model_type)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())
    avg_u65, avg_u80 = 0, 0
    seeds = generate_seeds(cv_n_fold) if seeds is None else seeds
    logger.info('Seeds generated %s', seeds)
    for time in range(cv_n_fold):
        # Generation a random k-fold validation.
        kf = KFold(n_splits=cv_n_fold, random_state=seeds[time], shuffle=True)
        model = __factory_model_precise(model_type, store_covariance=True)
        mean_u65, mean_u80 = 0, 0
        for idx_train, idx_test in kf.split(y):
            X_cv_train, y_cv_train = X[idx_train], y[idx_train]
            X_cv_test, y_cv_test = X[idx_test], y[idx_test]
            model.fit(X_cv_train, y_cv_train)
            n_test = len(idx_test)
            sum_u65, sum_u80 = 0, 0
            for i, test in enumerate(X_cv_test):
                evaluate = model.predict([test])
                if y_cv_test[i] in evaluate:
                    sum_u65 += u65(evaluate)
                    sum_u80 += u80(evaluate)
            mean_u65 += sum_u65 / n_test
            mean_u80 += sum_u80 / n_test
        logger.info("Time, seed, u65, u80 (%s, %s, %s, %s)", time, seeds[time], mean_u65 / cv_n_fold,
                    mean_u80 / cv_n_fold)
        avg_u65 += mean_u65 / cv_n_fold
        avg_u80 += mean_u80 / cv_n_fold
    logger.info("[Total:data-set:avgResults] (%s, %s, %s, %s)", in_path, kwargs, avg_u65 / cv_n_fold,
                avg_u80 / cv_n_fold)


def performance_qda_regularized(in_path=None, out_path=None, cv_n_fold=10, seeds=None,
                                from_alpha=0, to_alpha=2.0, by_alpha=0.01):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "Without output saving performance"
    data = pd.read_csv(in_path, header=None)
    logger = create_logger("performance_qda_regularized", True)
    logger.info('Training data set %s, cv_n_fold %s, model_type %s', in_path, cv_n_fold, model_type)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())

    seeds = generate_seeds(cv_n_fold) if seeds is None else seeds
    logger.info('Seeds generated %s', seeds)

    file_csv = open(out_path, 'a')
    writer = csv.writer(file_csv)

    alphas = np.arange(from_alpha, to_alpha, by_alpha)
    writer.writerow(alphas)

    qda_regularized = [None] * len(alphas)
    for idx, alpha in enumerate(alphas):
        qda_regularized[idx] = __factory_model_precise("qda", store_covariance=True, reg_param=alpha)
    # Generation a random k-fold validation.
    kf_second = KFold(n_splits=cv_n_fold, random_state=None, shuffle=True)
    ikfold, accuracy, best_alphas = 0, [0] * cv_n_fold, [0] * cv_n_fold
    for idx_learning, idx_testing in kf_second.split(y):
        X_training, y_training = X[idx_learning], y[idx_learning]
        X_testing, y_testing = X[idx_testing], y[idx_testing]
        kf = KFold(n_splits=cv_n_fold, random_state=seeds[ikfold], shuffle=True)
        acc_u80 = [0] * len(qda_regularized)
        for idx_train, idx_test in kf.split(y_training):
            X_cv_train, y_cv_train = X_training[idx_train], y_training[idx_train]
            X_cv_test, y_cv_test = X_training[idx_test], y_training[idx_test]
            for model in qda_regularized: model.fit(X_cv_train, y_cv_train)
            n_test = len(idx_test)
            for i, test in enumerate(X_cv_test):
                for im, model in enumerate(qda_regularized):
                    evaluate = model.predict([test])
                    if y_cv_test[i] in evaluate: acc_u80[im] += (u80(evaluate) / n_test) / cv_n_fold
        idx_best = np.argmax(acc_u80)
        logger.info("[1kfold:best_model:seed:u80] (%s, %s, %s, %s)", ikfold, alphas[idx_best], seeds[ikfold], acc_u80)
        writer.writerow(acc_u80)
        file_csv.flush()

        best_model = __factory_model_precise(model_type, store_covariance=True, reg_param=alphas[idx_best])
        best_model.fit(X_training, y_training)
        accuracy[ikfold], bn_test, best_alphas[ikfold] = 0, len(idx_testing), alphas[idx_best]
        for i, test in enumerate(X_testing):
            evaluate = best_model.predict([test])
            if y_testing[i] in evaluate: accuracy[ikfold] += u80(evaluate) / bn_test
        logger.info("[2kfold:best_model:seed:accuracy] (%s, %s, %s)", ikfold, alphas[idx_best], accuracy[ikfold])
        ikfold += 1
    logger.info("[total:data-set:avgResults] (%s, %s, %s, %s)", in_path, np.mean(accuracy), best_alphas, accuracy)


# in_path = sys.argv[1]
# performance_cv_accuracy(in_path, model_type='lda')

in_path = sys.argv[1]
out_path = sys.argv[2]
performance_qda_regularized(in_path, out_path)
