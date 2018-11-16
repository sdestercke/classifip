from classifip.models.qda import EuclideanDiscriminant, LinearDiscriminant, QuadraticDiscriminant
from classifip.dataset.uci_data_set import export_data_set
from classifip.utils import plot_classification as pc, create_logger
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from classifip.evaluation.measures import u65, u80
import random, os, csv, numpy as np, pandas as pd


## Server env:
# export LD_PRELOAD=/usr/local/MATLAB/R2018b/sys/os/glnxa64/libstdc++.so.6.0.22
QPBB_PATH_SERVER = ['/home/lab/ycarranz/QuadProgBB', '/opt/cplex128/cplex/matlab/x86-64_linux']
MODEL_TYPES = {'ieda': EuclideanDiscriminant, 'ilda': LinearDiscriminant, 'iqda': QuadraticDiscriminant}

def __factory_model(model_type, **kwargs):
    try:
        return MODEL_TYPES[model_type.lower()](**kwargs)
    except:
        raise Exception("Selected model does not exist")


def __test_imprecise_model(model, data, features=None, clazz=-1, hgrid=0.02, ell=2.0,
                           query=None, cmap_color=None, is_imprecise=True):
    features = list([1, 3]) if features is None else features
    X = data.iloc[:, features].values
    y = np.array(data.iloc[:, clazz].tolist())
    _, p = X.shape
    if is_imprecise:
        query_eval = np.array(np.ones(p)) if query is None else query
        model.learn(X=X, y=y, ell=ell)
        print("Evaluation ones features", query_eval, model.evaluate(query_eval), flush=True)
        pc.plot2D_classification(model, query)
        pc.plot2D_decision_boundary(model, h=hgrid, cmap_color=cmap_color)
        # same color for imprecise zone
        # newDic = dict()
        # newDic['Iris-setosa-Iris-versicolor'] = -1
        # newDic['Iris-setosa-Iris-virginica'] = -1
        # newDic['Iris-versicolor-Iris-virginica'] = -1
        # newDic['Iris-setosa-Iris-versicolor-Iris-virginica'] = -1
        # pc.plot2D_decision_boundary(lqa, h=0.01, new_multi_clazz=newDic)
    else:
        pc.plot2D_decision_boundary_det(X, y, h=hgrid)


def _test_IEuclideanDA(in_train=None, plotting=True, features=None):
    ieqa = EuclideanDiscriminant()
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(ieqa, data, plotting, features, hgrid=0.02)


def _test_ILDA(in_train=None, plotting=True, features=None):
    ilda = LinearDiscriminant()
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(ilda, data, plotting, features, hgrid=0.1)


def _test_IQDA(in_train=None, plotting=True, features=None):
    qlda = QuadraticDiscriminant()
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(qlda, data, plotting, features, hgrid=0.1)


def output_paper_result(model_type="ieda", ell=0.5, hgrid=0.1):
    data = export_data_set('bin_normal_rnd.data')
    model = __factory_model(model_type, DEBUG=True)
    __test_imprecise_model(model, data, features=[1, 2], hgrid=hgrid, ell=ell, clazz=0)


def output_paper_zone_im_precise(is_imprecise=True, model_type="ieda", in_train=None, ell=2,
                                 hgrid=0.1, features=None):
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    features = list([0, 1]) if features is None else features
    model = __factory_model(model_type) if is_imprecise else None
    __test_imprecise_model(model, data, features=features, hgrid=hgrid, ell=ell,
                           query=None, is_imprecise=is_imprecise, cmap_color=plt.cm.gist_ncar)


def prediction(model, model_type, X_train, y_train, ell_current, lib_path_server, splits):
    model, is_parallel = (__factory_model(model_type, init_matlab=True, add_path_matlab=lib_path_server), True) \
        if model is None else (model, False)
    idx_train, idx_test = splits
    print("Splits train %s", idx_train, flush=True)
    print("Splits test %s", idx_test, flush=True)
    X_cv_train, y_cv_train = X_train[idx_train], y_train[idx_train]
    X_cv_test, y_cv_test = X_train[idx_test], y_train[idx_test]
    model.learn(X=X_cv_train, y=y_cv_train, ell=ell_current)
    sum_u65, sum_u80 = 0, 0
    n_test = len(idx_test)
    for i, test in enumerate(X_cv_test):
        evaluate, _ = model.evaluate(test)
        print("testing, ell_current, prediction, ground-truth) (%s, %s, %s, %s)",
              i, ell_current, evaluate, y_cv_test[i])
        if y_cv_test[i] in evaluate:
            sum_u65 += u65(evaluate)
            sum_u80 += u80(evaluate)
    sum_u65 = sum_u65 / n_test
    sum_u80 = sum_u80 / n_test
    print("Partial-kfold (%s, %s, %s)", ell_current, sum_u65, sum_u80)
    if is_parallel: model.close_matlab()
    return dict({"u65": sum_u65, "u80": sum_u80})


def computing_best_imprecise_mean(in_path=None, out_path=None, cv_nfold=10, model_type="ieda", test_size=0.4,
                                  from_ell=0.1, to_ell=1.0, by_ell=0.1, seed=None, pl_process=False,
                                  lib_path_server=[]):
    assert os.path.exists(in_path), "Without training data, not testing"
    assert os.path.exists(out_path), "File for putting results does not exist"

    logger = create_logger("computing_best_imprecise_mean")

    data = pd.read_csv(in_path)
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
    file_csv = open(out_path, 'w')
    writer = csv.writer(file_csv)

    from functools import partial, reduce
    import multiprocessing

    model_static = None
    if not pl_process:
        model_static = __factory_model(model_type, init_matlab=True, add_path_matlab=lib_path_server, DEBUG=True)
    for ell_current in np.arange(from_ell, to_ell, by_ell):
        ell_u65[ell_current], ell_u80[ell_current] = 0, 0
        logger.info("ELL_CURRENT %s", ell_current)
        if pl_process:
            func = partial(prediction, model_static, model_type, X_train, y_train,
                           ell_current, lib_path_server)
            pool = multiprocessing.Pool(processes=1)
            z = pool.map(func, splits)
            pool.close()
            pool.join()
            z = reduce((lambda x, y: {'u65': x['u65'] + y['u65'], 'u80': x['u80'] + y['u80']}), z)
            ell_u65[ell_current] = z["u65"]
            ell_u80[ell_current] = z["u80"]
        else:
            for split in splits:
                z = prediction(model_static, model_type, X_train, y_train, ell_current, lib_path_server, split)
                ell_u65[ell_current] = z["u65"]
                ell_u80[ell_current] = z["u80"]
        ell_u65[ell_current] = ell_u65[ell_current] / cv_nfold
        ell_u80[ell_current] = ell_u80[ell_current] / cv_nfold
        writer.writerow([ell_current, ell_u65[ell_current], ell_u80[ell_current]])
        file_csv.flush()
        logger.debug("Partial-ell (%s, %s, %s)", ell_current, ell_u65, ell_u80)
    file_csv.close()
    logger.debug("Total-ell %s %s %s", in_path, ell_u65, ell_u80)


def computing_accuracy_imprecise(in_path=None, seeds=list([0]), ell_optimal=0.1):
    def u65(mod_Y):
        return 1.6 / mod_Y - 0.6 / mod_Y ** 2

    def u80(mod_Y):
        return 2.2 / mod_Y - 1.2 / mod_Y ** 2

    data = export_data_set('iris.data') if in_path is None else pd.read_csv(in_path)
    print("-----DATA SET TRAINING---", in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    n_time = len(seeds)
    mean_u65 = 0
    mean_u80 = 0
    lqa = LinearDiscriminant(init_matlab=True)
    for k in range(0, n_time):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=seeds[k])
        lqa.learn(X_train, y_train, ell=ell_optimal)
        sum_u65 = 0
        sum_u80 = 0
        n_test, _ = X_test.shape
        for i, test in enumerate(X_test):
            print("--TESTING-----", i, ell_optimal)
            evaluate, _ = lqa.evaluate(test)
            print(evaluate, "-----", y_test[i])
            if y_test[i] in evaluate:
                sum_u65 += u65(len(evaluate))
                sum_u80 += u80(len(evaluate))
        print("--ell_65_k_time---", k, sum_u65 / n_test)
        print("--ell_u80_k_time---", k, sum_u80 / n_test)
        mean_u65 += sum_u65 / n_test
        mean_u80 += sum_u80 / n_test
    mean_u65 = mean_u65 / n_time
    mean_u80 = mean_u80 / n_time
    print("--ell-->", ell_optimal, "--->", mean_u65, mean_u80)


def computing_cv_accuracy_imprecise(in_path=None, ell_optimal=0.1, cv_n_fold=10):
    def u65(mod_Y):
        return 1.6 / mod_Y - 0.6 / mod_Y ** 2

    def u80(mod_Y):
        return 2.2 / mod_Y - 1.2 / mod_Y ** 2

    data = export_data_set('iris.data') if in_path is None else pd.read_csv(in_path)
    print("-----DATA SET TRAINING---", in_path)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())
    mean_u65, mean_u80 = 0, 0
    lqa = LinearDiscriminant(init_matlab=True)
    kf = KFold(n_splits=cv_n_fold, random_state=None, shuffle=True)
    for idx_train, idx_test in kf.split(y):
        X_cv_train, y_cv_train = X[idx_train], y[idx_train]
        X_cv_test, y_cv_test = X[idx_test], y[idx_test]
        lqa.learn(X_cv_train, y_cv_train, ell=ell_optimal)
        sum_u65, sum_u80 = 0, 0
        n_test, _ = X_cv_test.shape
        for i, test in enumerate(X_cv_test):
            print("--TESTING-----", i, ell_optimal)
            evaluate, _ = lqa.evaluate(test)
            print(evaluate, "-----", y_cv_test[i])
            if y_cv_test[i] in evaluate:
                sum_u65 += u65(len(evaluate))
                sum_u80 += u80(len(evaluate))
        mean_u65 += sum_u65 / n_test
        mean_u80 += sum_u80 / n_test
    mean_u65 = mean_u65 / cv_n_fold
    mean_u80 = mean_u80 / cv_n_fold
    print("--ell-->", ell_optimal, "--->", mean_u65, mean_u80)


def computing_precise_vs_imprecise(in_path=None, ell_optimal=0.1, seeds=0):
    def u65(mod_Y):
        return 1.6 / mod_Y - 0.6 / mod_Y ** 2

    def u80(mod_Y):
        return 2.2 / mod_Y - 1.2 / mod_Y ** 2

    data = export_data_set('iris.data') if in_path is None else pd.read_csv(in_path)
    print("-----DATA SET TRAINING---", in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    n_time = len(seeds)
    lda_imp = LinearDiscriminant(init_matlab=True)
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    mean_u65_imp, mean_u80_imp, u_mean = 0, 0, 0
    for k in range(0, n_time):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=seeds[k])
        lda_imp.learn(X_train, y_train, ell=ell_optimal)
        lda.fit(X_train, y_train)
        sum_u65, sum_u80 = 0, 0
        u_precise, n_real_test = 0, 0
        n_test, _ = X_test.shape
        for i, test in enumerate(X_test):
            print("--TESTING-----", i)
            evaluate_imp, _ = lda_imp.evaluate(test)
            if len(evaluate_imp) > 1:
                n_real_test += 1
                if y_test[i] in evaluate_imp:
                    sum_u65 += u65(len(evaluate_imp))
                    sum_u80 += u80(len(evaluate_imp))
                evaluate = lda.predict([test])
                if y_test[i] in evaluate:
                    u_precise += u80(len(evaluate))
        mean_u65_imp += sum_u65 / n_real_test
        mean_u80_imp += sum_u80 / n_real_test
        u_mean += u_precise / n_real_test
        print("--time_k--u65-->", k, sum_u65 / n_real_test)
        print("--time_k--u80-->", k, sum_u80 / n_real_test)
        print("--time_k--precise-->", k, u_precise / n_real_test)
    print("--global--u65-->", mean_u65_imp / n_time)
    print("--global--u80-->", mean_u80_imp / n_time)
    print("--global--precise-->", u_mean / n_time)


def computing_performance_LDA(in_path=None, seeds=list([0])):
    def u65(mod_Y):
        return 1.6 / mod_Y - 0.6 / mod_Y ** 2

    def u80(mod_Y):
        return 2.2 / mod_Y - 1.2 / mod_Y ** 2

    data = export_data_set('iris.data') if in_path is None else pd.read_csv(in_path)
    print("-----DATA SET TRAINING---", in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    mean_u65, mean_u80 = 0, 0
    n_times = len(seeds)
    for k in range(0, n_times):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seeds[k])
        sum_u65, sum_u80 = 0, 0
        lda.fit(X_train, y_train)
        n, _ = X_test.shape
        for i, test in enumerate(X_test):
            evaluate = lda.predict([test])
            print("-----TESTING-----", i)
            if y_test[i] in evaluate:
                sum_u65 += u65(len(evaluate))
                sum_u80 += u80(len(evaluate))
        print("--k-->", k, sum_u65 / n, sum_u80 / n)
        mean_u65 += sum_u65 / n
        mean_u80 += sum_u80 / n
    print("--->", mean_u65 / n_times, mean_u80 / n_times)


def computing_cv_accuracy_LDA(in_path=None, cv_n_fold=10):
    def u65(mod_Y):
        return 1.6 / mod_Y - 0.6 / mod_Y ** 2

    def u80(mod_Y):
        return 2.2 / mod_Y - 1.2 / mod_Y ** 2

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    data = export_data_set('iris.data') if in_path is None else pd.read_csv(in_path)
    print("-----DATA SET TRAINING---", in_path)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())
    kf = KFold(n_splits=cv_n_fold, random_state=None, shuffle=True)
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    mean_u65, mean_u80 = 0, 0
    for idx_train, idx_test in kf.split(y):
        print("---k-FOLD-new-executing--")
        X_cv_train, y_cv_train = X[idx_train], y[idx_train]
        X_cv_test, y_cv_test = X[idx_test], y[idx_test]
        lda.fit(X_cv_train, y_cv_train)
        n_test = len(idx_test)
        sum_u65, sum_u80 = 0, 0
        for i, test in enumerate(X_cv_test):
            evaluate = lda.predict([test])
            print("-----TESTING-----", i)
            if y_cv_test[i] in evaluate:
                sum_u65 += u65(len(evaluate))
                sum_u80 += u80(len(evaluate))
        mean_u65 += sum_u65 / n_test
        mean_u80 += sum_u80 / n_test
    print("--->", mean_u65 / cv_n_fold, mean_u80 / cv_n_fold)


def computing_time_prediction(in_path=None):
    import time
    data = export_data_set('iris.data') if in_path is None else pd.read_csv(in_path)
    print("-----DATA SET TRAINING---", in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    lqa = LinearDiscriminant(init_matlab=True)
    lqa.learn(X=X_train, y=y_train, ell=0.01)
    sum_time = 0
    n, _ = X_test.shape
    for i, test in enumerate(X_test):
        start = time.time()
        evaluate, _ = lqa.evaluate(test)
        end = time.time()
        print(evaluate, "-----", y_test[i], '--time--', (end - start))
        sum_time += (end - start)
    print("--->", sum_time, '---n---', n)


# Simple testing methods
# _test_IEuclideanDA()
# _test_ILDA()
# _test_IQDA()
# output_paper_result()
# output_paper_zone_im_precise()

# Experiments with several datasets
QPBB_PATH_SERVER = [] # executed in host
in_path = "/Users/salmuz/Downloads/datasets/optdigits.csv"
out_path = "/Users/salmuz/Downloads/results.csv"
computing_best_imprecise_mean(in_path=in_path, out_path=out_path, model_type="ilda",
                              pl_process=False, lib_path_server=QPBB_PATH_SERVER)
# seeds = list([23, 10, 44, 31, 0, 17, 13, 29, 47, 87])
# seed_sampling_learn_ell = 23
# computing_best_imprecise_mean(in_path, seed=seed_sampling_learn_ell, from_ell=0.01, to_ell=0.1, by_ell=0.01)
# computing_accuracy_imprecise(in_path, ell_optimal=0.03, seeds=seeds)
# computing_cv_accuracy_imprecise(in_path, ell_optimal=0.03)
# computing_performance_LDA(in_path, seeds=seeds)
# computing_cv_accuracy_LDA(in_path)
# computing_precise_vs_imprecise(ell_best=0.1, seeds=seeds)
# output_paper_zone_imprecise()
# output_paper_result()
