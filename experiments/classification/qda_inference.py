from classifip.models.qda import EuclideanDiscriminant, LinearDiscriminant, QuadraticDiscriminant
from classifip.dataset.uci_data_set import export_data_set
from classifip.utils import plot_classification as pc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

QPBB_PATH_SERVER = ['/home/lab/ycarranz/QuadProgBB', '/opt/cplex128/cplex/matlab/x86-64_linux']


def __test_imprecise_model(model, data, plotting, features, clazz = -1, h=0.02, ell=2.0):
    X = data.iloc[:, features].values
    y = np.array(data.iloc[:, clazz].tolist())
    _, p = X.shape
    model.learn(X=X, y=y, ell=ell)
    print("Evaluation ones features", model.evaluate(np.ones(p)), flush=True)
    if plotting:
        pc.plot2D_classification(model, np.array(np.ones(p)))
        pc.plot2D_decision_boundary(model, h=h)


def _test_IEuclideanDA(in_train=None, plotting=True, features=list([1, 3])):
    ieqa = EuclideanDiscriminant()
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(ieqa, data, plotting, features, h=0.02)


def _test_ILDA(in_train=None, plotting=True, features=list([1, 3])):
    ilda = LinearDiscriminant()
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(ilda, data, plotting, features, h=0.1)


def _test_IQDA(in_train=None, plotting=True, features=list([1, 3])):
    qlda = QuadraticDiscriminant()
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(qlda, data, plotting, features, h=0.1)


def output_paper_result(in_path=None, model="edla", features=list([1,2]), ell=0.5, hgrid=0.1):
    data = export_data_set('bin_normal_rnd.data') if in_path is None else pd.read_csv(in_path)
    if model == "idla":
        learning = LinearDiscriminant(DEBUG=True)
    elif model == "edla":
        learning = EuclideanDiscriminant(DEBUG=True)
    elif model == "qdla":
        learning = QuadraticDiscriminant(DEBUG=True)
    else:
        raise Exception("Selected model does not exist")
    __test_imprecise_model(learning, data, True, features, h=hgrid, ell=ell, clazz=0)
    # pc.plot2D_decision_boundary(lqa, h=0.1, new_multi_clazz={'0-1': -1})
    # pc.plot2D_decision_boundary_det(X, y, h=0.01)


def output_paper_zone_imprecise(method="imprecise"):
    data = export_data_set('iris.data')
    X = data.iloc[:, 0:2].values
    y = data.iloc[:, -1].tolist()
    if method == "imprecise":
        lqa = LinearDiscriminant(init_matlab=True, DEBUG=True)
        lqa.learn(X, y, ell=5)
        pc.plot2D_decision_boundary(lqa, h=0.01, cmap_color=plt.cm.gist_ncar)
        # newDic = dict()
        # newDic['Iris-setosa-Iris-versicolor'] = -1
        # newDic['Iris-setosa-Iris-virginica'] = -1
        # newDic['Iris-versicolor-Iris-virginica'] = -1
        # newDic['Iris-setosa-Iris-versicolor-Iris-virginica'] = -1
        # pc.plot2D_decision_boundary(lqa, h=0.01, new_multi_clazz=newDic)
    else:
        pc.plot2D_decision_boundary_det(X, y, h=0.01)


def computing_best_imprecise_mean(in_path=None, seed=0, cv_n_fold=10,
                                  from_ell=0.1, to_ell=1.0, by_ell=0.1):
    def u65(mod_Y):
        return 1.6 / mod_Y - 0.6 / mod_Y ** 2

    def u80(mod_Y):
        return 2.2 / mod_Y - 1.2 / mod_Y ** 2

    data = export_data_set('iris.data') if in_path is None else pd.read_csv(in_path)
    print("-----DATA SET TRAINING---", in_path)
    X = data.iloc[:, :-1].values
    y = np.array(data.iloc[:, -1].tolist())
    ell_u65, ell_u80 = dict(), dict()
    lqa = LinearDiscriminant(init_matlab=True)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.4, random_state=seed)

    kf = KFold(n_splits=cv_n_fold, random_state=None, shuffle=True)
    splits = list([])
    for idx_train, idx_test in kf.split(y_train):
        splits.append((idx_train, idx_test))
    print("Splits --->", splits)

    for ell_current in np.arange(from_ell, to_ell, by_ell):
        ell_u65[ell_current], ell_u80[ell_current] = 0, 0
        for idx_train, idx_test in splits:
            print("---k-FOLD-new-executing--")
            X_cv_train, y_cv_train = X_train[idx_train], y_train[idx_train]
            X_cv_test, y_cv_test = X_train[idx_test], y_train[idx_test]
            lqa.learn(X_cv_train, y_cv_train, ell=ell_current)
            sum_u65, sum_u80 = 0, 0
            n_test = len(idx_test)
            for i, test in enumerate(X_cv_test):
                evaluate, _ = lqa.evaluate(test)
                print("----TESTING-----", i, ell_current, "|---|", evaluate, "-----", y_cv_test[i])
                if y_cv_test[i] in evaluate:
                    sum_u65 += u65(len(evaluate))
                    sum_u80 += u80(len(evaluate))
            ell_u65[ell_current] += sum_u65 / n_test
            ell_u80[ell_current] += sum_u80 / n_test
        print("-------ELL_CURRENT-----", ell_current)
        ell_u65[ell_current] = ell_u65[ell_current] / cv_n_fold
        ell_u80[ell_current] = ell_u80[ell_current] / cv_n_fold
        print("u65-->", ell_u65[ell_current])
        print("u80-->", ell_u80[ell_current])
    print("--->", ell_u65, ell_u80)


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


# _test_IEuclideanDA()
# _test_ILDA()
# _test_IQDA()
output_paper_result()

# seeds = list([23, 10, 44, 31, 0, 17, 13, 29, 47, 87])
# seed_sampling_learn_ell = 23
# in_path = "/Users/salmuz/Downloads/glass.csv"
# computing_best_imprecise_mean(in_path, seed=seed_sampling_learn_ell, from_ell=0.01, to_ell=0.1, by_ell=0.01)
# computing_accuracy_imprecise(in_path, ell_optimal=0.03, seeds=seeds)
# computing_cv_accuracy_imprecise(in_path, ell_optimal=0.03)
# computing_performance_LDA(in_path, seeds=seeds)
# computing_cv_accuracy_LDA(in_path)
# computing_precise_vs_imprecise(ell_best=0.1, seeds=seeds)
# output_paper_zone_imprecise()
# output_paper_result()
