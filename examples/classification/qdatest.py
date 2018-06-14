# Testing 02
from classifip.models.qda import LinearDiscriminant
from classifip.dataset.uci_data_set import export_data_set
import numpy as np
import pandas as pd
import feather
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def testingLargeDim(n, d):
    pass
    # def costFx(x, cov, query):
    # i_cov = inv(cov)
    # q = query.T @ i_cov
    # return 0.5 * (x.T @ i_cov @ x) + q.T @ x

    # e_mean = np.random.normal(size=d)
    # e_cov = normal(d, d)
    # e_cov = e_cov.T * e_cov
    # query = np.random.normal(size=d)
    # q = maximum_Fx(e_cov, e_mean, query, n, d)
    # print("--->", q["x"], costFx(np.array(q["x"]), e_cov, query))
    # bnb_search(e_cov, e_mean, query, n, d)
    # brute_force_search(e_cov, e_mean, query, n, d)


def output_paper_result(in_path=None):
    data = export_data_set('bin_normal_rnd.data') if in_path is None else pd.read_csv(in_path)
    X = data.loc[:, ['x1', 'x2']].values
    y = data.y.tolist()
    lqa = LinearDiscriminant(init_matlab=True)
    lqa.learn(X, y, ell=2)
    lqa.plot2D_classification(np.array([2, 2]), colors={0: 'red', 1: 'blue'})
    lqa.plot2D_decision_boundary()


def output_paper_zone_imprecise():
    in_path = "../../resources/iris.data"
    data = feather.read_dataframe(in_path)
    X = data.iloc[:, 0:2].values
    y = data.iloc[:, -1].tolist()
    lqa = LinearDiscriminant(init_matlab=True)
    lqa.learn(X, y, ell=5)
    query = np.array([5.0, 2])
    answer, _ = lqa.evaluate(query)
    # lqa.plot2D_classification(query)
    lqa.plot2D_decision_boundary()


def computing_best_imprecise_mean(seeds=list([0]), from_ell=0.1, to_ell=1.0, by_ell=0.1):
    def u65(mod_Y):
        return 1.6 / mod_Y - 0.6 / mod_Y ** 2

    def u80(mod_Y):
        return 2.2 / mod_Y - 1.2 / mod_Y ** 2

    in_path = "/Users/salmuz/Downloads/seeds.csv"
    print("--DATA SET---", in_path)
    data = pd.read_csv(in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    n_time = len(seeds)
    ell_u65 = dict()
    ell_u80 = dict()
    lqa = LinearDiscriminant(init_matlab=True)
    for ell_current in np.arange(from_ell, to_ell, by_ell):
        ell_u65[ell_current] = 0
        ell_u80[ell_current] = 0
        for k in range(0, n_time):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.4, random_state=seeds[k])
            lqa.learn(X_train, y_train, ell=ell_current)
            sum_u65 = 0
            sum_u80 = 0
            n_test, _ = X_test.shape
            for i, test in enumerate(X_test):
                print("--TESTING-----", i, ell_current)
                evaluate, _ = lqa.evaluate(test)
                print(evaluate, "-----", y_test[i])
                if y_test[i] in evaluate:
                    sum_u65 += u65(len(evaluate))
                    sum_u80 += u80(len(evaluate))
            print("--ell_65_k_time---", k, sum_u65, sum_u65 / n_test)
            print("--ell_u80_k_time---", k, sum_u80, sum_u80 / n_test)
            ell_u65[ell_current] += sum_u65 / n_test
            ell_u80[ell_current] += sum_u80 / n_test
        print("-------ELL_CURRENT-----", ell_current)
        print("u65-->", ell_u65[ell_current] / n_time)
        print("u80-->", ell_u80[ell_current] / n_time)
    print("--->", ell_u65, ell_u80)


def computing_precise_vs_imprecise(ell_best=0.1, seeds=list([0])):
    def u65(mod_Y):
        return 1.6 / mod_Y - 0.6 / mod_Y ** 2

    def u80(mod_Y):
        return 2.2 / mod_Y - 1.2 / mod_Y ** 2

    in_path = "/Users/salmuz/Downloads/seeds.csv"
    data = pd.read_csv(in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    n_time = len(seeds)
    lda_imp = LinearDiscriminant(init_matlab=True)
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    mean_u65_imp, mean_u80_imp, u_mean = 0, 0, 0
    for k in range(0, n_time):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=seeds[k])
        lda_imp.learn(X_train, y_train, ell=ell_best)
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


def computing_performing_LDA(seeds=list([0])):
    def u65(mod_Y):
        return 1.6 / mod_Y - 0.6 / mod_Y ** 2

    def u80(mod_Y):
        return 2.2 / mod_Y - 1.2 / mod_Y ** 2

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    in_path = "/Users/salmuz/Downloads/glass.csv"
    data = pd.read_csv(in_path)
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


def computing_time_prediction():
    import time
    in_path = "/Users/salmuz/Downloads/glass.csv"
    data = pd.read_csv(in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    lqa = LinearDiscriminant(init_matlab=True)
    lqa.learn(X_train, y_train, ell=0.01)
    sum_time = 0
    n, _ = X_test.shape
    for i, test in enumerate(X_test):
        start = time.time()
        evaluate, _ = lqa.evaluate(test)
        end = time.time()
        print(evaluate, "-----", y_test[i], '--time--', (end - start))
        sum_time += (end - start)
    print("--->", sum_time, '---n---', n)


seeds = list([23, 10, 44, 31, 0, 17, 13, 29, 47, 87])
# computing_precise_vs_imprecise(ell_best=0.1, seeds=seeds)
computing_best_imprecise_mean(seeds=seeds, from_ell=0.03, to_ell=0.05, by_ell=0.02)
# computing_performing_LDA(seeds=seeds)