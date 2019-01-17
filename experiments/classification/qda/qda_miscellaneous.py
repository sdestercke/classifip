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


# computing_precise_vs_imprecise(ell_best=0.1, seeds=seeds)
