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


# computing_accuracy_imprecise(in_path, ell_optimal=0.03, seeds=seeds)
# computing_cv_accuracy_imprecise(in_path, ell_optimal=0.03)
