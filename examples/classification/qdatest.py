# Testing 02
from classifip.models.qda import LinearDiscriminant
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


def testing01():
    ## Testing 01
    # current_dir = os.getcwd()
    # root_path = "/Users/salmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer_.csv"
    # root_path = "/Volumes/Data/DBSalmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv"
    root_path = "/Users/salmuz/Dropbox/PhD/testing/paper_qda.csv"
    # data = os.path.join(current_dir, root_path)
    df_train = pd.read_csv(root_path)
    X = df_train.loc[:, ['x1', 'x2']].values
    y = df_train.y.tolist()
    lqa = LinearDiscriminant(init_matlab=True)
    lqa.learn(X, y, ell=2)
    # lqa.testing_plot()
    # query = np.array([0.830031, 0.108776])
    # query = np.array([2, 2])
    # answer, _ = lqa.evaluate(query)
    # print(answer, _)
    # lqa.supremum_bf(query)
    # print(lqa.fit_max_likelihood(query))
    lqa.plot2D_classification(np.array([2, 2]), colors={0: 'red', 1: 'blue'})
    # lqa.plot2D_decision_boundary()
    # lqa.testing_plot()
    # Plots.plot2D_classification(X, y)
    # Plots.plot_cov_ellipse(X)
    # plt.show()


def testing02():
    import feather
    in_path = "../../resources/iris.data"
    data = feather.read_dataframe(in_path)
    X = data.iloc[:, 0:2].values
    y = data.iloc[:, -1].tolist()
    print(X, y)
    lqa = LinearDiscriminant(ell=5, init_matlab=True)
    lqa.learn(X, y)
    print(lqa)
    query = np.array([5.0, 2])
    answer, _ = lqa.evaluate(query)
    print(answer, _)
    lqa.plot2D_classification(query)
    # lqa.plot2D_decision_boundary(h=.1)


def testing03():

    def u65(mod_Y):
        return 1.6/mod_Y - 0.6/mod_Y**2

    def u80(mod_Y):
        return 2.2/mod_Y - 1.2/mod_Y**2

    import feather
    # in_path = "../../resources/iris.data"
    in_path = "/Users/salmuz/Downloads/glass.csv"
    # data = feather.read_dataframe(in_path)
    data = pd.read_csv(in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    # print(X_train, X_train.shape)
    # print(X_test, X_test.shape)
    ellu65 = dict()
    ellu80 = dict()
    lqa = LinearDiscriminant(init_matlab=True)
    for ell_current in np.arange(0.6,0.8,0.1):
        print("--ELL_CURRENT-----", ell_current)
        lqa.learn(X_train, y_train, ell=ell_current)
        sum_u65 = 0
        sum_u80 = 0
        for i, test in enumerate(X_test):
            print("--TESTING-----", i, ell_current)
            evaluate, _ = lqa.evaluate(test)
            print(evaluate, "-----", y_test[i])
            if y_test[i] in evaluate:
                sum_u65 += u65(len(evaluate))
                sum_u80 += u80(len(evaluate))
        ellu65[ell_current] = sum_u65
        ellu80[ell_current] = sum_u80
    print("--->", ellu65, ellu80)

def testing04():
    def u65(mod_Y):
        return 1.6/mod_Y - 0.6/mod_Y**2

    def u80(mod_Y):
        return 2.2/mod_Y - 1.2/mod_Y**2

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    in_path = "/Users/salmuz/Downloads/glass.csv"
    data = pd.read_csv(in_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].tolist()
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    sum_u65 = 0
    sum_u80 = 0
    lda.fit(X, y)
    n, _ = X_test.shape
    for i, test in enumerate(X_test):
        evaluate = lda.predict([test])
        print("--TESTING-----", i)
        if y_test[i] in evaluate:
            sum_u65 += u65(len(evaluate))
            sum_u80 += u80(len(evaluate))
    print("--->", sum_u65/n, sum_u80/n)

testing03()
