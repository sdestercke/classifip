# Testing 02
from classifip.models.qda import LinearDiscriminant
import numpy as np
import os
import pandas as pd

def testing01():
    ## Testing 01
    current_dir = os.getcwd()
    # root_path = "/Users/salmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer_.csv"
    root_path = "/Volumes/Data/DBSalmuz/Dropbox/PhD/code/idle-kaggle/resources/classifier_easer.csv"
    data = os.path.join(current_dir, root_path)
    df_train = pd.read_csv(data)
    X = df_train.loc[:, ['x1', 'x2']].values
    y = df_train.y.tolist()
    lqa = LinearDiscriminant(ell=5, init_matlab=True)
    lqa.learn(X, y)
    # lqa.testing_plot()
    # query = np.array([0.830031, 0.108776])
    # query = np.array([2, 2])
    # answer, _ = lqa.evaluate(query)
    # print(answer, _)
    # lqa.supremum_bf(query)
    # print(lqa.fit_max_likelihood(query))
    # lqa.plot2D_classification(query)
    lqa.plot2D_decision_boundary()
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
    query = np.array([6.7, 3.3])
    answer, _ = lqa.evaluate(query)
    print(answer, _)
    # lqa.plot2D_classification(query)
    lqa.plot2D_decision_boundary()


testing02()
