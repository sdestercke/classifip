import numpy as np, pandas as pd
from qda_common import __factory_model
import matplotlib.pyplot as plt
from classifip.models.qda import EuclideanDiscriminant, LinearDiscriminant, QuadraticDiscriminant, NaiveDiscriminant
from classifip.dataset.uci_data_set import export_data_set
from classifip.utils import plot_classification as pc

def __test_imprecise_model(model, data, features=None, clazz=-1, hgrid=0.02, ell=2.0,
                           query=None, cmap_color=None, is_imprecise=True, criterion=None):
    features = list([1, 3]) if features is None else features
    X = data.iloc[:, features].values
    y = np.array(data.iloc[:, clazz].tolist())
    _, p = X.shape
    if is_imprecise:
        query_eval = np.array(np.ones(p)) if query is None else query
        model.learn(X=X, y=y, ell=ell)
        print("Evaluation ones features", query_eval, model.evaluate(query_eval), flush=True)
        pc.plot2D_classification(model, query)
        pc.plot2D_decision_boundary(model, h=hgrid, cmap_color=cmap_color, criterion=criterion)
        # same color for imprecise zone
        # newDic = dict()
        # newDic['Iris-setosa-Iris-versicolor'] = -1
        # newDic['Iris-setosa-Iris-virginica'] = -1
        # newDic['Iris-versicolor-Iris-virginica'] = -1
        # newDic['Iris-setosa-Iris-versicolor-Iris-virginica'] = -1
        # pc.plot2D_decision_boundary(model, h=hgrid, new_multi_clazz=newDic)
    else:
        pc.plot2D_decision_boundary_det(X, y, h=hgrid)


def _test_IEuclideanDA(in_train=None, features=None):
    ieqa = EuclideanDiscriminant(DEBUG=True)
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(ieqa, data, features, hgrid=0.02)

def _test_INaiveDA(in_train=None, features=None):
    ieqa = NaiveDiscriminant(DEBUG=True)
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(ieqa, data, features, hgrid=0.02)


def _test_ILDA(in_train=None, features=None):
    ilda = LinearDiscriminant(DEBUG=True)
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(ilda, data, features, hgrid=0.1)


def _test_IQDA(in_train=None, features=None):
    qlda = QuadraticDiscriminant(DEBUG=True)
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    __test_imprecise_model(qlda, data, features, hgrid=0.1)


def output_paper_result(model_type="ieda", ell=0.5, hgrid=0.1):
    data = export_data_set('bin_normal_rnd.data')
    model = __factory_model(model_type, DEBUG=True)
    __test_imprecise_model(model, data, features=[1, 2], hgrid=hgrid, ell=ell, clazz=0)


def output_paper_zone_im_precise(is_imprecise=True, model_type="ieda", in_train=None, ell=2.0,
                                 hgrid=0.1, features=None, criterion=None):
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    features = list([0, 1]) if features is None else features
    model = __factory_model(model_type, DEBUG=True) if is_imprecise else None
    __test_imprecise_model(model, data, features=features, hgrid=hgrid, ell=ell,
                           query=None, is_imprecise=is_imprecise,
                           cmap_color=plt.cm.gist_ncar, criterion=criterion)

# Simple testing methods
# _test_IEuclideanDA()
# _test_ILDA()
# _test_IQDA()
# _test_INaiveDA()
# output_paper_result()
output_paper_zone_im_precise(model_type='iqda', hgrid=0.01, ell= 2,
                             criterion="maximality")
# output_paper_result()

