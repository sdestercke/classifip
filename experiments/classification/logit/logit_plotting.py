import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from classifip.dataset.uci_data_set import export_data_set
from classifip.utils import plot_classification as pc
from matplotlib.colors import ListedColormap
from classifip.models.logit import BinaryILogisticLasso


def fn_prediction(model, newClazz, clazz_by_index, query, criterion):
    answer = model.evaluate([query])
    answer = model.get_clazz()[answer[0].getmaximaldecision().astype('bool')]
    if len(answer) > 1 or len(answer) == 0:
        iClass = "-".join(str(clazz) for clazz in sorted(answer))
        return newClazz[iClass]
    else:
        return clazz_by_index[answer[0]]


def __test_imprecise_model(model, data, features=None, hgrid=0.02,
                           cmap_color=None, is_imprecise=True):
    features = list([1, 3]) if features is None else features
    X = data.iloc[:, features].values
    y = np.array(data.iloc[:, -1].tolist())
    _, p = X.shape
    if is_imprecise:
        model.learn(X=X, y=y, min_gamma=0, max_gamma=10)
        pc.plot2D_decision_boundary(model,
                                    h=hgrid,
                                    cmap_color=cmap_color,
                                    fn_prediction=fn_prediction)
    else:
        pc.plot2D_decision_boundary_det(X, y, h=hgrid)


def output_paper_result(model_type="ieda", ell=0.5, hgrid=0.1):
    data = export_data_set('bin_normal_rnd.data')
    model = __factory_model(model_type, DEBUG=True)
    __test_imprecise_model(model, data, features=[1, 2], hgrid=hgrid, ell=ell, clazz=0)


def output_paper_zone_im_precise(is_imprecise=True,
                                 in_train=None,
                                 hgrid=0.1,
                                 features=None,
                                 cmap_color=None):
    data = export_data_set('iris.data') if in_train is None else pd.read_csv(in_train)
    data = data[data['4'] != 'Iris-virginica']
    features = list([0, 1]) if features is None else features
    model = BinaryILogisticLasso()
    __test_imprecise_model(model, data,
                           features=features,
                           hgrid=hgrid,
                           is_imprecise=is_imprecise,
                           cmap_color=plt.cm.gist_ncar if cmap_color is None else cmap_color)


# output_paper_result()
cmap_light = ListedColormap(['#A7CDD0', '#B3E4C7', '#F2F1A7', '#E59C81', '#D2645D', '#D6DEF1', '#FBBDA6'])
output_paper_zone_im_precise(is_imprecise=True, hgrid=0.05, cmap_color=cmap_light)
