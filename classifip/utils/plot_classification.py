import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools as it
from functools import reduce


def __generate_all_multi_clazz(clazz):
    """
    source: https://markhneedham.com/blog/2013/11/06/python-generate-all-combinations-of-a-list/
    :param clazz:
    :return: all multi-classes
    """
    clazz = [str(clazz) for clazz in sorted(clazz)]
    last_idx = len(clazz)
    multi_clazz = reduce(lambda acc, x: acc + list(it.combinations(clazz, x)), range(2, len(clazz) + 1), [])
    new_clazz = dict()
    for new_idx, clazz in enumerate(multi_clazz, last_idx + 1):
        new_clazz["-".join(clazz)] = new_idx
    return new_clazz


def __check_data_available(data):
    X = data.iloc[:, :-1].as_matrix()
    y = data.y.tolist()
    if X is None: raise ValueError("It needs to learn one sample training")

    n_row, _ = X.shape
    if not isinstance(y, list): raise ValueError("Y isn't corrected form.")
    if n_row != len(y): raise ValueError("The number of column is not same in (X,y)")
    return X, np.array(y)


def plot2D_classification(model, query=None, colors=None, markers=None):
    markers = list(['*', 'v', 'o', '+', '-', '.', ',']) if markers is None else markers
    X, y = __check_data_available(model.get_data())
    n_row, n_col = X.shape
    _clazz = model.get_clazz()
    _nb_clazz = len(_clazz)

    c_map = plt.cm.get_cmap("hsv", _nb_clazz + 1)
    print(colors, markers, model.get_clazz())
    colors = dict((_clazz[idx], c_map(idx)) for idx in range(0, _nb_clazz)) \
        if colors is None else colors
    markers = dict((_clazz[idx], markers[idx]) for idx in range(0, _nb_clazz))

    def plot_constraints(lower, upper, line_style="solid"):
        plt.plot([lower[0], lower[0], upper[0], upper[0], lower[0]],
                 [lower[1], upper[1], upper[1], lower[1], lower[1]],
                 linestyle=line_style)
        plt.grid()

    def plot2D_scatter(X, y):
        for row in range(0, len(y)):
            plt.scatter(X[row, 0], X[row, 1], marker=markers[y[row]], c=colors[y[row]])

    def plot_ellipse(splot, mean, cov, color):
        from scipy import linalg

        v, w = linalg.eigh(cov)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5, 180 + angle,
                                  facecolor="none", edgecolor=color, linewidth=2, zorder=2)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.9)
        splot.add_artist(ell)

    if n_col == 2:
        for clazz in _clazz:
            mean = model.get_mean_by_clazz(clazz)
            # prior_mean_lower = mean - model.get_ell()
            # prior_mean_upper = mean + model.get_ell()
            # plot_constraints(prior_mean_lower, prior_mean_upper, line_style="dashed")
            post_mean_lower, post_mean_upper = model.get_bound_means(clazz)
            plot_constraints(post_mean_lower, post_mean_upper)

        if query is not None:
            ml_mean, ml_prob = model.fit_max_likelihood(query)
            plt.plot([query[0]], [query[1]], marker='h', markersize=5, color="black")
            model.evaluate(query)
            _bounds = model.get_bound_cond_probability()
            for clazz in _clazz:
                plt.plot([ml_mean[clazz][0]], [ml_mean[clazz][1]], marker='o', markersize=5, color=colors[clazz])
                _, est_mean_lower = _bounds[clazz]['inf']
                _, est_mean_upper = _bounds[clazz]['sup']
                plt.plot([est_mean_lower[0]], [est_mean_lower[1]], marker='x', markersize=4, color="black")
                plt.plot([est_mean_upper[0]], [est_mean_upper[1]], marker='x', markersize=4, color="black")

        s_plot = plt.subplot()
        for clazz in _clazz:
            cov, inv = model.get_cov_by_clazz(clazz)
            mean = model.get_mean_by_clazz(clazz)
            plot_ellipse(s_plot, mean, cov, colors[clazz])

    elif n_col > 2:
        if query is not None:
            inference = model.evaluate(query)
            X = np.vstack([X, query])
            y = np.append(y, inference[0])

        from sklearn.manifold import Isomap
        iso = Isomap(n_components=2)
        projection = iso.fit_transform(X)
        X = np.c_[projection[:, 0], projection[:, 1]]

        if query is not None:
            color_instance = colors[inference[0]] if len(inference) == 1 else 'black'
            plt.plot([X[n_row, 0]], [X[n_row, 1]], color='red', marker='o', mfc=color_instance)
    else:
        raise Exception("Not implemented for one feature yet.")

    plot2D_scatter(X, y)
    plt.show()


def plot2D_decision_boundary(model, h=.01, cmap_color=None, new_multi_clazz=None, markers=None,
                             criterion="maximality", savefig=False, fn_prediction=None):
    if fn_prediction is None:
        raise Exception("Not implemented prediction function!")

    markers = list(['+', '*', 'v', 'o', '-', '.', ',']) if markers is None else markers
    X, y = __check_data_available(model.get_data())
    _clazz = sorted(model.get_clazz())
    _nb_clazz = len(_clazz)
    _, n_col = X.shape

    if n_col > 2:
        raise Exception("Not implemented for n-dimension yet.")

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    clazz_by_index = dict((clazz, idx) for idx, clazz in enumerate(_clazz, 1))
    newClazz = __generate_all_multi_clazz(_clazz) if new_multi_clazz is None else new_multi_clazz
    z = np.array([])
    print("[DEBUG] How many queries:", len(yy.ravel()))
    for idx, query in enumerate(np.c_[xx.ravel(), yy.ravel()]):
        print("[Query] # current query:", idx)
        z = np.append(z, fn_prediction(model, newClazz, clazz_by_index, query, criterion))

    z = np.array(z)
    z = z.reshape(xx.shape)
    cmap_color = plt.cm.viridis if cmap_color is None else plt.cm.get_cmap(cmap_color, _nb_clazz + len(newClazz))
    plt.contourf(xx, yy, z, alpha=0.8, cmap=cmap_color)
    for row in range(0, len(y)):
        plt.scatter(X[row, 0], X[row, 1], c='black', s=40, marker=markers[clazz_by_index[y[row]]], edgecolor='k')
    if savefig:
        plt.savefig('model_plot.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot2D_decision_boundary_det(X, y, h=.01, markers=None):
    markers = list(['+', '*', 'v', 'o', '-', '.', ',']) if markers is None else markers
    _, n_col = X.shape
    _clazz = sorted(list(set(y)))
    _nb_clazz = len(_clazz)

    if n_col > 2:
        raise Exception("Not implemented for n-dimension yet.")

    import matplotlib.pyplot as plt

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    z = np.array([])
    clazz_by_index = dict((clazz, idx) for idx, clazz in enumerate(_clazz, 1))
    # @salmuz ToDo: modify this use base abstract classifier or precise version qda_precise.py
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as classifierLDA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as classifierQDA
    # lda = classifierLDA(solver="svd", store_covariance=True)
    lda = classifierQDA(store_covariance=True)
    lda.fit(X, y)
    for query in np.c_[xx.ravel(), yy.ravel()]:
        evaluate = lda.predict([query])
        z = np.append(z, clazz_by_index[evaluate[0]])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.4)
    for row in range(0, len(y)):
        plt.scatter(X[row, 0], X[row, 1], c='black', s=40, marker=markers[clazz_by_index[y[row]]], edgecolor='k')
    plt.show()
