# import scipy.optimize as optimize
import pandas as pd
import numpy as np
from sklearn import linear_model
import pymc3  as pm


def logistic_log_likelihood(X, y):
    logistic = linear_model.LogisticRegression(solver='newton-cg', fit_intercept=False)
    return logistic.fit(X, y)


def logit(X, betas):
    return np.exp(X @ betas) / (1 + np.exp(X @ betas))


def mcmc_conjugate_generalized_linear_model():
    """
    Implementation of method generalized linear model in
        Conjugate priors for generalized linear models - Ming-Hui Chen and Joseph G. Ibrahim

    **Problem of convergence with Iris-setosa (separate perfect logistic case)**
    :return:
    """

    def prior(betas, X, a0, y0):
        cov = X @ betas
        return a0 * (np.sum(y0 * cov) - np.sum(np.log(1.0 + np.exp(cov))))

    def loglikelihood(betas, X, y):
        cov = X @ betas
        return np.sum(y * cov) - np.sum(np.log(1.0 + np.exp(cov)))

    def acceptance(x, x_new):
        accept = np.random.uniform(0, 1)
        alpha = min(1, np.exp(x_new - x))
        return accept < alpha

    def metropolis_hastings(likelihood_computer, prior, transition_model,
                            param_init, iterations, acceptance, X, y, a0, y0):
        betas = param_init
        accepted, rejected = [], []
        for i in range(iterations):
            betas_new = transition_model(betas)
            x_lik = likelihood_computer(betas, X, y)
            x_new_lik = likelihood_computer(betas_new, X, y)
            if acceptance(x_lik + prior(betas, X, a0, y0), x_new_lik + prior(betas_new, X, a0, y0)):
                betas = betas_new
                accepted.append(betas_new)
            else:
                rejected.append(betas_new)

        return np.array(accepted), np.array(rejected)

    from classifip.utils import normalize_minmax

    in_path = "/Users/salmuz/Downloads/datasets/iris.csv"
    data = pd.read_csv(in_path, header=None)
    data.columns = ['x1', 'x2', 'x3', 'x4', 'y']
    X = data.iloc[:, :-1].values
    y = data.y.apply(lambda x: 1 if x == 'Iris-versicolor' else 0)

    fit_logit = logistic_log_likelihood(X, y)
    beta_hat = fit_logit.coef_[0]
    y0, a0 = logit(X, np.array(beta_hat)), 1
    # case with interception used
    # beta_hat = [fit_logit.intercept_[0]]
    # beta_hat.extend(fit_logit.coef_[0])
    # X = np.c_[np.ones(n), X]

    ## Equation (2.10), Theorem 2.3
    delta = np.diag(np.power(np.exp(X @ beta_hat) / (np.power((1 + np.exp(X @ beta_hat)), 2)), 2))
    v = np.diag(np.power(1 + np.exp(X @ beta_hat), 2))
    i_fisher = (1 / a0) * np.linalg.inv(X.T @ delta @ v @ X)

    transition_model = lambda betas: np.random.multivariate_normal(betas, 1 / a0 * i_fisher)
    accepted, rejected = metropolis_hastings(loglikelihood,
                                             prior,
                                             transition_model,
                                             beta_hat,
                                             10000,
                                             acceptance,
                                             X, y, a0, y0)
    # burn-in first 1000 samples
    accepted = accepted[1000:, ]
    import matplotlib.pyplot as plt
    plt.plot(accepted[:, 0], label='b1')
    plt.plot(accepted[:, 1], label='b2')
    plt.plot(accepted[:, 2], label='b3')
    plt.plot(accepted[:, 3], label='b4')
    plt.legend()
    plt.show()
    # plot histogram of posterior distribution
    post = [np.exp(loglikelihood(betas, X, y) + prior(betas, X, a0, y0)) for betas in accepted]
    plt.hist(post)
    plt.show()
    # plot correlation acf for each chain
    import statsmodels.api as sm
    for i in np.arange(0, 4):
        sm.graphics.tsa.plot_acf(accepted[:, i], lags=40)
        plt.show()
    # mean posterior prediction
    print(accepted.mean(axis=0))
    print(beta_hat)


def mcmc_pymc3_logistic():
    import theano
    from theano.compile.ops import as_op
    theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

    in_path = "/Users/salmuz/Downloads/datasets/iris.csv"
    data = pd.read_csv(in_path, header=None)
    data.columns = ['x1', 'x2', 'x3', 'x4', 'y']
    X = data.iloc[:, :-1].values
    y = data.y.apply(lambda x: 1 if x == 'Iris-setosa' else 0)

    n, p = X.shape
    basic_model = pm.Model()
    fit_logit = logistic_log_likelihood(X, y)
    beta_hat = fit_logit.coef_[0]
    with basic_model:
        betas = pm.MvNormal('betas', mu=beta_hat, cov=np.eye(p), shape=(p,))

        @as_op(itypes=[theano.tensor.dvector], otypes=[theano.tensor.dscalar])
        def likelihood(betas):
            return np.array(np.exp(np.sum(y * (X @ betas)) - np.sum(np.log(1.0 + np.exp(X @ betas)))))

        likelihood = pm.Potential('likelihood', likelihood(betas))
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(10000, step=step, start=start)

    import matplotlib.pyplot as plt

    pm.traceplot(trace)
    plt.show()
    pm.plots.autocorrplot(trace, figsize=(17, 5))
    plt.show()


mcmc_pymc3_logistic()
# mcmc_conjugate_generalized_linear_model()
