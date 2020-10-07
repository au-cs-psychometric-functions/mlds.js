import numpy as np
import scipy.stats
from scipy import special

FLOAT_EPS = np.finfo(float).eps

def default_clip(p):
    return np.clip(p, FLOAT_EPS, 1 - FLOAT_EPS)

def logit():
    def link(p):
        p = default_clip(p)
        return np.log(p / (1. - p))
    def inverse(z):
        z = np.assarray(z)
        t = np.exp(-z)
        return 1. / (1. + t)
    def deriv(p):
        p = default_clip(p)
        return 1. / (p * (1 - p))
    def inverse_deriv(z):
        t = np.exp(z)
        return t/(1 + t)**2
    link.inverse = inverse
    link.deriv = deriv
    link.inverse_deriv = inverse_deriv
    return link

def probit(dbn=scipy.stats.norm):
    def link(p):
        p = default_clip(p)
        return dbn.ppf(p)
    def inverse(z):
        return dbn.cdf(z)
    def deriv(p):
        p = default_clip(p)
        return 1 / dbn.pdf(dbn.ppf(p))
    def inverse_deriv(z):
        return 1 / deriv(inverse(z))
    link.inverse = inverse
    link.deriv = deriv
    link.inverse_deriv = inverse_deriv
    return link

def cauchy(dbn=scipy.stats.cauchy):
    return probit(dbn)

def log():
    def clean(x):
        return np.clip(x, FLOAT_EPS, np.inf)
    def link(p, **extra):
        x = clean(x)
        return np.log(x)
    def inverse(z):
        return np.exp(z)
    def deriv(p):
        p = clean(p)
        return 1. / p
    def inverse_deriv(z):
        return np.exp(z)
    link.inverse = inverse
    link.deriv = deriv
    link.inverse_deriv = inverse_deriv
    return link

def cloglog():
    def link(p):
        p = default_clip(p)
        return np.log(-np.log(1 - p))
    def inverse(z):
        return 1 - np.exp(-np.exp(z))
    def deriv(p):
        p = default_clip(p)
        return 1. / ((p - 1) * (np.log(1 - p)))
    def inverse_deriv(z):
        return np.exp(z - np.exp(z))
    link.inverse = inverse
    link.deriv = deriv
    link.inverse_deriv = inverse_deriv

class Binomial():
    def __init__(self, link=None):
        if link is None:
            link = logit()
        self.n = 1

        self.link = link

    def starting_mu(self, y):
        return (y + .5)/2

    def initialize(self, endog):
        return endog

    def weights(self, mu):
        p = default_clip(mu / self.n)
        variance = p * (1 - p) * self.n
        return 1. / (self.link.deriv(mu)**2 * variance)

    def deviance(self, endog, mu):
        endog_mu = self._clean(endog / mu)
        n_endog_mu = self._clean((1. - endog) / (1. - mu))
        resid_dev = endog * np.log(endog_mu) + (1 - endog) * np.log(n_endog_mu)
        return np.sum(2 * self.n * resid_dev)

    def fitted(self, lin_pred):
        fits = self.link.inverse(lin_pred)
        return fits

    def predict(self, mu):
        return self.link(mu)

    def loglike(self, endog, mu):
        ll_obs = self.loglike_obs(endog, mu)
        return np.sum(ll_obs)

    def _clean(self, x):
        return np.clip(x, FLOAT_EPS, np.inf)

    def loglike_obs(self, endog, mu):
        n = self.n     # Number of trials
        y = endog * n  # Number of successes

        # note that mu is still in (0,1), i.e. not converted back
        return (special.gammaln(n + 1) - special.gammaln(y + 1) -
                special.gammaln(n - y + 1) + y * np.log(mu / (1 - mu)) +
                n * np.log(1 - mu))

def _check_convergence(criterion, iteration, atol, rtol):
    return np.allclose(criterion[iteration], criterion[iteration + 1],
                       atol=atol, rtol=rtol)

class GLM():
    def __init__(self, endog, exog, linkname='probit'):
        self.linkname = linkname
        link = None
        if linkname == 'logit':
            link = logit()
        elif linkname == 'probit':
            link = probit()
        elif linkname == 'cauchy':
            link = cauchy()
        elif linkname == 'log':
            link = log()
        elif linkname == 'cloglog':
            link = cloglog()
        else:
            raise Exception('Invalid Link Name')

        missing = 'none'
        hasconst = None

        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)

        self.family = Binomial(link)

        # self.freq_weights = np.ones(len(endog))
        # self.var_weights = np.ones(len(endog))
        # self.iweights = np.asarray(self.freq_weights * self.var_weights)
        # self.nobs = self.endog.shape[0]

        self.endog = self.family.initialize(self.endog)

    def fit(self):
        maxiter = 100
        atol = 1e-8
        rtol = 0

        endog = self.endog
        wlsexog = self.exog

        start_params = np.zeros(self.exog.shape[1])
        mu = self.family.starting_mu(self.endog)
        lin_pred = self.family.predict(mu)
        
        converged = False

        dev = [np.inf, self.family.deviance(self.endog, mu)]

        for iteration in range(maxiter):
            self.weights = self.family.weights(mu)
            wlsendog = (lin_pred + self.family.link.deriv(mu) * (self.endog-mu))

            w_half = np.sqrt(self.weights)
            mendog = w_half * wlsendog
            mexog = np.asarray(w_half)[:, None] * wlsexog
            wls_results, _, _, _ = np.linalg.lstsq(mexog, mendog, rcond=-1)

            lin_pred = np.dot(self.exog, wls_results)
            mu = self.family.fitted(lin_pred)
            dev.append(self.family.deviance(self.endog, mu))
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)
            converged = _check_convergence(dev, iteration + 1, atol,
                                           rtol)
            if converged:
                break
        self.mu = mu

        wlsendog = np.asarray(wlsendog) * np.sqrt(self.weights)
        wlsexog = np.asarray(wlsexog) * np.sqrt(self.weights)[:, None]
        wlsexog = np.linalg.pinv(wlsexog, rcond=1e-15)
        wls_results = np.dot(wlsexog, wlsendog)

        logLike = self.family.loglike(self.endog, self.mu)
        return Summary(self.linkname, wls_results, logLike)

class Summary():
    def __init__(self, link, params, loglike):
        self.link = link
        self.params = [0] + params
        self.loglike = loglike

    def print(self):
        print('Method: GLM')
        print('Link: {}\n'.format(self.link))
        for param in self.params:
            print(param)
        print()
        print('logLik: {}'.format(self.loglike))

def mlds(filename):
    data = []
    with open('data.txt', 'r') as f:
        for line in f.readlines():
            data.append(list(map(int, line.strip().split('\t'))))

    for row in data:
        if row[1] > row[3]:
            row[1], row[3] = row[3], row[1]
            row[0] = 1 - row[0]

    mx = max(sum(data, []))
    table = []
    for row in data:
        arr = [0] * mx
        arr[row[1] - 1] = 1
        arr[row[2] - 1] = -2
        arr[row[3] - 1] = 1
        arr[0] = row[0]
        table.append(arr)

    y = []
    x = []
    for row in table:
        y.append(row[0])
        x.append(row[1:])

    summary = GLM(y, x).fit()
    summary.print()

if __name__ == '__main__':
    mlds('table.txt')