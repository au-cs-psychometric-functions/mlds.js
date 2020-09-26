import pandas as pd
import numpy as np
import scipy.stats
from scipy import special
# from statsmodels.base.data import handle_data
from statsmodels.base.data import ModelData, PandasData, PatsyData
import statsmodels.tools.data as data_util
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm

FLOAT_EPS = np.finfo(float).eps

def handle_data_class_factory(endog, exog):
    """
    Given inputs
    """
    if data_util._is_using_ndarray_type(endog, exog):
        print('model')
        klass = ModelData
    elif data_util._is_using_pandas(endog, exog):
        print('pandas')
        klass = PandasData
    elif data_util._is_using_patsy(endog, exog):
        print('patsy')
        klass = PatsyData
    # keep this check last
    elif data_util._is_using_ndarray(endog, exog):
        print('model')
        klass = ModelData
    else:
        raise ValueError('unrecognized data structures: %s / %s' %
                         (type(endog), type(exog)))
    return klass


def handle_data(endog, exog, missing='none', hasconst=None, **kwargs):
    # deal with lists and tuples up-front
    if isinstance(endog, (list, tuple)):
        print('endog')
        endog = np.asarray(endog)
    if isinstance(exog, (list, tuple)):
        print('exog')
        exog = np.asarray(exog)

    klass = handle_data_class_factory(endog, exog)
    return klass(endog, exog=exog, missing=missing, hasconst=hasconst,
                 **kwargs)


class Link(object):
    pass

class Logit(Link):
    def _clean(self, p):
        return np.clip(p, FLOAT_EPS, 1. - FLOAT_EPS)

    def __call__(self, p):
        p = self._clean(p)
        return np.log(p / (1. - p))

    def inverse(self, z):
        z = np.asarray(z)
        t = np.exp(-z)
        return 1. / (1. + t)

    def deriv(self, p):
        p = self._clean(p)
        return 1. / (p * (1 - p))

    def inverse_deriv(self, z):
        t = np.exp(z)
        return t/(1 + t)**2

class logit(Logit):
    pass

class CDFLink(Logit):
    def __init__(self, dbn=scipy.stats.norm):
        self.dbn = dbn

    def __call__(self, p):
        p = self._clean(p)
        return self.dbn.ppf(p)

    def inverse(self, z):
        return self.dbn.cdf(z)

    def deriv(self, p):
        p = self._clean(p)
        return 1. / self.dbn.pdf(self.dbn.ppf(p))

    def inverse_deriv(self, z):
        return 1/self.deriv(self.inverse(z))

class probit(CDFLink):
    pass

class cauchy(CDFLink):
    def __init__(self):
        super(cauchy, self).__init__(dbn=scipy.stats.cauchy)

class Log(Link):
    def _clean(self, x):
        return np.clip(x, FLOAT_EPS, np.inf)

    def __call__(self, p, **extra):
        x = self._clean(p)
        return np.log(x)

    def inverse(self, z):
        return np.exp(z)

    def deriv(self, p):
        p = self._clean(p)
        return 1. / p

    def inverse_deriv(self, z):
        return np.exp(z)


class log(Log):
    pass

class CLogLog(Logit):
    def __call__(self, p):
        p = self._clean(p)
        return np.log(-np.log(1 - p))

    def inverse(self, z):
        return 1 - np.exp(-np.exp(z))

    def deriv(self, p):
        p = self._clean(p)
        return 1. / ((p - 1) * (np.log(1 - p)))

    def inverse_deriv(self, z):
        return np.exp(z - np.exp(z))

class cloglog(CLogLog):
    pass

class BinomialVariance(object):
    def __init__(self, n=1):
        self.n = n

    def _clean(self, p):
        return np.clip(p, FLOAT_EPS, 1 - FLOAT_EPS)

    def __call__(self, mu):
        p = self._clean(mu / self.n)
        return p * (1 - p) * self.n

    # TODO: inherit from super
    def deriv(self, mu):
        return 1 - 2*mu

class Binomial():
    links = [logit, probit, cauchy, log, cloglog]

    def _setlink(self, link):
        self._link = link
        if not isinstance(link, Link):
            raise TypeError("The input should be a valid Link object.")
        if hasattr(self, "links"):
            validlink = max([isinstance(link, _) for _ in self.links])
            if not validlink:
                errmsg = "Invalid link for family, should be in %s. (got %s)"
                raise ValueError(errmsg % (repr(self.links), link))

    def _getlink(self):
        return self._link

    # link property for each family is a pointer to link instance
    link = property(_getlink, _setlink, doc="Link function for family")

    def __init__(self, link=None):  # , n=1.):
        if link is None:
            link = logit()
        self.n = 1

        self.link = link
        self.variance = BinomialVariance(n=self.n)

    def starting_mu(self, y):
        return (y + .5)/2

    def initialize(self, endog, freq_weights):
        if endog.ndim > 1 and endog.shape[1] > 2:
            raise ValueError('endog has more than 2 columns. The Binomial '
                             'link supports either a single response variable '
                             'or a paired response variable.')
        elif endog.ndim > 1 and endog.shape[1] > 1:
            y = endog[:, 0]
            # overwrite self.freq_weights for deviance below
            self.n = endog.sum(1)
            return y*1./self.n, self.n
        else:
            return endog, np.ones(endog.shape[0])

    def weights(self, mu):
        return 1. / (self.link.deriv(mu)**2 * self.variance(mu))

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

    def loglike(self, endog, mu, var_weights=1., freq_weights=1., scale=1.):
        ll_obs = self.loglike_obs(endog, mu, var_weights, scale)
        return np.sum(ll_obs * freq_weights)

    def _clean(self, x):
        return np.clip(x, FLOAT_EPS, np.inf)

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        n = self.n     # Number of trials
        y = endog * n  # Number of successes

        # note that mu is still in (0,1), i.e. not converted back
        return (special.gammaln(n + 1) - special.gammaln(y + 1) -
                special.gammaln(n - y + 1) + y * np.log(mu / (1 - mu)) +
                n * np.log(1 - mu)) * var_weights

def _check_convergence(criterion, iteration, atol, rtol):
    return np.allclose(criterion[iteration], criterion[iteration + 1],
                       atol=atol, rtol=rtol)

class GLM():
    _formula_max_endog = 2

    def __init__(self, endog, exog, family=None, freq_weights=None, var_weights=None,
                 missing='none', **kwargs):
        if freq_weights is not None:
            freq_weights = np.asarray(freq_weights)
        if var_weights is not None:
            var_weights = np.asarray(var_weights)

        self.freq_weights = freq_weights
        self.var_weights = var_weights

        kwargs['missing'] = missing
        kwargs['freq_weights'] = freq_weights
        kwargs['var_weights'] = var_weights

        missing = kwargs.pop('missing', 'none')
        hasconst = kwargs.pop('hasconst', None)
        self.data = self._handle_data(endog, exog, missing, hasconst,
                                      **kwargs)
        self.k_constant = self.data.k_constant
        self.exog = self.data.exog
        self.endog = self.data.endog
        self._data_attr = []
        self._data_attr.extend(['exog', 'endog', 'data.exog', 'data.endog'])
        if 'formula' not in kwargs:  # will not be able to unpickle without these
            self._data_attr.extend(['data.orig_endog', 'data.orig_exog'])
        # store keys for extras if we need to recreate model instance
        # we do not need 'missing', maybe we need 'hasconst'
        self._init_keys = list(kwargs.keys())
        if hasconst is not None:
            self._init_keys.append('hasconst')

        self.initialize()

        self._check_inputs(family, self.endog,
                           self.freq_weights, self.var_weights)

        self.nobs = self.endog.shape[0]

        # things to remove_data
        self._data_attr.extend(['weights', 'mu', 'freq_weights',
                                'var_weights', 'iweights',
                                'n_trials'])
        # register kwds for __init__, offset and exposure are added by super
        self._init_keys.append('family')

        self._setup_binomial()
        # internal usage for recreating a model
        if 'n_trials' in kwargs:
            self.n_trials = kwargs['n_trials']

    def initialize(self):
        self.df_model = np.linalg.matrix_rank(self.exog) - 1

        if (self.freq_weights is not None) and \
           (self.freq_weights.shape[0] == self.endog.shape[0]):
            self.wnobs = self.freq_weights.sum()
            self.df_resid = self.wnobs - self.df_model - 1
        else:
            self.wnobs = self.exog.shape[0]
            self.df_resid = self.exog.shape[0] - self.df_model - 1

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        data = handle_data(endog, exog, missing, hasconst, **kwargs)
        # kwargs arrays could have changed, easier to just attach here
        for key in kwargs:
            print('kwargs')
            if key in ['design_info', 'formula']:  # leave attached to data
                continue
            # pop so we do not start keeping all these twice or references
            try:
                setattr(self, key, data.__dict__.pop(key))
            except KeyError:  # panel already pops keys in data handling
                pass
        return data

    @property
    def endog_names(self):
        return self.data.ynames

    @property
    def exog_names(self):
        return self.data.xnames

    def _check_inputs(self, family, endog, freq_weights, var_weights):
        if family is None:
            family = Binomial()
        self.family = family

        if freq_weights is not None:
            if freq_weights.shape[0] != endog.shape[0]:
                raise ValueError("freq weights not the same length as endog")
            if len(freq_weights.shape) > 1:
                raise ValueError("freq weights has too many dimensions")

        # internal flag to store whether freq_weights were not None
        self._has_freq_weights = (self.freq_weights is not None)
        if self.freq_weights is None:
            self.freq_weights = np.ones((endog.shape[0]))
            # TODO: check do we want to keep None as sentinel for freq_weights

        if np.shape(self.freq_weights) == () and self.freq_weights > 1:
            self.freq_weights = (self.freq_weights *
                                 np.ones((endog.shape[0])))

        if var_weights is not None:
            if var_weights.shape[0] != endog.shape[0]:
                raise ValueError("var weights not the same length as endog")
            if len(var_weights.shape) > 1:
                raise ValueError("var weights has too many dimensions")

        # internal flag to store whether var_weights were not None
        self._has_var_weights = (var_weights is not None)
        if var_weights is None:
            self.var_weights = np.ones((endog.shape[0]))
            # TODO: check do we want to keep None as sentinel for var_weights
        self.iweights = np.asarray(self.freq_weights * self.var_weights)

    def loglike_mu(self, mu, scale=1.):
        scale = float_like(scale, "scale")
        return self.family.loglike(self.endog, mu, self.var_weights,
                                   self.freq_weights, scale)

    def loglike(self, params, scale=None):
        scale = float_like(scale, "scale", optional=True)
        lin_pred = np.dot(self.exog, params)
        expval = self.family.link.inverse(lin_pred)
        if scale is None:
            scale = 1
        llf = self.family.loglike(self.endog, expval, self.var_weights,
                                  self.freq_weights, scale)
        return llf

    def _update_history(self, tmp_result, mu, history):
        history['params'].append(tmp_result.params)
        history['deviance'].append(self.family.deviance(self.endog, mu))
        return history

    def _setup_binomial(self):
        # this checks what kind of data is given for Binomial.
        # family will need a reference to endog if this is to be removed from
        # preprocessing
        self.n_trials = np.ones((self.endog.shape[0]))  # For binomial
        if isinstance(self.family, Binomial):
            tmp = self.family.initialize(self.endog, self.freq_weights)
            self.endog = tmp[0]
            self.n_trials = tmp[1]
            self._init_keys.append('n_trials')

    def fit(self):
        maxiter = 100
        atol = 1e-8
        rtol = 0

        endog = self.endog
        wlsexog = self.exog

        start_params = np.zeros(self.exog.shape[1])
        mu = self.family.starting_mu(self.endog)
        lin_pred = self.family.predict(mu)
        self.scale = 1
        
        dev = self.family.deviance(self.endog, mu)

        # first guess on the deviance is assumed to be scaled by 1.
        # params are none to start, so they line up with the deviance
        history = dict(params=[np.inf, start_params], deviance=[np.inf, dev])
        converged = False
        criterion = history['deviance']

        for iteration in range(maxiter):
            self.weights = (self.iweights * self.n_trials *
                            self.family.weights(mu))
            wlsendog = (lin_pred + self.family.link.deriv(mu) * (self.endog-mu))
            wls_mod = reg_tools._MinimalWLS(wlsendog, wlsexog,
                                            self.weights, check_endog=True,
                                            check_weights=True)
            wls_results = wls_mod.fit(method='lstsq')
            lin_pred = np.dot(self.exog, wls_results.params)
            mu = self.family.fitted(lin_pred)
            history = self._update_history(wls_results, mu, history)
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)
            converged = _check_convergence(criterion, iteration + 1, atol,
                                           rtol)
            if converged:
                break
        self.mu = mu

        wls_model = lm.WLS(wlsendog, wlsexog, self.weights)
        wls_results = wls_model.fit(method='pinv')

        logLike = self.family.loglike(self.endog, self.mu, var_weights=self.var_weights, freq_weights=self.freq_weights, scale=self.scale)
        return Summary('probit', wls_results.params, self.scale, logLike)

class Summary():
    def __init__(self, link, params, scale, loglike):
        self.link = link
        self.params = [0] + params
        self.scale = scale
        self.loglike = loglike

    def print(self):
        print('Method: GLM')
        print('Link: {}\n'.format(self.link))
        for param in self.params:
            print(param)
        print()
        print('sigma: {}'.format(self.scale))
        print('logLik: {}'.format(self.loglike))

def mlds(filename):
    data = pd.read_table(filename, sep='\t')
    summary = GLM(data['resp'], data.drop('resp', axis=1), family=Binomial(probit()), data=data).fit()
    summary.print()

if __name__ == '__main__':
    mlds('table.txt')