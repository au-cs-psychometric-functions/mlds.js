import pandas as pd
import numpy as np
import scipy.stats
from scipy import special
from statsmodels.base.data import handle_data
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.compat.python import lzip

FLOAT_EPS = np.finfo(float).eps

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

    safe_links = [Logit, CDFLink]

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

    def deviance(self, endog, mu, var_weights=1., freq_weights=1., scale=1.):
        resid_dev = self._resid_dev(endog, mu)
        return np.sum(resid_dev * freq_weights * var_weights / scale)

    def resid_dev(self, endog, mu, var_weights=1., scale=1.):
        resid_dev = self._resid_dev(endog, mu)
        resid_dev *= var_weights / scale
        return np.sign(endog - mu) * np.sqrt(np.clip(resid_dev, 0., np.inf))

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

    def _resid_dev(self, endog, mu):
        endog_mu = self._clean(endog / mu)
        n_endog_mu = self._clean((1. - endog) / (1. - mu))
        resid_dev = endog * np.log(endog_mu) + (1 - endog) * np.log(n_endog_mu)
        return 2 * self.n * resid_dev

    def loglike_obs(self, endog, mu, var_weights=1., scale=1.):
        n = self.n     # Number of trials
        y = endog * n  # Number of successes

        # note that mu is still in (0,1), i.e. not converted back
        return (special.gammaln(n + 1) - special.gammaln(y + 1) -
                special.gammaln(n - y + 1) + y * np.log(mu / (1 - mu)) +
                n * np.log(1 - mu)) * var_weights

    def resid_anscombe(self, endog, mu, var_weights=1., scale=1.):
        endog = endog * self.n  # convert back to successes
        mu = mu * self.n  # convert back to successes

        def cox_snell(x):
            return special.betainc(2/3., 2/3., x) * special.beta(2/3., 2/3.)

        resid = (self.n ** (2/3.) * (cox_snell(endog * 1. / self.n) -
                                     cox_snell(mu * 1. / self.n)) /
                 (mu * (1 - mu * 1. / self.n) * scale ** 3) ** (1 / 6.))
        resid *= np.sqrt(var_weights)
        return resid

def _check_convergence(criterion, iteration, atol, rtol):
    return np.allclose(criterion[iteration], criterion[iteration + 1],
                       atol=atol, rtol=rtol)

class GLM():
    _formula_max_endog = 2

    def __init__(self, endog, exog, family=None, offset=None,
                 exposure=None, freq_weights=None, var_weights=None,
                 missing='none', **kwargs):
        if exposure is not None:
            exposure = np.log(exposure)
        if offset is not None:  # this should probably be done upstream
            offset = np.asarray(offset)

        if freq_weights is not None:
            freq_weights = np.asarray(freq_weights)
        if var_weights is not None:
            var_weights = np.asarray(var_weights)

        self.freq_weights = freq_weights
        self.var_weights = var_weights

        kwargs['missing'] = missing
        kwargs['offset'] = offset
        kwargs['exposure'] = exposure
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

        if offset is None:
            delattr(self, 'offset')
        if exposure is None:
            delattr(self, 'exposure')

        self.nobs = self.endog.shape[0]

        # things to remove_data
        self._data_attr.extend(['weights', 'mu', 'freq_weights',
                                'var_weights', 'iweights', '_offset_exposure',
                                'n_trials'])
        # register kwds for __init__, offset and exposure are added by super
        self._init_keys.append('family')

        self._setup_binomial()
        # internal usage for recreating a model
        if 'n_trials' in kwargs:
            self.n_trials = kwargs['n_trials']

        # Construct a combined offset/exposure term.  Note that
        # exposure has already been logged if present.
        offset_exposure = 0.
        if hasattr(self, 'offset'):
            offset_exposure = self.offset
        if hasattr(self, 'exposure'):
            offset_exposure = offset_exposure + self.exposure
        self._offset_exposure = offset_exposure

        self.scaletype = None

    def initialize(self):
        self.df_model = np.linalg.matrix_rank(self.exog) - 1

        if (self.freq_weights is not None) and \
           (self.freq_weights.shape[0] == self.endog.shape[0]):
            self.wnobs = self.freq_weights.sum()
            self.df_resid = self.wnobs - self.df_model - 1
        else:
            self.wnobs = self.exog.shape[0]
            self.df_resid = self.exog.shape[0] - self.df_model - 1

    def _get_init_kwds(self):
        kwds = dict(((key, getattr(self, key, None))
                     for key in self._init_keys))

        return kwds

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        data = handle_data(endog, exog, missing, hasconst, **kwargs)
        # kwargs arrays could have changed, easier to just attach here
        for key in kwargs:
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

    def _get_init_kwds(self):
        # this is a temporary fixup because exposure has been transformed
        # see #1609, copied from discrete_model.CountModel
        kwds = super(GLM, self)._get_init_kwds()
        if 'exposure' in kwds and kwds['exposure'] is not None:
            kwds['exposure'] = np.exp(kwds['exposure'])
        return kwds

    def loglike_mu(self, mu, scale=1.):
        scale = float_like(scale, "scale")
        return self.family.loglike(self.endog, mu, self.var_weights,
                                   self.freq_weights, scale)

    def loglike(self, params, scale=None):
        scale = float_like(scale, "scale", optional=True)
        lin_pred = np.dot(self.exog, params) + self._offset_exposure
        expval = self.family.link.inverse(lin_pred)
        if scale is None:
            scale = self.estimate_scale(expval)
        llf = self.family.loglike(self.endog, expval, self.var_weights,
                                  self.freq_weights, scale)
        return llf

    def _update_history(self, tmp_result, mu, history):
        history['params'].append(tmp_result.params)
        history['deviance'].append(self.family.deviance(self.endog, mu,
                                                        self.var_weights,
                                                        self.freq_weights,
                                                        self.scale))
        return history

    def estimate_scale(self, mu):
        if not self.scaletype:
            if isinstance(self.family, Binomial):
                return 1.
            else:
                return self._estimate_x2_scale(mu)

        if isinstance(self.scaletype, float):
            return np.array(self.scaletype)

        if isinstance(self.scaletype, str):
            if self.scaletype.lower() == 'x2':
                return self._estimate_x2_scale(mu)
            elif self.scaletype.lower() == 'dev':
                return (self.family.deviance(self.endog, mu, self.var_weights,
                                             self.freq_weights, 1.) /
                        (self.df_resid))
            else:
                raise ValueError("Scale %s with type %s not understood" %
                                 (self.scaletype, type(self.scaletype)))
        else:
            raise ValueError("Scale %s with type %s not understood" %
                             (self.scaletype, type(self.scaletype)))

    def _estimate_x2_scale(self, mu):
        resid = np.power(self.endog - mu, 2) * self.iweights
        return np.sum(resid / self.family.variance(mu)) / self.df_resid

    def predict(self, params, exog=None, exposure=None, offset=None,
                linear=False):
        # Use fit offset if appropriate
        if offset is None and exog is None and hasattr(self, 'offset'):
            offset = self.offset
        elif offset is None:
            offset = 0.

        if exposure is not None and not isinstance(self.family.link,
                                                   families.links.Log):
            raise ValueError("exposure can only be used with the log link "
                             "function")

        # Use fit exposure if appropriate
        if exposure is None and exog is None and hasattr(self, 'exposure'):
            # Already logged
            exposure = self.exposure
        elif exposure is None:
            exposure = 0.
        else:
            exposure = np.log(np.asarray(exposure))

        if exog is None:
            exog = self.exog

        linpred = np.dot(exog, params) + offset + exposure
        if linear:
            return linpred
        else:
            return self.family.fitted(linpred)

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

    def fit(self, start_params=None, maxiter=100, tol=1e-8,
            scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None,
            full_output=True, disp=False, max_start_irls=3, **kwargs):
        if isinstance(scale, str):
            scale = scale.lower()
            if scale not in ("x2", "dev"):
                raise ValueError(
                    "scale must be either X2 or dev when a string."
                )
        elif scale is not None:
            # GH-6627
            try:
                scale = float(scale)
            except Exception as exc:
                raise type(exc)(
                    "scale must be a float if given and no a string."
                )
        self.scaletype = scale

        attach_wls = kwargs.pop('attach_wls', False)
        atol = kwargs.get('atol')
        rtol = kwargs.get('rtol', 0.)
        tol_criterion = kwargs.get('tol_criterion', 'deviance')
        wls_method = kwargs.get('wls_method', 'lstsq')
        atol = tol if atol is None else atol

        endog = self.endog
        wlsexog = self.exog
        if start_params is None:
            start_params = np.zeros(self.exog.shape[1])
            mu = self.family.starting_mu(self.endog)
            lin_pred = self.family.predict(mu)
        else:
            lin_pred = np.dot(wlsexog, start_params) + self._offset_exposure
            mu = self.family.fitted(lin_pred)
        self.scale = self.estimate_scale(mu)
        dev = self.family.deviance(self.endog, mu, self.var_weights,
                                   self.freq_weights, self.scale)
        if np.isnan(dev):
            raise ValueError("The first guess on the deviance function "
                             "returned a nan.  This could be a boundary "
                             " problem and should be reported.")

        # first guess on the deviance is assumed to be scaled by 1.
        # params are none to start, so they line up with the deviance
        history = dict(params=[np.inf, start_params], deviance=[np.inf, dev])
        converged = False
        criterion = history[tol_criterion]
        # This special case is used to get the likelihood for a specific
        # params vector.
        if maxiter == 0:
            mu = self.family.fitted(lin_pred)
            self.scale = self.estimate_scale(mu)
            wls_results = lm.RegressionResults(self, start_params, None)
            iteration = 0
        for iteration in range(maxiter):
            self.weights = (self.iweights * self.n_trials *
                            self.family.weights(mu))
            wlsendog = (lin_pred + self.family.link.deriv(mu) * (self.endog-mu)
                        - self._offset_exposure)
            wls_mod = reg_tools._MinimalWLS(wlsendog, wlsexog,
                                            self.weights, check_endog=True,
                                            check_weights=True)
            wls_results = wls_mod.fit(method=wls_method)
            lin_pred = np.dot(self.exog, wls_results.params)
            lin_pred += self._offset_exposure
            mu = self.family.fitted(lin_pred)
            history = self._update_history(wls_results, mu, history)
            self.scale = self.estimate_scale(mu)
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)
            converged = _check_convergence(criterion, iteration + 1, atol,
                                           rtol)
            if converged:
                break
        self.mu = mu

        if maxiter > 0:  # Only if iterative used
            wls_method2 = 'pinv' if wls_method == 'lstsq' else wls_method
            wls_model = lm.WLS(wlsendog, wlsexog, self.weights)
            wls_results = wls_model.fit(method=wls_method2)

        print(wls_results.params)
        logLike = self.family.loglike(self.endog, self.mu, var_weights=self.var_weights, freq_weights=self.freq_weights, scale=self.scale)
        print(str(logLike))
        print(str(self.scale))

def mlds(filename):
    data = pd.read_table(filename, sep='\t')
    res = GLM(data['resp'], data.drop('resp', axis=1), family=Binomial(probit()), data=data).fit()
    # print(res.summary())

if __name__ == '__main__':
    mlds('table.txt')