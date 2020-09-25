import warnings
import inspect
import pandas as pd
import numpy as np
import scipy.stats
from scipy import special
import statsmodels.formula.api as smf
from statsmodels.tools.decorators import (cache_readonly, cached_data, cached_value)
from statsmodels.graphics._regressionplots_doc import (
    _plot_added_variable_doc,
    _plot_partial_residuals_doc,
    _plot_ceres_residuals_doc)
from statsmodels.base.data import handle_data
from statsmodels.formula import handle_formula_data
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.compat.python import lzip

FLOAT_EPS = np.finfo(float).eps

class Link(object):
    def __call__(self, p):
        return NotImplementedError

    def inverse(self, z):
        return NotImplementedError

    def deriv(self, p):
        return NotImplementedError

    def deriv2(self, p):
        from statsmodels.tools.numdiff import approx_fprime_cs
        # TODO: workaround proplem with numdiff for 1d
        return np.diag(approx_fprime_cs(p, self.deriv))

    def inverse_deriv(self, z):
        return 1 / self.deriv(self.inverse(z))

    def inverse_deriv2(self, z):
        iz = self.inverse(z)
        return -self.deriv2(iz) / self.deriv(iz)**3

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

    def deriv2(self, p):
        v = p * (1 - p)
        return (2*p - 1) / v**2

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

    def deriv2(self, p):
        from statsmodels.tools.numdiff import approx_fprime
        p = np.atleast_1d(p)
        # Note: special function for norm.ppf does not support complex
        return np.diag(approx_fprime(p, self.deriv, centered=True))

    def inverse_deriv(self, z):
        return 1/self.deriv(self.inverse(z))

class probit(CDFLink):
    pass

class cauchy(CDFLink):
    def __init__(self):
        super(cauchy, self).__init__(dbn=scipy.stats.cauchy)

    def deriv2(self, p):
        a = np.pi * (p - 0.5)
        d2 = 2 * np.pi**2 * np.sin(a) / np.cos(a)**3
        return d2

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

    def deriv2(self, p):
        p = self._clean(p)
        return -1. / p**2

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

    def deriv2(self, p):
        p = self._clean(p)
        fl = np.log(1 - p)
        d2 = -1 / ((1 - p)**2 * fl)
        d2 *= 1 + 1 / fl
        return d2

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

        if inspect.isclass(link):
            warnmssg = "Calling Family(..) with a link class as argument "
            warnmssg += "is deprecated.\n"
            warnmssg += "Use an instance of a link class instead."
            lvl = 2 if type(self) is Family else 3
            warnings.warn(warnmssg,
                          category=DeprecationWarning, stacklevel=lvl)
            self.link = link()
        else:
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

        if (family is not None) and not isinstance(family.link,
                                                   tuple(family.safe_links)):

            warnings.warn((f"The {type(family.link).__name__} link function "
                           "does not respect the domain of the "
                           f"{type(family).__name__} family."),
                          DomainWarning)

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

        self._check_inputs(family, self.offset, self.exposure, self.endog,
                           self.freq_weights, self.var_weights)
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

    def _fit_zeros(self, keep_index=None, start_params=None,
                   return_auxiliary=False, k_params=None, **fit_kwds):
        # we need to append index of extra params to keep_index as in
        # NegativeBinomial
        if hasattr(self, 'k_extra') and self.k_extra > 0:
            # we cannot change the original, TODO: should we add keep_index_params?
            keep_index = np.array(keep_index, copy=True)
            k = self.exog.shape[1]
            extra_index = np.arange(k, k + self.k_extra)
            keep_index_p = np.concatenate((keep_index, extra_index))
        else:
            keep_index_p = keep_index

        # not all models support start_params, drop if None, hide them in fit_kwds
        if start_params is not None:
            fit_kwds['start_params'] = start_params[keep_index_p]
            k_params = len(start_params)
            # ignore k_params in this case, or verify consisteny?

        # build auxiliary model and fit
        init_kwds = self._get_init_kwds()
        mod_constr = self.__class__(self.endog, self.exog[:, keep_index],
                                    **init_kwds)
        res_constr = mod_constr.fit(**fit_kwds)
        # switch name, only need keep_index for params below
        keep_index = keep_index_p

        if k_params is None:
            k_params = self.exog.shape[1]
            k_params += getattr(self, 'k_extra', 0)

        params_full = np.zeros(k_params)
        params_full[keep_index] = res_constr.params

        # create dummy results Instance, TODO: wire up properly
        # TODO: this could be moved into separate private method if needed
        # discrete L1 fit_regularized doens't reestimate AFAICS
        # RLM does not have method, disp nor warn_convergence keywords
        # OLS, WLS swallows extra kwds with **kwargs, but does not have method='nm'
        try:
            # Note: addding full_output=False causes exceptions
            res = self.fit(maxiter=0, disp=0, method='nm', skip_hessian=True,
                           warn_convergence=False, start_params=params_full)
            # we get a wrapper back
        except (TypeError, ValueError):
            res = self.fit()

        # Warning: make sure we are not just changing the wrapper instead of
        # results #2400
        # TODO: do we need to change res._results.scale in some models?
        if hasattr(res_constr.model, 'scale'):
            # Note: res.model is self
            # GLM problem, see #2399,
            # TODO: remove from model if not needed anymore
            res.model.scale = res._results.scale = res_constr.model.scale

        if hasattr(res_constr, 'mle_retvals'):
            res._results.mle_retvals = res_constr.mle_retvals
            # not available for not scipy optimization, e.g. glm irls
            # TODO: what retvals should be required?
            # res.mle_retvals['fcall'] = res_constr.mle_retvals.get('fcall', np.nan)
            # res.mle_retvals['iterations'] = res_constr.mle_retvals.get(
            #                                                 'iterations', np.nan)
            # res.mle_retvals['converged'] = res_constr.mle_retvals['converged']
        # overwrite all mle_settings
        if hasattr(res_constr, 'mle_settings'):
            res._results.mle_settings = res_constr.mle_settings

        res._results.params = params_full
        if (not hasattr(res._results, 'normalized_cov_params') or
                res._results.normalized_cov_params is None):
            res._results.normalized_cov_params = np.zeros((k_params, k_params))
        else:
            res._results.normalized_cov_params[...] = 0

        # fancy indexing requires integer array
        keep_index = np.array(keep_index)
        res._results.normalized_cov_params[keep_index[:, None], keep_index] = \
            res_constr.normalized_cov_params
        k_constr = res_constr.df_resid - res._results.df_resid
        if hasattr(res_constr, 'cov_params_default'):
            res._results.cov_params_default = np.zeros((k_params, k_params))
            res._results.cov_params_default[keep_index[:, None], keep_index] = \
                res_constr.cov_params_default
        if hasattr(res_constr, 'cov_type'):
            res._results.cov_type = res_constr.cov_type
            res._results.cov_kwds = res_constr.cov_kwds

        res._results.keep_index = keep_index
        res._results.df_resid = res_constr.df_resid
        res._results.df_model = res_constr.df_model

        res._results.k_constr = k_constr
        res._results.results_constrained = res_constr

        # special temporary workaround for RLM
        # need to be able to override robust covariances
        if hasattr(res.model, 'M'):
            del res._results._cache['resid']
            del res._results._cache['fittedvalues']
            del res._results._cache['sresid']
            cov = res._results._cache['bcov_scaled']
            # inplace adjustment
            cov[...] = 0
            cov[keep_index[:, None], keep_index] = res_constr.bcov_scaled
            res._results.cov_params_default = cov

        return res

    def _fit_collinear(self, atol=1e-14, rtol=1e-13, **kwds):
        x = self.exog
        tol = atol + rtol * x.var(0)
        r = np.linalg.qr(x, mode='r')
        mask = np.abs(r.diagonal()) < np.sqrt(tol)
        # TODO add to results instance
        # idx_collinear = np.where(mask)[0]
        idx_keep = np.where(~mask)[0]
        return self._fit_zeros(keep_index=idx_keep, **kwds)

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

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None,
                     *args, **kwargs):
        # TODO: provide a docs template for args/kwargs from child models
        # TODO: subset could use syntax. issue #469.
        if subset is not None:
            data = data.loc[subset]
        eval_env = kwargs.pop('eval_env', None)
        if eval_env is None:
            eval_env = 2
        elif eval_env == -1:
            from patsy import EvalEnvironment
            eval_env = EvalEnvironment({})
        elif isinstance(eval_env, int):
            eval_env += 1  # we're going down the stack again
        missing = kwargs.get('missing', 'drop')
        if missing == 'none':  # with patsy it's drop or raise. let's raise.
            missing = 'raise'

        tmp = handle_formula_data(data, None, formula, depth=eval_env,
                                  missing=missing)
        ((endog, exog), missing_idx, design_info) = tmp
        max_endog = cls._formula_max_endog
        if (max_endog is not None and
                endog.ndim > 1 and endog.shape[1] > max_endog):
            raise ValueError('endog has evaluated to an array with multiple '
                             'columns that has shape {0}. This occurs when '
                             'the variable converted to endog is non-numeric'
                             ' (e.g., bool or str).'.format(endog.shape))
        if drop_cols is not None and len(drop_cols) > 0:
            cols = [x for x in exog.columns if x not in drop_cols]
            if len(cols) < len(exog.columns):
                exog = exog[cols]
                cols = list(design_info.term_names)
                for col in drop_cols:
                    try:
                        cols.remove(col)
                    except ValueError:
                        pass  # OK if not present
                design_info = design_info.subset(cols)

        kwargs.update({'missing_idx': missing_idx,
                       'missing': missing,
                       'formula': formula,  # attach formula for unpckling
                       'design_info': design_info})
        mod = cls(endog, exog, *args, **kwargs)
        mod.formula = formula

        # since we got a dataframe, attach the original
        mod.data.frame = data
        return mod

    @property
    def endog_names(self):
        return self.data.ynames

    @property
    def exog_names(self):
        return self.data.xnames

    def _check_inputs(self, family, offset, exposure, endog, freq_weights,
                      var_weights):

        # Default family is Gaussian
        if family is None:
            family = Binomial()
        self.family = family

        if exposure is not None:
            if not isinstance(self.family.link, families.links.Log):
                raise ValueError("exposure can only be used with the log "
                                 "link function")
            elif exposure.shape[0] != endog.shape[0]:
                raise ValueError("exposure is not the same length as endog")

        if offset is not None:
            if offset.shape[0] != endog.shape[0]:
                raise ValueError("offset is not the same length as endog")

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

    def score_obs(self, params, scale=None):
        scale = float_like(scale, "scale", optional=True)
        score_factor = self.score_factor(params, scale=scale)
        return score_factor[:, None] * self.exog

    def score(self, params, scale=None):
        scale = float_like(scale, "scale", optional=True)
        score_factor = self.score_factor(params, scale=scale)
        return np.dot(score_factor, self.exog)

    def score_factor(self, params, scale=None):
        scale = float_like(scale, "scale", optional=True)
        mu = self.predict(params)
        if scale is None:
            scale = self.estimate_scale(mu)

        score_factor = (self.endog - mu) / self.family.link.deriv(mu)
        score_factor /= self.family.variance(mu)
        score_factor *= self.iweights * self.n_trials

        if not scale == 1:
            score_factor /= scale

        return score_factor

    def hessian_factor(self, params, scale=None, observed=True):
        # calculating eim_factor
        mu = self.predict(params)
        if scale is None:
            scale = self.estimate_scale(mu)

        eim_factor = 1 / (self.family.link.deriv(mu)**2 *
                          self.family.variance(mu))
        eim_factor *= self.iweights * self.n_trials

        if not observed:
            if not scale == 1:
                eim_factor /= scale
            return eim_factor

        # calculating oim_factor, eim_factor is with scale=1

        score_factor = self.score_factor(params, scale=1.)
        if eim_factor.ndim > 1 or score_factor.ndim > 1:
            raise RuntimeError('something wrong')

        tmp = self.family.variance(mu) * self.family.link.deriv2(mu)
        tmp += self.family.variance.deriv(mu) * self.family.link.deriv(mu)

        tmp = score_factor * tmp
        # correct for duplicatee iweights in oim_factor and score_factor
        tmp /= self.iweights * self.n_trials
        oim_factor = eim_factor * (1 + tmp)

        if tmp.ndim > 1:
            raise RuntimeError('something wrong')

        if not scale == 1:
            oim_factor /= scale

        return oim_factor

    def hessian(self, params, scale=None, observed=None):
        if observed is None:
            if getattr(self, '_optim_hessian', None) == 'eim':
                observed = False
            else:
                observed = True
        scale = float_like(scale, "scale", optional=True)
        tmp = getattr(self, '_tmp_like_exog', np.empty_like(self.exog, dtype=float))

        factor = self.hessian_factor(params, scale=scale, observed=observed)
        np.multiply(self.exog.T, factor, out=tmp.T)
        return -tmp.T.dot(self.exog)

    def information(self, params, scale=None):
        """
        Fisher information matrix.
        """
        scale = float_like(scale, "scale", optional=True)
        return self.hessian(params, scale=scale, observed=False)

    def _deriv_mean_dparams(self, params):
        lin_pred = self.predict(params, linear=True)
        idl = self.family.link.inverse_deriv(lin_pred)
        dmat = self.exog * idl[:, None]
        return dmat

    def _deriv_score_obs_dendog(self, params, scale=None):
        scale = float_like(scale, "scale", optional=True)
        mu = self.predict(params)
        if scale is None:
            scale = self.estimate_scale(mu)

        score_factor = 1 / self.family.link.deriv(mu)
        score_factor /= self.family.variance(mu)
        score_factor *= self.iweights * self.n_trials

        if not scale == 1:
            score_factor /= scale

        return score_factor[:, None] * self.exog

    def score_test(self, params_constrained, k_constraints=None,
                   exog_extra=None, observed=True):
        if exog_extra is None:
            if k_constraints is None:
                raise ValueError('if exog_extra is None, then k_constraints'
                                 'needs to be given')

            score = self.score(params_constrained)
            hessian = self.hessian(params_constrained, observed=observed)

        else:
            # exog_extra = np.asarray(exog_extra)
            if k_constraints is None:
                k_constraints = 0

            ex = np.column_stack((self.exog, exog_extra))
            k_constraints += ex.shape[1] - self.exog.shape[1]

            score_factor = self.score_factor(params_constrained)
            score = (score_factor[:, None] * ex).sum(0)
            hessian_factor = self.hessian_factor(params_constrained,
                                                 observed=observed)
            hessian = -np.dot(ex.T * hessian_factor, ex)

        from scipy import stats
        # TODO check sign, why minus?
        chi2stat = -score.dot(np.linalg.solve(hessian, score[:, None]))
        pval = stats.chi2.sf(chi2stat, k_constraints)
        # return a stats results instance instead?  Contrast?
        return chi2stat, pval, k_constraints

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

    def estimate_tweedie_power(self, mu, method='brentq', low=1.01, high=5.):
        if method == 'brentq':
            from scipy.optimize import brentq

            def psi_p(power, mu):
                scale = ((self.iweights * (self.endog - mu) ** 2 /
                          (mu ** power)).sum() / self.df_resid)
                return (np.sum(self.iweights * ((self.endog - mu) ** 2 /
                               (scale * (mu ** power)) - 1) *
                               np.log(mu)) / self.freq_weights.sum())
            power = brentq(psi_p, low, high, args=(mu))
        else:
            raise NotImplementedError('Only brentq can currently be used')
        return power

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

    def get_distribution(self, params, scale=1., exog=None, exposure=None,
                         offset=None):
        scale = float_like(scale, "scale", optional=True)
        fit = self.predict(params, exog, exposure, offset, linear=False)

        import scipy.stats.distributions as dist

        # if isinstance(self.family, families.Gaussian):
        #     return dist.norm(loc=fit, scale=np.sqrt(scale))

        if isinstance(self.family, Binomial):
            return dist.binom(n=1, p=fit)

        # elif isinstance(self.family, families.Poisson):
        #     return dist.poisson(mu=fit)

        # elif isinstance(self.family, families.Gamma):
        #     alpha = fit / float(scale)
        #     return dist.gamma(alpha, scale=scale)

        else:
            raise ValueError("get_distribution not implemented for %s" %
                             self.family.name)

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

    def fit(self, start_params=None, maxiter=100, method='IRLS', tol=1e-8,
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

        if method.lower() == "irls":
            if cov_type.lower() == 'eim':
                cov_type = 'nonrobust'
            return self._fit_irls(start_params=start_params, maxiter=maxiter,
                                  tol=tol, scale=scale, cov_type=cov_type,
                                  cov_kwds=cov_kwds, use_t=use_t, **kwargs)
        else:
            self._optim_hessian = kwargs.get('optim_hessian')
            self._tmp_like_exog = np.empty_like(self.exog, dtype=float)
            fit_ = self._fit_gradient(start_params=start_params,
                                      method=method,
                                      maxiter=maxiter,
                                      tol=tol, scale=scale,
                                      full_output=full_output,
                                      disp=disp, cov_type=cov_type,
                                      cov_kwds=cov_kwds, use_t=use_t,
                                      max_start_irls=max_start_irls,
                                      **kwargs)
            del self._optim_hessian
            del self._tmp_like_exog
            return fit_

    def _fit_gradient(self, start_params=None, method="newton",
                      maxiter=100, tol=1e-8, full_output=True,
                      disp=True, scale=None, cov_type='nonrobust',
                      cov_kwds=None, use_t=None, max_start_irls=3,
                      **kwargs):
        # fix scale during optimization, see #4616
        scaletype = self.scaletype
        self.scaletype = 1.

        if (max_start_irls > 0) and (start_params is None):
            irls_rslt = self._fit_irls(start_params=start_params,
                                       maxiter=max_start_irls,
                                       tol=tol, scale=1., cov_type='nonrobust',
                                       cov_kwds=None, use_t=None,
                                       **kwargs)
            start_params = irls_rslt.params
            del irls_rslt

        rslt = super(GLM, self).fit(start_params=start_params, tol=tol,
                                    maxiter=maxiter, full_output=full_output,
                                    method=method, disp=disp, **kwargs)

        # reset scaletype to original
        self.scaletype = scaletype

        mu = self.predict(rslt.params)
        scale = self.estimate_scale(mu)

        if rslt.normalized_cov_params is None:
            cov_p = None
        else:
            cov_p = rslt.normalized_cov_params / scale

        if cov_type.lower() == 'eim':
            oim = False
            cov_type = 'nonrobust'
        else:
            oim = True

        try:
            cov_p = np.linalg.inv(-self.hessian(rslt.params, observed=oim)) / scale
        except LinAlgError:
            warnings.warn('Inverting hessian failed, no bse or cov_params '
                          'available', HessianInversionWarning)
            cov_p = None

        results_class = getattr(self, '_results_class', GLMResults)
        results_class_wrapper = getattr(self, '_results_class_wrapper', GLMResultsWrapper)
        glm_results = results_class(self, rslt.params,
                                    cov_p,
                                    scale,
                                    cov_type=cov_type, cov_kwds=cov_kwds,
                                    use_t=use_t)

        # TODO: iteration count is not always available
        history = {'iteration': 0}
        if full_output:
            glm_results.mle_retvals = rslt.mle_retvals
            if 'iterations' in rslt.mle_retvals:
                history['iteration'] = rslt.mle_retvals['iterations']
        glm_results.method = method
        glm_results.fit_history = history

        return results_class_wrapper(glm_results)

    def _fit_irls(self, start_params=None, maxiter=100, tol=1e-8,
                  scale=None, cov_type='nonrobust', cov_kwds=None,
                  use_t=None, **kwargs):
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

        glm_results = GLMResults(self, wls_results.params,
                                 wls_results.normalized_cov_params,
                                 self.scale,
                                 cov_type=cov_type, cov_kwds=cov_kwds,
                                 use_t=use_t)

        glm_results.method = "IRLS"
        glm_results.mle_settings = {}
        glm_results.mle_settings['wls_method'] = wls_method
        glm_results.mle_settings['optimizer'] = glm_results.method
        if (maxiter > 0) and (attach_wls is True):
            glm_results.results_wls = wls_results
        history['iteration'] = iteration + 1
        glm_results.fit_history = history
        glm_results.converged = converged
        return GLMResultsWrapper(glm_results)

    def fit_regularized(self, method="elastic_net", alpha=0.,
                        start_params=None, refit=False,
                        opt_method="bfgs", **kwargs):
        if kwargs.get("L1_wt", 1) == 0:
            return self._fit_ridge(alpha, start_params, opt_method)

        from statsmodels.base.elastic_net import fit_elasticnet

        if method != "elastic_net":
            raise ValueError("method for fit_regularied must be elastic_net")

        defaults = {"maxiter": 50, "L1_wt": 1, "cnvrg_tol": 1e-10,
                    "zero_tol": 1e-10}
        defaults.update(kwargs)

        result = fit_elasticnet(self, method=method,
                                alpha=alpha,
                                start_params=start_params,
                                refit=refit,
                                **defaults)

        self.mu = self.predict(result.params)
        self.scale = self.estimate_scale(self.mu)

        if not result.converged:
            warnings.warn("Elastic net fitting did not converge")

        return result

    def _fit_ridge(self, alpha, start_params, method):

        if start_params is None:
            start_params = np.zeros(self.exog.shape[1])

        def fun(x):
            return -(self.loglike(x) / self.nobs - np.sum(alpha * x**2) / 2)

        def grad(x):
            return -(self.score(x) / self.nobs - alpha * x)

        from scipy.optimize import minimize
        from statsmodels.base.elastic_net import (RegularizedResults,
            RegularizedResultsWrapper)

        mr = minimize(fun, start_params, jac=grad, method=method)
        params = mr.x

        if not mr.success:
            import warnings
            ngrad = np.sqrt(np.sum(mr.jac**2))
            msg = "GLM ridge optimization may have failed, |grad|=%f" % ngrad
            warnings.warn(msg)

        results = RegularizedResults(self, params)
        results = RegularizedResultsWrapper(results)

        return results

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        from patsy import DesignInfo
        from statsmodels.base._constraints import (fit_constrained,
                                                   LinearConstraints)

        # same pattern as in base.LikelihoodModel.t_test
        lc = DesignInfo(self.exog_names).linear_constraint(constraints)
        R, q = lc.coefs, lc.constants

        # TODO: add start_params option, need access to tranformation
        #       fit_constrained needs to do the transformation
        params, cov, res_constr = fit_constrained(self, R, q,
                                                  start_params=start_params,
                                                  fit_kwds=fit_kwds)
        # create dummy results Instance, TODO: wire up properly
        res = self.fit(start_params=params, maxiter=0)  # we get a wrapper back
        res._results.params = params
        res._results.cov_params_default = cov
        cov_type = fit_kwds.get('cov_type', 'nonrobust')
        if cov_type != 'nonrobust':
            res._results.normalized_cov_params = cov / res_constr.scale
        else:
            res._results.normalized_cov_params = None
        res._results.scale = res_constr.scale
        k_constr = len(q)
        res._results.df_resid += k_constr
        res._results.df_model -= k_constr
        res._results.constraints = LinearConstraints.from_patsy(lc)
        res._results.k_constr = k_constr
        res._results.results_constrained = res_constr
        return res

class PredictionResults(object):
    def __init__(self, predicted_mean, var_pred_mean, var_resid=None,
                 df=None, dist=None, row_labels=None, linpred=None, link=None):
        # TODO: is var_resid used? drop from arguments?
        self.predicted_mean = predicted_mean
        self.var_pred_mean = var_pred_mean
        self.df = df
        self.var_resid = var_resid
        self.row_labels = row_labels
        self.linpred = linpred
        self.link = link

        if dist is None or dist == 'norm':
            self.dist = scipy.stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = scipy.stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    @property
    def se_obs(self):
        raise NotImplementedError
        return np.sqrt(self.var_pred_mean + self.var_resid)

    @property
    def se_mean(self):
        return np.sqrt(self.var_pred_mean)

    @property
    def tvalues(self):
        return self.predicted_mean / self.se_mean

    def t_test(self, value=0, alternative='two-sided'):
        # assumes symmetric distribution
        stat = (self.predicted_mean - value) / self.se_mean

        if alternative in ['two-sided', '2-sided', '2s']:
            pvalue = self.dist.sf(np.abs(stat), *self.dist_args)*2
        elif alternative in ['larger', 'l']:
            pvalue = self.dist.sf(stat, *self.dist_args)
        elif alternative in ['smaller', 's']:
            pvalue = self.dist.cdf(stat, *self.dist_args)
        else:
            raise ValueError('invalid alternative')
        return stat, pvalue

    def conf_int(self, method='endpoint', alpha=0.05, **kwds):
        tmp = np.linspace(0, 1, 6)
        is_linear = (self.link.inverse(tmp) == tmp).all()
        if method == 'endpoint' and not is_linear:
            ci_linear = self.linpred.conf_int(alpha=alpha, obs=False)
            ci = self.link.inverse(ci_linear)
        elif method == 'delta' or is_linear:
            se = self.se_mean
            q = self.dist.ppf(1 - alpha / 2., *self.dist_args)
            lower = self.predicted_mean - q * se
            upper = self.predicted_mean + q * se
            ci = np.column_stack((lower, upper))
            # if we want to stack at a new last axis, for lower.ndim > 1
            # np.concatenate((lower[..., None], upper[..., None]), axis=-1)

        return ci

    def summary_frame(self, alpha=0.05):
        """Summary frame"""
        # TODO: finish and cleanup
        #ci_obs = self.conf_int(alpha=alpha, obs=True) # need to split
        ci_mean = self.conf_int(alpha=alpha)
        to_include = {}
        to_include['mean'] = self.predicted_mean
        to_include['mean_se'] = self.se_mean
        to_include['mean_ci_lower'] = ci_mean[:, 0]
        to_include['mean_ci_upper'] = ci_mean[:, 1]

        self.table = to_include
        # pandas dict does not handle 2d_array
        #data = np.column_stack(list(to_include.values()))
        #names = ....
        res = pd.DataFrame(to_include, index=self.row_labels,
                           columns=to_include.keys())
        return res


def get_prediction_glm(self, exog=None, transform=True, weights=None,
                       row_labels=None, linpred=None, link=None,
                       pred_kwds=None):
    # prepare exog and row_labels, based on base Results.predict
    if transform and hasattr(self.model, 'formula') and exog is not None:
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            exog = pd.DataFrame(exog)
        exog = dmatrix(self.model.data.design_info, exog)

    if exog is not None:
        if row_labels is None:
            row_labels = getattr(exog, 'index', None)
            if callable(row_labels):
                row_labels = None

        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                               self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]
    else:
        exog = self.model.exog
        if weights is None:
            weights = getattr(self.model, 'weights', None)

        if row_labels is None:
            row_labels = getattr(self.model.data, 'row_labels', None)

    # need to handle other arrays, TODO: is delegating to model possible ?
    if weights is not None:
        weights = np.asarray(weights)
        if (weights.size > 1 and
           (weights.ndim != 1 or weights.shape[0] == exog.shape[1])):
            raise ValueError('weights has wrong shape')

    ### end

    pred_kwds['linear'] = False
    predicted_mean = self.model.predict(self.params, exog, **pred_kwds)

    covb = self.cov_params()

    link_deriv = self.model.family.link.inverse_deriv(linpred.predicted_mean)
    var_pred_mean = link_deriv**2 * (exog * np.dot(covb, exog.T).T).sum(1)
    var_resid = self.scale  # self.mse_resid / weights

    # TODO: check that we have correct scale, Refactor scale #???
    # special case for now:
    if self.cov_type == 'fixed scale':
        var_resid = self.cov_kwds['scale']

    if weights is not None:
        var_resid /= weights

    dist = ['norm', 't'][self.use_t]
    return PredictionResults(predicted_mean, var_pred_mean, var_resid,
                             df=self.df_resid, dist=dist,
                             row_labels=row_labels, linpred=linpred, link=link)


def params_transform_univariate(params, cov_params, link=None, transform=None,
                                row_labels=None):
    from statsmodels.genmod.families import links
    if link is None and transform is None:
        link = links.Log()

    if row_labels is None and hasattr(params, 'index'):
        row_labels = params.index

    params = np.asarray(params)

    predicted_mean = link.inverse(params)
    link_deriv = link.inverse_deriv(params)
    var_pred_mean = link_deriv**2 * np.diag(cov_params)
    # TODO: do we want covariance also, or just var/se

    dist = scipy.stats.norm

    # TODO: need ci for linear prediction, method of `lin_pred
    linpred = PredictionResults(params, np.diag(cov_params), dist=dist,
                                row_labels=row_labels, link=links.identity())

    res = PredictionResults(predicted_mean, var_pred_mean, dist=dist,
                            row_labels=row_labels, linpred=linpred, link=link)

    return res

class GLMResults():
    def __init__(self, model, params, normalized_cov_params, scale,
                 cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        self.__dict__.update(kwargs)
        self.initialize(model, params, **kwargs)
        self._data_attr = []
        # Variables to clear from cache
        self._data_in_cache = ['fittedvalues', 'resid', 'wresid']

        self.normalized_cov_params = normalized_cov_params
        self.scale = scale
        self._use_t = False
        # robust covariance
        # We put cov_type in kwargs so subclasses can decide in fit whether to
        # use this generic implementation
        if 'use_t' in kwargs:
            use_t = kwargs['use_t']
            self.use_t = use_t if use_t is not None else False
        if 'cov_type' in kwargs:
            cov_type = kwargs.get('cov_type', 'nonrobust')
            cov_kwds = kwargs.get('cov_kwds', {})

            if cov_type == 'nonrobust':
                self.cov_type = 'nonrobust'
                self.cov_kwds = {'description': 'Standard Errors assume that the ' +
                                 'covariance matrix of the errors is correctly ' +
                                 'specified.'}
            else:
                from statsmodels.base.covtype import get_robustcov_results
                if cov_kwds is None:
                    cov_kwds = {}
                use_t = self.use_t
                # TODO: we should not need use_t in get_robustcov_results
                get_robustcov_results(self, cov_type=cov_type, use_self=True,
                                      use_t=use_t, **cov_kwds)

        self.family = model.family
        self._endog = model.endog
        self.nobs = model.endog.shape[0]
        self._freq_weights = model.freq_weights
        self._var_weights = model.var_weights
        self._iweights = model.iweights
        if isinstance(self.family, Binomial):
            self._n_trials = self.model.n_trials
        else:
            self._n_trials = 1
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self._cache = {}
        # are these intermediate results needed or can we just
        # call the model's attributes?

        # for remove data and pickle without large arrays
        self._data_attr.extend(['results_constrained', '_freq_weights',
                                '_var_weights', '_iweights'])
        self._data_in_cache.extend(['null', 'mu'])
        self._data_attr_model = getattr(self, '_data_attr_model', [])
        self._data_attr_model.append('mu')

        # robust covariance
        from statsmodels.base.covtype import get_robustcov_results
        if use_t is None:
            self.use_t = False    # TODO: class default
        else:
            self.use_t = use_t

        # temporary warning
        ct = (cov_type == 'nonrobust') or (cov_type.upper().startswith('HC'))
        if self.model._has_freq_weights and not ct:
            import warnings
            from statsmodels.tools.sm_exceptions import SpecificationWarning
            warnings.warn('cov_type not fully supported with freq_weights',
                          SpecificationWarning)

        if self.model._has_var_weights and not ct:
            import warnings
            from statsmodels.tools.sm_exceptions import SpecificationWarning
            warnings.warn('cov_type not fully supported with var_weights',
                          SpecificationWarning)

        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description': 'Standard Errors assume that the' +
                             ' covariance matrix of the errors is correctly ' +
                             'specified.'}

        else:
            if cov_kwds is None:
                cov_kwds = {}
            get_robustcov_results(self, cov_type=cov_type, use_self=True,
                                  use_t=use_t, **cov_kwds)

    def initialize(self, model, params, **kwargs):
        self.params = params
        self.model = model
        if hasattr(model, 'k_constant'):
            self.k_constant = model.k_constant

    def predict(self, exog=None, transform=True, *args, **kwargs):
        import pandas as pd

        is_pandas = _is_using_pandas(exog, None)

        exog_index = exog.index if is_pandas else None

        if transform and hasattr(self.model, 'formula') and (exog is not None):
            design_info = self.model.data.design_info
            from patsy import dmatrix
            if isinstance(exog, pd.Series):
                # we are guessing whether it should be column or row
                if (hasattr(exog, 'name') and isinstance(exog.name, str) and
                        exog.name in design_info.describe()):
                    # assume we need one column
                    exog = pd.DataFrame(exog)
                else:
                    # assume we need a row
                    exog = pd.DataFrame(exog).T
            orig_exog_len = len(exog)
            is_dict = isinstance(exog, dict)
            try:
                exog = dmatrix(design_info, exog, return_type="dataframe")
            except Exception as exc:
                msg = ('predict requires that you use a DataFrame when '
                       'predicting from a model\nthat was created using the '
                       'formula api.'
                       '\n\nThe original error message returned by patsy is:\n'
                       '{0}'.format(str(str(exc))))
                raise exc.__class__(msg)
            if orig_exog_len > len(exog) and not is_dict:
                if exog_index is None:
                    warnings.warn('nan values have been dropped', ValueWarning)
                else:
                    exog = exog.reindex(exog_index)
            exog_index = exog.index

        if exog is not None:
            exog = np.asarray(exog)
            if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                                   self.model.exog.shape[1] == 1):
                exog = exog[:, None]
            exog = np.atleast_2d(exog)  # needed in count model shape[1]

        predict_results = self.model.predict(self.params, exog, *args,
                                             **kwargs)

        if exog_index is not None and not hasattr(predict_results,
                                                  'predicted_values'):
            if predict_results.ndim == 1:
                return pd.Series(predict_results, index=exog_index)
            else:
                return pd.DataFrame(predict_results, index=exog_index)
        else:
            return predict_results

    def normalized_cov_params(self):
        """See specific model class docstring"""
        raise NotImplementedError

    def _get_robustcov_results(self, cov_type='nonrobust', use_self=True,
                               use_t=None, **cov_kwds):
        if use_self is False:
            raise ValueError("use_self should have been removed long ago.  "
                             "See GH#4401")
        from statsmodels.base.covtype import get_robustcov_results
        if cov_kwds is None:
            cov_kwds = {}

        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description': 'Standard Errors assume that the ' +
                             'covariance matrix of the errors is correctly ' +
                             'specified.'}
        else:
            # TODO: we should not need use_t in get_robustcov_results
            get_robustcov_results(self, cov_type=cov_type, use_self=True,
                                  use_t=use_t, **cov_kwds)
    @property
    def use_t(self):
        """Flag indicating to use the Student's distribution in inference."""
        return self._use_t

    @use_t.setter
    def use_t(self, value):
        self._use_t = bool(value)

    @cached_value
    def bse(self):
        """The standard errors of the parameter estimates."""
        # Issue 3299
        if ((not hasattr(self, 'cov_params_default')) and
                (self.normalized_cov_params is None)):
            bse_ = np.empty(len(self.params))
            bse_[:] = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bse_ = np.sqrt(np.diag(self.cov_params()))
        return bse_

    @cached_value
    def tvalues(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self.params / self.bse

    @cached_value
    def pvalues(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.use_t:
                df_resid = getattr(self, 'df_resid_inference', self.df_resid)
                return scipy.stats.t.sf(np.abs(self.tvalues), df_resid) * 2
            else:
                return scipy.stats.norm.sf(np.abs(self.tvalues)) * 2

    def cov_params(self, r_matrix=None, column=None, scale=None, cov_p=None,
                   other=None):
        if (hasattr(self, 'mle_settings') and
                self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            dot_fun = nan_dot
        else:
            dot_fun = np.dot

        if (cov_p is None and self.normalized_cov_params is None and
                not hasattr(self, 'cov_params_default')):
            raise ValueError('need covariance of parameters for computing '
                             '(unnormalized) covariances')
        if column is not None and (r_matrix is not None or other is not None):
            raise ValueError('Column should be specified without other '
                             'arguments.')
        if other is not None and r_matrix is None:
            raise ValueError('other can only be specified with r_matrix')

        if cov_p is None:
            if hasattr(self, 'cov_params_default'):
                cov_p = self.cov_params_default
            else:
                if scale is None:
                    scale = self.scale
                cov_p = self.normalized_cov_params * scale

        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return cov_p[column, column]
            else:
                return cov_p[column[:, None], column]
        elif r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
            if r_matrix.shape == ():
                raise ValueError("r_matrix should be 1d or 2d")
            if other is None:
                other = r_matrix
            else:
                other = np.asarray(other)
            tmp = dot_fun(r_matrix, dot_fun(cov_p, np.transpose(other)))
            return tmp
        else:  # if r_matrix is None and column is None:
            return cov_p

    # TODO: make sure this works as needed for GLMs
    def t_test(self, r_matrix, cov_p=None, scale=None, use_t=None):
        if scale is not None:
            warnings.warn('scale is has no effect and is deprecated. It will'
                          'be removed in the next version.',
                          DeprecationWarning)

        from patsy import DesignInfo
        names = self.model.data.cov_names
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants
        num_ttests = r_matrix.shape[0]
        num_params = r_matrix.shape[1]

        if (cov_p is None and self.normalized_cov_params is None and
                not hasattr(self, 'cov_params_default')):
            raise ValueError('Need covariance of parameters for computing '
                             'T statistics')
        params = self.params.ravel()
        if num_params != params.shape[0]:
            raise ValueError('r_matrix and params are not aligned')
        if q_matrix is None:
            q_matrix = np.zeros(num_ttests)
        else:
            q_matrix = np.asarray(q_matrix)
            q_matrix = q_matrix.squeeze()
        if q_matrix.size > 1:
            if q_matrix.shape[0] != num_ttests:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")

        if use_t is None:
            # switch to use_t false if undefined
            use_t = (hasattr(self, 'use_t') and self.use_t)

        _effect = np.dot(r_matrix, params)

        # Perform the test
        if num_ttests > 1:
            _sd = np.sqrt(np.diag(self.cov_params(
                r_matrix=r_matrix, cov_p=cov_p)))
        else:
            _sd = np.sqrt(self.cov_params(r_matrix=r_matrix, cov_p=cov_p))
        _t = (_effect - q_matrix) * recipr(_sd)

        df_resid = getattr(self, 'df_resid_inference', self.df_resid)

        if use_t:
            return ContrastResults(effect=_effect, t=_t, sd=_sd,
                                   df_denom=df_resid)
        else:
            return ContrastResults(effect=_effect, statistic=_t, sd=_sd,
                                   df_denom=df_resid,
                                   distribution='norm')

    def f_test(self, r_matrix, cov_p=None, scale=1.0, invcov=None):
        if scale != 1.0:
            warnings.warn('scale is has no effect and is deprecated. It will'
                          'be removed in the next version.',
                          DeprecationWarning)

        res = self.wald_test(r_matrix, cov_p=cov_p, invcov=invcov, use_f=True)
        return res

    # TODO: untested for GLMs?
    def wald_test(self, r_matrix, cov_p=None, scale=1.0, invcov=None,
                  use_f=None, df_constraints=None):
        if scale != 1.0:
            warnings.warn('scale is has no effect and is deprecated. It will'
                          'be removed in the next version.',
                          DeprecationWarning)

        if use_f is None:
            # switch to use_t false if undefined
            use_f = (hasattr(self, 'use_t') and self.use_t)

        from patsy import DesignInfo
        names = self.model.data.cov_names
        params = self.params.ravel()
        LC = DesignInfo(names).linear_constraint(r_matrix)
        r_matrix, q_matrix = LC.coefs, LC.constants

        if (self.normalized_cov_params is None and cov_p is None and
                invcov is None and not hasattr(self, 'cov_params_default')):
            raise ValueError('need covariance of parameters for computing '
                             'F statistics')

        cparams = np.dot(r_matrix, params[:, None])
        J = float(r_matrix.shape[0])  # number of restrictions

        if q_matrix is None:
            q_matrix = np.zeros(J)
        else:
            q_matrix = np.asarray(q_matrix)
        if q_matrix.ndim == 1:
            q_matrix = q_matrix[:, None]
            if q_matrix.shape[0] != J:
                raise ValueError("r_matrix and q_matrix must have the same "
                                 "number of rows")
        Rbq = cparams - q_matrix
        if invcov is None:
            cov_p = self.cov_params(r_matrix=r_matrix, cov_p=cov_p)
            if np.isnan(cov_p).max():
                raise ValueError("r_matrix performs f_test for using "
                                 "dimensions that are asymptotically "
                                 "non-normal")
            invcov = np.linalg.pinv(cov_p)
            J_ = np.linalg.matrix_rank(cov_p)
            if J_ < J:
                warnings.warn('covariance of constraints does not have full '
                              'rank. The number of constraints is %d, but '
                              'rank is %d' % (J, J_), ValueWarning)
                J = J_

        # TODO streamline computation, we do not need to compute J if given
        if df_constraints is not None:
            # let caller override J by df_constraint
            J = df_constraints

        if (hasattr(self, 'mle_settings') and
                self.mle_settings['optimizer'] in ['l1', 'l1_cvxopt_cp']):
            F = nan_dot(nan_dot(Rbq.T, invcov), Rbq)
        else:
            F = np.dot(np.dot(Rbq.T, invcov), Rbq)

        df_resid = getattr(self, 'df_resid_inference', self.df_resid)
        if use_f:
            F /= J
            return ContrastResults(F=F, df_denom=df_resid,
                                   df_num=J) #invcov.shape[0])
        else:
            return ContrastResults(chi2=F, df_denom=J, statistic=F,
                                   distribution='chi2', distargs=(J,))

    def wald_test_terms(self, skip_single=False, extra_constraints=None,
                        combine_terms=None):
        # lazy import
        from collections import defaultdict

        result = self
        if extra_constraints is None:
            extra_constraints = []
        if combine_terms is None:
            combine_terms = []
        design_info = getattr(result.model.data, 'design_info', None)

        if design_info is None and extra_constraints is None:
            raise ValueError('no constraints, nothing to do')

        identity = np.eye(len(result.params))
        constraints = []
        combined = defaultdict(list)
        if design_info is not None:
            for term in design_info.terms:
                cols = design_info.slice(term)
                name = term.name()
                constraint_matrix = identity[cols]

                # check if in combined
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(constraint_matrix)

                k_constraint = constraint_matrix.shape[0]
                if skip_single:
                    if k_constraint == 1:
                        continue

                constraints.append((name, constraint_matrix))

            combined_constraints = []
            for cname in combine_terms:
                combined_constraints.append((cname, np.vstack(combined[cname])))
        else:
            # check by exog/params names if there is no formula info
            for col, name in enumerate(result.model.exog_names):
                constraint_matrix = np.atleast_2d(identity[col])

                # check if in combined
                for cname in combine_terms:
                    if cname in name:
                        combined[cname].append(constraint_matrix)

                if skip_single:
                    continue

                constraints.append((name, constraint_matrix))

            combined_constraints = []
            for cname in combine_terms:
                combined_constraints.append((cname, np.vstack(combined[cname])))

        use_t = result.use_t
        distribution = ['chi2', 'F'][use_t]

        res_wald = []
        index = []
        for name, constraint in constraints + combined_constraints + extra_constraints:
            wt = result.wald_test(constraint)
            row = [wt.statistic.item(), wt.pvalue.item(), constraint.shape[0]]
            if use_t:
                row.append(wt.df_denom)
            res_wald.append(row)
            index.append(name)

        # distribution nerutral names
        col_names = ['statistic', 'pvalue', 'df_constraint']
        if use_t:
            col_names.append('df_denom')
        # TODO: maybe move DataFrame creation to results class
        from pandas import DataFrame
        table = DataFrame(res_wald, index=index, columns=col_names)
        res = WaldTestResults(None, distribution, None, table=table)
        # TODO: remove temp again, added for testing
        res.temp = constraints + combined_constraints + extra_constraints
        return res

    def t_test_pairwise(self, term_name, method='hs', alpha=0.05,
                        factor_labels=None):
        res = t_test_pairwise(self, term_name, method=method, alpha=alpha,
                              factor_labels=factor_labels)
        return res

    def conf_int(self, alpha=.05, cols=None):
        bse = self.bse

        if self.use_t:
            dist = scipy.stats.t
            df_resid = getattr(self, 'df_resid_inference', self.df_resid)
            q = dist.ppf(1 - alpha / 2, df_resid)
        else:
            dist = scipy.stats.norm
            q = dist.ppf(1 - alpha / 2)

        params = self.params
        lower = params - q * bse
        upper = params + q * bse
        if cols is not None:
            cols = np.asarray(cols)
            lower = lower[cols]
            upper = upper[cols]
        return np.asarray(lzip(lower, upper))

    def save(self, fname, remove_data=False):
        from statsmodels.iolib.smpickle import save_pickle

        if remove_data:
            self.remove_data()

        save_pickle(self, fname)

    @classmethod
    def load(cls, fname):
        from statsmodels.iolib.smpickle import load_pickle
        return load_pickle(fname)

    @cached_data
    def resid_response(self):
        return self._n_trials * (self._endog-self.mu)

    @cached_data
    def resid_pearson(self):
        return (np.sqrt(self._n_trials) * (self._endog-self.mu) *
                np.sqrt(self._var_weights) /
                np.sqrt(self.family.variance(self.mu)))

    @cached_data
    def resid_working(self):
        # Isn't self.resid_response is already adjusted by _n_trials?
        val = (self.resid_response * self.family.link.deriv(self.mu))
        val *= self._n_trials
        return val

    @cached_data
    def resid_anscombe(self):
        import warnings
        warnings.warn('Anscombe residuals currently unscaled. After the 0.12 '
                      'release, they will be scaled.', category=FutureWarning)
        return self.family.resid_anscombe(self._endog, self.fittedvalues,
                                          var_weights=self._var_weights,
                                          scale=1.)

    @cached_data
    def resid_anscombe_scaled(self):
        return self.family.resid_anscombe(self._endog, self.fittedvalues,
                                          var_weights=self._var_weights,
                                          scale=self.scale)

    @cached_data
    def resid_anscombe_unscaled(self):
        return self.family.resid_anscombe(self._endog, self.fittedvalues,
                                          var_weights=self._var_weights,
                                          scale=1.)

    @cached_data
    def resid_deviance(self):
        dev = self.family.resid_dev(self._endog, self.fittedvalues,
                                    var_weights=self._var_weights,
                                    scale=1.)
        return dev

    @cached_value
    def pearson_chi2(self):
        chisq = (self._endog - self.mu)**2 / self.family.variance(self.mu)
        chisq *= self._iweights * self._n_trials
        chisqsum = np.sum(chisq)
        return chisqsum

    @cached_data
    def fittedvalues(self):
        return self.mu

    @cached_data
    def mu(self):
        return self.model.predict(self.params)

    @cache_readonly
    def null(self):
        endog = self._endog
        model = self.model
        exog = np.ones((len(endog), 1))

        kwargs = model._get_init_kwds()
        kwargs.pop('family')
        if hasattr(self.model, '_offset_exposure'):
            return GLM(endog, exog, family=self.family,
                       **kwargs).fit().fittedvalues
        else:
            # correct if fitted is identical across observations
            wls_model = lm.WLS(endog, exog,
                               weights=self._iweights * self._n_trials)
            return wls_model.fit().fittedvalues

    @cache_readonly
    def deviance(self):
        return self.family.deviance(self._endog, self.mu, self._var_weights,
                                    self._freq_weights)

    @cache_readonly
    def null_deviance(self):
        return self.family.deviance(self._endog, self.null, self._var_weights,
                                    self._freq_weights)

    @cache_readonly
    def llnull(self):
        return self.family.loglike(self._endog, self.null,
                                   var_weights=self._var_weights,
                                   freq_weights=self._freq_weights,
                                   scale=self.scale)

    @cached_value
    def llf(self):
        _modelfamily = self.family
        val = _modelfamily.loglike(self._endog, self.mu,
                                   var_weights=self._var_weights,
                                   freq_weights=self._freq_weights,
                                   scale=self.scale)
        return val

    @cached_value
    def aic(self):
        return -2 * self.llf + 2 * (self.df_model + 1)

    @property
    def bic(self):
        if _use_bic_helper.use_bic_llf not in (True, False):
            warnings.warn(
                "The bic value is computed using the deviance formula. After "
                "0.13 this will change to the log-likelihood based formula. "
                "This change has no impact on the relative rank of models "
                "compared using BIC. You can directly access the "
                "log-likelihood version using the `bic_llf` attribute. You "
                "can suppress this message by calling "
                "statsmodels.genmod.generalized_linear_model.SET_USE_BIC_LLF "
                "with True to get the LLF-based version now or False to retain"
                "the deviance version.",
                FutureWarning
            )
        if bool(_use_bic_helper.use_bic_llf):
            return self.bic_llf

        return self.bic_deviance

    @cached_value
    def bic_deviance(self):
        return (self.deviance -
                (self.model.wnobs - self.df_model - 1) *
                np.log(self.model.wnobs))

    @cached_value
    def bic_llf(self):
        return -2*self.llf + (self.df_model+1)*np.log(
            self.df_model+self.df_resid+1
        )

    def get_prediction(self, exog=None, exposure=None, offset=None,
                       transform=True, linear=False,
                       row_labels=None):

        pred_kwds = {'exposure': exposure, 'offset': offset, 'linear': True}

        # two calls to a get_prediction duplicates exog generation if patsy
        res_linpred = get_prediction(self, exog=exog,
                                             transform=transform,
                                             row_labels=row_labels,
                                             pred_kwds=pred_kwds)

        pred_kwds['linear'] = False
        res = get_prediction_glm(self, exog=exog, transform=transform,
                                      row_labels=row_labels,
                                      linpred=res_linpred,
                                      link=self.model.family.link,
                                      pred_kwds=pred_kwds)

        return res

    def get_hat_matrix_diag(self, observed=True):
        weights = self.model.hessian_factor(self.params, observed=observed)
        wexog = np.sqrt(weights)[:, None] * self.model.exog

        hd = (wexog * np.linalg.pinv(wexog).T).sum(1)
        return hd

    def get_influence(self, observed=True):
        from statsmodels.stats.outliers_influence import GLMInfluence

        weights = self.model.hessian_factor(self.params, observed=observed)
        weights_sqrt = np.sqrt(weights)
        wexog = weights_sqrt[:, None] * self.model.exog
        wendog = weights_sqrt * self.model.endog

        # using get_hat_matrix_diag has duplicated computation
        hat_matrix_diag = self.get_hat_matrix_diag(observed=observed)
        infl = GLMInfluence(self, endog=wendog, exog=wexog,
                         resid=self.resid_pearson,
                         hat_matrix_diag=hat_matrix_diag)
        return infl

    def remove_data(self):
        # GLM has alias/reference in result instance
        self._data_attr.extend([i for i in self.model._data_attr
                                if '_data.' not in i])
        super(self.__class__, self).remove_data()

        cls = self.__class__
        # Note: we cannot just use `getattr(cls, x)` or `getattr(self, x)`
        # because of redirection involved with property-like accessors
        cls_attrs = {}
        for name in dir(cls):
            try:
                attr = object.__getattribute__(cls, name)
            except AttributeError:
                pass
            else:
                cls_attrs[name] = attr
        data_attrs = [x for x in cls_attrs
                      if isinstance(cls_attrs[x], cached_data)]
        value_attrs = [x for x in cls_attrs
                       if isinstance(cls_attrs[x], cached_value)]
        # make sure the cached for value_attrs are evaluated; this needs to
        # occur _before_ any other attributes are removed.
        for name in value_attrs:
            getattr(self, name)
        for name in data_attrs:
            self._cache[name] = None

        def wipe(obj, att):
            # get to last element in attribute path
            p = att.split('.')
            att_ = p.pop(-1)
            try:
                obj_ = reduce(getattr, [obj] + p)
                if hasattr(obj_, att_):
                    setattr(obj_, att_, None)
            except AttributeError:
                pass

        model_only = ['model.' + i for i in getattr(self, "_data_attr_model", [])]
        model_attr = ['model.' + i for i in self.model._data_attr]
        for att in self._data_attr + model_attr + model_only:
            if att in data_attrs:
                # these have been handled above, and trying to call wipe
                # would raise an Exception anyway, so skip these
                continue
            wipe(self, att)

        for key in self._data_in_cache:
            try:
                self._cache[key] = None
            except (AttributeError, KeyError):
                pass

        # TODO: what are these in results?
        self._endog = None
        self._freq_weights = None
        self._var_weights = None
        self._iweights = None
        self._n_trials = None

    def plot_added_variable(self, focus_exog, resid_type=None,
                            use_glm_weights=True, fit_kwargs=None,
                            ax=None):

        from statsmodels.graphics.regressionplots import plot_added_variable

        fig = plot_added_variable(self, focus_exog,
                                  resid_type=resid_type,
                                  use_glm_weights=use_glm_weights,
                                  fit_kwargs=fit_kwargs, ax=ax)

        return fig

    def plot_partial_residuals(self, focus_exog, ax=None):

        from statsmodels.graphics.regressionplots import plot_partial_residuals

        return plot_partial_residuals(self, focus_exog, ax=ax)

    def plot_ceres_residuals(self, focus_exog, frac=0.66, cond_means=None,
                             ax=None):

        from statsmodels.graphics.regressionplots import plot_ceres_residuals

        return plot_ceres_residuals(self, focus_exog, frac,
                                    cond_means=cond_means, ax=ax)

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Model Family:', [self.family.__class__.__name__]),
                    ('Link Function:', [self.family.link.__class__.__name__]),
                    ('Method:', [self.method]),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Iterations:',
                     ["%d" % self.fit_history['iteration']]),
                    ]

        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ('Scale:', ["%#8.5g" % self.scale]),
                     ('Log-Likelihood:', None),
                     ('Deviance:', ["%#8.5g" % self.deviance]),
                     ('Pearson chi2:', ["%#6.3g" % self.pearson_chi2])
                     ]

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        if title is None:
            title = "Generalized Linear Model Regression Results"

        # create summary tables
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                              use_t=self.use_t)

        if hasattr(self, 'constraints'):
            smry.add_extra_txt(['Model has been estimated subject to linear '
                                'equality constraints.'])
        return smry

    def summary2(self, yname=None, xname=None, title=None, alpha=.05,
                 float_format="%.4f"):
        self.method = 'IRLS'
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            smry.add_base(results=self, alpha=alpha, float_format=float_format,
                          xname=xname, yname=yname, title=title)
        if hasattr(self, 'constraints'):
            smry.add_text('Model has been estimated subject to linear '
                          'equality constraints.')

        return smry

class GLMResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {
        'resid_anscombe': 'rows',
        'resid_deviance': 'rows',
        'resid_pearson': 'rows',
        'resid_response': 'rows',
        'resid_working': 'rows'
    }
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs,
                                   _attrs)

wrap.populate_wrapper(GLMResultsWrapper, GLMResults)

def mlds(filename):
    data = pd.read_table(filename, sep='\t')
    res = GLM.from_formula('resp ~ S2 + S3 + S4 + S5 + S6 + S7 + S8 + S9 + S10 + S10 + S11 - 1', family=Binomial(probit()), data=data).fit()
    print(res.summary())

if __name__ == '__main__':
    mlds('data.txt')