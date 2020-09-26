import pandas as pd
import numpy as np
import scipy.stats
from scipy import special
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.decorators import (cache_readonly,
                                          cache_writable)
from statsmodels.tools.decorators import (cache_readonly,
                                          cached_value, cached_data)
import statsmodels.base.wrapper as wrap
from statsmodels.base.data import handle_data

FLOAT_EPS = np.finfo(float).eps

class Model(object):
    _formula_max_endog = 1

    def __init__(self, endog, exog=None, **kwargs):
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

    def fit(self):
        raise NotImplementedError

    def predict(self, params, exog=None, *args, **kwargs):
        raise NotImplementedError


class LikelihoodModel(Model):
    def __init__(self, endog, exog=None, **kwargs):
        super(LikelihoodModel, self).__init__(endog, exog, **kwargs)
        self.initialize()

    def initialize(self):
        pass

    # TODO: if the intent is to re-initialize the model with new data then this
    # method needs to take inputs...

    def loglike(self, params):
        raise NotImplementedError

    def score(self, params):
        raise NotImplementedError

    def information(self, params):
        raise NotImplementedError

    def hessian(self, params):
        raise NotImplementedError

    def fit(self, start_params=None, method='newton', maxiter=100,
            full_output=True, disp=True, fargs=(), callback=None, retall=False,
            skip_hessian=False, **kwargs):
        Hinv = None  # JP error if full_output=0, Hinv not defined

        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            elif self.exog is not None:
                # fails for shape (K,)?
                start_params = [0] * self.exog.shape[1]
            else:
                raise ValueError("If exog is None, then start_params should "
                                 "be specified")

        # TODO: separate args from nonarg taking score and hessian, ie.,
        # user-supplied and numerically evaluated estimate frprime does not take
        # args in most (any?) of the optimize function

        nobs = self.endog.shape[0]
        # f = lambda params, *args: -self.loglike(params, *args) / nobs

        def f(params, *args):
            return -self.loglike(params, *args) / nobs

        if method == 'newton':
            # TODO: why are score and hess positive?
            def score(params, *args):
                return self.score(params, *args) / nobs

            def hess(params, *args):
                return self.hessian(params, *args) / nobs
        else:
            def score(params, *args):
                return -self.score(params, *args) / nobs

            def hess(params, *args):
                return -self.hessian(params, *args) / nobs

        warn_convergence = kwargs.pop('warn_convergence', True)
        optimizer = Optimizer()
        xopt, retvals, optim_settings = optimizer._fit(f, score, start_params,
                                                       fargs, kwargs,
                                                       hessian=hess,
                                                       method=method,
                                                       disp=disp,
                                                       maxiter=maxiter,
                                                       callback=callback,
                                                       retall=retall,
                                                       full_output=full_output)

        # NOTE: this is for fit_regularized and should be generalized
        cov_params_func = kwargs.setdefault('cov_params_func', None)
        if cov_params_func:
            Hinv = cov_params_func(self, xopt, retvals)
        elif method == 'newton' and full_output:
            Hinv = np.linalg.inv(-retvals['Hessian']) / nobs
        elif not skip_hessian:
            H = -1 * self.hessian(xopt)
            invertible = False
            if np.all(np.isfinite(H)):
                eigvals, eigvecs = np.linalg.eigh(H)
                if np.min(eigvals) > 0:
                    invertible = True

            if invertible:
                Hinv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
                Hinv = np.asfortranarray((Hinv + Hinv.T) / 2.0)
            else:
                warnings.warn('Inverting hessian failed, no bse or cov_params '
                              'available', HessianInversionWarning)
                Hinv = None

        if 'cov_type' in kwargs:
            cov_kwds = kwargs.get('cov_kwds', {})
            kwds = {'cov_type': kwargs['cov_type'], 'cov_kwds': cov_kwds}
        else:
            kwds = {}
        if 'use_t' in kwargs:
            kwds['use_t'] = kwargs['use_t']
        # TODO: add Hessian approximation and change the above if needed
        mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1., **kwds)

        # TODO: hardcode scale?
        mlefit.mle_retvals = retvals
        if isinstance(retvals, dict):
            if warn_convergence and not retvals['converged']:
                from statsmodels.tools.sm_exceptions import ConvergenceWarning
                warnings.warn("Maximum Likelihood optimization failed to "
                              "converge. Check mle_retvals",
                              ConvergenceWarning)

        mlefit.mle_settings = optim_settings
        return mlefit

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
        # ------ copied from PR #2380 remove when merged
        x = self.exog
        tol = atol + rtol * x.var(0)
        r = np.linalg.qr(x, mode='r')
        mask = np.abs(r.diagonal()) < np.sqrt(tol)
        # TODO add to results instance
        # idx_collinear = np.where(mask)[0]
        idx_keep = np.where(~mask)[0]
        return self._fit_zeros(keep_index=idx_keep, **kwds)

class RegressionModel(LikelihoodModel):
    def __init__(self, endog, exog, **kwargs):
        super(RegressionModel, self).__init__(endog, exog, **kwargs)
        self._data_attr.extend(['pinv_wexog', 'weights'])

    def initialize(self):
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        # overwrite nobs from class Model:
        self.nobs = float(self.wexog.shape[0])

        self._df_model = None
        self._df_resid = None
        self.rank = None

    @property
    def df_model(self):
        if self._df_model is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_model = float(self.rank - self.k_constant)
        return self._df_model

    @df_model.setter
    def df_model(self, value):
        self._df_model = value

    @property
    def df_resid(self):
        if self._df_resid is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_resid = self.nobs - self.rank
        return self._df_resid

    @df_resid.setter
    def df_resid(self, value):
        self._df_resid = value

    def whiten(self, x):
        raise NotImplementedError("Subclasses must implement.")

    def fit(self, method="pinv", cov_type='nonrobust', cov_kwds=None,
            use_t=None, **kwargs):
        if method == "pinv":
            if not (hasattr(self, 'pinv_wexog') and
                    hasattr(self, 'normalized_cov_params') and
                    hasattr(self, 'rank')):

                self.pinv_wexog, singular_values = pinv_extended(self.wexog)
                self.normalized_cov_params = np.dot(
                    self.pinv_wexog, np.transpose(self.pinv_wexog))

                # Cache these singular values for use later.
                self.wexog_singular_values = singular_values
                self.rank = np.linalg.matrix_rank(np.diag(singular_values))

            beta = np.dot(self.pinv_wexog, self.wendog)

        elif method == "qr":
            if not (hasattr(self, 'exog_Q') and
                    hasattr(self, 'exog_R') and
                    hasattr(self, 'normalized_cov_params') and
                    hasattr(self, 'rank')):
                Q, R = np.linalg.qr(self.wexog)
                self.exog_Q, self.exog_R = Q, R
                self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))

                # Cache singular values from R.
                self.wexog_singular_values = np.linalg.svd(R, 0, 0)
                self.rank = np.linalg.matrix_rank(R)
            else:
                Q, R = self.exog_Q, self.exog_R

            # used in ANOVA
            self.effects = effects = np.dot(Q.T, self.wendog)
            beta = np.linalg.solve(R, effects)
        else:
            raise ValueError('method has to be "pinv" or "qr"')

        if self._df_model is None:
            self._df_model = float(self.rank - self.k_constant)
        if self._df_resid is None:
            self.df_resid = self.nobs - self.rank

        lfit = RegressionResults(
            self, beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
            **kwargs)
        return RegressionResultsWrapper(lfit)

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog

        return np.dot(exog, params)

    def get_distribution(self, params, scale, exog=None, dist_class=None):
        fit = self.predict(params, exog)
        if dist_class is None:
            from scipy.stats.distributions import norm
            dist_class = norm
        gen = dist_class(loc=fit, scale=np.sqrt(scale))
        return gen

class WLS(RegressionModel):
    def __init__(self, endog, exog, weights=1., missing='none', hasconst=None,
                 **kwargs):
        weights = np.array(weights)
        if weights.shape == ():
            if (missing == 'drop' and 'missing_idx' in kwargs and
                    kwargs['missing_idx'] is not None):
                # patsy may have truncated endog
                weights = np.repeat(weights, len(kwargs['missing_idx']))
            else:
                weights = np.repeat(weights, len(endog))
        # handle case that endog might be of len == 1
        if len(weights) == 1:
            weights = np.array([weights.squeeze()])
        else:
            weights = weights.squeeze()
        super(WLS, self).__init__(endog, exog, missing=missing,
                                  weights=weights, hasconst=hasconst, **kwargs)
        nobs = self.exog.shape[0]
        weights = self.weights
        # Experimental normalization of weights
        weights = weights / np.sum(weights) * nobs
        if weights.size != nobs and weights.shape[0] != nobs:
            raise ValueError('Weights must be scalar or same length as design')

    def whiten(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            return x * np.sqrt(self.weights)
        elif x.ndim == 2:
            return np.sqrt(self.weights)[:, None] * x

    def loglike(self, params):
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params))**2, axis=0)
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with constant
        llf += 0.5 * np.sum(np.log(self.weights))
        return llf

    def hessian_factor(self, params, scale=None, observed=True):
        return self.weights

    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):
        # Docstring attached below
        if not np.isscalar(alpha):
            alpha = np.asarray(alpha)
        # Need to adjust since RSS/n in elastic net uses nominal n in
        # denominator
        alpha = alpha * np.sum(self.weights) / len(self.weights)

        rslt = OLS(self.wendog, self.wexog).fit_regularized(
            method=method, alpha=alpha,
            L1_wt=L1_wt,
            start_params=start_params,
            profile_scale=profile_scale,
            refit=refit, **kwargs)

        from statsmodels.base.elastic_net import (
            RegularizedResults, RegularizedResultsWrapper)
        rrslt = RegularizedResults(self, rslt.params)
        return RegularizedResultsWrapper(rrslt)

class Results():
    def __init__(self, model, params, **kwd):
        self.__dict__.update(kwd)
        self.initialize(model, params, **kwd)
        self._data_attr = []
        # Variables to clear from cache
        self._data_in_cache = ['fittedvalues', 'resid', 'wresid']

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

    def summary(self):
        """
        Summary

        Not implemented
        """
        raise NotImplementedError


# TODO: public method?
class LikelihoodModelResults(Results):

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
                 **kwargs):
        super(LikelihoodModelResults, self).__init__(model, params)
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

    def normalized_cov_params(self):
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
        return self._use_t

    @use_t.setter
    def use_t(self, value):
        self._use_t = bool(value)

    @cached_value
    def llf(self):
        return self.model.loglike(self.params)

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
                return stats.t.sf(np.abs(self.tvalues), df_resid) * 2
            else:
                return stats.norm.sf(np.abs(self.tvalues)) * 2

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
            dist = stats.t
            df_resid = getattr(self, 'df_resid_inference', self.df_resid)
            q = dist.ppf(1 - alpha / 2, df_resid)
        else:
            dist = stats.norm
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

    def remove_data(self):
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

class RegressionResults(LikelihoodModelResults):
    _cache = {}  # needs to be a class attribute for scale setter?

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
                 cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        super(RegressionResults, self).__init__(
            model, params, normalized_cov_params, scale)

        # Keep wresid since needed by predict
        self._data_in_cache.remove("wresid")
        self._cache = {}
        if hasattr(model, 'wexog_singular_values'):
            self._wexog_singular_values = model.wexog_singular_values
        else:
            self._wexog_singular_values = None

        self.df_model = model.df_model
        self.df_resid = model.df_resid

        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {
                'description': 'Standard Errors assume that the ' +
                'covariance matrix of the errors is correctly ' +
                'specified.'}
            if use_t is None:
                use_t = True  # TODO: class default
            self.use_t = use_t
        else:
            if cov_kwds is None:
                cov_kwds = {}
            if 'use_t' in cov_kwds:
                # TODO: we want to get rid of 'use_t' in cov_kwds
                use_t_2 = cov_kwds.pop('use_t')
                if use_t is None:
                    use_t = use_t_2
                # TODO: warn or not?
            self.get_robustcov_results(cov_type=cov_type, use_self=True,
                                       use_t=use_t, **cov_kwds)
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def conf_int(self, alpha=.05, cols=None):
        # keep method for docstring for now
        ci = super(RegressionResults, self).conf_int(alpha=alpha, cols=cols)
        return ci

    @cache_readonly
    def nobs(self):
        """Number of observations n."""
        return float(self.model.wexog.shape[0])

    @cache_readonly
    def fittedvalues(self):
        """The predicted values for the original (unwhitened) design."""
        return self.model.predict(self.params, self.model.exog)

    @cache_readonly
    def wresid(self):
        return self.model.wendog - self.model.predict(
            self.params, self.model.wexog)

    @cache_readonly
    def resid(self):
        return self.model.endog - self.model.predict(
            self.params, self.model.exog)

    # TODO: fix writable example
    @cache_writable()
    def scale(self):
        wresid = self.wresid
        return np.dot(wresid, wresid) / self.df_resid

    @cache_readonly
    def ssr(self):
        wresid = self.wresid
        return np.dot(wresid, wresid)

    @cache_readonly
    def centered_tss(self):
        model = self.model
        weights = getattr(model, 'weights', None)
        sigma = getattr(model, 'sigma', None)
        if weights is not None:
            mean = np.average(model.endog, weights=weights)
            return np.sum(weights * (model.endog - mean)**2)
        elif sigma is not None:
            # Exactly matches WLS when sigma is diagonal
            iota = np.ones_like(model.endog)
            iota = model.whiten(iota)
            mean = model.wendog.dot(iota) / iota.dot(iota)
            err = model.endog - mean
            err = model.whiten(err)
            return np.sum(err**2)
        else:
            centered_endog = model.wendog - model.wendog.mean()
            return np.dot(centered_endog, centered_endog)

    @cache_readonly
    def uncentered_tss(self):
        wendog = self.model.wendog
        return np.dot(wendog, wendog)

    @cache_readonly
    def ess(self):
        if self.k_constant:
            return self.centered_tss - self.ssr
        else:
            return self.uncentered_tss - self.ssr

    @cache_readonly
    def rsquared(self):
        if self.k_constant:
            return 1 - self.ssr/self.centered_tss
        else:
            return 1 - self.ssr/self.uncentered_tss

    @cache_readonly
    def rsquared_adj(self):
        return 1 - (np.divide(self.nobs - self.k_constant, self.df_resid)
                    * (1 - self.rsquared))

    @cache_readonly
    def mse_model(self):
        if np.all(self.df_model == 0.0):
            return np.full_like(self.ess, np.nan)
        return self.ess/self.df_model

    @cache_readonly
    def mse_resid(self):
        if np.all(self.df_resid == 0.0):
            return np.full_like(self.ssr, np.nan)
        return self.ssr/self.df_resid

    @cache_readonly
    def mse_total(self):
        if np.all(self.df_resid + self.df_model == 0.0):
            return np.full_like(self.centered_tss, np.nan)
        if self.k_constant:
            return self.centered_tss / (self.df_resid + self.df_model)
        else:
            return self.uncentered_tss / (self.df_resid + self.df_model)

    @cache_readonly
    def fvalue(self):
        if hasattr(self, 'cov_type') and self.cov_type != 'nonrobust':
            # with heteroscedasticity or correlation robustness
            k_params = self.normalized_cov_params.shape[0]
            mat = np.eye(k_params)
            const_idx = self.model.data.const_idx
            # TODO: What if model includes implicit constant, e.g. all
            #       dummies but no constant regressor?
            # TODO: Restats as LM test by projecting orthogonalizing
            #       to constant?
            if self.model.data.k_constant == 1:
                # if constant is implicit, return nan see #2444
                if const_idx is None:
                    return np.nan

                idx = lrange(k_params)
                idx.pop(const_idx)
                mat = mat[idx]  # remove constant
                if mat.size == 0:  # see  #3642
                    return np.nan
            ft = self.f_test(mat)
            # using backdoor to set another attribute that we already have
            self._cache['f_pvalue'] = float(ft.pvalue)
            return float(ft.fvalue)
        else:
            # for standard homoscedastic case
            return self.mse_model/self.mse_resid

    @cache_readonly
    def f_pvalue(self):
        # Special case for df_model 0
        if self.df_model == 0:
            return np.full_like(self.fvalue, np.nan)
        return stats.f.sf(self.fvalue, self.df_model, self.df_resid)

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def aic(self):
        return -2 * self.llf + 2 * (self.df_model + self.k_constant)

    @cache_readonly
    def bic(self):
        return (-2 * self.llf + np.log(self.nobs) * (self.df_model +
                                                     self.k_constant))

    @cache_readonly
    def eigenvals(self):
        if self._wexog_singular_values is not None:
            eigvals = self._wexog_singular_values ** 2
        else:
            eigvals = np.linalg.linalg.eigvalsh(np.dot(self.model.wexog.T,
                                                       self.model.wexog))
        return np.sort(eigvals)[::-1]

    @cache_readonly
    def condition_number(self):
        eigvals = self.eigenvals
        return np.sqrt(eigvals[0]/eigvals[-1])

    # TODO: make these properties reset bse
    def _HCCM(self, scale):
        H = np.dot(self.model.pinv_wexog,
                   scale[:, None] * self.model.pinv_wexog.T)
        return H

    def _abat_diagonal(self, a, b):
        # equivalent to np.diag(a @ b @ a.T)
        return np.einsum('ij,ik,kj->i', a, a, b)

    @cache_readonly
    def cov_HC0(self):
        self.het_scale = self.wresid**2
        cov_HC0 = self._HCCM(self.het_scale)
        return cov_HC0

    @cache_readonly
    def cov_HC1(self):
        self.het_scale = self.nobs/(self.df_resid)*(self.wresid**2)
        cov_HC1 = self._HCCM(self.het_scale)
        return cov_HC1

    @cache_readonly
    def cov_HC2(self):
        wexog = self.model.wexog
        h = self._abat_diagonal(wexog, self.normalized_cov_params)
        self.het_scale = self.wresid**2/(1-h)
        cov_HC2 = self._HCCM(self.het_scale)
        return cov_HC2

    @cache_readonly
    def cov_HC3(self):
        wexog = self.model.wexog
        h = self._abat_diagonal(wexog, self.normalized_cov_params)
        self.het_scale = (self.wresid / (1 - h))**2
        cov_HC3 = self._HCCM(self.het_scale)
        return cov_HC3

    @cache_readonly
    def HC0_se(self):
        return np.sqrt(np.diag(self.cov_HC0))

    @cache_readonly
    def HC1_se(self):
        return np.sqrt(np.diag(self.cov_HC1))

    @cache_readonly
    def HC2_se(self):
        return np.sqrt(np.diag(self.cov_HC2))

    @cache_readonly
    def HC3_se(self):
        return np.sqrt(np.diag(self.cov_HC3))

    @cache_readonly
    def resid_pearson(self):
        if not hasattr(self, 'resid'):
            raise ValueError('Method requires residuals.')
        eps = np.finfo(self.wresid.dtype).eps
        if np.sqrt(self.scale) < 10 * eps * self.model.endog.mean():
            # do not divide if scale is zero close to numerical precision
            warnings.warn(
                "All residuals are 0, cannot compute normed residuals.",
                RuntimeWarning
            )
            return self.wresid
        else:
            return self.wresid / np.sqrt(self.scale)

    def _is_nested(self, restricted):
        if self.model.nobs != restricted.model.nobs:
            return False

        full_rank = self.model.rank
        restricted_rank = restricted.model.rank
        if full_rank <= restricted_rank:
            return False

        restricted_exog = restricted.model.wexog
        full_wresid = self.wresid

        scores = restricted_exog * full_wresid[:, None]
        score_l2 = np.sqrt(np.mean(scores.mean(0) ** 2))
        # TODO: Could be improved, and may fail depending on scale of
        # regressors
        return np.allclose(score_l2, 0)

    def compare_lm_test(self, restricted, demean=True, use_lr=False):
        import statsmodels.stats.sandwich_covariance as sw
        from numpy.linalg import inv

        if not self._is_nested(restricted):
            raise ValueError("Restricted model is not nested by full model.")

        wresid = restricted.wresid
        wexog = self.model.wexog
        scores = wexog * wresid[:, None]

        n = self.nobs
        df_full = self.df_resid
        df_restr = restricted.df_resid
        df_diff = (df_restr - df_full)

        s = scores.mean(axis=0)
        if use_lr:
            scores = wexog * self.wresid[:, None]
            demean = False

        if demean:
            scores = scores - scores.mean(0)[None, :]
        # Form matters here.  If homoskedastics can be sigma^2 (X'X)^-1
        # If Heteroskedastic then the form below is fine
        # If HAC then need to use HAC
        # If Cluster, should use cluster

        cov_type = getattr(self, 'cov_type', 'nonrobust')
        if cov_type == 'nonrobust':
            sigma2 = np.mean(wresid**2)
            xpx = np.dot(wexog.T, wexog) / n
            s_inv = inv(sigma2 * xpx)
        elif cov_type in ('HC0', 'HC1', 'HC2', 'HC3'):
            s_inv = inv(np.dot(scores.T, scores) / n)
        elif cov_type == 'HAC':
            maxlags = self.cov_kwds['maxlags']
            s_inv = inv(sw.S_hac_simple(scores, maxlags) / n)
        elif cov_type == 'cluster':
            # cluster robust standard errors
            groups = self.cov_kwds['groups']
            # TODO: Might need demean option in S_crosssection by group?
            s_inv = inv(sw.S_crosssection(scores, groups))
        else:
            raise ValueError('Only nonrobust, HC, HAC and cluster are ' +
                             'currently connected')

        lm_value = n * (s @ s_inv @ s.T)
        p_value = stats.chi2.sf(lm_value, df_diff)
        return lm_value, p_value, df_diff

    def compare_f_test(self, restricted):
        has_robust1 = getattr(self, 'cov_type', 'nonrobust') != 'nonrobust'
        has_robust2 = (getattr(restricted, 'cov_type', 'nonrobust') !=
                       'nonrobust')

        if has_robust1 or has_robust2:
            warnings.warn('F test for comparison is likely invalid with ' +
                          'robust covariance, proceeding anyway',
                          InvalidTestWarning)

        ssr_full = self.ssr
        ssr_restr = restricted.ssr
        df_full = self.df_resid
        df_restr = restricted.df_resid

        df_diff = (df_restr - df_full)
        f_value = (ssr_restr - ssr_full) / df_diff / ssr_full * df_full
        p_value = stats.f.sf(f_value, df_diff, df_full)
        return f_value, p_value, df_diff

    def compare_lr_test(self, restricted, large_sample=False):
        # TODO: put into separate function, needs tests

        # See mailing list discussion October 17,

        if large_sample:
            return self.compare_lm_test(restricted, use_lr=True)

        has_robust1 = (getattr(self, 'cov_type', 'nonrobust') != 'nonrobust')
        has_robust2 = (
            getattr(restricted, 'cov_type', 'nonrobust') != 'nonrobust')

        if has_robust1 or has_robust2:
            warnings.warn('Likelihood Ratio test is likely invalid with ' +
                          'robust covariance, proceeding anyway',
                          InvalidTestWarning)

        llf_full = self.llf
        llf_restr = restricted.llf
        df_full = self.df_resid
        df_restr = restricted.df_resid

        lrdf = (df_restr - df_full)
        lrstat = -2*(llf_restr - llf_full)
        lr_pvalue = stats.chi2.sf(lrstat, lrdf)

        return lrstat, lr_pvalue, lrdf

    def get_robustcov_results(self, cov_type='HC1', use_t=None, **kwargs):
        import statsmodels.stats.sandwich_covariance as sw
        from statsmodels.base.covtype import normalize_cov_type, descriptions

        cov_type = normalize_cov_type(cov_type)

        if 'kernel' in kwargs:
            kwargs['weights_func'] = kwargs.pop('kernel')
        if 'weights_func' in kwargs and not callable(kwargs['weights_func']):
            kwargs['weights_func'] = sw.kernel_dict[kwargs['weights_func']]

        # TODO: make separate function that returns a robust cov plus info
        use_self = kwargs.pop('use_self', False)
        if use_self:
            res = self
        else:
            res = self.__class__(
                self.model, self.params,
                normalized_cov_params=self.normalized_cov_params,
                scale=self.scale)

        res.cov_type = cov_type
        # use_t might already be defined by the class, and already set
        if use_t is None:
            use_t = self.use_t
        res.cov_kwds = {'use_t': use_t}  # store for information
        res.use_t = use_t

        adjust_df = False
        if cov_type in ['cluster', 'hac-panel', 'hac-groupsum']:
            df_correction = kwargs.get('df_correction', None)
            # TODO: check also use_correction, do I need all combinations?
            if df_correction is not False:  # i.e. in [None, True]:
                # user did not explicitely set it to False
                adjust_df = True

        res.cov_kwds['adjust_df'] = adjust_df

        # verify and set kwargs, and calculate cov
        # TODO: this should be outsourced in a function so we can reuse it in
        #       other models
        # TODO: make it DRYer   repeated code for checking kwargs
        if cov_type in ['fixed scale', 'fixed_scale']:
            res.cov_kwds['description'] = descriptions['fixed_scale']

            res.cov_kwds['scale'] = scale = kwargs.get('scale', 1.)
            res.cov_params_default = scale * res.normalized_cov_params
        elif cov_type.upper() in ('HC0', 'HC1', 'HC2', 'HC3'):
            if kwargs:
                raise ValueError('heteroscedasticity robust covariance '
                                 'does not use keywords')
            res.cov_kwds['description'] = descriptions[cov_type.upper()]
            res.cov_params_default = getattr(self, 'cov_' + cov_type.upper())
        elif cov_type.lower() == 'hac':
            # TODO: check if required, default in cov_hac_simple
            maxlags = kwargs['maxlags']
            res.cov_kwds['maxlags'] = maxlags
            weights_func = kwargs.get('weights_func', sw.weights_bartlett)
            res.cov_kwds['weights_func'] = weights_func
            use_correction = kwargs.get('use_correction', False)
            res.cov_kwds['use_correction'] = use_correction
            res.cov_kwds['description'] = descriptions['HAC'].format(
                maxlags=maxlags,
                correction=['without', 'with'][use_correction])

            res.cov_params_default = sw.cov_hac_simple(
                self, nlags=maxlags, weights_func=weights_func,
                use_correction=use_correction)
        elif cov_type.lower() == 'cluster':
            # cluster robust standard errors, one- or two-way
            groups = kwargs['groups']
            if not hasattr(groups, 'shape'):
                groups = np.asarray(groups).T

            if groups.ndim >= 2:
                groups = groups.squeeze()

            res.cov_kwds['groups'] = groups
            use_correction = kwargs.get('use_correction', True)
            res.cov_kwds['use_correction'] = use_correction
            if groups.ndim == 1:
                if adjust_df:
                    # need to find number of groups
                    # duplicate work
                    self.n_groups = n_groups = len(np.unique(groups))
                res.cov_params_default = sw.cov_cluster(
                    self, groups, use_correction=use_correction)

            elif groups.ndim == 2:
                if hasattr(groups, 'values'):
                    groups = groups.values

                if adjust_df:
                    # need to find number of groups
                    # duplicate work
                    n_groups0 = len(np.unique(groups[:, 0]))
                    n_groups1 = len(np.unique(groups[:, 1]))
                    self.n_groups = (n_groups0, n_groups1)
                    n_groups = min(n_groups0, n_groups1)  # use for adjust_df

                # Note: sw.cov_cluster_2groups has 3 returns
                res.cov_params_default = sw.cov_cluster_2groups(
                    self, groups, use_correction=use_correction)[0]
            else:
                raise ValueError('only two groups are supported')
            res.cov_kwds['description'] = descriptions['cluster']

        elif cov_type.lower() == 'hac-panel':
            # cluster robust standard errors
            res.cov_kwds['time'] = time = kwargs.get('time', None)
            res.cov_kwds['groups'] = groups = kwargs.get('groups', None)
            # TODO: nlags is currently required
            # nlags = kwargs.get('nlags', True)
            # res.cov_kwds['nlags'] = nlags
            # TODO: `nlags` or `maxlags`
            res.cov_kwds['maxlags'] = maxlags = kwargs['maxlags']
            use_correction = kwargs.get('use_correction', 'hac')
            res.cov_kwds['use_correction'] = use_correction
            weights_func = kwargs.get('weights_func', sw.weights_bartlett)
            res.cov_kwds['weights_func'] = weights_func
            if groups is not None:
                groups = np.asarray(groups)
                tt = (np.nonzero(groups[:-1] != groups[1:])[0] + 1).tolist()
                nobs_ = len(groups)
            elif time is not None:
                time = np.asarray(time)
                # TODO: clumsy time index in cov_nw_panel
                tt = (np.nonzero(time[1:] < time[:-1])[0] + 1).tolist()
                nobs_ = len(time)
            else:
                raise ValueError('either time or groups needs to be given')
            groupidx = lzip([0] + tt, tt + [nobs_])
            self.n_groups = n_groups = len(groupidx)
            res.cov_params_default = sw.cov_nw_panel(self, maxlags, groupidx,
                                                     weights_func=weights_func,
                                                     use_correction=use_correction)
            res.cov_kwds['description'] = descriptions['HAC-Panel']

        elif cov_type.lower() == 'hac-groupsum':
            # Driscoll-Kraay standard errors
            res.cov_kwds['time'] = time = kwargs['time']
            # TODO: nlags is currently required
            # nlags = kwargs.get('nlags', True)
            # res.cov_kwds['nlags'] = nlags
            # TODO: `nlags` or `maxlags`
            res.cov_kwds['maxlags'] = maxlags = kwargs['maxlags']
            use_correction = kwargs.get('use_correction', 'cluster')
            res.cov_kwds['use_correction'] = use_correction
            weights_func = kwargs.get('weights_func', sw.weights_bartlett)
            res.cov_kwds['weights_func'] = weights_func
            if adjust_df:
                # need to find number of groups
                tt = (np.nonzero(time[1:] < time[:-1])[0] + 1)
                self.n_groups = n_groups = len(tt) + 1
            res.cov_params_default = sw.cov_nw_groupsum(
                self, maxlags, time, weights_func=weights_func,
                use_correction=use_correction)
            res.cov_kwds['description'] = descriptions['HAC-Groupsum']
        else:
            raise ValueError('cov_type not recognized. See docstring for ' +
                             'available options and spelling')

        if adjust_df:
            # Note: df_resid is used for scale and others, add new attribute
            res.df_resid_inference = n_groups - 1

        return res

    def get_prediction(self, exog=None, transform=True, weights=None,
                       row_labels=None, **kwargs):

        return pred.get_prediction(
            self, exog=exog, transform=transform, weights=weights,
            row_labels=row_labels, **kwargs)

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        from statsmodels.stats.stattools import (
            jarque_bera, omni_normtest, durbin_watson)

        jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)

        eigvals = self.eigenvals
        condno = self.condition_number

        # TODO: Avoid adding attributes in non-__init__
        self.diagn = dict(jb=jb, jbpv=jbpv, skew=skew, kurtosis=kurtosis,
                          omni=omni, omnipv=omnipv, condno=condno,
                          mineigval=eigvals[-1])

        # TODO not used yet
        # diagn_left_header = ['Models stats']
        # diagn_right_header = ['Residual stats']

        # TODO: requiring list/iterable is a bit annoying
        #   need more control over formatting
        # TODO: default do not work if it's not identically spelled

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Least Squares']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Observations:', None),
                    ('Df Residuals:', None),
                    ('Df Model:', None),
                    ]

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        rsquared_type = '' if self.k_constant else ' (uncentered)'
        top_right = [('R-squared' + rsquared_type + ':',
                      ["%#8.3f" % self.rsquared]),
                     ('Adj. R-squared' + rsquared_type + ':',
                      ["%#8.3f" % self.rsquared_adj]),
                     ('F-statistic:', ["%#8.4g" % self.fvalue]),
                     ('Prob (F-statistic):', ["%#6.3g" % self.f_pvalue]),
                     ('Log-Likelihood:', None),
                     ('AIC:', ["%#8.4g" % self.aic]),
                     ('BIC:', ["%#8.4g" % self.bic])
                     ]

        diagn_left = [('Omnibus:', ["%#6.3f" % omni]),
                      ('Prob(Omnibus):', ["%#6.3f" % omnipv]),
                      ('Skew:', ["%#6.3f" % skew]),
                      ('Kurtosis:', ["%#6.3f" % kurtosis])
                      ]

        diagn_right = [('Durbin-Watson:',
                        ["%#8.3f" % durbin_watson(self.wresid)]
                        ),
                       ('Jarque-Bera (JB):', ["%#8.3f" % jb]),
                       ('Prob(JB):', ["%#8.3g" % jbpv]),
                       ('Cond. No.', ["%#8.3g" % condno])
                       ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        # create summary table instance
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                              use_t=self.use_t)

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                             yname=yname, xname=xname,
                             title="")

        # add warnings/notes, added to text format only
        etext = []
        if not self.k_constant:
            etext.append(
                "R² is computed without centering (uncentered) since the "
                "model does not contain a constant."
            )
        if hasattr(self, 'cov_type'):
            etext.append(self.cov_kwds['description'])
        if self.model.exog.shape[0] < self.model.exog.shape[1]:
            wstr = "The input rank is higher than the number of observations."
            etext.append(wstr)
        if eigvals[-1] < 1e-10:
            wstr = "The smallest eigenvalue is %6.3g. This might indicate "
            wstr += "that there are\n"
            wstr += "strong multicollinearity problems or that the design "
            wstr += "matrix is singular."
            wstr = wstr % eigvals[-1]
            etext.append(wstr)
        elif condno > 1000:  # TODO: what is recommended?
            wstr = "The condition number is large, %6.3g. This might "
            wstr += "indicate that there are\n"
            wstr += "strong multicollinearity or other numerical "
            wstr += "problems."
            wstr = wstr % condno
            etext.append(wstr)

        if etext:
            etext = ["[{0}] {1}".format(i + 1, text)
                     for i, text in enumerate(etext)]
            etext.insert(0, "Notes:")
            smry.add_extra_txt(etext)

        return smry

    def summary2(self, yname=None, xname=None, title=None, alpha=.05,
                 float_format="%.4f"):
        # Diagnostics
        from statsmodels.stats.stattools import (jarque_bera,
                                                 omni_normtest,
                                                 durbin_watson)

        jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
        omni, omnipv = omni_normtest(self.wresid)
        dw = durbin_watson(self.wresid)
        eigvals = self.eigenvals
        condno = self.condition_number
        eigvals = np.sort(eigvals)  # in increasing order
        diagnostic = dict([
            ('Omnibus:',  "%.3f" % omni),
            ('Prob(Omnibus):', "%.3f" % omnipv),
            ('Skew:', "%.3f" % skew),
            ('Kurtosis:', "%.3f" % kurtosis),
            ('Durbin-Watson:', "%.3f" % dw),
            ('Jarque-Bera (JB):', "%.3f" % jb),
            ('Prob(JB):', "%.3f" % jbpv),
            ('Condition No.:', "%.0f" % condno)
            ])

        # Summary
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_base(results=self, alpha=alpha, float_format=float_format,
                      xname=xname, yname=yname, title=title)
        smry.add_dict(diagnostic)

        # Warnings
        if eigvals[-1] < 1e-10:
            warn = "The smallest eigenvalue is %6.3g. This might indicate that\
            there are strong multicollinearity problems or that the design\
            matrix is singular." % eigvals[-1]
            smry.add_text(warn)
        if condno > 1000:
            warn = "* The condition number is large (%.g). This might indicate \
            strong multicollinearity or other numerical problems." % condno
            smry.add_text(warn)

        return smry

class LikelihoodResultsWrapper(wrap.ResultsWrapper):
    _attrs = {
        'params': 'columns',
        'bse': 'columns',
        'pvalues': 'columns',
        'tvalues': 'columns',
        'resid': 'rows',
        'fittedvalues': 'rows',
        'normalized_cov_params': 'cov',
    }

    _wrap_attrs = _attrs
    _wrap_methods = {
        'cov_params': 'cov',
        'conf_int': 'columns'
    }

wrap.populate_wrapper(LikelihoodResultsWrapper,  # noqa:E305
                      LikelihoodModelResults)

class RegressionResultsWrapper(wrap.ResultsWrapper):

    _attrs = {
        'chisq': 'columns',
        'sresid': 'rows',
        'weights': 'rows',
        'wresid': 'rows',
        'bcov_unscaled': 'cov',
        'bcov_scaled': 'cov',
        'HC0_se': 'columns',
        'HC1_se': 'columns',
        'HC2_se': 'columns',
        'HC3_se': 'columns',
        'norm_resid': 'rows',
    }

    _wrap_attrs = wrap.union_dicts(LikelihoodResultsWrapper._attrs,
                                   _attrs)

    _methods = {}

    _wrap_methods = wrap.union_dicts(
                        LikelihoodResultsWrapper._wrap_methods,
                        _methods)


wrap.populate_wrapper(RegressionResultsWrapper,
                      RegressionResults)

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

    def initialize(self, endog, freq_weights):
        return endog, np.ones(endog.shape[0])

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

        self.df_model = np.linalg.matrix_rank(self.exog) - 1
        self.wnobs = self.exog.shape[0]
        self.df_resid = self.exog.shape[0] - self.df_model - 1

        self.family = Binomial(link)

        self.freq_weights = np.ones(len(endog))
        self.var_weights = np.ones(len(endog))
        self.iweights = np.asarray(self.freq_weights * self.var_weights)

        self.nobs = self.endog.shape[0]

        self.endog, self.n_trials = self.family.initialize(self.endog, self.freq_weights)

    def _update_history(self, tmp_result, mu, history):
        history['params'].append(tmp_result.params)
        history['deviance'].append(self.family.deviance(self.endog, mu))
        return history

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

        wls_model = WLS(wlsendog, wlsexog, self.weights)
        wls_results = wls_model.fit(method='pinv')

        logLike = self.family.loglike(self.endog, self.mu, var_weights=self.var_weights, freq_weights=self.freq_weights, scale=self.scale)
        return Summary(self.linkname, wls_results.params, self.scale, logLike)

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