import warnings
import inspect
import pandas as pd
import numpy as np
import scipy.stats
from scipy import special
import statsmodels.formula.api as smf

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
            link = sm.families.links.logit()
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
        self.variance = sm.families.varfuncs.Binomial(n=self.n)

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

def mlds(filename):
    data = pd.read_table(filename, sep='\t')
    res = smf.glm('resp ~ S2 + S3 + S4 + S5 + S6 + S7 + S8 + S9 + S10 + S10 + S11 - 1', family=Binomial(probit()), data=data).fit()
    print(res.summary())

if __name__ == '__main__':
    mlds('data.txt')