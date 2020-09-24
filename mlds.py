import warnings
import inspect
import pandas as pd
import numpy as np
from scipy import special
import statsmodels.api as sm
import statsmodels.formula.api as smf

FLOAT_EPS = np.finfo(float).eps

class Binomial():
    links = [
        sm.families.links.logit,
        sm.families.links.probit,
        sm.families.links.cauchy,
        sm.families.links.log,
        sm.families.links.cloglog,
        sm.families.links.identity
    ]

    safe_links = [
        sm.families.links.Logit,
        sm.families.links.CDFLink
    ]

    def _setlink(self, link):
        self._link = link
        if not isinstance(link, sm.families.links.Link):
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
        # if not np.all(np.asarray(freq_weights) == 1):
        #     self.variance = V.Binomial(n=freq_weights)
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
    res = smf.glm('resp ~ S2 + S3 + S4 + S5 + S6 + S7 + S8 + S9 + S10 + S10 + S11 - 1', family=Binomial(sm.families.links.probit()), data=data).fit()
    print(res.summary())

if __name__ == '__main__':
    mlds('data.txt')