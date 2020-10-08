import math
import numpy as np

FLOAT_EPS = np.finfo(float).eps

def default_clip(p):
    return np.asarray([max(FLOAT_EPS, min(1 - FLOAT_EPS, e)) for e in p])

def inf_clip(p):
    return np.asarray([max(FLOAT_EPS, min(float('inf'), e)) for e in p])

s2pi = 2.50662827463100050242E0

P0 = [
    -5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0,
]

Q0 = [
    1,
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0,
]

P1 = [
    4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
]

Q1 = [
    1,
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
]

P2 = [
    3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9,
]

Q2 = [
    1,
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9,
]

def ndtri(y0):
    if y0 <= 0 or y0 >= 1:
        raise ValueError("ndtri(x) needs 0 < x < 1")
    negate = True
    y = y0
    if y > 1.0 - 0.13533528323661269189:
        y = 1.0 - y
        negate = False

    if y > 0.13533528323661269189:
        y = y - 0.5
        y2 = y * y
        x = y + y * (y2 * polevl(y2, P0) / polevl(y2, Q0))
        x = x * s2pi
        return x

    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x

    z = 1.0 / x
    if x < 8.0:
        x1 = z * polevl(z, P1) / polevl(z, Q1)
    else:
        x1 = z * polevl(z, P2) / polevl(z, Q2)
    x = x0 - x1
    if negate:
        x = -x
    return x

def polevl(x, coef):
    accum = 0
    for c in coef:
        accum = x * accum + c
    return accum

def erf(x):
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * math.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r

def cdf(x):
    return 1. - 0.5*erf(x/(2**0.5))

def pdf(x):
    return (1/s2pi)*math.exp(-x*x/2)

def probit():
    def link(p):
        p = default_clip(p)
        return np.asarray([ndtri(e) for e in p])
    def inverse(z):
        return np.asarray([cdf(e) for e in z])
    def deriv(p):
        p = default_clip(p)
        return 1 / np.asarray([pdf(ndtri(e)) for e in p])
    def inverse_deriv(z):
        return 1 / deriv(inverse(z))
    link.inverse = inverse
    link.deriv = deriv
    link.inverse_deriv = inverse_deriv
    return link

def deviance(y, mu):
    y_mu = inf_clip(y / mu)
    n_y_mu = inf_clip((1. - y) / (1. - mu))
    resid_dev = y * np.log(y_mu) + (1 - y) * np.log(n_y_mu)
    return np.sum(2 * resid_dev)    

def check_convergence(criterion, iteration, atol, rtol):
    return np.allclose(criterion[iteration], criterion[iteration + 1], atol=atol, rtol=rtol)

def glm(y, x):
    link = probit()
    y = np.asarray(y)
    x = np.asarray(x)

    maxiter = 100
    atol = 1e-8
    rtol = 0

    wls_x = x

    mu = np.asarray([(e + .5) / 2 for e in y])
    lin_pred = link(mu)
    
    converged = False

    dev = [np.inf, deviance(y, mu)]

    for iteration in range(maxiter):
        p = default_clip(mu)
        variance = p * (1 - p)
        weights = 1. / (link.deriv(mu)**2 * variance)

        wls_y = (lin_pred + link.deriv(mu) * (y - mu))

        w_half = np.sqrt(weights)
        m_y = w_half * wls_y
        m_x = np.asarray(w_half)[:, None] * wls_x
        wls_results, _, _, _ = np.linalg.lstsq(m_x, m_y, rcond=-1)

        lin_pred = np.dot(x, wls_results)
        mu = link.inverse(lin_pred)
        dev.append(deviance(y, mu))
        converged = check_convergence(dev, iteration + 1, atol, rtol)
        if converged:
            break

    wls_y = np.asarray(wls_y) * np.sqrt(weights)
    wls_x = np.asarray(wls_x) * np.sqrt(weights)[:, None]
    wls_x = np.linalg.pinv(wls_x, rcond=1e-15)
    wls_results = np.dot(wls_x, wls_y)

    logLike = sum([math.lgamma(2) - math.lgamma(y[i] + 1) -
            math.lgamma(2 - y[i]) + y[i] * math.log(mu[i] / (1 - mu[i])) +
            math.log(1 - mu[i]) for i in range(len(y))])
    return Summary('probit', wls_results, logLike)

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

    summary = glm(y, x)
    summary.print()

if __name__ == '__main__':
    mlds('table.txt')