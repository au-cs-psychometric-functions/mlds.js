(function(exports) {
    const default_clip = p => p.map(e => Math.max(Math.EPSILON, Math.min(1 - Math.EPSILON, e)));
    
    const inf_clip = p => p.map(e => Math.max(Math.EPSILON, Math.min(Math.POSITIVE_INFINITY, e)));

    const s2pi = 2.50662827463100050242E0;
    
    const P0 = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ];

    const Q0 = [
        1,
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ];

    const P1 = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ];

    const Q1 = [
        1,
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ];

    const P2 = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ];

    const Q2 = [
        1,
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ];

    const ndtri = y0 => {
        let negate = true;
        y = y0;
        if (y > 1.0 - 0.13533528323661269189) {
            y = 1.0 - y;
            negate = false;
        }

        if (y > 0.13533528323661269189) {
            y = y - 0.5;
            y2 = y * y;
            x = y + y * (y2 * polevl(y2, P0) / polevl(y2, Q0));
            x = x * s2pi;
            return x;
        }

        x = Math.sqrt(-2.0 * Math.log(y));
        x0 = x - Math.log(x) / x;

        z = 1.0 / x;
        if (x < 8.0) {
            x1 = z * polevl(z, P1) / polevl(z, Q1);
        } else {
            x1 = z * polevl(z, P2) / polevl(z, Q2);
        }
        x = x0 - x1;
        if (negate) {
            x = -x;
        }
        return x;
    }

    const polevl = (x, coef) => {
        let accum = 0;
        coef.forEach(c => {
            acum = x * accum + c;
        });
        return accum;
    }

    const erf = x => {
        z = Math.abs(x);
        t = 1. / (1. + 0.5 * z)
        r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
            t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
            t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 +
            t * 0.17087277)))))))));
        if (x >= 0.0) {
            return r
        }
        else {
            return 2.0 - r;
        }
    }

    const cdf = x => {
        return 1.0 - 0.5 * erf(x / Math.sqrt(2));
    }

    const pdf = x => {
        return (1 / s2pi) * Math.exp(-x * x / 2);
    }

    const probit = () => {
        const link = p => default_clip(p).map(e => ndtri(e));
        link.inverse = z => z.map(e => cdf(e));
        link.deriv = p => default_clip(p).map(e => 1 / pdf(ndtri(e)));
        link.inverse_deriv = z => link.deriv(link.inverse(z)).map(e => 1 / e);
        return link;
    }

    const deviance = (y, mu) => {
        const y_mu = inf_clip(y.map((e, i) => e / mu[i]));
        const n_y_mu = inf_clip(y.map((e, i) => (1.0 - e) / (1.0 - mu[i])));
        return y.map((e, i) => 2 * (e * Math.log(y_mu[i]) + (1 - y[i]) * Math.log(n_y_mu[i]))).reduce((a, b) => a + b, 0);
    }

    const allclose = (a, b, atol, rtol) => {
        return Math.abs(a - b) <= (atol + rtol * Math.abs(b));
    }

    const check_convergence = (criterion, iteration, atol, rtol) => {
        return allclose(criterion[iteration], criterion[iteration + 1], atol, rtol);
    }

    const glm = (y, x) => {
        const link = probit();
        
        let wls_x = x;

        let mu = y.map(e => (e + 0.5) / 2);
        let lin_pred = link(mu);

        const dev = [Math.POSITIVE_INFINITY, deviance(y, mu)];

        let iteration = 0;
        while (true) {
            iteration++;
            if (iteration > 100) {
                break;
            }

            variance = default_clip(mu).map(e => e * (1 - e));
            weights = link.deriv(mu).map((e, i) => 1.0 / (e * e * variance[i]));

            let wls_y = link.deriv(mu).map((e, i) => lin_pred[i] + e * (y[i] - mu[i]));

            let w_half = weights.map(w => Math.sqrt(w));
            let m_y = wls_y.map((e, i) => e * w_half[i]);
            let m_x = w_half.map((w, i) => wls_x[i].map(e => w * e));
            let wls_results = lstsq(m_x, m_y, -1);

            lin_pred = x.map(r => r.map((e, i) => e * wls_results[i]).reduce((a, b) => a + b, 0));
            mu = link.inverse(lin_pred);
            dev.append(deviance(y, mu));
            converged = check_convergence(dev, iteration, 1e-8, 0);
            if (converged) {
                break;
            }
        }

        wls_y = wls_y.map((e, i) => e * Math.sqrt(weights[i]));
        wls_x = weights.map((w, i) => wls_x[i].map(e => Math.sqrt(w) * e));
        wls_x = pinv(wls_x, 1e-15);
        wls_results = wls_x.map(r => r.map((e, i) => e * wls_y[i]).reduce((a, b) => a + b, 0));

        const log_like = y.map((e, i) => Math.lgamma(2) - Math.lgamma(e + 1) -
                        Math.lgamma(2 - e) + e * Math.log(mu[i] / (1 - mu[i])) +
                        Math.log(1 - mu[i])).reduce((a, b) => a + b, 0);
        return [wls_results, log_like];
    }
})();