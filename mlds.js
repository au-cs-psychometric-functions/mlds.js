const mlds = (function() {
    const default_clip = p => p.map(e => Math.max(Number.EPSILON, Math.min(1 - Number.EPSILON, e)))

    const inf_clip = p => p.map(e => Math.max(Number.EPSILON, Math.min(Number.POSITIVE_INFINITY, e)));
    
    const transpose = a => {
        const m = a.length;
        const n = a[0].length;
        let at = [];
        for (let i = 0; i < n; i++) {
            at.push(Array.from(Array(m), () => 0.0));
        }
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                at[j][i] = a[i][j];
            }
        }
        return at;
    }

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
        let y = y0;
        if (y > 1.0 - 0.13533528323661269189) {
            y = 1.0 - y;
            negate = false;
        }
        
        if (y > 0.13533528323661269189) {
            y = y - 0.5;
            let y2 = y * y;
            let x = y + y * (y2 * polevl(y2, P0) / polevl(y2, Q0));
            x = x * s2pi;
            return x;
        }

        let x = Math.sqrt(-2.0 * Math.log(y));
        let x0 = x - Math.log(x) / x;

        let z = 1.0 / x;
        let x1 = 0;
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
            accum = x * accum + c;
        });
        return accum;
    }
    
    const erf = x => {
        const z = Math.abs(x)
        const t = 1.0 / (1.0 + 0.5 * z)
        const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
            t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
            t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 +
            t * 0.17087277)))))))))
        if (x >= 0.0) {
            return r;
        } else {
            return 2.0 - r;
        }
    }

    const cdf = x => 1.0 - 0.5 * erf(x / Math.sqrt(2));

    const pdf = x => (1 / s2pi) * Math.exp(-x * x / 2);

    const probit = () => {
        const link = p => default_clip(p).map(e => ndtri(e));
        link.inverse = z => z.map(e => cdf(e));
        link.deriv = p => default_clip(p).map(e => 1 / pdf(ndtri(e)));
        link.inverse_deriv = z => deriv(inverse(p)).map(e => 1 / e);
        return link;
    }

    const deviance = (y, mu) => {
        y_mu = inf_clip(y.map((e, i) => e / mu[i]));
        n_y_mu = inf_clip(y.map((e, i) => (1.0 - e) / (1.0 - mu[i])));
        return y.map((e, i) => 2 * (e * Math.log(y_mu[i]) + (1 - e) * Math.log(n_y_mu[i]))).reduce((a, b) => a + b, 0);
    }

    const allclose = (a, b, atol, rtol) => Math.abs(a - b) <= (atol + rtol * Math.abs(b));

    const check_convergence = (criterion, iteration, atol, rtol) => allclose(criterion[iteration], criterion[iteration + 1], atol, rtol);

    const rref = a => {
        let lead = 0;
        const m = a.length;
        const n = a[0].length;
        for (let r = 0; r < m; r++) {
            if (n <= lead) {
                return;
            }
            let i = r;
            while (a[i][lead] == 0) {
                i++;
                if (m === i) {
                    i = r;
                    lead++;
                    if (n === lead) {
                        return;
                    }
                }
            }
     
            [a[i], a[r]] = [a[r], a[i]];
     
            let val = a[r][lead];
            for (let j = 0; j < n; j++) {
                a[r][j] /= val;
            }
     
            for (let i = 0; i < m; i++) {
                if (i === r) {
                    continue;
                }
                val = a[i][lead];
                for (var j = 0; j < n; j++) {
                    a[i][j] -= val * a[r][j];
                }
            }
            lead++;
        }
        return a;
    }
    
    const lstsq = (a, b) => {
        at = transpose(a);
        ata = at.map(x => at.map(y => x.map((e, i) => e * y[i]).reduce((a, b) => a + b, 0)));
        atb = at.map(x => x.map((e, i) => e * b[i]).reduce((a, b) => a + b, 0));
        return rref(ata.map((e, i) => e.concat(atb[i]))).map(e => e[e.length - 1]);
    }

    const svd = a => {
        let eps = Number.EPSILON;
        const tol = 1e-64 / eps;
      
        const m = a.length;
        const n = a[0].length;
      
        if (m < n) {
          throw new Error('m < n');
        }
      
        let u = a;

        let e = Array.from(Array(n), () => 0);
        let q = Array.from(Array(n), () => 0);
        let v = Array.from(Array(n), () => Array.from(Array(n), () => 0));

        let i, j, k, l, l1, c, f, g, h, s, x, y, z;
      
        g = 0;
        x = 0;
      
        for (i = 0; i < n; i++) {
            e[i] = g;
            s = 0;
            l = i + 1;
            for (j = i; j < m; j++) {
                s += Math.pow(u[j][i], 2);
            }
            if (s < tol) {
                g = 0;
            } else {
                f = u[i][i];
                g = f < 0 ? Math.sqrt(s) : -Math.sqrt(s);
                h = f * g - s;
                u[i][i] = f - g;
                for (j = l; j < n; j++) {
                    s = 0;
                    for (k = i; k < m; k++) {
                        s += u[k][i] * u[k][j];
                    }
                    f = s / h;
                    for (k = i; k < m; k++) {
                        u[k][j] = u[k][j] + f * u[k][i];
                    }
                }
            }
            q[i] = g;
            s = 0;
            for (j = l; j < n; j++) {
                s += Math.pow(u[i][j], 2);
            }
            if (s < tol) {
                g = 0;
            } else {
                f = u[i][i + 1];
                g = f < 0 ? Math.sqrt(s) : -Math.sqrt(s);
                h = f * g - s;
                u[i][i + 1] = f - g;
                for (j = l; j < n; j++) {
                    e[j] = u[i][j] / h;
                }
                for (j = l; j < m; j++) {
                    s = 0;
                    for (k = l; k < n; k++) {
                        s += u[j][k] * u[i][k];
                    }
                    for (k = l; k < n; k++) {
                        u[j][k] = u[j][k] + s * e[k];
                    }
                }
            }
            y = Math.abs(q[i]) + Math.abs(e[i]);
            if (y > x) {
                x = y;
            }
        }
       
        for (i = n - 1; i >= 0; i--) {
            if (g !== 0) {
                h = u[i][i + 1] * g;
                for (j = l; j < n; j++) {
                    v[j][i] = u[i][j] / h;
                }
                for (j = l; j < n; j++) {
                    s = 0;
                    for (k = l; k < n; k++) {
                        s += u[i][k] * v[k][j];
                    }
                    for (k = l; k < n; k++) {
                        v[k][j] = v[k][j] + s * v[k][i];
                    }
                }
            }
            for (j = l; j < n; j++) {
                v[i][j] = 0;
                v[j][i] = 0;
            }
            v[i][i] = 1;
            g = e[i];
            l = i;
        }

        for (i = n - 1; i >= 0; i--) {
            l = i + 1;
            g = q[i];
            for (j = l; j < n; j++) {
                u[i][j] = 0;
            }
            if (g !== 0) {
                h = u[i][i] * g;
                for (j = l; j < n; j++) {
                    s = 0;
                    for (k = l; k < m; k++) {
                        s += u[k][i] * u[k][j];
                    }
                    f = s / h;
                    for (k = i; k < m; k++) {
                        u[k][j] = u[k][j] + f * u[k][i];
                    }
                }
                for (j = i; j < m; j++) {
                    u[j][i] = u[j][i] / g;
                }
            } else {
                for (j = i; j < m; j++) {
                    u[j][i] = 0;
                }
            }
            u[i][i] = u[i][i] + 1;
        }
      
        eps = eps * x;
        let test_f;
        for (k = n - 1; k >= 0; k--) {
            for (let iteration = 0; iteration < 50; iteration++) {
                test_f = false;
                for (l = k; l >= 0; l--) {
                    if (Math.abs(e[l]) <= eps) {
                        test_f = true;
                        break;
                    }
                    if (Math.abs(q[l - 1]) <= eps) {
                        break;
                    }
                }
                if (!test_f) {
                    c = 0;
                    s = 1;
                    l1 = l - 1;
                    for (i = l; i < k + 1; i++) {
                        f = s * e[i];
                        e[i] = c * e[i];
                        if (Math.abs(f) <= eps) {
                            break;
                        }
                        g = q[i];
                        q[i] = Math.sqrt(f * f + g * g);
                        h = q[i];
                        c = g / h;
                        s = -f / h;
                        for (j = 0; j < m; j++) {
                            y = u[j][l1];
                            z = u[j][i];
                            u[j][l1] = y * c + (z * s);
                            u[j][i] = -y * s + (z * c);
                        }
                    }
                }
                z = q[k];
                if (l === k) {
                    if (z < 0) {
                        q[k] = -z;
                        for (j = 0; j < n; j++) {
                            v[j][k] = -v[j][k];
                        }
                    }
                    break;
                }        
                x = q[l];
                y = q[k - 1];
                g = e[k - 1];
                h = e[k];
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
                g = Math.sqrt(f * f + 1);
                f = ((x - z) * (x + z) + h * (y / (f < 0 ? (f - g) : (f + g)) - h)) / x;
                c = 1;
                s = 1;
                for (i = l + 1; i < k + 1; i++) {
                    g = e[i];
                    y = q[i];
                    h = s * g;
                    g = c * g;
                    z = Math.sqrt(f * f + h * h);
                    e[i - 1] = z;
                    c = f / z;
                    s = h / z;
                    f = x * c + g * s;
                    g = -x * s + g * c;
                    h = y * s;
                    y = y * c;
                    for (j = 0; j < n; j++) {
                        x = v[j][i - 1];
                        z = v[j][i];
                        v[j][i - 1] = x * c + z * s;
                        v[j][i] = -x * s + z * c;
                    }
                    z = Math.sqrt(f * f + h * h);
                    q[i - 1] = z;
                    c = f / z;
                    s = h / z;
                    f = c * g + s * y;
                    x = -s * g + c * y;
                    for (j = 0; j < m; j++) {
                        y = u[j][i - 1];
                        z = u[j][i];
                        u[j][i - 1] = y * c + z * s;
                        u[j][i] = -y * s + z * c;
                    }
                }
                e[l] = 0;
                e[k] = f;
                q[k] = x;
            }
        }

        vt = transpose(v);
        return [u, q, vt];
      }

    const pinv = a => {
        let [u, s, vt] = svd(a);
        const cutoff = 1e-15 * Math.max.apply(null, s);
        s = s.map(e => e > cutoff ? 1 / e : 0);
        let ut = transpose(u);
        let v = transpose(vt);
        s = ut.map((m, i) => m.map(n => n * s[i]));
        let st = transpose(s);
        res = v.map(vm => st.map(stm => vm.map((e, i) => e * stm[i]).reduce((a, b) => a + b, 0)));
        return res;
    }

    const glm = (y, x) => {
        const link = probit();

        let wls_x = x;
        let wls_y = y;
        let weights = [];

        let mu = y.map(e => (e + 0.5) / 2);
        let lin_pred = link(mu);

        let converged = false;

        let dev = [Number.POSITIVE_INFINITY, deviance(y, mu)];

        let iteration = 0;
        while (true) {
            iteration++;
            if (iteration > 100) {
                break;
            }

            let variance = default_clip(mu).map(e => e * (1 - e));
            weights = link.deriv(mu).map((e, i) => 1.0 / (e * e * variance[i]));

            wls_y = link.deriv(mu).map((e, i) => lin_pred[i] + e * (y[i] - mu[i]));

            let w_half = weights.map(e => Math.sqrt(e));
            let m_y = wls_y.map((e, i) => e * w_half[i]);
            let m_x = w_half.map((e, i) => wls_x[i].map(x => e * x));
            let wls_results = lstsq(m_x, m_y);

            lin_pred = x.map(r => r.map((e, i) => e * wls_results[i]).reduce((a, b) => a + b, 0));
            mu = link.inverse(lin_pred);
            dev.push(deviance(y, mu));
            converged = check_convergence(dev, iteration, 1e-8, 0);
            if (converged) {
                break;
            }
        }

        wls_y = wls_y.map((e, i) => e * Math.sqrt(weights[i]));
        wls_x = weights.map((e, i) => wls_x[i].map(x => Math.sqrt(e) * x));
        wls_x = pinv(wls_x);
        wls_results = wls_x.map(r => r.map((e, i) => e * wls_y[i]).reduce((a, b) => a + b, 0));
        return wls_results;
    }
    
    return {
        mlds: data => {
            data = data.map(row => {
                if (row[1] > row[3]) {
                    [row[1], row[3]] = [row[3], row[1]];
                    row[0] = 1 - row[0];
                }
                return row;
            }).filter(e => e.length == 4);
    
            const mx = Math.max.apply(null, data.map(row => Math.max.apply(Math, row)));
            const table = data.map(row => {
                let arr = Array.from(Array(mx), () => 0);
                arr[row[1] - 1] = 1;
                arr[row[2] - 1] = -2;
                arr[row[3] - 1] = 1;
                arr[0] = row[0];
                return arr;
            });
    
            const y = table.map(row => row[0]);
            const x = table.map(row => row.slice(1));
            return [0, ...glm(y, x)];
        }
    }
})();