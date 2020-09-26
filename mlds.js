(function(exports) {
    const links = {
        LOGIT: 'logit',
        PROBIT: 'probit',
        CAUCHY: 'cauchy',
        LOG: 'log',
        CLOGLOG: 'cloglog',
    };

    const logit = () => {
        const link = a => {
            a = link.clean(a);
            return map(a, e => log(e / (1 - e)));
        }
        link.clean = a => clip(a, Number.EPSILON, 1 - Number.EPSILON);
        link.inverse = a => map(a, e => 1 / (1 + e));
        link.derivative = a => {
            a = link.clean(a);
            return map(a, e => 1 / (e * (1 - e)))
        }
        link.inverseDerivative = a => {
            a = exp(a);
            return map(a, e => e / pow(1 + e, 2));
        }
        return link;
    }

    const probit = () => {

    }

    const cauchy = () => {

    }

    const logLink = () => {

    }

    const cloglog = () => {

    }

    const linkBuilder = function(link) {
        switch (link) {
            case links.LOGIT:
                return logit();
            case links.PROBIT:
                return probit();
            case links.CAUCHY:
                return cauchy();
            case links.LOG:
                return logLink();
            case links.CLOGLOG:
                return cloglog();
        }
    }

    const map = (a, f) => a.map(e => f(e));

    const clip = (a, aMin, aMax) => a.map(e => a < aMin ? aMin : e > aMax ? aMax : e);

    const log = x => Math.log(x);

    const exp = x => Math.exp(x);

    const pow = (x, y) => Math.pow(x, y);

    exports.test = function() {
        const logit = linkBuilder('logit');
        console.log(logit([5]));
    }
})(this);