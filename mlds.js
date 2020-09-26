(function(exports) {
    const links = {
        LOGIT: 'logit',
        PROBIT: 'probit',
        CAUCHY: 'cauchy',
        LOG: 'log',
        CLOGLOG: 'cloglog',
    };

    const logit = () => {
        const link = a => link.clean(a).map(e => Math.log(e / (1 - e)));
        link.clean = a => clip(a, Number.EPSILON, 1 - Number.EPSILON);
        link.inverse = a => a.map(e => 1 / (1 + e));
        link.derivative = a => link.clean(a).map(e => 1 / (e * (1 - e)));
        link.inverseDerivative = a=> a.map(e => Math.exp(e)).map(e => e / Math.pow(1 + e, 2));
        return link;
    }

    const probit = () => {
    }

    const cauchy = () => {
    }

    const log = () => {
        const link = a => link.clean(a).map(e => Math.log(e));
        link.clean = a => clip(a, Number.EPSILON, Number.POSITIVE_INFINITY);
        link.inverse = a => a.map(e => Math.exp(e));
        link.derivative = a => link.clean(a).map(e => 1 / e);
        link.inverseDerivative = a => a.map(e => Math.exp(e));
        return link;
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
                return log();
            case links.CLOGLOG:
                return cloglog();
        }
    }

    const clip = (a, aMin, aMax) => a.map(e => a < aMin ? aMin : e > aMax ? aMax : e);

    exports.test = function() {
        const logit = linkBuilder('logit');
        console.log(logit([5]));
    }
})(this);