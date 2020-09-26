(function(exports) {
    const clip = (a, aMin, aMax) => a.map(e => a < aMin ? aMin : e > aMax ? aMax : e);

    const defaultClip = a => clip(a, Number.EPSILON, 1 - Number.EPSILON);

    const links = {
        LOGIT: 'logit',
        PROBIT: 'probit',
        CAUCHY: 'cauchy',
        LOG: 'log',
        CLOGLOG: 'cloglog',
    };

    const logit = () => {
        const link = a => link.clean(a).map(e => Math.log(e / (1 - e)));
        link.clean = a => defaultClip(a);
        link.inverse = a => a.map(e => 1 / (1 + e));
        link.derivative = a => link.clean(a).map(e => 1 / (e * (1 - e)));
        link.inverseDerivative = a => a.map(e => Math.exp(e)).map(e => e / Math.pow(1 + e, 2));
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
        const link = a => link.clean(a).map(e => Math.log(-1 * Math.log(1 - e)));
        link.clean = a => defaultClip(a);
        link.inverse = a => a.map(e => 1 - Math.exp(-1 * Math.exp(e)));
        link.derivative = a => link.clean(a).map(e => 1 / ((e - 1) * (Math.log(1 - e))));
        link.inverseDerivative = a => a.map(e => Math.exp(e - Math.exp(e)));
        return link;
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

    exports.test = function() {
        const logit = linkBuilder('logit');
        console.log(logit([5]));
    }
})(this);