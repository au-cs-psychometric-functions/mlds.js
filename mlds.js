(function(exports) {
    const links = {
        LOGIT: 'logit',
        PROBIT: 'probit',
        CAUCHY: 'cauchy',
        LOG: 'log',
        CLOGLOG: 'cloglog',
    };

    const logit = () => {

    }

    const probit = () => {

    }

    const cauchy = () => {

    }

    const log = () => {

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

    exports.test = function() {
        console.log('Hello World');
    }
})(this);