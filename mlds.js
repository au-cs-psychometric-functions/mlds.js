(function(exports) {
    const default_clip = p => p.map(e => Math.max(Number.EPSILON, Math.min(1 - Number.EPSILON, e)));
    const inf_clip = p => p.map(e => Math.max(Number.EPSILON, Math.min(Number.POSITIVE_INFINITY, e)))
})();