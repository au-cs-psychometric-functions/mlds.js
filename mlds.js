(function(exports) {
    const GLM = function(y, x) {
        console.log(x);
        console.log(y);
    }

    exports.mlds = function(data) {
        data = data.map(row1 => {
            if (row[1] > row[3]) {
                [row[1], row[3]] = row[3], row[1];
                row[0] = 1 - row[0];
            }
            return row;
        });

        mx = Math.max.apply(null, data.map(row => Math.max.apply(Math, row)));
        data = data.map(row => {
            const arr = Array.from({length: mx}, () => 0);
            arr[row[1] - 1] = 1;
            arr[row[2] - 1] = -2;
            arr[row[3] - 1] = 1;
            arr[0] = row[0];
            return arr;
        });

        y = data.map(row => (row[0]));
        x = data.map(row => {
            [, ...arr] = row;
            return arr;
        });

        return GLM(y, x);
    }
})(this);