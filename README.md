# mlds.js

Implementation of [Maximum Likelihood Difference Scaling](https://cran.r-project.org/web/packages/MLDS/vignettes/MLDS.pdf) in python and javascript.

## Examples

Python

```python
from mlds import mlds
print(mlds('data.txt'))
```

Javascript

```javascript
const fs = require('fs');
const data = [];
fs.readFileSync('./data.txt', 'utf-8').split("\n").forEach((row) => {
    data.push(row.split("\t").map(n => +n));
});
console.log(mlds.mlds(data));
```