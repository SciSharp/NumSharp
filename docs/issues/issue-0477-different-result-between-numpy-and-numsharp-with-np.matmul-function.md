# #477: Different Result between NumPy and NumSharp with np.matmul Function

- **URL:** https://github.com/SciSharp/NumSharp/issues/477
- **State:** OPEN
- **Author:** @Koyamin
- **Created:** 2022-04-05T15:29:06Z
- **Updated:** 2022-09-09T18:42:50Z

## Description

I am a new learner of NumSharp and now I want to calculate the matmul product of two NDArrays by using `np.matmul` function:
```Csharp
using NumSharp;

var a = np.arange(2 * 2 * 3).reshape((2, 2, 3));
var b = np.array(new double[] { 1, 2, 3 });
var res = np.matmul(a, b);
```
The value of `res` is as follow:
```
[[6],  [24]]
```
However I have tried the same code in Python:
```Python
import numpy as np

a = np.arange(2*2*3).reshape((2,2,3))
b = np.array([1,2,3])
res = np.matmul(a, b)
```
Now the value of `res` is
```
[[ 8 26]
 [44 62]]
```
I have no idea about it. I want to get the result in Python, what should I do?

## Comments

### Comment 1 by @ChengYen-Tang (2022-09-09T18:42:50Z)

這個專案好像已經沒有在維護了，我有發現一個更完善的專案
https://github.com/Quansight-Labs/numpy.net
