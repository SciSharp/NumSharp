# #427: Performance on np.matmul

- **URL:** https://github.com/SciSharp/NumSharp/issues/427
- **State:** OPEN
- **Author:** @Banyc
- **Created:** 2020-10-31T12:26:03Z
- **Updated:** 2020-10-31T14:10:34Z

## Description

The shape of `x` is [200, 1000], of `w` is [1000, 500], and of `b` is [500]

`b` is filled with zeros, `x` and `w` are random float64/double.

Example code of NumSharp:

```csharp
var out = np.matmul(x, w) + b;
```

... takes 3-4 seconds.

Example code of numpy:

```python
out = x @ w + b
```

... finishes immediately.


## Comments

### Comment 1 by @Oceania2018 (2020-10-31T13:53:38Z)

@Banyc Can you test it in `TensorFlow.NET` eager mode?

### Comment 2 by @Banyc (2020-10-31T14:07:13Z)

I use only this package to implement neural network layers from the stretch, without using other packages like `Tensorflow.NET`.

### Comment 3 by @Oceania2018 (2020-10-31T14:10:34Z)

It will have performance issue. Should use other more mature package.
