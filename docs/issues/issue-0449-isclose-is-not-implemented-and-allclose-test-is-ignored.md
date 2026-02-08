# #449: IsClose is not implemented and allclose test is ignored

- **URL:** https://github.com/SciSharp/NumSharp/issues/449
- **State:** OPEN
- **Author:** @koliyo
- **Created:** 2021-04-28T17:23:19Z
- **Updated:** 2021-04-29T09:56:42Z
- **Labels:** missing feature/s

## Description

These should probably be removed from the API if they are not properly implemented?

```cs
        [Ignore("TODO: fix this test")]
        [TestMethod]
        public void np_allclose_1D()
```

```cs
        public override NDArray<bool> IsClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false)
        {
            // ... lots of commeted out code
           return null;
        }
```
