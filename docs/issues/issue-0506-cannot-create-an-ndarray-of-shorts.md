# #506: Cannot create an NDArray of shorts

- **URL:** https://github.com/SciSharp/NumSharp/issues/506
- **State:** OPEN
- **Author:** @NickBotelho
- **Created:** 2024-01-24T14:19:58Z
- **Updated:** 2024-01-24T14:19:58Z

## Description

When I try to initialize an NDArray with the short type, it throws a System.NotSupportedException. It seems like it only does this with shorts, every other type works fine. Heres a line of code to run to reproduce

`var nd = new NDArray(dtype: np.int16, shape: new Shape(new []{1, 2}));`
