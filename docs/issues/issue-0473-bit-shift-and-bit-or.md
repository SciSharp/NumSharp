# #473: Bit shift and bit or

- **URL:** https://github.com/SciSharp/NumSharp/issues/473
- **State:** OPEN
- **Author:** @MichielMans
- **Created:** 2021-12-28T13:21:27Z
- **Updated:** 2021-12-28T13:21:27Z

## Description

Is it possible to bit-shift and bit-or all values in a NDArray with NumSharp, whiteout a for loop? With Numpy (python) is works as follows:

r_shift = r<<8
g_shift = g<<0
rg_concat = r_shift|g_shift

