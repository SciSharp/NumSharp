# #351: Proper way to iterate using IEnumerable<T>

- **URL:** https://github.com/SciSharp/NumSharp/issues/351
- **State:** OPEN
- **Author:** @Nucs
- **Created:** 2019-09-28T15:42:47Z
- **Updated:** 2019-09-28T15:42:47Z
- **Labels:** enhancement
- **Assignees:** @Nucs

## Description

There should be an approachable way to perform fast `foreach` on a `NDArray`.
Currently `NDArray` implements non-generic `IEnumerable` which returns a boxed value that can be either the `NDArray.dtype` or an `NDArray`.
Boxing causes `O(n)` operations to be significantly slower on large datasets.
