# #363: Add `NDIterator<T>` overload with support for specific axis.

- **URL:** https://github.com/SciSharp/NumSharp/issues/363
- **State:** OPEN
- **Author:** @Nucs
- **Created:** 2019-10-12T11:11:32Z
- **Updated:** 2019-10-12T11:11:32Z
- **Labels:** missing feature/s

## Description

NDIterator is useful, we should add an overload that handles specific axis iterator:
usage:
```C#
new NDIterator<T>(ndarray, axis: 1);
```

Also add an overload to extension `ndarray.AsIterator<T>(axis: 1);`
