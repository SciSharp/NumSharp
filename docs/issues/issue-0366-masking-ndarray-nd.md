# #366: Masking (ndarray[nd])

- **URL:** https://github.com/SciSharp/NumSharp/issues/366
- **State:** OPEN
- **Author:** @henon
- **Created:** 2019-10-12T19:49:33Z
- **Updated:** 2019-10-12T19:51:59Z

## Description

Doesn't work in ndarray[nd]?

![image](https://user-images.githubusercontent.com/1705364/66706800-4335ef80-ecfd-11e9-9a4d-4631255241d8.png)

```shell
System.NotSupportedException: Specified method is not supported.
   at NumSharp.NDArray.set_Item(Object[] indices_or_slices, NDArray value) in D:\SciSharp\NumSharp\src\NumSharp.Core\Selection\NDArray.Indexing.cs:line 202
```

_Originally posted by @Oceania2018 in https://github.com/SciSharp/NumSharp/issues/359#issuecomment-541355337_

## Comments

### Comment 1 by @henon (2019-10-12T19:51:59Z)

we need this for it: #365
