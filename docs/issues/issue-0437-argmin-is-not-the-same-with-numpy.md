# #437: argmin is not the same with numpy

- **URL:** https://github.com/SciSharp/NumSharp/issues/437
- **State:** OPEN
- **Author:** @tomachristian
- **Created:** 2021-01-30T09:08:48Z
- **Updated:** 2021-01-30T09:08:48Z

## Description

https://github.com/SciSharp/NumSharp/blob/00d8700b00e815f321238536e0d6b4dbc9af8d6a/src/NumSharp.Core/Statistics/NDArray.argmin.cs#L20

this does not look right because it returns int, also np.argmin does not seem to behave correctly (like its numpy counterpart)
