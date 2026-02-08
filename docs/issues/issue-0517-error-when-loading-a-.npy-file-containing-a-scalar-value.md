# #517: Error when loading a `.npy` file containing a scalar value

- **URL:** https://github.com/SciSharp/NumSharp/issues/517
- **State:** OPEN
- **Author:** @thalesfm
- **Created:** 2024-09-24T17:56:29Z
- **Updated:** 2024-09-24T17:56:29Z

## Description

Due to an off-by-one error, `np.load` fails to parse the file's header when `shape = ()` and triggers an `ArgumentOutOfRangeException`
(Also, the function doesn't consider this possibility when calculating the size of the underlying buffer).
