# #448: Debug.Assert(...) causes tests to stop the entire process

- **URL:** https://github.com/SciSharp/NumSharp/issues/448
- **State:** OPEN
- **Author:** @Nucs
- **Created:** 2021-04-23T12:03:33Z
- **Updated:** 2023-02-27T19:47:40Z
- **Labels:** bug
- **Assignees:** @Nucs

## Description

_No description provided._

## Comments

### Comment 1 by @bojake (2023-02-27T19:47:40Z)

The problem with this test is that the values array is not being broadcast/broadened properly. The assert expects the values array to match the size of the indices array. In this test case, though, the selector ("np < 3") is choosing 2 elements out of the 6 and the result is applying "-2" (which is another bug, btw). Changing the result value to "-2.0" gets past another bug but then you see where the actual bug in the SetIndiceND method resides. Before SetIndicesND can be called the "values" must be made to match the expected buffer size through implicit broadening.

If you give "-2" as the value then the framework thinks you are setting the "size" of the NDArray for values instead of setting an actual value array of INTs. Oops.
