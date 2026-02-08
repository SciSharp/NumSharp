# #340: Memory Limitations

- **URL:** https://github.com/SciSharp/NumSharp/issues/340
- **State:** OPEN
- **Author:** @Nucs
- **Created:** 2019-08-08T15:39:23Z
- **Updated:** 2019-08-12T13:39:23Z
- **Labels:** enhancement
- **Assignees:** @Nucs

## Description

Allocation currently supports up to 2^32 bytes due to using int and not IntPtr and long.

## Comments

### Comment 1 by @Nucs (2019-08-08T15:52:05Z)

Unit test
```C#
[TestMethod]
public void MyTestMethod()
{
    NDArray x = np.zeros(new Shape(600, 1000, 1000), np.float32);
}
```

### Comment 2 by @Nucs (2019-08-12T13:38:48Z)

Ported UnmanagedMemoryBlock and ArraySlice to use long in commit https://github.com/SciSharp/NumSharp/commit/539683f23af53a8b1e31023f8472880cbcc69517 .
Next is to port Shape to use long and all the algorithms with it. Mainly refactoring job
