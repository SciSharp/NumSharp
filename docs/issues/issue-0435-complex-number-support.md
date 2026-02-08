# #435: Complex number support?

- **URL:** https://github.com/SciSharp/NumSharp/issues/435
- **State:** OPEN
- **Author:** @cgranade
- **Created:** 2021-01-07T22:43:31Z
- **Updated:** 2023-08-19T14:29:33Z

## Description

When attempting to create a new `NDArray` of complex numbers, I get an exception in the `Allocate` method at https://github.com/SciSharp/NumSharp/blob/00d8700b00e815f321238536e0d6b4dbc9af8d6a/src/NumSharp.Core/Backends/Unmanaged/ArraySlice.cs#L387:

![image](https://user-images.githubusercontent.com/31516/103953204-89435400-50f6-11eb-9ec7-13e522da1445.png)

Are complex numbers supported as the dtype of `NDArray` objects, and if so, how do I allocate them? Thanks for the help, and for the awesome project!

## Comments

### Comment 1 by @dcuccia (2021-06-29T00:48:34Z)

+1. Just got to the end of a port, and realized Complex is not supported. Are there plans for this?

### Comment 2 by @LetGo (2022-03-10T06:33:07Z)

+1. Just got to the end of a port, and realized Complex is not supported. Are there plans for this?

### Comment 3 by @gsgou (2023-08-19T14:26:53Z)

Any way to workaround this one?
UnmanagedStorage also doesnt support Complex.

```
System.NotSupportedException: Specified method is not supported.
  at NumSharp.NPTypeCodeExtensions.AsType (NumSharp.NPTypeCode typeCode) [0x00097] in D:\SciSharp\NumSharp\src\NumSharp.Core\Backends\NPTypeCode.cs:144 
  at NumSharp.Backends.UnmanagedStorage..ctor (NumSharp.NPTypeCode typeCode) [0x00014] in D:\SciSharp\NumSharp\src\NumSharp.Core\Backends\Unmanaged\UnmanagedStorage.cs:181 
  at NumSharp.Backends.DefaultEngine.GetStorage (NumSharp.NPTypeCode typeCode) [0x00000] in D:\SciSharp\NumSharp\src\NumSharp.Core\Backends\Default\Allocation\Default.Allocation.cs:14 
  at NumSharp.NDArray..ctor (NumSharp.NPTypeCode typeCode, NumSharp.TensorEngine engine) [0x0000d] in D:\SciSharp\NumSharp\src\NumSharp.Core\Backends\NDArray.cs:102 
  at NumSharp.NDArray..ctor (NumSharp.NPTypeCode typeCode) [0x00000] in D:\SciSharp\NumSharp\src\NumSharp.Core\Backends\NDArray.cs:119 
  at NumSharp.NDArray..ctor (NumSharp.NPTypeCode dtype, NumSharp.Shape shape, System.Boolean fillZeros) [0x00000] in D:\SciSharp\NumSharp\src\NumSharp.Core\Backends\NDArray.cs:234 
  at NumSharp.np.zeros (NumSharp.Shape shape, NumSharp.NPTypeCode typeCode) [0x0000e] in D:\SciSharp\NumSharp\src\NumSharp.Core\Creation\np.zeros.cs:54 
```


