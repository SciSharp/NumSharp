# #491: ToBitmap() - datatype mistmatch

- **URL:** https://github.com/SciSharp/NumSharp/issues/491
- **State:** OPEN
- **Author:** @davidvct
- **Created:** 2023-03-08T09:12:45Z
- **Updated:** 2023-03-08T09:12:45Z

## Description

I tried to convert a numsharp array with integers to bitmap.

```
array_rgb_numsharp = array_rgb_numsharp.reshape(1, 300, 300, 3);
Bitmap image = array_rgb_numsharp.ToBitmap();
```
and encounter this error:
System.InvalidCastException: 'Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.'

How can I fix it?

