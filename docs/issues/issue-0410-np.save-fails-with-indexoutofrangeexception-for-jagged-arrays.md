# #410: np.save fails with IndexOutOfRangeException for jagged arrays

- **URL:** https://github.com/SciSharp/NumSharp/issues/410
- **State:** OPEN
- **Author:** @Jmerk523
- **Created:** 2020-05-08T01:00:52Z
- **Updated:** 2020-05-08T01:00:52Z

## Description

Hi,
I am seeing this exception when trying to save a jagged array to a stream:

Index was outside the bounds of the array.
   at NumSharp.np.<Enumerate>d__109`1.MoveNext()
   at NumSharp.np.writeValueJagged(BinaryWriter reader, Array matrix, Int32 bytes, Int32[] shape)
   at NumSharp.np.Save(Array array, Stream stream)

I haven't pinpointed the root cause specifically, but it seems that there is some issue with jagged arrays when the outer array has rank 1:

```
        static IEnumerable<T> Enumerate<T>(Array a, int[] dimensions, int pos)
        {
            if (pos == dimensions.Length - 1)
            {
                for (int i = 0; i < dimensions[pos]; i++)
                    yield return (T)a.GetValue(i);
            }
            else
            {
                for (int i = 0; i < dimensions[pos]; i++)
                    foreach (var subArray in Enumerate<T>(a.GetValue(i) as Array, dimensions, pos + 1))
                        yield return subArray;
            }
        }
```
When pos == 0 and dimensions[] has length 0, the else will execute and pos(0) will be out of bounds.
dimensions[] has length 0 from the caller, which passes:
`            int[] first = shape.Take(shape.Length - 1).ToArray();`
so first[] will be an empty array when shape has length 1 (as in, the single dimension of the 1 dimensional outer jagged array).
