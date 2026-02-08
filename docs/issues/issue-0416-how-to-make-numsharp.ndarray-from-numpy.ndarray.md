# #416: how to make NumSharp.NDArray from Numpy.NDarray?

- **URL:** https://github.com/SciSharp/NumSharp/issues/416
- **State:** OPEN
- **Author:** @djagatiya
- **Created:** 2020-07-06T13:45:49Z
- **Updated:** 2020-07-07T05:43:56Z

## Description

_No description provided._

## Comments

### Comment 1 by @djagatiya (2020-07-07T05:43:56Z)

I found a solution, but is this the right way to do it?

```
Numpy.NDarray numpyArray = Numpy.np.random.randn(224,224,3);
byte[] v = numpyArray.GetData<byte>();
fixed (byte* packet = v)
{
    var block = new UnmanagedMemoryBlock<byte>(packet, v.Length);
    var storage = new UnmanagedStorage(new ArraySlice<byte>(block), numpyArray.shape.Dimensions);
    NumSharp.NDArray numSharpArray = new NumSharp.NDArray(storage);
}
```
