# #468: np_array.convolve returning Null

- **URL:** https://github.com/SciSharp/NumSharp/issues/468
- **State:** OPEN
- **Author:** @dklein9500
- **Created:** 2021-12-06T10:57:13Z
- **Updated:** 2022-06-26T22:42:49Z

## Description

Hi,
I am using this function to smooth some measurement data, sadly it returns null for some reason.
Here is the example code that produced the same result:

```
int filter_length = 5;
List<double> data = new List<double>() {1, 34, 3, 4, 22, 4, 24, 42, 24, 22, 4 };  // Just some random numbers for testing
var array = data.ToArray();
NDArray np_array = new NDArray(array);
var filter = np.ones(filterLength);
NDArray filtered_array = np_array.convolve(filter, "same");   // Here null is returned
```
There is probably a better way to construct the NDArray, but I think the code should still work. 
What am I doing wrong? 
Thanks in advance! 

## Comments

### Comment 1 by @abbefus (2022-04-28T18:02:26Z)

This is because someone put "return null" a few lines down in the code:

```
public NDArray convolve(NDArray rhs, string mode = "full")
{
          var lhs = this;
          int nf = lhs.shape[0];
          int ng = rhs.shape[0];

          if (ndim > 1 || rhs.ndim > 1)
              throw new IncorrectShapeException();
          var retType = np._FindCommonType(lhs, rhs);
          return null;
```

That's enough to guarantee you get null every time. Who knows if the rest of the code ever worked.


### Comment 2 by @guillermoe7 (2022-06-24T18:01:36Z)

Also facing this problem. 
Have any idea if this function was in working condition before?

### Comment 3 by @abbefus (2022-06-24T19:14:29Z)

This function works when it is rewritten, leading me to believe it did work before. Here is the function as I rewrote it -- verified against the python version. Sorry for my comments.

```
public static class NumpyExtensions
{

    // NOTE: lhs must always be bigger than rhs --
    public static NDArray LinearConvolution(this NDArray lhs, NDArray rhs, ConvolveModes mode = ConvolveModes.Full)
    {
        if (lhs.ndim > 1 || rhs.ndim > 1)
            throw new IncorrectShapeException("Both arrays must be 1-dimensional");

        if (lhs.Shape.Size < rhs.Shape.Size)
            throw new IncorrectShapeException("Right-hand side array must be smaller than left-hand side.");


        // NOTE:
        // NDArray.GetData just runs NDArray.Storage.GetData
        // which returns NDArray.Storage.InternalArray == NDArray.Array
        // so all three methods are practically interchangeable so why they had to make things complicated is beyond me

        ArraySlice<double> lhsarr = lhs.GetData<double>();
        ArraySlice<double> rhsarr = rhs.GetData<double>();

        int nf = lhs.shape[0];
        int ng = rhs.shape[0];


        switch (mode)
        {
            case ConvolveModes.Full:
                {
                    int n = nf + ng - 1;

                    NDArray<double> ret = new NDArray<double>(Shape.Vector(n), true);
                    ArraySlice<double> outArray = ret.GetData<double>();

                    for (int idx = 0; idx < n; ++idx)
                    {
                        int jmn = (idx >= ng - 1) ? (idx - (ng - 1)) : 0;
                        int jmx = (idx < nf - 1) ? idx : nf - 1;

                        for (int jdx = jmn; jdx <= jmx; ++jdx)
                        {
                            outArray[idx] += lhsarr[jdx] * rhsarr[idx - jdx];
                        }
                    }

                    return ret;
                }

            case ConvolveModes.Valid:
                {
                    var min_v = (nf < ng) ? lhsarr : rhsarr;
                    var max_v = (nf < ng) ? rhsarr : lhsarr;

                    int n = Math.Max(nf, ng) - Math.Min(nf, ng) + 1;

                    var ret = new NDArray(typeof(double), Shape.Vector(n), true);
                    ArraySlice<double> outArray = ret.GetData<double>();

                    for (int idx = 0; idx < n; ++idx)
                    {
                        int kdx = idx;

                        for (int jdx = (min_v.Count - 1); jdx >= 0; --jdx)
                        {
                            outArray[idx] += min_v[jdx] * max_v[kdx];
                            ++kdx;
                        }
                    }

                    return ret;
                }

            case ConvolveModes.Same:
                {
                    // https://stackoverflow.com/questions/38194270/matlab-convolution-same-to-numpy-convolve
                    var npad = rhs.shape[0] - 1;

                    if (npad % 2 == 1)
                    {
                        unsafe
                        {
                            npad = (int)Math.Floor(((double)npad) / 2.0);

                            ArraySlice<double> arr = ArraySlice<double>.Allocate(npad + lhsarr.Count);
                            Span<double> span = new Span<double>(arr.VoidAddress, arr.Count);
                            lhsarr.CopyTo(span, npad);
                            var retnd = new NDArray(new UnmanagedStorage(arr, Shape.Vector(lhsarr.Count)));
                            return retnd.LinearConvolution(rhs, ConvolveModes.Valid);
                        }
                    }
                    else
                    {
                        throw new NotImplementedException("Cannot implement because NDArray.Address is protected.");
                        // I suppose we could extend NDArray and create a getter for Address
                        //{
                        //    unsafe
                        //    {
                        //        npad = npad / 2;

                        //        NPTypeCode retType = NPTypeCode.Double;
                        //        NDArray puffer = new NDArray(retType, Shape.Vector(npad + lhsarr.Count), true);
                        //        ArraySlice<double> puffslice = puffer.Data<double>(); // not sure this is equal to storage
                        //        Span<double> span = new Span<double>(puffslice.VoidAddress, puffslice.Count);
                        //        //lhsarr.CopyTo(puffer.Storage.AsSpan <#202>(), npad);
                        //        lhsarr.CopyTo(span, npad);
                        //        NDArray np1New = puffer;

                        //        puffer = new NDArray(retType, Shape.Vector(npad + np1New.size), true);
                        //        int cpylen = np1New.size * sizeof(double);
                        //        Buffer.MemoryCopy(np1New.Address, (double)puffer.Address) + npad, cpylen, cpylen);
                        //        return puffer.convolve(rhs, "valid");
                        //    }
                        //}
                    }
                }
            default:
                return lhs.LinearConvolution(rhs);
        }
    }
}
```

```
public enum ConvolveModes
{
    Full,
    Same,
    Valid
}
```

### Comment 4 by @guillermoe7 (2022-06-26T22:42:49Z)

Hello abbefus,

Thanks a lot. That's quite a good code example.
Just one more bit of help. Seems my files are not up to date as there are some resources not available in my SliceAndDice install:
ArraySlice.Allocate()
ArraySlice.CopyTo()
Shape.Vector()

Can you please show me where to get the proper version where these methods may be found?

Thank you very much.
