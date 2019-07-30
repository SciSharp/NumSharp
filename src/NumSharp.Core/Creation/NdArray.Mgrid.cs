using System;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NDArray
    {
        public (NDArray, NDArray) mgrid(NDArray nd2)
        {
            if (!(this.ndim == 1 || nd2.ndim == 1))
                throw new IncorrectShapeException();

            IArraySlice nd1Data = this.Storage.GetData();
            IArraySlice nd2Data = nd2.Storage.GetData();

            int[] resultDims = new int[] { this.Storage.Shape.Dimensions[0], nd2.Storage.Shape.Dimensions[0] };

            NDArray res1 = new NDArray(this.dtype, resultDims);
            NDArray res2 = new NDArray(this.dtype, resultDims);

            IArraySlice res1Arr = res1.Storage.GetData();
            IArraySlice res2Arr = res2.Storage.GetData();

            int counter = 0;

            for (int row = 0; row < nd1Data.Count; row++)
            {
                for (int col = 0; col < nd2Data.Count; col++)
                {
                    res1Arr[counter] = nd1Data[row];
                    res2Arr[counter] = nd2Data[col];
                    counter++;
                }
            }


            return (res1, res2);
        }
    }
}
