using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        public static (NDArray, NDArray) mgrid(NDArray lhs, NDArray rhs)
        {
            if (!(lhs.ndim == 1 || rhs.ndim == 1))
                throw new IncorrectShapeException();

            IArraySlice nd1Data = lhs.Storage.GetData();
            IArraySlice nd2Data = rhs.Storage.GetData();

            int[] resultDims = new int[] { lhs.Storage.Shape.Dimensions[0], rhs.Storage.Shape.Dimensions[0] };

            NDArray res1 = new NDArray(lhs.dtype, resultDims);
            NDArray res2 = new NDArray(lhs.dtype, resultDims);

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
