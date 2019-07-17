using System;
using NumSharp;

namespace NumSharp
{
    public partial class NDArray
    {
        public (NDArray,NDArray) mgrid(NDArray nd2)
        {
            if( !(this.ndim == 1 || nd2.ndim == 1))
                throw new IncorrectShapeException("mgrid is implemented only for two single dimension arrays");

            Array nd1Data = this.Storage.GetData();
            Array nd2Data = nd2.Storage.GetData();

            int[] resultDims = new int[]{this.Storage.Shape.Dimensions[0],nd2.Storage.Shape.Dimensions[0]};

            NDArray res1 = new NDArray(this.dtype,resultDims);
            NDArray res2 = new NDArray(this.dtype,resultDims);

            Array res1Arr = res1.Storage.GetData();
            Array res2Arr = res2.Storage.GetData();

            int counter = 0;
            for (int row = 0; row < nd1Data.Length; row++)
            {
                for (int col = 0; col < nd2Data.Length; col++)
                {
                    res1Arr.SetValue(nd1Data.GetValue(row), counter);
                    res2Arr.SetValue(nd2Data.GetValue(col),counter);
                    counter++;
                }
            }

            return (res1,res2);
        }
    }
}
