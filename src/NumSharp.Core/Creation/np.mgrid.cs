using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {

        //TODO! implement mgrid that takes (string lhs, string rhs) and (Slice lhs, Slice rhs) fallbacking to each other.

        /// <summary>
        ///     nd_grid instance which returns a dense multi-dimensional “meshgrid”.
        ///     An instance of numpy.lib.index_tricks.nd_grid which returns an dense (or fleshed out) mesh-grid when indexed, so that each returned argument has the same shape.
        ///     The dimensions and number of the output arrays are equal to the number of indexing dimensions.If the step length is not a complex number, then the stop is not inclusive.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>mesh-grid `ndarrays` all of the same dimensions</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.mgrid.html</remarks>
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
