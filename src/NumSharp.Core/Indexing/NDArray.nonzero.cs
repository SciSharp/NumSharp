using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return the indices of the elements that are non-zero. Refer to <see cref="np.nonzero(NDArray)"/>
        ///     for full documentation.
        /// </summary>
        /// <returns>One index array per dimension, together selecting the non-zero elements in C (row-major) order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nonzero.html</remarks>
        public NDArray<long>[] nonzero()
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return np.nonzero(this);
        }
    }
}
