using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the indices of array elements that are non-zero, grouped by element.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>
        ///     Indices of elements that are non-zero, grouped per element. The result is a
        ///     2-D array of shape <c>(N, a.ndim)</c> where <c>N</c> is the number of non-zero
        ///     elements. Each row contains the coordinates of one non-zero element.
        ///     Result dtype is <c>int64</c>. For 0-d input, returns shape <c>(1, 0)</c> when
        ///     the value is truthy and <c>(0, 0)</c> otherwise. For empty input of shape
        ///     <c>(0, d1, ..., dn)</c>, returns shape <c>(0, ndim)</c>.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
        ///     Equivalent to <c>np.transpose(np.nonzero(a))</c> with a special case for 0-d input.
        ///     Distinct from <see cref="nonzero"/> which returns a tuple of column arrays.
        /// </remarks>
        public static NDArray argwhere(NDArray a)
            => a.TensorEngine.Argwhere(a);
    }
}
