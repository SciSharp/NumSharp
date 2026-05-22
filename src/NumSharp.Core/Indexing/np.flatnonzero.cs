using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return indices that are non-zero in the flattened version of <paramref name="a"/>.
        ///     This is equivalent to <c>np.nonzero(np.ravel(a))[0]</c>.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <returns>
        ///     1-D <see cref="NDArray{T}"/> of <see cref="long"/> (NumPy <c>intp</c>) containing
        ///     the indices of elements of <c>a.ravel()</c> that are non-zero. For 0-d input,
        ///     returns <c>[0]</c> when the value is truthy and an empty array otherwise.
        ///     For empty input, returns an empty 1-D array.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.flatnonzero.html
        ///     <para>
        ///     Faster than the literal composition <c>nonzero(ravel(a))[0]</c>: the engine
        ///     runs the SIMD popcount + bit-scan straight into the 1-D result buffer
        ///     without materializing the per-axis coordinate arrays produced by
        ///     <see cref="nonzero"/>. For multi-dim inputs the cost is the same as a 1-D
        ///     input of equal element count (the layout is collapsed by materializing to
        ///     C-contig when needed).
        ///     </para>
        /// </remarks>
        public static NDArray<long> flatnonzero(NDArray a)
            => a.TensorEngine.FlatNonZero(a);
    }
}
