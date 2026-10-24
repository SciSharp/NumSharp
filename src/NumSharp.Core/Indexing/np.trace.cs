using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the sum along diagonals of the array. For a 2-D array,
        ///     <c>trace(a) == sum(a.diagonal())</c>. For an N-D array, traces
        ///     the diagonals along the 2-D sub-arrays defined by
        ///     <paramref name="axis1"/> / <paramref name="axis2"/> and reduces
        ///     them, leaving an array with those two axes removed.
        /// </summary>
        /// <param name="a">Source array. Must have at least 2 dimensions.</param>
        /// <param name="offset">
        ///     Offset of the diagonal from the main diagonal.
        ///     See <see cref="diagonal"/> for details.
        /// </param>
        /// <param name="axis1">First axis of the 2-D sub-array. Default 0.</param>
        /// <param name="axis2">Second axis of the 2-D sub-array. Default 1.</param>
        /// <param name="dtype">
        ///     Output dtype. <c>null</c> (default) preserves <c>a.dtype</c>,
        ///     except narrow integer dtypes widen per NumPy's add.reduce rule:
        ///     signed ints and bool promote to <see cref="long"/>, UNSIGNED ints
        ///     (and Char, the uint16 twin) promote to <see cref="ulong"/> —
        ///     trace(uint8) is uint64, exactly as in NumPy.
        /// </param>
        /// <param name="out">
        ///     Optional output array. Shape must equal the natural reduction
        ///     output; values are copied with unsafe casting and the method
        ///     returns <paramref name="out"/> itself.
        /// </param>
        /// <returns>
        ///     Sum along the diagonal. 2-D input → 0-d scalar. N-D input →
        ///     array with <c>a.shape</c> minus <paramref name="axis1"/> and
        ///     <paramref name="axis2"/>.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.trace.html
        ///     <para>
        ///     Mirrors NumPy's <c>PyArray_Trace</c>: <c>sum(diagonal(a,
        ///     offset, axis1, axis2), axis=-1, dtype=dtype, out=out)</c>. The
        ///     diagonal view is the heavy lifting; the sum is a regular
        ///     reduction that goes through NumSharp's IL reduction kernels.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError"><paramref name="a"/> has fewer than two dimensions, or the two axes are the same
        ///     axis ("axis1 and axis2 cannot be the same").</exception>
        /// <exception cref="AxisError">An axis is out of range: "axis1: axis …" / "axis2: axis …" (axis1 checked
        ///     first).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="TypeError">NumPy's loop resolution refuses text: a text operand with another out class, a
        ///     text out for numbers, a string <paramref name="dtype"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape ("output array does
        ///     not match result of trace").</exception>
        [NDScoped] // the boundary of the plain region and the engine call (the engine reclaims its own temporaries)
        public static NDArray trace(
            NDArray a, int offset = 0, int axis1 = 0, int axis2 = 1,
            DType dtype = null, NDArray @out = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (a is null) throw new ArgumentNullException(nameof(a));

            // NumPy's trace IS PyArray_Trace (the diagonal's add.reduce over its last axis with dtype= and out=), and the
            // whole call (the fused 2-D / 3-D walk, the text routes, the sum, the out= store) is the engine's:
            // DefaultEngine.Trace.
            return a.TensorEngine.Trace(a, offset, axis1, axis2, dtype, @out);
        }
    }
}
