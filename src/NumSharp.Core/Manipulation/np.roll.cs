using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Roll array elements along a given axis.
        ///
        /// Elements that roll beyond the last position are re-introduced at the first.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="shift">The number of places by which elements are shifted.</param>
        /// <param name="axis">Axis along which elements are shifted. By default, the array
        /// is flattened before shifting, after which the original shape is restored.</param>
        /// <returns>Output array, with the same shape as <paramref name="a"/>.</returns>
        /// <remarks>
        /// Matches NumPy's algorithm: empty_like + slice-copy pairs.
        /// https://numpy.org/doc/stable/reference/generated/numpy.roll.html
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>'s rank.</exception>
        [NDScoped]
        public static NDArray roll(NDArray a, long shift, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, shift, axis), static s => roll(s.a, s.shift, s.axis));
            if (axis == null)
                return roll(a.ravel(), shift, 0).reshape(Shape.Adopt(a.Shape.dimensions));

            // normalize_axis_tuple(axis, a.ndim, allow_duplicate=True): NumPy's AxisError, the axis as given.
            int ax = Backends.DefaultEngine.check_and_adjust_axis(a.ndim, axis.Value);

            long n = a.Shape.dimensions[ax];

            // Python-style modulo: always non-negative when n > 0.
            // If n == 0 (empty axis), offset stays 0 — nothing to roll.
            long offset = n == 0 ? 0 : ((shift % n) + n) % n;

            // NumPy still runs empty_like + the slice copies for a zero shift, so the result keeps a's memory order
            // (order='K': a transposed / broadcast input stays transposed / F), which copy('K') reproduces.
            if (offset == 0)
                return a.copy('K');

            var result = np.empty_like(a);

            // Build Slice arrays for each dimension.
            // For the rolled axis we split into body + tail:
            //   result[..., offset:, ...] = a[..., :-offset, ...]   (body shifts)
            //   result[..., :offset, ...] = a[..., -offset:, ...]   (tail wraps)
            // All other axes use Slice.All (":").

            var srcBody = new Slice[a.ndim];
            var dstBody = new Slice[a.ndim];
            var srcTail = new Slice[a.ndim];
            var dstTail = new Slice[a.ndim];

            for (int i = 0; i < a.ndim; i++)
            {
                if (i == ax)
                {
                    srcBody[i] = new Slice(null, -offset);   // :-offset
                    dstBody[i] = new Slice(offset, null);    // offset:
                    srcTail[i] = new Slice(-offset, null);   // -offset:
                    dstTail[i] = new Slice(null, offset);    // :offset
                }
                else
                {
                    srcBody[i] = dstBody[i] = srcTail[i] = dstTail[i] = Slice.All;
                }
            }

            result[dstBody] = a[srcBody];
            result[dstTail] = a[srcTail];

            return result;
        }

        /// <inheritdoc cref="roll(NDArray, long, int?)"/>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray roll(NDArray a, int shift, int? axis = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((a, shift, axis), static s => roll(s.a, s.shift, s.axis))
            : roll(a, (long)shift, axis);
    }
}
