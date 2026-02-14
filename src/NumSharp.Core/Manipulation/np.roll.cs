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
        public static NDArray roll(NDArray a, int shift, int? axis = null)
        {
            if (axis == null)
                return roll(a.ravel(), shift, 0).reshape(a.shape);

            int ax = axis.Value;
            if (ax < 0) ax += a.ndim;
            if (ax < 0 || ax >= a.ndim)
                throw new ArgumentException(
                    $"axis {axis.Value} is out of bounds for array of dimension {a.ndim}");

            int n = a.shape[ax];

            // Python-style modulo: always non-negative when n > 0.
            // If n == 0 (empty axis), offset stays 0 — nothing to roll.
            int offset = n == 0 ? 0 : ((shift % n) + n) % n;

            if (offset == 0)
                return a.copy();

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
                    srcBody[i] = new Slice(null, -offset);  // :-offset
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
    }
}
