using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Remove values along a dimension which are zero along all other dimensions.
        ///     Convenience overload for a single <paramref name="axis"/>.
        /// </summary>
        /// <param name="filt">Input array.</param>
        /// <param name="trim">'f' trims from the front, 'b' from the back; "fb" (default) trims both.</param>
        /// <param name="axis">The single dimension to trim.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html</remarks>
        public static NDArray trim_zeros(NDArray filt, string trim, int axis)
            => trim_zeros(filt, trim, new[] {axis});

        /// <summary>
        ///     Remove values along a dimension which are zero along all other dimensions.
        /// </summary>
        /// <param name="filt">Input array.</param>
        /// <param name="trim">
        ///     A string with 'f' representing trim from front and 'b' to trim from back. By default,
        ///     zeros are trimmed on both sides ("fb"). Case-insensitive; "bf" is accepted as an alias of "fb".
        /// </param>
        /// <param name="axis">
        ///     If <c>null</c>, <paramref name="filt"/> is cropped to the smallest bounding box that still
        ///     contains all non-zero values. If axes are specified, <paramref name="filt"/> is sliced in
        ///     those dimensions only, on the sides selected by <paramref name="trim"/>. An empty array of
        ///     axes leaves the input unmodified.
        /// </param>
        /// <returns>
        ///     A view of <paramref name="filt"/> with leading/trailing all-zero hyperplanes removed. The number
        ///     of dimensions and the input dtype are preserved.
        /// </returns>
        /// <exception cref="ArgumentException">If <paramref name="trim"/> contains unexpected characters, or an axis is repeated.</exception>
        /// <exception cref="AxisOutOfRangeException">If an axis is out of bounds for <paramref name="filt"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html</remarks>
        public static NDArray trim_zeros(NDArray filt, string trim = "fb", int[] axis = null)
        {
            // NumPy lower-cases and validates the trim spec against the four accepted spellings.
            trim = (trim ?? string.Empty).ToLowerInvariant();
            if (trim != "fb" && trim != "bf" && trim != "f" && trim != "b")
                throw new ArgumentException($"unexpected character(s) in `trim`: '{trim}'");

            int ndim = filt.ndim;

            // Resolve the axis tuple. axis == null means "all axes" (bounding box over the whole array);
            // an explicit list is normalized (negative wrap + out-of-range check) and must not repeat an axis.
            var trimAxis = new bool[ndim];
            bool anyAxis;
            if (axis == null)
            {
                for (int i = 0; i < ndim; i++)
                    trimAxis[i] = true;
                anyAxis = ndim > 0;
            }
            else
            {
                anyAxis = axis.Length > 0;
                foreach (var ax in axis)
                {
                    int a = DefaultEngine.check_and_adjust_axis(ndim, ax);
                    if (trimAxis[a])
                        throw new ArgumentException("repeated axis in `axis` argument");
                    trimAxis[a] = true;
                }
            }

            // No trimming requested (axis == () or a 0-d input) -> return the input unmodified, like NumPy.
            if (!anyAxis)
                return filt;

            bool trimFront = trim.IndexOf('f') >= 0;
            bool trimBack = trim.IndexOf('b') >= 0;

            // The bounding box of the non-zero elements is SEPARABLE per dimension: the first/last non-zero
            // coordinate along axis d equals the first/last index whose perpendicular hyperplane contains any
            // non-zero value. So rather than materializing every non-zero coordinate (NumPy's argwhere.min/max
            // over the whole array), reduce "any non-zero" over all OTHER axes down to a 1-D mask and take the
            // first/last of its non-zero positions. Only the axes actually being trimmed are projected.
            var start = new long?[ndim]; // null => open slice edge (Slice.All for untrimmed axes)
            var stop = new long?[ndim];

            var others = new int[ndim > 0 ? ndim - 1 : 0];
            for (int d = 0; d < ndim; d++)
            {
                if (!trimAxis[d])
                    continue; // untrimmed axis keeps null/null -> Slice.All

                // positions: sorted indices of the non-zero hyperplanes along axis d.
                //   1-D  -> the non-zero element indices directly (skip the projection).
                //   N-D  -> np.any over every other axis gives a 1-D mask; its non-zero indices are the edges.
                NDArray positions;
                if (ndim == 1)
                {
                    positions = np.nonzero(filt)[0];
                }
                else
                {
                    int oi = 0;
                    for (int k = 0; k < ndim; k++)
                        if (k != d)
                            others[oi++] = k;
                    positions = np.nonzero(np.any(filt, others))[0];
                }

                long n = positions.size;
                if (n == 0)
                {
                    // All-zero (or empty) along d -> empty slice; trim spec ignored, matching NumPy (start == stop).
                    start[d] = 0L;
                    stop[d] = 0L;
                }
                else
                {
                    long first = (long)positions[0];         // ascending -> ends are the first/last non-zero
                    long last = (long)positions[(int)n - 1];
                    start[d] = trimFront ? first : (long?)null; // 'f' absent -> keep the front (open start)
                    stop[d] = trimBack ? last + 1 : (long?)null; // 'b' absent -> keep the back  (open stop)
                }
            }

            // Trimmed axes are sliced to [start, stop); untouched axes span the full dimension.
            var slices = new Slice[ndim];
            for (int ax = 0; ax < ndim; ax++)
                slices[ax] = trimAxis[ax] ? new Slice(start[ax], stop[ax]) : Slice.All;

            return filt[slices];
        }
    }
}
