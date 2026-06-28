using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Range of values (maximum - minimum) along an axis.
        ///     Equivalent to <c>np.amax(a, axis) - np.amin(a, axis)</c>; dtype is preserved,
        ///     so unsigned/signed integer overflow wraps the same way NumPy does
        ///     (e.g. <c>ptp(uint8[0,255]) == 255</c>, <c>ptp(int8[-128,127]) == -1</c>).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ptp.html</remarks>
        public static NDArray ptp(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            var maxRes = np.amax(a, axis, keepdims);
            var minRes = np.amin(a, axis, keepdims);
            var diff = maxRes - minRes;

            return WriteOrReturn(diff, @out);
        }

        public static NDArray ptp(NDArray a, int[] axis, NDArray @out = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (axis is null) return ptp(a, (int?)null, @out, keepdims);
            if (axis.Length == 1) return ptp(a, (int?)axis[0], @out, keepdims);

            int ndim = a.ndim;
            var normalized = new int[axis.Length];
            for (int i = 0; i < axis.Length; i++)
            {
                int ax = axis[i];
                if (ax < 0) ax += ndim;
                if (ax < 0 || ax >= ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis),
                        $"axis {axis[i]} is out of bounds for array of dimension {ndim}");
                normalized[i] = ax;
            }

            var sortedDesc = (int[])normalized.Clone();
            Array.Sort(sortedDesc);
            for (int i = 1; i < sortedDesc.Length; i++)
                if (sortedDesc[i] == sortedDesc[i - 1])
                    throw new ArgumentException("duplicate value in 'axis'");
            Array.Reverse(sortedDesc);

            NDArray maxRes = a;
            NDArray minRes = a;
            foreach (var ax in sortedDesc)
            {
                maxRes = np.amax(maxRes, ax, keepdims: true);
                minRes = np.amin(minRes, ax, keepdims: true);
            }

            NDArray diff = maxRes - minRes;

            if (!keepdims)
            {
                var kept = new List<long>(ndim - normalized.Length);
                for (int i = 0; i < ndim; i++)
                {
                    bool reduced = false;
                    for (int j = 0; j < normalized.Length; j++)
                        if (normalized[j] == i) { reduced = true; break; }
                    if (!reduced) kept.Add(a.shape[i]);
                }
                diff = diff.reshape(kept.ToArray());
            }

            return WriteOrReturn(diff, @out);
        }

        private static NDArray WriteOrReturn(NDArray diff, NDArray @out)
        {
            if (@out is null) return diff;
            if (!diff.shape.SequenceEqual(@out.shape))
                throw new ArgumentException(
                    $"out has wrong shape; expected=[{string.Join(",", diff.shape)}] got=[{string.Join(",", @out.shape)}]");
            np.copyto(@out, diff);
            return @out;
        }
    }
}
