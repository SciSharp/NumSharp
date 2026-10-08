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
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray ptp(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, axis, @out, keepdims), static s => ptp(s.a, s.axis, s.@out, s.keepdims), @out);
            // Text (input or out=): NumPy's maximum.reduce(a, axis, out=out) parses out (read-only), validates the axis (a
            // 0-d input accepts 0 and -1), then resolves its loop on (out, a) and refuses text with its verbatim text; see
            // NDStringUfuncs.ReduceError / ReduceOutError.
            if (a is null) throw new ArgumentNullException(nameof(a));
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("ptp", a, axis, @out);

            var maxRes = np.amax(a, axis, keepdims);
            var minRes = np.amin(a, axis, keepdims);
            var diff = maxRes - minRes;

            // Fresh 0-d ptp is a numpy SCALAR (read-only); out= returns the writeable out.
            return @out is null ? diff.MarkReductionScalar() : WriteOrReturn(diff, @out);
        }

        /// <summary>
        ///     Range of values (maximum - minimum) over the axes of a tuple: NumPy's
        ///     <c>np.ptp(a, axis=(…))</c>, i.e. <c>maximum.reduce(a, axis) - minimum.reduce(a, axis)</c>.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">The reduced axes (negative values count from the end), or null for every axis. An empty array
        ///     reduces nothing (the result is <c>a - a</c>).</param>
        /// <param name="out">An optional output; the result is written into it and it is returned.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <returns>The range, dtype preserved (integer overflow wraps as in NumPy).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range (a 0-d input has no axis for a tuple to name).</exception>
        /// <exception cref="ValueError">An axis appears twice ("duplicate value in 'axis'").</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> has the wrong shape.</exception>
        [NDScoped]
        public static NDArray ptp(NDArray a, int[] axis, NDArray @out = null, bool keepdims = false)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, axis, @out, keepdims), static s => ptp(s.a, s.axis, s.@out, s.keepdims), @out);
            // Text (input or out=): NumPy's maximum.reduce checks the tuple (every axis, then "duplicate value in
            // 'axis'"), then refuses text on (out, a) with its verbatim text; see NDStringUfuncs.ReduceError / ReduceOutError.
            if (a is null) throw new ArgumentNullException(nameof(a));
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("ptp", a, axis, @out);
            if (axis is null) return ptp(a, (int?)null, @out, keepdims);

            // maximum.reduce's tuple axis (PyUFunc_GenericReduction's tuple branch, then PyUFunc_Reduce's flags): every
            // axis normalized first — NumPy's AxisError, with no 0-d exception for a tuple — then "duplicate value in
            // 'axis'" (NumPy's ValueError). Checked before the one-axis delegation, which would let (0,) through on a 0-d
            // input.
            var flags = Backends.ReductionArguments.AxisFlags(a.ndim, axis);
            if (axis.Length == 1) return ptp(a, (int?)axis[0], @out, keepdims);

            // A tuple naming EVERY axis is ufunc.reduce's axis=None (PyUFunc_Reduce reduces the same axis set either
            // way), so it takes the flat reductions - NumPy's own schedule for the whole operand, one pass each - instead
            // of the axis-by-axis chain below (four reductions and their temporaries: 7.7 us against 3.8 us on (100, 100)).
            bool everyAxis = flags.Length > 0;
            foreach (bool reduced in flags)
                everyAxis &= reduced;
            if (everyAxis)
                return ptp(a, (int?)null, @out, keepdims);

            // The reduced axes, highest first, so each keepdims reduction leaves the lower axes' positions unchanged.
            var sortedDesc = new List<int>(axis.Length);
            for (int d = flags.Length - 1; d >= 0; d--)
                if (flags[d])
                    sortedDesc.Add(d);

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
                var kept = new List<long>(a.ndim - sortedDesc.Count);
                for (int i = 0; i < a.ndim; i++)
                    if (!flags[i])
                        kept.Add(a.Shape.dimensions[i]);
                diff = diff.reshape(Shape.Adopt(kept.ToArray()));
            }

            return WriteOrReturn(diff, @out);
        }

        private static NDArray WriteOrReturn(NDArray diff, NDArray @out)
        {
            if (@out is null) return diff;
            if (!diff.Shape.dimensions.SequenceEqual(@out.Shape.dimensions))
                throw new ArgumentException(
                    $"out has wrong shape; expected=[{string.Join(",", diff.Shape.dimensions)}] got=[{string.Join(",", @out.Shape.dimensions)}]");
            np.copyto(@out, diff);
            return @out;
        }
    }
}
