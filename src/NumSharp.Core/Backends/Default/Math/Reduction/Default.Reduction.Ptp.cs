using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>np.ptp</c> body (<c>_methods._ptp</c>): <c>subtract(maximum.reduce(a, axis, out, keepdims),
        ///     minimum.reduce(a, axis, keepdims), out)</c>, with NumPy's text routes first. The facade applies the
        ///     <c>numpy.matrix</c> rule and calls this for everything else.
        /// </summary>
        /// <param name="a">The array (any dtype; StringDType computes both extremes, fixed-width text is refused).</param>
        /// <param name="axis">The reduced axis (negative counts from the end; a 0-d array accepts 0 and -1), or null for
        ///     every axis.</param>
        /// <param name="out">An optional output of the result's shape; the result is written into it and it is returned.</param>
        /// <param name="keepdims">Whether the reduced axis stays with length one.</param>
        /// <returns>The range, dtype preserved (integer overflow wraps as in NumPy): <paramref name="out"/> itself when
        ///     given, else a fresh array, read-only when 0-d (NumPy's scalar).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">Fixed-width text, or a text out: NumPy's maximum.reduce refuses the loop with its
        ///     verbatim text.</exception>
        /// <exception cref="ValueError">A read-only out (NumPy parses out first), an empty reduction (no identity), or a
        ///     StringDType subtraction NumPy refuses.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <remarks>
        ///     Moved from the public <c>np.ptp(a, int? axis, out, keepdims)</c> by the np-layer dtype plan's P1b, statement for
        ///     statement. NumPy's ptp is a Python function, so its body is an internal member here rather than a
        ///     TensorEngine one (decision D1, as np.angle's). The facade's <c>np.amax</c> / <c>np.amin</c> calls became the
        ///     engine members they reach for a plain operand (<see cref="TensorEngine.ReduceAMax"/> /
        ///     <see cref="TensorEngine.ReduceAMin"/>: no <c>out=</c>, <c>initial=</c> or <c>where=</c>, and the matrix rule
        ///     has already been applied).
        /// </remarks>
        [NDScoped] // the two extremes are this call's own temporaries (the difference, or out, is yielded)
        internal static NDArray Ptp(NDArray a, int? axis, NDArray @out, bool keepdims)
        {
            // Text (input or out=): NumPy's maximum.reduce(a, axis, out=out) parses out (read-only), validates the axis (a
            // 0-d input accepts 0 and -1), then resolves its loop on (out, a) and refuses text with its verbatim text; see
            // NDStringUfuncs.ReduceError / ReduceOutError.
            if (a is null) throw new ArgumentNullException(nameof(a));
            // StringDType: _ptp computes both extremes (maximum.reduce into the out, minimum.reduce) and then meets NumPy's
            // subtraction of what Python holds: NDVStringStatistics.Ptp.
            if (Strings.NDStringUfuncs.VStringReduces("ptp", a, axis, null, @out))
                return Strings.NDVStringStatistics.Ptp(a, axis, @out, keepdims);
            Strings.NDStringUfuncs.ThrowIfReduce("ptp", a, axis, @out);

            var maxRes = a.TensorEngine.ReduceAMax(a, axis, keepdims);
            var minRes = a.TensorEngine.ReduceAMin(a, axis, keepdims);
            var diff = maxRes - minRes;

            // Fresh 0-d ptp is a numpy SCALAR (read-only); out= returns the writeable out.
            return @out is null ? diff.MarkReductionScalar() : PtpOut(diff, @out);
        }

        /// <summary>
        ///     NumPy's <c>np.ptp(a, axis=(…))</c> body: <c>maximum.reduce(a, axis) - minimum.reduce(a, axis)</c> over the axes
        ///     of a tuple, with NumPy's text routes and tuple checks first.
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
        /// <exception cref="TypeError">Fixed-width text, or a text out (NumPy's loop refusal).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> has the wrong shape.</exception>
        /// <remarks>Moved from the public tuple-axis <c>np.ptp</c> by the np-layer dtype plan's P1b, statement for
        ///     statement; its re-entries into the one-axis overload call the one-axis member above.</remarks>
        [NDScoped] // the chained extremes and the kept-dims difference are this call's own (the result, or out, is yielded)
        internal static NDArray Ptp(NDArray a, int[] axis, NDArray @out, bool keepdims)
        {
            // Text (input or out=): NumPy's maximum.reduce checks the tuple (every axis, then "duplicate value in
            // 'axis'"), then refuses text on (out, a) with its verbatim text; see NDStringUfuncs.ReduceError / ReduceOutError.
            if (a is null) throw new ArgumentNullException(nameof(a));
            // StringDType: maximum.reduce's tuple checks and loop resolution, then one axis computes (NDVStringStatistics.Ptp);
            // no axis reduces nothing, so the extremes are the elements themselves (an ndarray: subtract has no loop).
            if (axis is not null && Strings.NDStringUfuncs.VStringReducesAxes("ptp", a, axis, @out))
            {
                if (axis.Length == 1)
                    return Strings.NDVStringStatistics.Ptp(a, axis[0], @out, keepdims);
                return Strings.NDVStringStatistics.PtpOverNoAxis(a, @out);
            }
            Strings.NDStringUfuncs.ThrowIfReduce("ptp", a, axis, @out);
            if (axis is null) return Ptp(a, (int?)null, @out, keepdims);

            // maximum.reduce's tuple axis (PyUFunc_GenericReduction's tuple branch, then PyUFunc_Reduce's flags): every
            // axis normalized first — NumPy's AxisError, with no 0-d exception for a tuple — then "duplicate value in
            // 'axis'" (NumPy's ValueError). Checked before the one-axis delegation, which would let (0,) through on a 0-d
            // input.
            var flags = ReductionArguments.AxisFlags(a.ndim, axis);
            if (axis.Length == 1) return Ptp(a, (int?)axis[0], @out, keepdims);

            // A tuple naming EVERY axis is ufunc.reduce's axis=None (PyUFunc_Reduce reduces the same axis set either
            // way), so it takes the flat reductions - NumPy's own schedule for the whole operand, one pass each - instead
            // of the axis-by-axis chain below (four reductions and their temporaries: 7.7 us against 3.8 us on (100, 100)).
            bool everyAxis = flags.Length > 0;
            foreach (bool reduced in flags)
                everyAxis &= reduced;
            if (everyAxis)
                return Ptp(a, (int?)null, @out, keepdims);

            // The reduced axes, highest first, so each keepdims reduction leaves the lower axes' positions unchanged.
            var sortedDesc = new List<int>(axis.Length);
            for (int d = flags.Length - 1; d >= 0; d--)
                if (flags[d])
                    sortedDesc.Add(d);

            NDArray maxRes = a;
            NDArray minRes = a;
            foreach (var ax in sortedDesc)
            {
                maxRes = maxRes.TensorEngine.ReduceAMax(maxRes, ax, keepdims: true);
                minRes = minRes.TensorEngine.ReduceAMin(minRes, ax, keepdims: true);
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

            return PtpOut(diff, @out);
        }

        /// <summary>
        ///     The ptp's <c>out=</c> store: <paramref name="out"/> must have the result's dimensions and receives it under
        ///     np.copyto's default <c>same_kind</c> rule.
        /// </summary>
        /// <param name="diff">The range.</param>
        /// <param name="out">The caller's out, or null.</param>
        /// <returns><paramref name="diff"/> when <paramref name="out"/> is null, else <paramref name="out"/> after the
        ///     store.</returns>
        /// <exception cref="ArgumentException"><paramref name="out"/>'s dimensions are not the result's ("out has wrong
        ///     shape").</exception>
        /// <exception cref="InvalidCastException">The result's dtype does not cast to <paramref name="out"/>'s under
        ///     <c>same_kind</c> (np.copyto's text).</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (np.copyto's text).</exception>
        /// <remarks>Moved from <c>np.ptp</c>'s <c>WriteOrReturn</c> by the np-layer dtype plan's P1b. The store still goes
        ///     through <c>np.copyto</c>, a structural call into the facade that the plan's P6 replaces with an engine copy;
        ///     the up-call ratchet pins it with that reason.</remarks>
        private static NDArray PtpOut(NDArray diff, NDArray @out)
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
