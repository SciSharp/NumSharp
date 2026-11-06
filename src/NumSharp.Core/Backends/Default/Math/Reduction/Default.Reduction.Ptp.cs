using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Iteration;

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
        ///     given (the subtraction then runs in NEP 50's loop over the out's dtype and the minima's, so an int8 range into
        ///     an int64 out does not wrap), else a fresh array, read-only when 0-d (NumPy's scalar).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">Fixed-width text, or a text out: NumPy's maximum.reduce refuses the loop with its
        ///     verbatim text.</exception>
        /// <exception cref="ValueError">A read-only out (NumPy parses out first), an empty reduction (no identity), or a
        ///     StringDType subtraction NumPy refuses; an out of the wrong rank or with a length-1 axis where the result is
        ///     longer: maximum.reduce's texts (<c>output parameter for reduction operation maximum has the wrong number of
        ///     dimensions: Found 2 but expected 1</c>, parity QA-31), checked before anything is computed.</exception>
        /// <exception cref="IncorrectShapeException">An out whose axes do not broadcast with the result's: maximum.reduce's
        ///     iterator text (<c>operands could not be broadcast together with remapped shapes [original-&gt;remapped]:
        ///     (3,)-&gt;(3) (3,4) </c>), NumPy's ValueError under the house type of a broadcast failure.</exception>
        /// <exception cref="TypeError">An out that is a C# scalar turned into a 0-d array (NumPy's <c>return arrays must be
        ///     of ArrayType</c>).</exception>
        /// <exception cref="TypeError">An out the range cannot reach by a same_kind cast: subtract's UFuncTypeError
        ///     (<c>Cannot cast ufunc 'subtract' output from dtype('float64') to dtype('int64') with casting rule
        ///     'same_kind'</c>), raised after maximum.reduce stored the maxima into it, as NumPy leaves them (parity
        ///     QA-55).</exception>
        /// <remarks>
        ///     Moved from the public <c>np.ptp(a, int? axis, out, keepdims)</c> by the np-layer dtype plan's P1b, statement for
        ///     statement; the facade's <c>np.amax</c> / <c>np.amin</c> calls became the engine members they reach for a
        ///     plain operand (<see cref="TensorEngine.ReduceAMax"/> / <see cref="TensorEngine.ReduceAMin"/>: no
        ///     <c>out=</c>, <c>initial=</c> or <c>where=</c>, and the matrix rule has already been applied). The plan's phase
        ///     PE (E1) made it DefaultEngine's override of <see cref="TensorEngine.Ptp(NDArray, int?, NDArray, bool)"/>; it
        ///     was an internal static.
        /// </remarks>
        [NDScoped] // the two extremes are this call's own temporaries (the difference, or out, is yielded)
        public override NDArray Ptp(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
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
            // With an out, _ptp's three statements in NumPy's order: `subtract(umr_maximum(a, axis, None, out, keepdims),
            // umr_minimum(a, axis, None, None, keepdims), out)`.
            if (@out is not null)
            {
                // maximum.reduce(out=out) first: its prologue checks the out before anything is computed (read-only, the
                // axis, the rank text naming 'maximum', the iterator's shapes; parity QA-31), so a misshapen out is reported
                // before an empty reduction's "no identity", and its unsafe store leaves the maxima in the out.
                NDArray maxima = a.TensorEngine.ReduceAMax(a, axis, keepdims, null, @out);
                NDArray minima = a.TensorEngine.ReduceAMin(a, axis, keepdims);
                // subtract(out, min, out): NEP 50's loop over the out's dtype and the minima's (an int8 range into an int64
                // out subtracts in int64, without int8's wrap), and its same_kind out cast — NumPy's UFuncTypeError for an
                // integer out of float data, raised with the maxima already stored, as NumPy leaves them (parity QA-55).
                return a.TensorEngine.Subtract(maxima, minima, null, @out);
            }

            var maxRes = a.TensorEngine.ReduceAMax(a, axis, keepdims);
            var minRes = a.TensorEngine.ReduceAMin(a, axis, keepdims);
            var diff = maxRes - minRes;

            // Fresh 0-d ptp is a numpy SCALAR (read-only).
            return diff.MarkReductionScalar();
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
        /// <returns>The range, dtype preserved (integer overflow wraps as in NumPy); <paramref name="out"/> itself when given,
        ///     the subtraction then in NEP 50's loop over the out's dtype and the minima's.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range (a 0-d input has no axis for a tuple to name).</exception>
        /// <exception cref="ValueError">An axis appears twice ("duplicate value in 'axis'").</exception>
        /// <exception cref="TypeError">Fixed-width text, or a text out (NumPy's loop refusal); an out that is a C# scalar
        ///     turned into a 0-d array (<c>return arrays must be of ArrayType</c>).</exception>
        /// <exception cref="ValueError">A read-only out, or one of the wrong rank or with a length-1 axis where the result is
        ///     longer: maximum.reduce's prologue texts (parity QA-31), before the axes are reduced.</exception>
        /// <exception cref="IncorrectShapeException">An out whose axes do not broadcast with the result's (maximum.reduce's
        ///     <c>operands could not be broadcast together with remapped shapes ...</c>).</exception>
        /// <exception cref="TypeError">An out the range cannot reach by a same_kind cast: subtract's UFuncTypeError, raised
        ///     after the maxima were stored into it (parity QA-55).</exception>
        /// <remarks>Moved from the public tuple-axis <c>np.ptp</c> by the np-layer dtype plan's P1b, statement for
        ///     statement; its re-entries into the one-axis overload call the one-axis member above, through virtual
        ///     dispatch since phase PE made both of them DefaultEngine's overrides of
        ///     <see cref="TensorEngine.Ptp(NDArray, int[], NDArray, bool)"/> and its one-axis sibling.</remarks>
        [NDScoped] // the chained extremes and the kept-dims difference are this call's own (the result, or out, is yielded)
        public override NDArray Ptp(NDArray a, int[] axis, NDArray @out = null, bool keepdims = false)
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
            // _ptp's maximum.reduce(a, axis, out=out, keepdims) checks the out first, in its prologue's order: read-only, then
            // the tuple's axes, then the out's rank and the iterator's shapes (parity QA-31). The one-axis and every-axis
            // delegations below run the same check again, which passes.
            if (@out is not null)
                ReductionArguments.Validate(a, axis, keepdims, @out, null, "maximum");

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

            // The result's dimensions without keepdims: the kept axes alone (the keepdims chain left length-1 axes in place).
            Shape resultShape = default;
            if (!keepdims)
            {
                var kept = new List<long>(a.ndim - sortedDesc.Count);
                for (int i = 0; i < a.ndim; i++)
                    if (!flags[i])
                        kept.Add(a.Shape.dimensions[i]);
                resultShape = Shape.Adopt(kept.ToArray());
            }

            if (@out is not null)
            {
                // _ptp's statements with the tuple (parity QA-55): maximum.reduce's unsafe store leaves the maxima in the out
                // (its prologue checked the out above), then subtract(out, min, out) takes NEP 50's loop over the out's dtype
                // and the minima's and casts into the out under same_kind (NumPy's UFuncTypeError text for an out it cannot
                // reach, the maxima already stored).
                NDArray maxima = keepdims ? maxRes : maxRes.reshape(resultShape);
                NDArray minima = keepdims ? minRes : minRes.reshape(resultShape);
                UnmanagedStorage.Copyto(@out, maxima, NPY_CASTING.NPY_UNSAFE_CASTING);
                return a.TensorEngine.Subtract(@out, minima, null, @out);
            }

            NDArray diff = maxRes - minRes;
            return keepdims ? diff : diff.reshape(resultShape);
        }
    }
}
