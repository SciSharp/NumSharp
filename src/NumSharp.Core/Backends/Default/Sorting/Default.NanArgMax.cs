// NumPy's np.nanargmax and np.nanargmin (numpy/lib/_nanfunctions_impl.py) as DefaultEngine's members: the axis check,
// text's argmax / argmin checks, _replace_nan with the all-NaN slice guard, and the arg reduction with NumPy's out=
// contract (ReduceArgMax / ReduceArgMin's).

using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped]
        public override NDArray NanArgMax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            // This member's scope reclaims the NaN mask, the all-NaN-guard reductions and the replaced copy; the
            // result (or the caller's @out — untracked, a Returns no-op) is yielded.
            ValidateNanArgAxis(a, axis);
            // Text: no NaN mask (not inexact), so this is NumPy's argmax, which runs its checks — the empty length, then
            // out= — before it compares the text; the engine then compares (or, where nothing is compared, computes the zero
            // indices on a bool stand-in), and out= receives the indices as NumPy writes them.
            if (Backends.Strings.NDStringUfuncs.IsVString(a) || Backends.Strings.NDStringUfuncs.IsString(a))
            {
                ArgTextChecks(a, axis, @out, keepdims, "argmax");
                return a.TensorEngine.ReduceArgMax(a, axis, keepdims, @out);
            }
            var prepared = NanArgReplace(a, double.NegativeInfinity, axis);
            return prepared.TensorEngine.ReduceArgMax(prepared, axis, keepdims, @out);
        }

        /// <inheritdoc />
        [NDScoped]
        public override NDArray NanArgMin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            // This member's scope works as NanArgMax's does.
            ValidateNanArgAxis(a, axis);
            // Text: NumPy's argmin checks the empty length and out= before it compares (see NanArgMax); the engine
            // compares, and out= receives the indices.
            if (Backends.Strings.NDStringUfuncs.IsVString(a) || Backends.Strings.NDStringUfuncs.IsString(a))
            {
                ArgTextChecks(a, axis, @out, keepdims, "argmin");
                return a.TensorEngine.ReduceArgMin(a, axis, keepdims, @out);
            }
            var prepared = NanArgReplace(a, double.PositiveInfinity, axis);
            return prepared.TensorEngine.ReduceArgMin(prepared, axis, keepdims, @out);
        }

        /// <summary>
        ///     Validate the reduction axis against NumPy's rule, up front and for every dtype path:
        ///     allowed iff <c>-max(ndim,1) &lt;= axis &lt; max(ndim,1)</c> — a 0-d array accepts axis 0 and -1
        ///     (NumPy's legacy reduction quirk, probed against 2.4.2). Reports the ORIGINAL axis.
        ///     Doing it here (rather than trusting the delegates) keeps the integer passthrough path from
        ///     inheriting the engine's silent negative-axis wrap-around.
        /// </summary>
        /// <param name="a">The input (its dtype decides which NumPy statement reports a bad axis).</param>
        /// <param name="axis">The axis, or null for the flattened array (always legal).</param>
        /// <exception cref="AxisError">The axis is out of bounds: the message names the input's own rank for an inexact
        ///     dtype and the ravelled rank (at least 1) otherwise, as NumPy's raising statement does.</exception>
        private static void ValidateNanArgAxis(NDArray a, int? axis)
        {
            if (axis is not int ax)
                return;
            // NumPy's argmax ravels a 0-d input first, so the bound is 1. The dimension the message names depends on who
            // raises: for an inexact input NumPy's all-NaN guard runs np.all(mask, axis) on the array as it is, so a 0-d
            // float reports dimension 0; any other input (integers, bool, text) reaches argmax directly, which reports the
            // ravelled dimension 1 (probed: nanargmax(np.array(5.0), axis=1) vs nanargmax(np.array(5), axis=1)).
            int bound = Math.Max(a.ndim, 1);
            if (ax < -bound || ax >= bound)
                throw new AxisError(ax, IsNanArgInexact(a.typecode) ? a.ndim : bound);
        }

        /// <summary>
        ///     Whether NumPy's <c>_replace_nan</c> builds a NaN mask for this dtype (the inexact family: float16, float32,
        ///     float64, complex128), so that its all-NaN guard — not argmax — sees the axis first.
        /// </summary>
        /// <param name="code">The input's dtype.</param>
        /// <returns>True for the inexact dtypes.</returns>
        private static bool IsNanArgInexact(NPTypeCode code)
            => code is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex;

        /// <summary>
        ///     Port of NumPy's <c>_replace_nan</c> + the all-NaN slice guard from
        ///     <c>numpy/lib/_nanfunctions_impl.py</c> (nanargmin/nanargmax): for inexact dtypes
        ///     (Half/Single/Double/Complex) replace every NaN with <paramref name="fill"/> in a copy and
        ///     raise <see cref="ValueError"/> when any slice along <paramref name="axis"/> is entirely NaN.
        ///     Every other dtype (NumPy's <c>mask=None</c> branch — integral/bool, and NumSharp's
        ///     Char/Decimal which carry no NaN) passes through untouched.
        /// </summary>
        /// <param name="a">The input.</param>
        /// <param name="fill">The value a NaN becomes: -inf for nanargmax, +inf for nanargmin.</param>
        /// <param name="axis">The reduced axis, or null for the flattened array (decides the all-NaN guard).</param>
        /// <returns><paramref name="a"/> itself when it holds no NaN or is not inexact; otherwise a copy with every NaN
        ///     replaced.</returns>
        /// <exception cref="ValueError">A slice (or the whole flattened array) is entirely NaN (<c>All-NaN slice
        ///     encountered</c>).</exception>
        private NDArray NanArgReplace(NDArray a, double fill, int? axis)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    break;
                default:
                    return a;
            }

            // np.isnan / np.any / np.all as the operands' engine members (a static helper: the operand's engine).
            var mask = a.TensorEngine.IsNan(a);
            // NumPy checks `mask is not None and mask.size` — an empty array skips the all-NaN
            // guard and lets argmax/argmin raise their own empty-sequence error. A NaN-free array
            // needs no replacement copy (argmax only reads, so returning `a` is observably identical).
            if (mask.size == 0 || !mask.TensorEngine.Any(mask))
                return a;

            var engine = mask.TensorEngine;
            bool anyAllNanSlice = axis is int ax
                ? engine.Any(engine.All(mask, ax))
                : engine.All(mask);
            if (anyAllNanSlice)
                throw new ValueError("All-NaN slice encountered");

            // np.copyto(a_copy, val, where=mask): the double scalar casts into the operand's own
            // dtype (Half/float/double; Complex becomes (fill, 0) — probed identical to NumPy).
            var replaced = a.copy();
            UnmanagedStorage.Copyto(replaced, NewScalar(fill), NPY_CASTING.NPY_SAME_KIND_CASTING, mask);
            return replaced;
        }
    }
}
