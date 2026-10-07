using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the index of the maximum value in the flattened array, ignoring NaNs.
        ///     For an all-NaN array a <see cref="ValueError"/> is raised. Warning: the result
        ///     cannot be trusted if the array contains only NaNs and -Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <returns>Index of the maximum value in the flattened array (NumPy intp = int64).</returns>
        /// <exception cref="ValueError">All-NaN slice encountered.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html</remarks>
        [NDScoped]
        public static long nanargmax(NDArray a)
        {
            // Scalar return — scope only: reclaims the NaN mask and the full-size replaced copy
            // (a NaN-free or integer input passes through NanArgReplace untracked and unharmed).
            var prepared = NanArgReplace(a, double.NegativeInfinity, axis: null);
            return (long)prepared.TensorEngine.ArgMax(prepared);
        }

        /// <summary>
        ///     Return the indices of the maximum values in the specified axis, ignoring NaNs.
        ///     For all-NaN slices a <see cref="ValueError"/> is raised. Warning: the results
        ///     cannot be trusted if a slice contains only NaNs and -Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis along which to operate. If <c>null</c>, the flattened input is used.</param>
        /// <param name="out">If provided, the result is inserted into this array and the SAME instance is
        ///     returned (NumPy semantics): the shape must equal the result's exactly, and the dtype must be
        ///     an integer family safe-castable to int64 (bool through uint32 — uint64/floats raise NumPy's
        ///     verbatim cast TypeError); the int64 indices then cast UNSAFELY into it (wrap/truncate).</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one
        ///     (with <paramref name="axis"/> <c>null</c> the result has shape <c>(1,) * a.ndim</c>, like NumPy).</param>
        /// <returns>Array of int64 indices (or a 0-d scalar for the flattened form); <paramref name="out"/> when given.</returns>
        /// <exception cref="ValueError">All-NaN slice encountered.</exception>
        /// <exception cref="AxisError">Axis out of bounds (reports the original axis, like NumPy).</exception>
        /// <exception cref="StringComputationNotSupportedException">A string array whose lines hold two or more elements: it is
        ///     <c>np.argmax</c> / <c>np.argmin</c> of the text, which NumPy compares (the next phase of NumSharp's string
        ///     support). An empty result or lines of one element give their zero indices (<paramref name="out"/> included).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html</remarks>
        [NDScoped]
        public static NDArray nanargmax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, keepdims), static s => nanargmax(s.a, s.axis, s.@out, s.keepdims), axis, keepdims, null);
            // Boundary scope: reclaims the NaN mask, the all-NaN-guard reductions and the replaced
            // copy; the result (or the caller's @out — untracked, a Returns no-op) is yielded.
            ValidateNanArgAxis(a, axis);
            // Text: no NaN mask (not inexact), so NumPy's argmax runs its checks — the empty length, then out= — before it
            // compares the text. Where it compares nothing (an empty result, lines of one element) the bool stand-in of the
            // same shape computes the zeros, out= included.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                if (NanArgTextAnswer(a, axis, @out, keepdims, "argmax") is { } refusal)
                    throw refusal;
                using var standIn = Backends.Strings.NDStringUfuncs.BoolStandIn(a);
                return nanargmax(standIn, axis, @out, keepdims);
            }
            var prepared = NanArgReplace(a, double.NegativeInfinity, axis);
            var res = prepared.TensorEngine.ReduceArgMax(prepared, axis, keepdims);
            return NanArgWriteOut(res, @out, "argmax");
        }

        /// <summary>
        ///     Return the index of the minimum value in the flattened array, ignoring NaNs.
        ///     For an all-NaN array a <see cref="ValueError"/> is raised. Warning: the result
        ///     cannot be trusted if the array contains only NaNs and Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <returns>Index of the minimum value in the flattened array (NumPy intp = int64).</returns>
        /// <exception cref="ValueError">All-NaN slice encountered.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html</remarks>
        [NDScoped]
        public static long nanargmin(NDArray a)
        {
            // Scalar return — scope only (see nanargmax(NDArray)).
            var prepared = NanArgReplace(a, double.PositiveInfinity, axis: null);
            return (long)prepared.TensorEngine.ArgMin(prepared);
        }

        /// <summary>
        ///     Return the indices of the minimum values in the specified axis, ignoring NaNs.
        ///     For all-NaN slices a <see cref="ValueError"/> is raised. Warning: the results
        ///     cannot be trusted if a slice contains only NaNs and Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis along which to operate. If <c>null</c>, the flattened input is used.</param>
        /// <param name="out">If provided, the result is inserted into this array and the SAME instance is
        ///     returned (NumPy semantics — see <see cref="nanargmax(NDArray, int?, NDArray, bool)"/>).</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one
        ///     (with <paramref name="axis"/> <c>null</c> the result has shape <c>(1,) * a.ndim</c>, like NumPy).</param>
        /// <returns>Array of int64 indices (or a 0-d scalar for the flattened form); <paramref name="out"/> when given.</returns>
        /// <exception cref="ValueError">All-NaN slice encountered.</exception>
        /// <exception cref="AxisError">Axis out of bounds (reports the original axis, like NumPy).</exception>
        /// <exception cref="StringComputationNotSupportedException">A string array whose lines hold two or more elements: it is
        ///     <c>np.argmax</c> / <c>np.argmin</c> of the text, which NumPy compares (the next phase of NumSharp's string
        ///     support). An empty result or lines of one element give their zero indices (<paramref name="out"/> included).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html</remarks>
        [NDScoped]
        public static NDArray nanargmin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, keepdims), static s => nanargmin(s.a, s.axis, s.@out, s.keepdims), axis, keepdims, null);
            // Boundary scope (see nanargmax(NDArray, int?, NDArray, bool)).
            ValidateNanArgAxis(a, axis);
            // Text: NumPy's argmin checks the empty length and out= before it compares (see nanargmax); a shape that
            // compares nothing runs on the bool stand-in.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                if (NanArgTextAnswer(a, axis, @out, keepdims, "argmin") is { } refusal)
                    throw refusal;
                using var standIn = Backends.Strings.NDStringUfuncs.BoolStandIn(a);
                return nanargmin(standIn, axis, @out, keepdims);
            }
            var prepared = NanArgReplace(a, double.PositiveInfinity, axis);
            var res = prepared.TensorEngine.ReduceArgMin(prepared, axis, keepdims);
            return NanArgWriteOut(res, @out, "argmin");
        }

        /// <summary>
        ///     NumPy's argmax/argmin <c>out=</c> contract (probed 2.4.2): the out array's dtype must be
        ///     safe-castable TO intp — every bool/int dtype through uint32 (uint64 is not: verbatim
        ///     <c>Cannot cast array data from dtype('uint64') to dtype('int64') according to the rule
        ///     'safe'</c>, a TypeError) — its shape must EQUAL the (keepdims-adjusted) result shape
        ///     (<c>output array does not match result of np.argmax.</c> — nanargmax leaks "argmax",
        ///     nanargmin "argmin"), and the int64 indices then cast UNSAFELY into it (an int8 out wraps
        ///     index 200 to -56; a bool out reads index != 0 — probed). Returns the SAME instance.
        ///     Char (no NumPy dtype) is integer-family and safe to int64, so it is allowed (house call).
        /// </summary>
        private static NDArray NanArgWriteOut(NDArray result, NDArray @out, string name)
        {
            if (@out is null)
                return result;

            ValidateNanArgOut(@out, result.Shape.dimensions, name);
            np.copyto(@out, result, casting: "unsafe");
            return @out;
        }

        /// <summary>
        ///     <c>PyArray_ArgMaxWithKeepdims</c>'s two <c>out=</c> checks, in its order: the out must have the result's
        ///     shape exactly (<c>output array does not match result of np.argmax.</c>), then it must reach intp by a SAFE cast
        ///     — bool through uint32 (and NumSharp's char) do; uint64, floats, complex and text do not (NumPy's cast TypeError,
        ///     which words a 0-d out as a <c>scalar</c>).
        /// </summary>
        /// <param name="out">The out array.</param>
        /// <param name="dims">The result's dimensions.</param>
        /// <param name="name"><c>argmax</c> or <c>argmin</c> (the nan forms leak the plain name, as NumPy does).</param>
        /// <exception cref="ValueError">The shape differs from the result's.</exception>
        /// <exception cref="TypeError">The out's dtype does not cast safely to int64.</exception>
        /// <remarks>
        ///     The shape is checked FIRST: NumPy compares the out's dimensions before it converts the out to intp (an out of
        ///     the wrong shape AND the wrong dtype reports the shape). The numeric path used to check the dtype first.
        /// </remarks>
        private static void ValidateNanArgOut(NDArray @out, long[] dims, string name)
        {
            if (@out.ndim != dims.Length)
                throw new ValueError($"output array does not match result of np.{name}.");
            for (int d = 0; d < dims.Length; d++)
                if (@out.Shape.dimensions[d] != dims[d])
                    throw new ValueError($"output array does not match result of np.{name}.");

            switch (@out.typecode)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                    return;
                default:
                    throw new TypeError(
                        $"Cannot cast {(@out.ndim == 0 ? "scalar" : "array data")} from {@out.dtype.ToString(true)} to " +
                        "dtype('int64') according to the rule 'safe'");
            }
        }

        /// <summary>
        ///     NumPy's <c>np.argmax</c> / <c>np.argmin</c> of text (what <c>nanargmax</c> / <c>nanargmin</c> are for text: its
        ///     <c>_replace_nan</c> leaves a non-inexact array alone), once the axis passed: an empty reduced length refuses,
        ///     then <c>out=</c> is checked (<see cref="ValidateNanArgOut"/>), and only then the text is compared.
        /// </summary>
        /// <param name="a">The text array.</param>
        /// <param name="axis">The validated axis, or null for the flattened array.</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one (decides the expected out shape).</param>
        /// <param name="name"><c>argmax</c> or <c>argmin</c>.</param>
        /// <returns>The exception to throw, or null when NumPy compares nothing: an empty result (the reduced length is not
        ///     0) or lines of at most one element, whose indices are all 0 (the caller computes them on a bool stand-in).</returns>
        /// <exception cref="ValueError">An out of the wrong shape (thrown).</exception>
        /// <exception cref="TypeError">An out that does not cast safely to int64 (thrown).</exception>
        private static Exception NanArgTextAnswer(NDArray a, int? axis, NDArray @out, bool keepdims, string name)
        {
            int nd = a.ndim;
            long[] dims;
            long reduced;
            if (axis is not int ax)
            {
                // The flattened array: a 0-d input ravels to one element.
                reduced = nd == 0 ? 1 : a.size;
                dims = new long[keepdims ? nd : 0];
                Array.Fill(dims, 1L);
            }
            else if (nd == 0)
            {
                reduced = 1;
                dims = Array.Empty<long>();
            }
            else
            {
                int norm = ax < 0 ? ax + nd : ax;
                reduced = a.Shape.dimensions[norm];
                dims = new long[keepdims ? nd : nd - 1];
                for (int d = 0, k = 0; d < nd; d++)
                {
                    if (d == norm)
                    {
                        if (keepdims)
                            dims[k++] = 1;
                        continue;
                    }
                    dims[k++] = a.Shape.dimensions[d];
                }
            }
            if (reduced == 0)
                return new ValueError($"attempt to get {name} of an empty sequence");
            if (@out is not null)
                ValidateNanArgOut(@out, dims, name);
            // The arg loop returns before its first comparison for an empty input or a line of one element.
            if (a.size == 0 || reduced == 1)
                return null;
            return Backends.Strings.NDStringUfuncs.NextPhase(name, a);
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
        private static NDArray NanArgReplace(NDArray a, double fill, int? axis)
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

            var mask = np.isnan(a);
            // NumPy checks `mask is not None and mask.size` — an empty array skips the all-NaN
            // guard and lets argmax/argmin raise their own empty-sequence error. A NaN-free array
            // needs no replacement copy (argmax only reads, so returning `a` is observably identical).
            if (mask.size == 0 || !np.any(mask))
                return a;

            bool anyAllNanSlice = axis is int ax
                ? np.any(np.all(mask, ax))
                : np.all(mask);
            if (anyAllNanSlice)
                throw new ValueError("All-NaN slice encountered");

            // np.copyto(a_copy, val, where=mask): the double scalar casts into the operand's own
            // dtype (Half/float/double; Complex becomes (fill, 0) — probed identical to NumPy).
            var replaced = a.copy();
            np.copyto(replaced, NDArray.Scalar(fill), where: mask);
            return replaced;
        }
    }
}
