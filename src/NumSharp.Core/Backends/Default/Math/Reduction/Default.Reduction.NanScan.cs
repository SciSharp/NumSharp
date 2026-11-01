using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>np.nancumsum</c> body (<c>numpy/lib/_nanfunctions_impl.py</c>): <c>a, mask = _replace_nan(a, 0)</c>
        ///     then <c>np.cumsum(a, axis, dtype, out)</c>, with NumPy's text routes first. The facade applies the
        ///     <c>numpy.matrix</c> rule and calls this for everything else.
        /// </summary>
        /// <param name="a">The array (any dtype; text is not inexact, so it is np.cumsum's text: StringDType concatenates,
        ///     fixed-width text is refused or parsed under a numeric dtype=).</param>
        /// <param name="axis">The accumulated axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="dtype">The loop / result dtype, or null for NEP 50's accumulator of the input.</param>
        /// <param name="out">NumPy's <c>out=</c>, or null.</param>
        /// <returns><paramref name="out"/> itself when given, else a fresh array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A text operand or request NumPy's loop resolution refuses.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>, or text a numeric loop cannot parse.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <remarks>Moved from the public <c>np.nancumsum</c> by the np-layer dtype plan's P1b, statement for statement;
        ///     its <c>np.cumsum</c> call became the engine member it reaches for a plain operand,
        ///     <see cref="ReduceCumAdd"/>. The plan's phase PE (E1) made it DefaultEngine's override of
        ///     <see cref="TensorEngine.NanCumSum"/>; it was an internal static.</remarks>
        [NDScoped] // reclaims the full-size NaN-replaced copy; the scan result (or the caller's out) is yielded
        public override NDArray NanCumSum(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null)
        {
            // StringDType is not inexact: _replace_nan leaves it alone and nancumsum IS np.cumsum, which computes.
            if (Strings.NDStringUfuncs.VStringReduces("nancumsum", a, axis, dtype, @out))
                return Strings.NDVStringReduction.CumSum(a, axis, @out);
            // Fixed-width text, or a string dtype= request: NumPy's _replace_nan leaves text alone and its accumulate
            // validates the axis (a 0-d input ravels first), then resolves the loop on (out, a): a U input into a
            // StringDType out concatenates (np.cumsum's route), anything else is the verbatim refusal (see
            // NDStringUfuncs.ReduceError / ReduceOutError).
            switch (Strings.NDStringUfuncs.ReduceIntoOutRoute("nancumsum", a, axis, @out, dtype, out var resolution))
            {
                case Strings.NDStringUfuncs.TextOutReduce.Concatenate:
                    return Strings.NDVStringReduction.CumSumOfCast(a, axis, @out);
                case Strings.NDStringUfuncs.TextOutReduce.NumericCast:
                    // A numeric loop over a text operand (np.cumsum's route: _replace_nan leaves text alone).
                    return Strings.NDTextNumericReduction.Accumulate("add", a, axis, @out, resolution.Descriptors[0]);
            }
            // NumPy: a, mask = _replace_nan(a, 0); return np.cumsum(a, axis, dtype, out). The mask is discarded by nancumsum
            // (unlike nanmean/nanvar), so only the NaN-replaced copy is needed.
            NDArray replaced = ReplaceNanForScan(a, 0);
            return replaced.TensorEngine.ReduceCumAdd(replaced, axis, dtype, @out);
        }

        /// <summary>
        ///     NumPy's <c>np.nancumprod</c> body: <c>a, mask = _replace_nan(a, 1)</c> then <c>np.cumprod(a, axis, dtype,
        ///     out)</c>, with NumPy's text routes first. The facade applies the <c>numpy.matrix</c> rule and calls this for
        ///     everything else.
        /// </summary>
        /// <param name="a">The array (any dtype; text is not inexact, so a numeric dtype= parses it and a parsed 'nan' is
        ///     MULTIPLIED, not replaced by 1).</param>
        /// <param name="axis">The accumulated axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="dtype">The loop / result dtype, or null for NEP 50's accumulator of the input.</param>
        /// <param name="out">NumPy's <c>out=</c>, or null.</param>
        /// <returns><paramref name="out"/> itself when given, else a fresh array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A text operand or request NumPy's loop resolution refuses.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>, or text a numeric dtype cannot parse.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <remarks>Moved from the public <c>np.nancumprod</c> by the np-layer dtype plan's P1b, statement for statement;
        ///     its <c>np.cumprod</c> calls became <see cref="ReduceCumMul"/>, the member they reach for a plain
        ///     operand. The plan's phase PE (E1) made it DefaultEngine's override of <see cref="TensorEngine.NanCumProd"/>;
        ///     it was an internal static.</remarks>
        [NDScoped] // reclaims the NaN-replaced copy and a parse of text; the scan result (or the caller's out) is yielded
        public override NDArray NanCumProd(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null)
        {
            // Text, or a string dtype= request: NumPy's _replace_nan leaves text alone and its accumulate validates the
            // axis (a 0-d input ravels first), then refuses with its verbatim text; see NDStringUfuncs.ReduceError.
            // A numeric dtype= over text: NumPy's _replace_nan leaves text alone (it masks only inexact arrays), and cumprod's
            // cast to the requested number type parses it — so a parsed 'nan' is MULTIPLIED, not replaced by 1 (probed:
            // np.nancumprod(['inf', 'nan'], dtype='f8') is [inf, nan]). The scope reclaims the parsed array.
            // The loop resolution answers first (np.cumprod's order): a numeric loop over a text operand with out= casts
            // around the accumulation, without out= the parse below runs.
            if (Strings.NDStringUfuncs.ReduceIntoOutRoute("nancumprod", a, axis, @out, dtype, out var resolution)
                == Strings.NDStringUfuncs.TextOutReduce.NumericCast)
                return Strings.NDTextNumericReduction.Accumulate("multiply", a, axis, @out, resolution.Descriptors[0]);
            if (Strings.NDStringUfuncs.ParseForNumericReduce("nancumprod", a, axis, dtype) is { } parsed)
                return parsed.TensorEngine.ReduceCumMul(parsed, axis, dtype, @out);
            Strings.NDStringUfuncs.ThrowIfReduce("nancumprod", a, axis, @out, dtype);
            // NumPy: a, mask = _replace_nan(a, 1); return np.cumprod(a, axis, dtype, out).
            NDArray replaced = ReplaceNanForScan(a, 1);
            return replaced.TensorEngine.ReduceCumMul(replaced, axis, dtype, @out);
        }

        /// <summary>
        ///     NumPy's <c>_replace_nan(a, val)</c> for the NaN scans: a copy of an inexact array with every NaN replaced by
        ///     <paramref name="val"/> in its own dtype; any other array is returned itself (it holds no NaN).
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="val">The replacement (0 for nancumsum, 1 for nancumprod).</param>
        /// <returns>The NaN-replaced copy (float16 / float32 / float64 / complex128), or <paramref name="a"/>
        ///     itself.</returns>
        /// <remarks>Moved from <c>np._replace_nan_for_scan</c> by the np-layer dtype plan's P1b. The mask still goes
        ///     through <c>np.isnan</c>'s engine member and the select through <c>np.where</c>, a computational call into the
        ///     facade that the plan's P6 replaces with an engine select; the up-call ratchet pins it with that
        ///     reason.</remarks>
        [NDScopedCovered] // only callers: NanCumSum / NanCumProd, [NDScoped] (the isnan mask and the superseded fill are scope-reclaimed)
        private NDArray ReplaceNanForScan(NDArray a, int val)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    // Hot path: a fused single-pass SIMD fill for a C-contiguous float32/float64 operand
                    // (isnan + where would be two passes plus a bool-mask temp — see NanReplace). Every
                    // other inexact case (Half, Complex, or any non-C-contiguous view — F-order/strided/
                    // transposed/broadcast/reversed) takes the composed where(isnan(a), val, a) path, which
                    // is bit-identical and fully gated but reads logical order correctly for those layouts.
                    if ((a.typecode == NPTypeCode.Single || a.typecode == NPTypeCode.Double)
                        && a.Shape.IsContiguous && !a.Shape.IsBroadcasted)
                        return NanReplace.ReplaceContiguousFloat(a, val);

                    NDArray fill = NewScalar(val);
                    if (fill.typecode != a.typecode)
                        fill = fill.astype(a.typecode);   // 0/1 -> (Half)0/1, 0+0j/1+0j — dtype-preserving
                    return NDArray.where(a.TensorEngine.IsNan(a), fill, a);
                default:
                    return a;                             // integer / bool / decimal / char cannot be NaN
            }
        }
    }
}
