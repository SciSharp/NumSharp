using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.NanVar"/>: NumPy's <c>nanvar</c>, run by
        ///     <see cref="NanVarStdNumPy"/>.
        /// </summary>
        /// <param name="a">The array to reduce, any layout.</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for a flat reduction.</param>
        /// <param name="dtype">The loop of both sums (NumPy's <c>dtype=</c>), or null for the input's dtype.</param>
        /// <param name="ddof">Delta degrees of freedom.</param>
        /// <param name="keepdims">Keep the reduced axis as a length-1 dimension.</param>
        /// <returns>The variance of the non-NaN elements (see <see cref="NanVarStdNumPy"/>).</returns>
        /// <exception cref="TypeError">An inexact input with a non-inexact <paramref name="dtype"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> outside the operand.</exception>
        /// <remarks>
        ///     A public entry point since the np-layer dtype plan's phase PE (E1), so it owns its scope: the count, the
        ///     NaN-zeroed copy, the deviations and the sums are reclaimed here even when no scoped facade called it.
        /// </remarks>
        [NDScoped] // every intermediate of nanvar is this call's own; the variance is yielded
        public override NDArray NanVar(NDArray a, int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false)
            => NanVarStdNumPy(a, axis, keepdims, ddof, dtype, takeSqrt: false);

        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.NanStd"/>: NumPy's <c>nanstd</c>, the square root of
        ///     <c>nanvar</c>, run by <see cref="NanVarStdNumPy"/>.
        /// </summary>
        /// <param name="a">The array to reduce, any layout.</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for a flat reduction.</param>
        /// <param name="dtype">The loop of both sums (NumPy's <c>dtype=</c>), or null for the input's dtype.</param>
        /// <param name="ddof">Delta degrees of freedom.</param>
        /// <param name="keepdims">Keep the reduced axis as a length-1 dimension.</param>
        /// <returns>The standard deviation of the non-NaN elements (see <see cref="NanVarStdNumPy"/>).</returns>
        /// <exception cref="TypeError">An inexact input with a non-inexact <paramref name="dtype"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> outside the operand.</exception>
        /// <remarks>A public entry point since the np-layer dtype plan's phase PE (E1), so it owns its scope.</remarks>
        [NDScoped] // every intermediate of nanstd is this call's own; the deviation is yielded
        public override NDArray NanStd(NDArray a, int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false)
            => NanVarStdNumPy(a, axis, keepdims, ddof, dtype, takeSqrt: true);

        /// <summary>
        ///     NumPy 2.4.2's <c>nanvar</c> / <c>nanstd</c> (<c>numpy/lib/_nanfunctions_impl.py</c>) statement for
        ///     statement, the engine behind <see cref="np.nanvar"/> and <see cref="np.nanstd"/>. For an inexact input NumPy
        ///     runs, in this order:
        ///     <code>
        ///     arr, mask = _replace_nan(a, 0)                    # a copy in K order, NaN -> 0
        ///     cnt = np.sum(~mask, axis, dtype=intp, keepdims=True)
        ///     avg = np.sum(arr, axis, dtype, keepdims=True); avg = divide(avg, cnt, out=avg, casting='unsafe')
        ///     np.subtract(arr, avg, out=arr, casting='unsafe'); arr = _copyto(arr, 0, mask)
        ///     sqr = np.multiply(arr, arr, out=arr)              # complex: np.multiply(arr, arr.conj(), out=arr).real
        ///     var = np.sum(sqr, axis, dtype, keepdims)
        ///     dof = cnt (squeezed) - ddof; var = divide(var, dof, out=var, casting='unsafe')
        ///     var[dof &lt;= 0] = nan                             # nanstd: sqrt(var, out=var)
        ///     </code>
        ///     A non-inexact input has no NaN and no mask: NumPy returns <c>np.var</c> (nanstd its square root, which is
        ///     <c>np.std</c>), and so does this method.
        /// </summary>
        /// <param name="a">The array to reduce, any layout.</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for a flat reduction.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is the non-NaN count minus <paramref name="ddof"/>.</param>
        /// <param name="dtype">The loop of both sums (NumPy's <c>dtype=</c>), or null for the input's dtype. Must be inexact
        /// for an inexact input.</param>
        /// <param name="takeSqrt">True for nanstd: the square root of the variance, as <c>nanstd</c> takes it.</param>
        /// <returns>The variance (or standard deviation): the loop dtype, float64 for a complex input without
        /// <paramref name="dtype"/>; 0-d (NumPy's scalar) for a flat reduction without keepdims.</returns>
        /// <exception cref="TypeError">An inexact input with a non-inexact <paramref name="dtype"/>: "If a is inexact, then
        /// dtype must be inexact" (raised before the axis is checked, as in NumPy).</exception>
        /// <exception cref="AxisError"><paramref name="axis_"/> outside the operand (a 0-d operand included).</exception>
        /// <remarks>
        ///     <para>
        ///     How NumPy's statements decide the bits, all reproduced:
        ///     </para>
        ///     <list type="bullet">
        ///     <item>Both sums run over NumPy's K-order copy (<see cref="Shape.KeepOrder"/>, positive strides), so their
        ///     schedule — one pairwise call per run, slab order along a non-innermost axis, buffered fills of a cast loop —
        ///     is <see cref="ReduceAdd"/>'s over an array of that layout.</item>
        ///     <item>The deviation is computed in <c>result_type(input, loop)</c> and rounded into the INPUT's dtype
        ///     (<c>out=arr</c>), then squared in the input's dtype: with <c>dtype=float64</c> a float32 input still squares
        ///     float32 deviations (np.var squares float64 ones).</item>
        ///     <item>A complex square is the real part of NumPy's complex multiply (<c>simd_cmul</c>: one fused rounding of
        ///     <c>re*re + round(im*im)</c>), not np.var's <c>re² + im²</c>.</item>
        ///     <item>Both divisions take the float64 loop for a real dtype (<c>cnt</c> is an int64 array), the complex loop
        ///     (Smith) for a complex one, then round once into the result's dtype.</item>
        ///     <item><c>dof &lt;= 0</c> (an all-NaN slice, an empty input, <c>ddof</c> at or past the count) gives NaN — NumPy
        ///     overwrites whatever the division made (np.var gives +inf there) — as the positive quiet NaN of
        ///     <c>np.nan</c>.</item>
        ///     </list>
        ///     <para>
        ///     Engines: the count of non-NaN elements is one streamed bool fold (order-free integers). When the input is one
        ///     memory run (flat) or C/F-contiguous (an axis), the loop is the input's dtype and dynamic code exists, the
        ///     mean's sum streams over <c>a</c> with its NaNs zeroed (no copy: a dense <c>a</c> lies in the K order), and a
        ///     flat variance sum streams the squared deviations as well — three reads of <c>a</c>, nothing written. Otherwise
        ///     the K-order copy is materialized and summed by <see cref="ReduceAdd"/>, and the squared deviations are
        ///     written by one fused <c>np.evaluate</c> pass into a K-order buffer that <see cref="ReduceAdd"/> sums. float16
        ///     (whose <c>np.evaluate</c> arithmetic is scalar) and hosts without dynamic code take NumPy's own eager
        ///     statements instead — the same HALF loops, so the same bits.
        ///     </para>
        /// </remarks>
        internal NDArray NanVarStdNumPy(NDArray a, int? axis_, bool keepdims, int ddof, DType dtype, bool takeSqrt)
        {
            // Text is not inexact, so _replace_nan leaves it alone and nanvar / nanstd ARE np.var / np.std: the delegation
            // below reaches VarStdNumPy's guard (the axis, then add.reduce's text, the same for both).
            var shape = a.Shape;

            // The uninitialized-shape sentinel (not a size-0 array) has nothing to reduce; it passes through as before.
            if (shape.IsEmpty)
                return a;

            // _replace_nan(a, 0): mask is None unless a is inexact, and then nanvar IS np.var with the same keywords —
            // nanstd takes the square root of that variance with np.std's own statements, so it IS np.std. Decimal holds no
            // NaN (its mask would be all false); it takes np.var's route, whose DivideByZeroException at ddof >= count
            // stands in for the NaN a decimal cannot hold.
            var engine = a.TensorEngine;
            NPTypeCode inT = a.GetTypeCode;
            if (inT is not (NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex))
                return takeSqrt
                    ? engine.ReduceStd(a, axis_, keepdims, ddof, dtype)
                    : engine.ReduceVar(a, axis_, keepdims, ddof, dtype);

            // `if dtype is not None and not issubclass(dtype.type, np.inexact)` — checked before anything reads the axis,
            // so this TypeError wins over an AxisError. NumSharp's hierarchy counts Decimal as floating (inexact).
            if (dtype is not null && !(dtype.GetTypeCode().IsFloatingPoint() || dtype.GetTypeCode() == NPTypeCode.Complex))
                throw new TypeError("If a is inexact, then dtype must be inexact");

            // cnt = np.sum(~mask, axis=axis, ...) is the first statement that reads the axis, so ufunc.reduce's rule
            // applies: on a 0-d operand axis 0 and -1 mean "every axis" (PyArray_ConvertMultiAxis) — a flat reduction,
            // keepdims or not (probed: np.nanvar(np.array(2.5), axis=-1) is 0.0) — and any other axis is AxisError. (np.var
            // raises for both: its _count_reduce_items normalizes the axis first.)
            int? axis = null;
            if (axis_ is not null && !(shape.NDim == 0 && (axis_.Value == 0 || axis_.Value == -1)))
                axis = NormalizeAxis(axis_.Value, shape.NDim);
            bool flat = axis is null;

            // NumPy's dtypes: arr is the input's T; avg = np.sum(arr, dtype) is the loop L (T without dtype=); sqr is T,
            // or float64 for a complex input (.real of the complex product); var = np.sum(sqr, dtype) is L, or sqr's.
            NPTypeCode avgT = UfuncLoopCode(dtype) ?? inT;
            NPTypeCode sqT = inT == NPTypeCode.Complex ? NPTypeCode.Double : inT;
            NPTypeCode varT = UfuncLoopCode(dtype) ?? sqT;

            // float16 runs np.evaluate's scalar half arithmetic (~8 ns an element) where the eager SIMD half kernels run the
            // same HALF loops; a host without dynamic code has no fused kernels at all.
            bool eager = !DirectILKernelGenerator.Enabled || inT == NPTypeCode.Half;

            // np.array(a, copy=True): PyArray_NewLikeArray's KEEPORDER layout — the order both of NumPy's sums walk.
            var kl = shape.KeepOrder();

            NDArray mask = null;
            NDArray arr0 = null;
            NDArray cnt;
            NDArray avg;
            var ae = NDExpr.Arr(a);
            var zeroed = NDExpr.Where(NDExpr.IsNaN(ae), NDExpr.Const(0), ae);

            // A dense operand already lies in the K order, so a sum streamed over it walks NumPy's copy: a flat reduction
            // over one memory run (StreamsAsOneRun), an axis reduction over a C- or F-contiguous operand (np.evaluate's
            // exact axis streams). A cast loop's buffered fills are ReduceAdd's to reproduce, so the loop must be T.
            bool dense = flat ? StreamsAsOneRun(a) : shape.IsContiguous || shape.IsFContiguous;
            bool stream = !eager && dense && avgT == inT;

            if (eager)
            {
                // NumPy's own statements: the mask, the copy with zeros, the count of ~mask.
                mask = engine.IsNan(a);
                arr0 = NewArray(inT, kl, false);
                UnmanagedStorage.Copyto(arr0, a, NPY_CASTING.NPY_SAME_KIND_CASTING);
                ZeroWhereMask(arr0, mask);
                var notMask = engine.LogicalNot(mask);
                cnt = engine.ReduceAdd(notMask, axis, keepdims: true, dtype: np.int64, @out: null);
                avg = engine.ReduceAdd(arr0, axis, keepdims: true, dtype: dtype, @out: null);
            }
            else
            {
                // The non-NaN count per output: an order-free integer fold, streamed over a in any layout.
                var notNaN = NDExpr.LogicalNot(NDExpr.IsNaN(ae));
                cnt = engine.Evaluate(flat ? NDExpr.CountNonzero(notNaN, keepdims: true) : NDExpr.CountNonzero(notNaN, axis.Value, keepdims: true));
                if (stream)
                {
                    avg = engine.Evaluate(flat ? NDExpr.Sum(zeroed, keepdims: true) : NDExpr.Sum(zeroed, axis.Value, keepdims: true));
                }
                else
                {
                    arr0 = NewArray(inT, kl, false);
                    engine.Evaluate(zeroed, @out: arr0);
                    avg = engine.ReduceAdd(arr0, axis, keepdims: true, dtype: dtype, @out: null);
                }
            }

            // avg = _divide_by_count(avg, cnt): in place, like NumPy's out=avg. A 0-d sum can come back read-only (the
            // reduction scalar); 0-d has no layout to keep, so a copy is exact.
            if (!avg.Shape.IsWriteable)
                avg = avg.copy();
            DivideByCountsLikeNumPy(avg, cnt);

            // A flat reduction hoists the mean as a 0-d PARAMETER of the fused passes (read once); an axis reduction
            // broadcasts the keepdims mean.
            NDArray avgOperand = flat ? avg.reshape(Shape.NewScalar()) : avg;

            NDArray var;
            if (stream && flat && varT == sqT)
            {
                // var = np.sum(sqr) with sqr NumPy's fresh K-order array: one memory run summed as one pairwise call, which
                // np.evaluate's flat Sum streams over the fused child — the squared deviations never reach memory.
                var sq = SquaredNanDeviationExpr(a, avgOperand, inT);
                var = engine.Evaluate(keepdims ? NDExpr.Sum(sq, keepdims: true) : NDExpr.Sum(sq));
            }
            else if (eager)
            {
                var = EagerNanDeviationSum(arr0, avg, mask, axis, keepdims, dtype, inT, engine);
            }
            else
            {
                // The squared deviations in NumPy's K-order layout, then NumPy's sum over them. A real input reuses the
                // copy's buffer (same dtype and layout); a complex one writes float64 squares.
                var x2 = arr0 is not null && sqT == inT ? arr0 : NewArray(sqT, kl, false);
                engine.Evaluate(SquaredNanDeviationExpr(a, avgOperand, inT), @out: x2);
                var = engine.ReduceAdd(x2, axis, keepdims, dtype, @out: null);
            }

            // dof = cnt - ddof, cnt squeezed to var's shape when the reduced axes were dropped (NumPy's
            // cnt.squeeze(axis)); then var = _divide_by_count(var, dof), in place.
            bool scalarResult = var.ndim == 0;
            if (!var.Shape.IsWriteable)
                var = var.copy();
            var cntOut = keepdims ? cnt : cnt.reshape(Shape.Adopt((long[])var.Shape.dimensions.Clone()));
            var dof = engine.Subtract(cntOut, NewScalar((long)ddof));
            DivideByCountsLikeNumPy(var, dof);

            // isbad = dof <= 0: NaN, inf or a negative quotient are all replaced by np.nan — the POSITIVE quiet NaN
            // (.NET's double.NaN is the negative one), cast into var's dtype.
            var isbad = engine.LessEqual(dof, NewScalar(0L));
            if (engine.Any(isbad))
                UnmanagedStorage.Copyto(var, NewScalar(BitConverter.Int64BitsToDouble(0x7FF8000000000000L)), NPY_CASTING.NPY_UNSAFE_CASTING, isbad);

            if (takeSqrt)
                SqrtLikeNumPy(var, scalarResult);
            return scalarResult ? var.MarkReductionScalar() : var;
        }

        /// <summary>
        ///     NumPy's squared deviation of each element, as one fused expression over the input: <c>arr - avg</c> computed
        ///     in <c>result_type(input, avg)</c> and rounded into the input's dtype (<c>out=arr</c>), squared in that dtype
        ///     (a complex input: the real part of the complex product with its conjugate), and 0 where the input is NaN
        ///     (NumPy zeroes the copy there before it squares, so the sum adds +0.0 at those positions).
        /// </summary>
        /// <param name="a">The input (its NaN positions are the mask).</param>
        /// <param name="avg">The mean: 0-d (a hoisted parameter) or the keepdims mean that broadcasts.</param>
        /// <param name="inT">The input's dtype: Single, Double or Complex.</param>
        /// <returns>The expression; float64 for a complex input, the input's dtype otherwise.</returns>
        private static NDExpr SquaredNanDeviationExpr(NDArray a, NDArray avg, NPTypeCode inT)
        {
            var ae = NDExpr.Arr(a);
            var d = NDExpr.Cast(ae - NDExpr.Arr(avg), inT);
            var sq = inT == NPTypeCode.Complex ? NDExpr.Real(d * NDExpr.Conj(d)) : NDExpr.Square(d);
            return NDExpr.Where(NDExpr.IsNaN(ae), NDExpr.Const(0), sq);
        }

        /// <summary>
        ///     NumPy's deviation statements with the eager ufuncs, on the K-order copy in place: <c>subtract(arr, avg,
        ///     out=arr, casting='unsafe')</c>, <c>copyto(arr, 0, where=mask)</c>, the square (<c>multiply(arr, arr,
        ///     out=arr)</c>, or <c>multiply(arr, conj(arr), out=arr).real</c> for complex), then <c>np.sum(sqr, axis, dtype,
        ///     keepdims)</c>. Used for float16 (the SIMD half kernels are NumPy's HALF loops) and on hosts without dynamic
        ///     code.
        /// </summary>
        /// <param name="arr0">The K-order copy with its NaNs zeroed; overwritten with the squares (the real input case).</param>
        /// <param name="avg">The keepdims mean (any inexact dtype).</param>
        /// <param name="mask">The input's NaN mask (bool, the input's shape).</param>
        /// <param name="axis">The normalized axis, or null for a flat reduction.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="dtype">The sum's loop, or null.</param>
        /// <param name="inT">The input's dtype.</param>
        /// <param name="engine">The engine whose <see cref="ReduceAdd"/> sums the squares.</param>
        /// <returns>The sum of the squared deviations (NumPy's <c>var</c> before the division).</returns>
        private static NDArray EagerNanDeviationSum(NDArray arr0, NDArray avg, NDArray mask, int? axis, bool keepdims,
            DType dtype, NPTypeCode inT, TensorEngine engine)
        {
            // subtract(arr, avg, out=arr, casting='unsafe'): the loop is result_type(arr, avg); its result is cast into arr.
            // NumSharp's out= checks same_kind, so a complex loop over a real copy (a complex dtype=) goes through a
            // temporary and NumPy's unsafe cast (the real part).
            var loopT = DTypePromotion.ResultType(arr0.dtype, avg.dtype);
            if (DTypeCasting.CanCastTypeTo(loopT, arr0.dtype, NPY_CASTING.NPY_SAME_KIND_CASTING))
                engine.Subtract(arr0, avg, null, arr0);
            else
                UnmanagedStorage.Copyto(arr0, engine.Subtract(arr0, avg), NPY_CASTING.NPY_UNSAFE_CASTING);
            ZeroWhereMask(arr0, mask);

            NDArray sqr;
            if (inT == NPTypeCode.Complex)
            {
                engine.Multiply(arr0, engine.Conjugate(arr0), null, arr0);
                // np.real of the complex copy: a view of its real lane.
                sqr = new NDArray(arr0.Storage.AliasComplexLane(imaginary: false));
            }
            else
            {
                engine.Multiply(arr0, arr0, null, arr0);
                sqr = arr0;
            }
            return engine.ReduceAdd(sqr, axis, keepdims, dtype, @out: null);
        }

        /// <summary>
        ///     NumPy's <c>copyto(arr, 0, where=mask)</c> (<c>_replace_nan</c>'s and nanvar's zeroing of the NaN positions) for
        ///     a <paramref name="mask"/> of <paramref name="arr"/>'s shape: every masked element becomes +0.0 (0 cast to the
        ///     dtype), the others keep their value.
        /// </summary>
        /// <param name="arr">The array to zero in place; writeable, any layout.</param>
        /// <param name="mask">The bool mask, the same shape as <paramref name="arr"/>.</param>
        /// <remarks>
        ///     NumPy's own statement (<c>_copyto(arr, 0, mask)</c>: a weak Python 0, <c>casting='unsafe'</c>). Since the
        ///     masked copy is a bit-copy kernel (plan X11, 2026-10-06) it is also the fastest route: 3.7 µs over a float16
        ///     <c>(250, 400)</c> with every 7th element masked, where <c>np.putmask</c> with one scalar (the route A7 took
        ///     while the masked copy cost 253 µs) takes 22.7 µs. The masked kernel has a C# fallback, so a host without dynamic
        ///     code runs the same statement.
        ///     <para>It calls the array object's assignment of a weak Python int (<see cref="UnmanagedStorage.CopytoPythonScalar"/>,
        ///     what np.copyto runs for a C# int) rather than the public np.copyto; <see cref="NDAnchoredAttribute"/> keeps the
        ///     two operands rooted while the masked kernel reads them, as np.copyto's own attribute did.</para>
        /// </remarks>
        [NDAnchored]
        private static void ZeroWhereMask(NDArray arr, NDArray mask)
            => UnmanagedStorage.CopytoPythonScalar(arr, 0, PyScalarKind.Int, NPY_CASTING.NPY_UNSAFE_CASTING, mask);

        /// <summary>
        ///     NumPy's <c>_divide_by_count(a, b)</c> for an array divisor: <c>divide(a, b, out=a, casting='unsafe')</c> (and
        ///     <c>a.dtype.type(a / b)</c> for a scalar, the same arithmetic), in place. <paramref name="counts"/> is int64, so
        ///     every real dtype divides in the float64 loop and rounds once into <paramref name="a"/>'s dtype; complex
        ///     divides in the complex loop by <c>count + 0j</c> (Smith's algorithm); Decimal (NumSharp's) in decimal.
        /// </summary>
        /// <param name="a">The dividend; overwritten with the quotient. Writeable, an inexact dtype.</param>
        /// <param name="counts">The int64 divisor, broadcastable to <paramref name="a"/>; 0 gives NumPy's NaN / ±inf.</param>
        private static void DivideByCountsLikeNumPy(NDArray a, NDArray counts)
        {
            switch (a.GetTypeCode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                    // The float64 loop, then a same_kind cast into the narrower float (one rounding from double).
                    a.TensorEngine.Divide(a, counts, np.float64, a);
                    break;
                default:
                    a.TensorEngine.Divide(a, counts, null, a);
                    break;
            }
        }
    }
}
