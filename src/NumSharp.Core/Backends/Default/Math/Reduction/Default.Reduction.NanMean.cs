using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.NanMean"/>: NumPy's <c>nanmean</c>, run by
        ///     <see cref="NanMeanNumPy"/>.
        /// </summary>
        /// <param name="a">The array to reduce, any layout.</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for a flat reduction.</param>
        /// <param name="dtype">The loop of the sum (NumPy's <c>dtype=</c>), or null for the input's dtype.</param>
        /// <param name="keepdims">Keep the reduced axis as a length-1 dimension.</param>
        /// <param name="out">NumPy's <c>out=</c> (returned), or null.</param>
        /// <param name="where">NumPy's <c>where=</c> mask, or null.</param>
        /// <returns>The mean of the non-NaN elements (see <see cref="NanMeanNumPy"/>); <paramref name="out"/> itself when
        ///     given.</returns>
        /// <exception cref="TypeError">An inexact input with a non-inexact <paramref name="dtype"/> or
        ///     <paramref name="out"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> outside the operand.</exception>
        /// <remarks>
        ///     A public entry point since the np-layer dtype plan's phase PE (E1), so it owns its scope: the count, the
        ///     NaN-zeroed copy and the sums are reclaimed here even when no scoped facade called it.
        /// </remarks>
        [NDScoped] // the count, the K-order copy and the sum before the division are this call's own; the mean is yielded
        public override NDArray NanMean(NDArray a, int? axis = null, DType dtype = null, bool keepdims = false,
            NDArray @out = null, NDArray @where = null)
            // out= / where= run NumPy's statements as written (NanMeanKeywords); the keyword-free call keeps the fused route.
            => @out is null && @where is null ? NanMeanNumPy(a, axis, keepdims, dtype) : NanMeanKeywords(a, axis, keepdims, dtype, @out, @where);

        /// <summary>
        ///     NumPy's <c>nanmean</c> with <c>out=</c> / <c>where=</c>, statement for statement:
        ///     <code>
        ///     arr, mask = _replace_nan(a, 0)
        ///     if mask is None: return np.mean(arr, axis, dtype, out, keepdims, where)
        ///     (the dtype / out inexact checks)
        ///     cnt = np.sum(~mask, axis, dtype=np.intp, keepdims=keepdims, where=where)
        ///     tot = np.sum(arr, axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
        ///     avg = _divide_by_count(tot, cnt, out=out)
        ///     </code>
        /// </summary>
        /// <param name="a">The array to reduce, any layout.</param>
        /// <param name="axis">The reduced axis, or null for a flat reduction (np.sum's axis rules).</param>
        /// <param name="keepdims">Keep the reduced axis as a length-1 dimension.</param>
        /// <param name="dtype">The loop of the sum, or null.</param>
        /// <param name="out">The sum's and the division's destination (returned), or null.</param>
        /// <param name="where">The bool mask both sums read, or null.</param>
        /// <returns>The mean; <paramref name="out"/> itself when given; a 0-d scalar for a flat reduction without
        ///     <paramref name="keepdims"/> and <paramref name="out"/>.</returns>
        /// <exception cref="TypeError">A non-inexact <paramref name="dtype"/> or <paramref name="out"/> for an inexact
        ///     input (NumPy's texts, before the axis is read).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> outside the operand (the count's np.sum).</exception>
        /// <remarks>A slice without a non-NaN element is 0 / 0: NaN, which NumPy keeps (it only warns). The scope reclaims
        ///     the mask, the copy and the count; the mean is yielded.</remarks>
        [NDScoped]
        private NDArray NanMeanKeywords(NDArray a, int? axis, bool keepdims, DType dtype, NDArray @out, NDArray @where)
        {
            var engine = a.TensorEngine;
            NPTypeCode inT = a.GetTypeCode;
            // _replace_nan's mask is None unless a is inexact, and then nanmean IS np.mean with every keyword.
            if (inT is not (NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex))
                return engine.ReduceMean(a, axis, keepdims, dtype, @out, @where);

            // The two inexact checks, in NumPy's order, before anything reads the axis.
            if (dtype is not null && !(dtype.GetTypeCode().IsFloatingPoint() || dtype.GetTypeCode() == NPTypeCode.Complex))
                throw new TypeError("If a is inexact, then dtype must be inexact");
            if (@out is not null && !(@out.GetTypeCode.IsFloatingPoint() || @out.GetTypeCode == NPTypeCode.Complex))
                throw new TypeError("If a is inexact, then out must be inexact");

            // arr = the K-order copy with zeros at the NaNs.
            var mask = engine.IsNan(a);
            var arr0 = NewArray(inT, a.Shape.KeepOrder(), false);
            UnmanagedStorage.Copyto(arr0, a, NPY_CASTING.NPY_SAME_KIND_CASTING);
            ZeroWhereMask(arr0, mask);

            var cnt = engine.ReduceAdd(engine.LogicalNot(mask), axis, keepdims, np.int64, null, null, @where);
            var tot = engine.ReduceAdd(arr0, axis, keepdims, dtype, @out, null, @where);

            // avg = _divide_by_count(tot, cnt, out=out): in place into tot, which IS out when it was given. A 0-d sum can come
            // back read-only (the reduction scalar); 0-d has no layout to keep, so a copy is exact.
            bool scalarResult = @out is null && tot.ndim == 0;
            if (!tot.Shape.IsWriteable)
                tot = tot.copy();
            DivideByCountsLikeNumPy(tot, cnt);
            return scalarResult ? tot.MarkReductionScalar() : tot;
        }

        /// <summary>
        ///     NumPy 2.4.2's <c>nanmean</c> (<c>numpy/lib/_nanfunctions_impl.py</c>) statement for statement, the engine
        ///     behind <see cref="np.nanmean"/>. For an inexact input NumPy runs, in this order:
        ///     <code>
        ///     arr, mask = _replace_nan(a, 0)                    # a copy in K order, NaN -> 0
        ///     cnt = np.sum(~mask, axis, dtype=intp, keepdims=keepdims)
        ///     tot = np.sum(arr, axis, dtype, keepdims=keepdims)
        ///     avg = _divide_by_count(tot, cnt)                  # divide(tot, cnt, out=tot, casting='unsafe')
        ///     </code>
        ///     A non-inexact input has no NaN and no mask: NumPy returns <c>np.mean</c>, and so does this method.
        /// </summary>
        /// <param name="a">The array to reduce, any layout.</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for a flat reduction.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="dtype">The loop of the sum (NumPy's <c>dtype=</c>), or null for the input's dtype. Must be inexact
        /// for an inexact input.</param>
        /// <returns>The mean of the non-NaN elements, in the sum's dtype; NaN for a slice without one (the division's
        /// <c>0 / 0</c>, which NumPy does not overwrite); 0-d (NumPy's scalar) for a flat reduction without keepdims.</returns>
        /// <exception cref="TypeError">An inexact input with a non-inexact <paramref name="dtype"/>: "If a is inexact, then
        /// dtype must be inexact" (raised before the axis is checked, as in NumPy).</exception>
        /// <exception cref="AxisError"><paramref name="axis_"/> outside the operand (except 0 / -1 on a 0-d inexact
        /// operand, which reduce flat: ufunc.reduce's rule, since nanmean's first axis consumer is np.sum).</exception>
        /// <remarks>
        ///     <para>
        ///     The bits follow NumPy's statements: the sum runs over NumPy's K-order copy (<see cref="Shape.KeepOrder"/>)
        ///     in its own schedule (a float16 / float32 input sums in its own dtype, never in double); the quotient takes the
        ///     float64 loop for a real dtype (the count is int64) and Smith's complex loop (<c>count + 0j</c>) for complex,
        ///     so an infinite part turns the other part into NaN as NumPy's does.
        ///     </para>
        ///     <para>
        ///     Engines, as <see cref="NanVarStdNumPy"/>: the count streams over the input; a dense input (one memory run flat,
        ///     C/F along an axis) without a cast streams its NaN-zeroed sum too; otherwise the K-order copy is materialized for
        ///     <see cref="ReduceAdd"/>. float16 and hosts without dynamic code take NumPy's eager statements.
        ///     </para>
        /// </remarks>
        internal NDArray NanMeanNumPy(NDArray a, int? axis_, bool keepdims, DType dtype)
        {
            // Text is not inexact, so _replace_nan leaves it alone and nanmean IS np.mean: the delegation below reaches
            // ReduceMean's guard, which validates the axis (a 0-d input refuses every axis) and refuses with add.reduce's
            // text — or, under a string dtype= request on an integer input, with the text of its REQUIRED_STR_LEN.
            var shape = a.Shape;

            // The uninitialized-shape sentinel (not a size-0 array) has nothing to reduce; it passes through as before.
            if (shape.IsEmpty)
                return a;

            // _replace_nan(a, 0): mask is None unless a is inexact, and then nanmean IS np.mean with the same keywords.
            // Decimal holds no NaN (its mask would be all false) and takes np.mean's route as well.
            var engine = a.TensorEngine;
            NPTypeCode inT = a.GetTypeCode;
            if (inT is not (NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex))
                return engine.ReduceMean(a, axis_, keepdims, dtype);

            // The dtype check runs before anything reads the axis, so this TypeError wins over an AxisError.
            if (dtype is not null && !(dtype.GetTypeCode().IsFloatingPoint() || dtype.GetTypeCode() == NPTypeCode.Complex))
                throw new TypeError("If a is inexact, then dtype must be inexact");

            // cnt = np.sum(~mask, axis, ...) reads the axis first: ufunc.reduce's rule (a 0-d operand takes 0 / -1 as every
            // axis; anything else outside the operand is AxisError).
            int? axis = null;
            if (axis_ is not null && !(shape.NDim == 0 && (axis_.Value == 0 || axis_.Value == -1)))
                axis = NormalizeAxis(axis_.Value, shape.NDim);
            bool flat = axis is null;

            NPTypeCode totT = UfuncLoopCode(dtype) ?? inT;
            // float16's np.evaluate arithmetic is scalar; a host without dynamic code has no fused kernels.
            bool eager = !DirectILKernelGenerator.Enabled || inT == NPTypeCode.Half;
            // A dense operand already lies in the K order, so a sum streamed over it walks NumPy's copy; a cast loop's
            // buffered fills are ReduceAdd's to reproduce.
            bool dense = flat ? StreamsAsOneRun(a) : shape.IsContiguous || shape.IsFContiguous;
            bool stream = !eager && dense && totT == inT;

            NDArray cnt;
            NDArray tot;
            if (eager)
            {
                // NumPy's own statements: the mask, the copy with zeros, the count of ~mask, the sum.
                var mask = engine.IsNan(a);
                var arr0 = NewArray(inT, shape.KeepOrder(), false);
                UnmanagedStorage.Copyto(arr0, a, NPY_CASTING.NPY_SAME_KIND_CASTING);
                ZeroWhereMask(arr0, mask);
                cnt = engine.ReduceAdd(engine.LogicalNot(mask), axis, keepdims, np.int64, @out: null);
                tot = engine.ReduceAdd(arr0, axis, keepdims, dtype, @out: null);
            }
            else
            {
                var ae = NDExpr.Arr(a);
                var notNaN = NDExpr.LogicalNot(NDExpr.IsNaN(ae));
                cnt = engine.Evaluate(flat
                    ? (keepdims ? NDExpr.CountNonzero(notNaN, keepdims: true) : NDExpr.CountNonzero(notNaN))
                    : NDExpr.CountNonzero(notNaN, axis.Value, keepdims));
                var zeroed = NDExpr.Where(NDExpr.IsNaN(ae), NDExpr.Const(0), ae);
                if (stream)
                {
                    tot = engine.Evaluate(flat
                        ? (keepdims ? NDExpr.Sum(zeroed, keepdims: true) : NDExpr.Sum(zeroed))
                        : NDExpr.Sum(zeroed, axis.Value, keepdims));
                }
                else
                {
                    var arr0 = NewArray(inT, shape.KeepOrder(), false);
                    engine.Evaluate(zeroed, @out: arr0);
                    tot = engine.ReduceAdd(arr0, axis, keepdims, dtype, @out: null);
                }
            }

            // avg = _divide_by_count(tot, cnt), in place like NumPy's out=tot. A 0-d sum can come back read-only (the
            // reduction scalar); 0-d has no layout to keep, so a copy is exact. A slice without a non-NaN element is
            // 0 / 0: NaN, which NumPy keeps (only warns).
            bool scalarResult = tot.ndim == 0;
            if (!tot.Shape.IsWriteable)
                tot = tot.copy();
            DivideByCountsLikeNumPy(tot, cnt);
            return scalarResult ? tot.MarkReductionScalar() : tot;
        }
    }
}
