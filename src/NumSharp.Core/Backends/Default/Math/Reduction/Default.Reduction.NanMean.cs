using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
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
        internal static NDArray NanMeanNumPy(NDArray a, int? axis_, bool keepdims, DType dtype)
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
                var mask = np.isnan(a);
                var arr0 = new NDArray(inT, shape.KeepOrder(), false);
                np.copyto(arr0, a);
                ZeroWhereMask(arr0, mask);
                cnt = engine.ReduceAdd(np.logical_not(mask), axis, keepdims, np.int64, @out: null);
                tot = engine.ReduceAdd(arr0, axis, keepdims, dtype, @out: null);
            }
            else
            {
                var ae = NDExpr.Arr(a);
                var notNaN = NDExpr.LogicalNot(NDExpr.IsNaN(ae));
                cnt = np.evaluate(flat
                    ? (keepdims ? NDExpr.CountNonzero(notNaN, keepdims: true) : NDExpr.CountNonzero(notNaN))
                    : NDExpr.CountNonzero(notNaN, axis.Value, keepdims));
                var zeroed = NDExpr.Where(NDExpr.IsNaN(ae), NDExpr.Const(0), ae);
                if (stream)
                {
                    tot = np.evaluate(flat
                        ? (keepdims ? NDExpr.Sum(zeroed, keepdims: true) : NDExpr.Sum(zeroed))
                        : NDExpr.Sum(zeroed, axis.Value, keepdims));
                }
                else
                {
                    var arr0 = new NDArray(inT, shape.KeepOrder(), false);
                    np.evaluate(zeroed, @out: arr0);
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
