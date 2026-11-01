using System;
using NumSharp.Backends;

namespace NumSharp
{
    // =============================================================================
    // np.var: ONE overload, NumPy's whole signature
    //     var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>, mean=<no value>,
    //         correction=<no value>)
    // (np-layer dtype plan, P10 item 1, decision D10; plan docs/plans/api-parameter-parity.md, item B4). The engine member is
    // TensorEngine.ReduceVar (NumPy's _methods._var plus fromnumeric's correction= rule), which takes every keyword.
    //
    // NumSharp's earlier forms — (a, keepdims, ddof, dtype) and (a, int axis, keepdims, ddof, dtype) — are gone (decision
    // D19). ddof is an int defaulting to 0 (it was a nullable int); NumPy also takes a float ddof / correction, which C#'s
    // int does not.
    // =============================================================================

    public static partial class np
    {
        /// <summary>
        ///     Compute the variance along the specified axis — NumPy's <c>np.var(a, axis=None, dtype=None, out=None, ddof=0,
        ///     keepdims=&lt;no value&gt;, *, where=&lt;no value&gt;, mean=&lt;no value&gt;, correction=&lt;no value&gt;)</c>
        ///     (<c>_methods._var</c>): the mean of the squared deviations from the mean, divided by <c>max(N - ddof, 0)</c>.
        ///     This is the only overload: NumPy's whole signature, with NumPy's names, order and defaults.
        /// </summary>
        /// <param name="a">The values whose variance is desired.</param>
        /// <param name="axis">The axis to compute along (negative counts from the end); <c>null</c> uses every element. A 0-d
        ///     array takes no axis unless <paramref name="where"/> is a mask.</param>
        /// <param name="dtype">The loop of both sums. <c>null</c> is NumPy's default: float64 for bool and integer input, the
        ///     input's own dtype otherwise; an integer dtype truncates the mean, as in NumPy.</param>
        /// <param name="out">The array the variance is written into and returned (the result's shape). NumPy hands it to the
        ///     LAST sum only (whose loop it joins) and divides it in place. Footgun: the FOURTH positional parameter, as in
        ///     NumPy; write <c>ddof:</c> and <c>keepdims:</c> by name.</param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is <c>max(N - ddof, 0)</c>, so <c>ddof &gt;= N</c> gives
        ///     +inf (NaN when every deviation is 0), never a negative variance.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="where">A bool mask broadcast against <paramref name="a"/>: each output takes its selected elements
        ///     only (its own count, mean and divisor). The C# <c>true</c> literal is NumPy's default. <c>null</c> is no
        ///     mask.</param>
        /// <param name="mean">The mean to subtract instead of computing it: an array that broadcasts against
        ///     <paramref name="a"/> (the shape <c>np.mean(a, axis, keepdims: true)</c> has), or a C# number standing for a
        ///     Python scalar. Footgun: it is used as given, so the deviations take <c>a - mean</c>'s dtype (an int32 array
        ///     with an integer mean subtracts and squares in int32, wrapping), and a mean that broadcasts to a larger shape
        ///     makes the result larger. <c>null</c> computes it.</param>
        /// <param name="correction">The Array API's name of <paramref name="ddof"/>: with it given, <paramref name="ddof"/>
        ///     must be 0, and the variance divides by <c>max(N - correction, 0)</c>. <c>null</c> is none.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) when the
        ///     result has no dimensions.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">A nonzero <paramref name="ddof"/> together with <paramref name="correction"/>
        ///     (<c>ddof and correction can't be provided simultaneously.</c>, raised first); a read-only or misshapen
        ///     <paramref name="out"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/>, <paramref name="where"/> or
        ///     <paramref name="mean"/> does not broadcast against <paramref name="a"/>.</exception>
        /// <exception cref="TypeError">A bool input with <c>dtype: bool</c>, or a bool <paramref name="mean"/> of a bool
        ///     input (NumPy's boolean-subtract refusal); a C# scalar passed as <paramref name="out"/>; NumPy's text refusals;
        ///     a <c>numpy.matrix</c> operand with <paramref name="keepdims"/>, <paramref name="where"/> or
        ///     <paramref name="mean"/> (<c>matrix.var() got an unexpected keyword argument 'keepdims'</c>).</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain (non-matrix) <paramref name="out"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.var.html</remarks>
        public static NDArray var(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null, int ddof = 0,
            bool keepdims = false, NDArray @where = null, NDArray mean = null, int? correction = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (NDMatrixBoundary.Sees(a))
            {
                // fromnumeric folds correction= into ddof before it looks at the operand's class; then NumPy hands a
                // numpy.matrix to matrix.var(axis, dtype, out, ddof) with the other keywords the caller passed, which the
                // method does not take (the first one passed is its TypeError).
                int d = DefaultEngine.DdofWithCorrection(ddof, correction);
                if (keepdims)
                    throw NDMatrixBoundary.KeywordError("var", "keepdims");
                if (@where is not null)
                    throw NDMatrixBoundary.KeywordError("var", "where");
                if (mean is not null)
                    throw NDMatrixBoundary.KeywordError("var", "mean");
                if (@out is not null)
                    return NDMatrixBoundary.ReducedInto(a, @out, (axis, dtype, d),
                        static (m, o, s) => m.var(s.axis, s.dtype, o, s.d),
                        static (m, o, s) => m.TensorEngine.ReduceVar(m, s.axis, keepdims: true, s.d, s.dtype, o));
                // a row, a column or the scalar, as NumPy's matrix method returns
                return NDMatrixBoundary.Reduced((a, axis, dtype, d), static s => var(s.a, s.axis, s.dtype, ddof: s.d), axis, false, "var");
            }

            // The whole call is the engine's: TensorEngine.ReduceVar of a's engine takes every keyword, correction= first.
            return a.TensorEngine.ReduceVar(a, axis, keepdims, ddof, dtype, @out, @where, mean, correction);
        }
    }
}
