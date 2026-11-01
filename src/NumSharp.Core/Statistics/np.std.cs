using System;
using NumSharp.Backends;

namespace NumSharp
{
    // =============================================================================
    // np.std: ONE overload, NumPy's whole signature
    //     std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>, mean=<no value>,
    //         correction=<no value>)
    // (np-layer dtype plan, P10 item 1, decision D10; plan docs/plans/api-parameter-parity.md, item B4). The engine member is
    // TensorEngine.ReduceStd (NumPy's _methods._std plus fromnumeric's correction= rule), which takes every keyword.
    //
    // NumSharp's earlier forms — (a, keepdims, ddof, dtype) and (a, int axis, keepdims, ddof, dtype) — are gone (decision
    // D19). ddof is an int defaulting to 0 (it was a nullable int).
    // =============================================================================

    public static partial class np
    {
        /// <summary>
        ///     Compute the standard deviation along the specified axis — NumPy's <c>np.std(a, axis=None, dtype=None,
        ///     out=None, ddof=0, keepdims=&lt;no value&gt;, *, where=&lt;no value&gt;, mean=&lt;no value&gt;,
        ///     correction=&lt;no value&gt;)</c> (<c>_methods._std</c>): the square root of <see cref="var"/>. This is the only
        ///     overload: NumPy's whole signature, with NumPy's names, order and defaults.
        /// </summary>
        /// <param name="a">The values whose standard deviation is desired.</param>
        /// <param name="axis">The axis to compute along (negative counts from the end); <c>null</c> uses every element. A 0-d
        ///     array takes no axis unless <paramref name="where"/> is a mask.</param>
        /// <param name="dtype">The loop of both sums (see <see cref="var"/>). An integer or bool dtype with a result that has
        ///     dimensions is NumPy's sqrt output-cast error; a 0-d result truncates the root instead.</param>
        /// <param name="out">The FLOAT or complex array the result is written into and returned. NumPy runs the variance into
        ///     it and takes the square root in place (an integer out is refused after the variance is stored). Footgun: the
        ///     FOURTH positional parameter, as in NumPy; write <c>ddof:</c> and <c>keepdims:</c> by name.</param>
        /// <param name="ddof">Delta degrees of freedom: the variance divides by <c>max(N - ddof, 0)</c>.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="where">A bool mask of the elements each output takes (see <see cref="var"/>). The C# <c>true</c>
        ///     literal is NumPy's default; <c>null</c> is no mask.</param>
        /// <param name="mean">The mean to subtract instead of computing it, used as given (see <see cref="var"/>), or
        ///     <c>null</c>.</param>
        /// <param name="correction">The Array API's name of <paramref name="ddof"/> (see <see cref="var"/>), or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) when the
        ///     result has no dimensions.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">A nonzero <paramref name="ddof"/> together with <paramref name="correction"/> (raised
        ///     first); a read-only or misshapen <paramref name="out"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="dtype"/> with a result that has dimensions,
        ///     or an integer or bool <paramref name="out"/>: NumPy's <c>Cannot cast ufunc 'sqrt' output from
        ///     dtype('float64') to dtype('int64') with casting rule 'same_kind'</c>; a non-bool <paramref name="where"/>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/>, <paramref name="where"/> or
        ///     <paramref name="mean"/> does not broadcast against <paramref name="a"/>.</exception>
        /// <exception cref="TypeError">A boolean subtraction (see <see cref="var"/>); a C# scalar passed as
        ///     <paramref name="out"/>; NumPy's text refusals; a <c>numpy.matrix</c> operand with <paramref name="keepdims"/>,
        ///     <paramref name="where"/> or <paramref name="mean"/>.</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain (non-matrix) <paramref name="out"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.std.html</remarks>
        public static NDArray std(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null, int ddof = 0,
            bool keepdims = false, NDArray @where = null, NDArray mean = null, int? correction = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (NDMatrixBoundary.Sees(a))
            {
                // fromnumeric folds correction= into ddof first; then matrix.std(axis, dtype, out, ddof) refuses every other
                // keyword the caller passed (the first one passed is its TypeError).
                int d = DefaultEngine.DdofWithCorrection(ddof, correction);
                if (keepdims)
                    throw NDMatrixBoundary.KeywordError("std", "keepdims");
                if (@where is not null)
                    throw NDMatrixBoundary.KeywordError("std", "where");
                if (mean is not null)
                    throw NDMatrixBoundary.KeywordError("std", "mean");
                if (@out is not null)
                    return NDMatrixBoundary.ReducedInto(a, @out, (axis, dtype, d),
                        static (m, o, s) => m.std(s.axis, s.dtype, o, s.d),
                        static (m, o, s) => m.TensorEngine.ReduceStd(m, s.axis, keepdims: true, s.d, s.dtype, o));
                // a row, a column or the scalar, as NumPy's matrix method returns
                return NDMatrixBoundary.Reduced((a, axis, dtype, d), static s => std(s.a, s.axis, s.dtype, ddof: s.d), axis, false, "std");
            }

            // The whole call is the engine's: TensorEngine.ReduceStd of a's engine takes every keyword, correction= first.
            return a.TensorEngine.ReduceStd(a, axis, keepdims, ddof, dtype, @out, @where, mean, correction);
        }
    }
}
