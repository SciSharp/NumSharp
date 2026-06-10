using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evaluate an expression tree over NDArrays in ONE fused pass —
        ///     no intermediate arrays, one read of each operand, one write of
        ///     the result (NumSharp extension; the NumPy-ecosystem equivalent
        ///     is numexpr.evaluate).
        /// </summary>
        /// <param name="expr">
        ///     Expression with embedded array leaves. NDArrays convert
        ///     implicitly, so one cast lights up the whole operator set:
        ///     <code>
        ///     NDArray r = np.evaluate((NpyExpr)a * b + 2);            // a*b+2 fused
        ///     NDArray d = np.evaluate((NpyExpr.Arr(a) - b) / (NpyExpr.Arr(a) + b));
        ///     NDArray s = np.evaluate(NpyExpr.Sum((NpyExpr)a * b));   // one-pass sum(a*b)
        ///     </code>
        ///     A repeated NDArray reference becomes ONE iterator operand.
        /// </param>
        /// <param name="out">
        ///     Optional pre-allocated result (ufunc out= rules: joins the
        ///     broadcast but is never stretched; same_kind cast from the
        ///     resolved dtype; may alias an input — overlap-safe).
        /// </param>
        /// <returns>
        ///     The evaluated array at the tree's NumPy result_type — dtypes
        ///     match the equivalent unfused NumPy expression node-for-node
        ///     (NEP50, including weak python-scalar literals). Root reductions
        ///     (<see cref="NpyExpr.Sum(NpyExpr)"/> / Prod / Min / Max / Mean)
        ///     return a 0-d scalar array.
        /// </returns>
        public static NDArray evaluate(NpyExpr expr, NDArray @out = null)
            => BackendFactory.GetEngine().Evaluate(expr, @out);

        /// <summary>
        ///     Evaluate an expression built over positional
        ///     <see cref="NpyExpr.Input"/> leaves against an explicit operand
        ///     list: <c>np.evaluate(NpyExpr.Input(0) * NpyExpr.Input(1), new[] { a, b })</c>.
        /// </summary>
        public static NDArray evaluate(NpyExpr expr, NDArray[] operands, NDArray @out = null)
            => BackendFactory.GetEngine().Evaluate(expr, operands, @out);
    }
}
