using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Solves the linear system <c>a x = b</c> for x.
            /// </summary>
            /// <param name="a">Coefficient matrix, <c>(..., M, M)</c>.</param>
            /// <param name="b">Ordinate values, <c>(..., M)</c> or <c>(..., M, K)</c>.</param>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
            ///     <para>
            ///     NumPy 2.0 tightened the b-is-a-vector rule: <paramref name="b"/> is treated as a
            ///     stack of vectors ONLY when it is exactly 1-D. Anything else is a stack of matrices,
            ///     so the two gufuncs — <c>(m,m),(m)->(m)</c> and <c>(m,m),(m,n)->(m,n)</c> — are
            ///     selected by rank alone and report their own core-dimension errors.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>);
            ///     a core dimension of <paramref name="b"/> that does not match <paramref name="a"/>.</exception>
            /// <returns>The solution, a fresh array (float64 for bool / integer operands).</returns>
            /// <exception cref="LinAlgError">Fewer than two axes, a non-square <paramref name="a"/>, or a singular matrix.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand.</exception>
            /// <exception cref="ArgumentException">A text operand (NumPy's gufunc input-cast refusal).</exception>
            // Scope: the class boundary's temporaries; LinalgEngine.Solve reclaims the two ToCommon dtype-cast temps.
            [NDScoped]
            public static NDArray solve(NDArray a, NDArray b)
            {
                // NumPy takes its wrap from b (`a, _ = _makearray(a); b, wrap = _makearray(b)`): a numpy.matrix b
                // gives a matrix, a matrix a with a plain b a plain array (probed 2.4.2).
                if (NDMatrixBoundary.Sees(a, b))
                    return NDMatrixBoundary.IsMatrix(b)
                        ? NDMatrixBoundary.Wrapped((a, b), static s => solve(s.a, s.b))
                        : NDMatrixBoundary.Plain((a, b), static s => solve(s.a, s.b));
                return LinalgEngine.Solve(a, b);
            }

            /// <summary>
            ///     Solves the tensor equation <c>a x = b</c> for x.
            /// </summary>
            /// <param name="a">The coefficient tensor; its leading <c>b.ndim</c> axes (after <paramref name="axes"/> moved
            ///     some to the end) index the equations.</param>
            /// <param name="b">The right-hand side.</param>
            /// <param name="axes">
            ///     Axes of <paramref name="a"/> to move to the right before inversion. NumPy uses them as LIST VALUES
            ///     (<c>allaxes.remove(k)</c>), never normalizing them: a negative axis is not found, so only
            ///     <c>0 .. a.ndim - 1</c> are accepted.
            /// </param>
            /// <returns>The solution, of NumPy's <c>oldshape = a.shape[-(a.ndim - b.ndim):]</c> (after the axis move): the
            ///     trailing <c>a.ndim - b.ndim</c> axes, or the WHOLE shape of <paramref name="a"/> when the two have as many
            ///     axes. It owns its data (NumPy's <c>res.shape = oldshape</c>, not a reshape).
            ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
            ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorsolve.html
            ///     <para>
            ///     A reshape of <see cref="solve"/>: <paramref name="a"/> is collapsed to a square
            ///     matrix whose side is <c>prod(oldshape)</c>, solved against <c>b.ravel()</c>, and the
            ///     solution relabelled to <c>oldshape</c> in place. The slice's edges are NumPy's: with
            ///     as many axes in <paramref name="a"/> as in <paramref name="b"/> only an operand of 0
            ///     or 1 elements passes the size check (an empty or 1x1 system), and with more axes in
            ///     <paramref name="b"/> the system is 1x1. The solve needs no matrix backend (the managed
            ///     LU computes it).
            ///     </para>
            /// </remarks>
            /// <exception cref="ValueError">An axis that is not one of <c>0 .. a.ndim - 1</c>: Python's
            ///     "list.remove(x): x not in list" (a negative axis included; NumPy raises no AxisError here); or a
            ///     right-hand side that does not match the square system (solve's gufunc text).</exception>
            /// <exception cref="LinAlgError"><c>a.size</c> is not <c>prod(oldshape) ** 2</c>, or the system is
            ///     singular.</exception>
            /// <exception cref="TypeError">A float16, decimal or char operand (solve's <c>_commonType</c>).</exception>
            /// <exception cref="ArgumentException">A text operand (solve's gufunc input cast, NumPy's UFuncTypeError).</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            [NDScoped] // the class boundary's temporaries; LinalgEngine.Tensorsolve scopes the computation's own
            public static NDArray tensorsolve(NDArray a, NDArray b, int[] axes = null)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped((a, b, axes), static s => tensorsolve(s.a, s.b, s.axes));
                // a numpy.char.chararray operand: NumPy's asarray(a) / asarray(b) drop the class (NDCharArrayBoundary)
                if (NDCharArrayBoundary.Sees(a, b))
                    return NDCharArrayBoundary.Plain((a, b, axes), static s => tensorsolve(s.a, s.b, s.axes));
                return LinalgEngine.Tensorsolve(a, b, axes);
            }
        }
    }
}
