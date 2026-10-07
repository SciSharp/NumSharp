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
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            // Scope: solve casts BOTH operands via ToCommon (two potential dtype-cast temps) before
            // the engine call — [NDScoped] reclaims whichever were minted (a passthrough operand
            // already at the common type is never tracked), yielding the engine's fresh solution.
            [NDScoped]
            public static NDArray solve(NDArray a, NDArray b)
            {
                // NumPy takes its wrap from b (`a, _ = _makearray(a); b, wrap = _makearray(b)`): a numpy.matrix b
                // gives a matrix, a matrix a with a plain b a plain array (probed 2.4.2).
                if (NDMatrixBoundary.Sees(a, b))
                    return NDMatrixBoundary.IsMatrix(b)
                        ? NDMatrixBoundary.Wrapped((a, b), static s => solve(s.a, s.b))
                        : NDMatrixBoundary.Plain((a, b), static s => solve(s.a, s.b));
                AssertStackedSquare(a);

                bool oneDimensional = b.ndim == 1;
                // NumPy's _commonType(a, b) runs before the gufunc, and the gufunc resolves its loop (where its input
                // casts refuse text) before it checks the core dimensions.
                var common = CommonType(a, b);
                ThrowIfTextGufunc(oneDimensional ? "solve1" : "solve", a, b);
                if (oneDimensional)
                {
                    GufuncGuard.RequireCoreSize("solve1", "(m,m),(m)->(m)", 1, 0,
                        b.Shape.dimensions[0], a.Shape.dimensions[a.ndim - 1]);
                }
                else
                {
                    GufuncGuard.RequireRank("solve", "(m,m),(m,n)->(m,n)", 1, b, 2);
                    GufuncGuard.RequireCoreSize("solve", "(m,m),(m,n)->(m,n)", 1, 0,
                        b.Shape.dimensions[b.ndim - 2], a.Shape.dimensions[a.ndim - 1]);
                }

                return a.TensorEngine.Solve(ToCommon(a, common), ToCommon(b, common), oneDimensional);
            }

            /// <summary>
            ///     Solves the tensor equation <c>a x = b</c> for x.
            /// </summary>
            /// <param name="axes">
            ///     Axes of <paramref name="a"/> to move to the right before inversion.
            /// </param>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorsolve.html
            ///     <para>
            ///     A reshape of <see cref="solve"/>: <paramref name="a"/> is collapsed to a square
            ///     matrix whose side is <c>prod(b.shape)</c>, solved, and the answer reshaped to the
            ///     trailing block of <c>a.shape</c>.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            [NDScoped]
            public static NDArray tensorsolve(NDArray a, NDArray b, int[] axes = null)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped((a, b, axes), static s => tensorsolve(s.a, s.b, s.axes));
                var work = a;
                if (axes is not null)
                {
                    // NumPy removes each named axis from the identity permutation and re-appends it,
                    // so the requested axes end up rightmost IN THE ORDER GIVEN.
                    var order = new System.Collections.Generic.List<int>();
                    for (int k = 0; k < a.ndim; k++)
                        order.Add(k);
                    foreach (int k in axes)
                    {
                        order.Remove(k);
                        order.Add(k);
                    }

                    work = np.transpose(a, order.ToArray());
                }

                var prod = work.Shape.dimensions;
                long trailing = 1;
                for (int i = b.ndim; i < work.ndim; i++)
                    trailing *= prod[i];
                long leading = 1;
                for (int i = 0; i < b.ndim && i < work.ndim; i++)
                    leading *= prod[i];

                if (trailing != leading)
                {
                    // Verbatim, run of spaces included: NumPy's literal is a backslash-continued
                    // Python string, so the source's indentation lands inside the message.
                    throw new LinAlgError(
                        "Input arrays must satisfy the requirement             " +
                        "prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])");
                }

                var resultShape = new long[work.ndim - b.ndim];
                for (int i = b.ndim; i < work.ndim; i++)
                    resultShape[i - b.ndim] = prod[i];

                var square = np.reshape(work, new[] {trailing, trailing});
                var rhs = np.ravel(b);

                return np.reshape(solve(square, rhs), resultShape);
            }
        }
    }
}
