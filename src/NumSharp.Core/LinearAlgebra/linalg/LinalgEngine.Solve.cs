using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.solve</c> past the class boundary: <c>_assert_stacked_square</c>, <c>_commonType</c>,
        ///     the gufunc's text refusal and its core-dimension checks (the 1-D <paramref name="b"/> picks the
        ///     <c>(m,m),(m)-&gt;(m)</c> gufunc, any other rank <c>(m,m),(m,n)-&gt;(m,n)</c>), then
        ///     <see cref="TensorEngine.Solve"/> at the loop dtype.
        /// </summary>
        /// <param name="a">Coefficient matrix, <c>(..., M, M)</c> (plain: the facade applied the class boundary).</param>
        /// <param name="b">Ordinate values, <c>(..., M)</c> or <c>(..., M, K)</c>.</param>
        /// <returns>The solution, a fresh array of the loop dtype.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square <paramref name="a"/>, or a singular matrix.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (the gufunc's input cast).</exception>
        /// <exception cref="ValueError">A core dimension of <paramref name="b"/> that does not match, or a <paramref name="b"/>
        ///     of too few axes for the matrix gufunc (NumPy's gufunc texts).</exception>
        /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
        // Scope: solve casts BOTH operands via ToCommon (two potential dtype-cast temps) before
        // the engine call — [NDScoped] reclaims whichever were minted (a passthrough operand
        // already at the common type is never tracked), yielding the engine's fresh solution.
        [NDScoped]
        internal static NDArray Solve(NDArray a, NDArray b)
        {
            AssertStackedSquare(a);

            bool oneDimensional = b.ndim == 1;
            // NumPy's _commonType(a, b) runs before the gufunc, and the gufunc resolves its loop (where its input
            // casts refuse text) before it checks the core dimensions.
            var common = CommonType(a, b);
            ThrowIfTextGufunc(oneDimensional ? "solve1" : "solve", a, b);
            if (oneDimensional)
            {
                Backends.GufuncGuard.RequireCoreSize("solve1", "(m,m),(m)->(m)", 1, 0,
                    b.Shape.dimensions[0], a.Shape.dimensions[a.ndim - 1]);
            }
            else
            {
                Backends.GufuncGuard.RequireRank("solve", "(m,m),(m,n)->(m,n)", 1, b, 2);
                Backends.GufuncGuard.RequireCoreSize("solve", "(m,m),(m,n)->(m,n)", 1, 0,
                    b.Shape.dimensions[b.ndim - 2], a.Shape.dimensions[a.ndim - 1]);
            }

            return a.TensorEngine.Solve(ToCommon(a, common), ToCommon(b, common), oneDimensional);
        }

        /// <summary>
        ///     The body of <c>np.linalg.tensorsolve</c> past the class boundary: NumPy's axis move, the shape check, the
        ///     reshape to a square system, its solution (through the public <c>np.linalg.solve</c>) and the reshape of the
        ///     answer.
        /// </summary>
        /// <param name="a">The coefficient tensor (plain: the facade applied the class boundary).</param>
        /// <param name="b">The right-hand side.</param>
        /// <param name="axes">Axes of <paramref name="a"/> to move to the right before inversion, as LIST VALUES (never
        ///     normalized), or null.</param>
        /// <returns>The solution, of shape <c>a.shape[b.ndim:]</c> (after the axis move).</returns>
        /// <exception cref="ValueError">An axis that is not one of <c>0 .. a.ndim - 1</c> (Python's
        ///     <c>list.remove(x): x not in list</c>).</exception>
        /// <exception cref="LinAlgError">The moved operand is not square in the equation / unknown split, or the system
        ///     is singular.</exception>
        /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
        [NDScoped]
        internal static NDArray Tensorsolve(NDArray a, NDArray b, int[] axes)
        {
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
                    // Python's list.remove raises for a value the list does not hold; List.Remove only returns
                    // false, which used to append a stray axis and surface as transpose's "axes don't match array".
                    if (!order.Remove(k))
                        throw new ValueError("list.remove(x): x not in list");
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

            return np.reshape(np.linalg.solve(square, rhs), resultShape);
        }
    }
}
