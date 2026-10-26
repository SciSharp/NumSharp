using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Chained matrix product, evaluated in the cheapest association order.
            /// </summary>
            /// <param name="arrays">
            ///     Two or more arrays. With three or more, every one must be 2-D EXCEPT the endpoints,
            ///     which may be 1-D (a 1-D first operand is a row, a 1-D last a column); a 0-D, a
            ///     3-D-or-higher, or a 1-D operand anywhere in the MIDDLE raises
            ///     <see cref="LinAlgError"/> "{ndim}-dimensional array given. Array must be
            ///     two-dimensional". With exactly two arrays this is a plain <see cref="np.dot"/> and
            ///     imposes no such restriction.
            /// </param>
            /// <param name="out">
            ///     Where to deposit the answer. NumPy hands <c>out</c> to the final <c>dot</c>, so it
            ///     receives the TWO-DIMENSIONAL product and the returned array is a reshaped view of
            ///     it — <c>out</c> for <c>multi_dot([v, B, C])</c> is shaped <c>(1, k)</c>, not
            ///     <c>(k,)</c>.
            /// </param>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html
            ///     <para>
            ///     Matrix multiplication is associative but its COST is not: chaining
            ///     <c>(10000,100) @ (100,1000) @ (1000,5)</c> left to right costs about 10⁹
            ///     multiplications and right to left about 10⁷. This picks the order with the
            ///     classic O(n³) dynamic program (Cormen et al. 15.2) — except for exactly three
            ///     matrices, where the single comparison is done directly.
            ///     </para>
            ///     <para>
            ///     A 1-D first operand is treated as a ROW vector and a 1-D last operand as a COLUMN
            ///     vector, with the added axis removed again afterwards — so the chain's endpoints
            ///     behave like <c>np.dot</c>'s.
            ///     </para>
            /// </remarks>
            /// <returns>The chain's product: <paramref name="out"/> itself for a two-array chain given one, else a fresh
            ///     array or a reshaped view of <paramref name="out"/>.</returns>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>);
            ///     fewer than two arrays.</exception>
            /// <exception cref="LinAlgError">With three or more arrays, an operand other than a vector end that is not
            ///     two-dimensional.</exception>
            [NDScoped] // the class boundary's temporaries; LinalgEngine.MultiDot scopes the computation's own
            public static NDArray multi_dot(NDArray[] arrays, NDArray @out = null)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(arrays))
                    return NDMatrixBoundary.Wrapped((arrays, @out), static s => multi_dot(s.arrays, s.@out), @out);
                return LinalgEngine.MultiDot(arrays, @out);
            }

            /// <inheritdoc cref="multi_dot(NDArray[], NDArray)"/>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            public static NDArray multi_dot(params NDArray[] arrays) => NDMatrixBoundary.Sees(arrays)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                ? NDMatrixBoundary.Wrapped(arrays, static s => multi_dot(s))
                : multi_dot(arrays, null);
        }
    }
}
