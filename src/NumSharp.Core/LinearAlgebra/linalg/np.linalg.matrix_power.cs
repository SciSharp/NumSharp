using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Raises a square matrix to the (integer) power <paramref name="n"/>.
            /// </summary>
            /// <param name="a">The square matrix, or a stack of them (the last two axes).</param>
            /// <param name="n">
            ///     Any integer. <c>0</c> gives the identity, a NEGATIVE power inverts first.
            /// </param>
            /// <returns>
            ///     The power. <b>Footgun (NumPy's own):</b> <c>n == 1</c> returns <paramref name="a"/> ITSELF, not a copy
            ///     (<c>np.linalg.matrix_power(a, 1) is a</c>), so writing to the result writes to <paramref name="a"/>;
            ///     <c>n == -1</c> returns the fresh inverse.
            /// </returns>
            /// <exception cref="LinAlgError"><paramref name="a"/> is not at least 2-D, or its last two axes are not square;
            ///     a negative power of a singular matrix.</exception>
            /// <exception cref="TypeError">A text matrix with <c>n</c> of 2 or more (matmul has no string loop) or a negative
            ///     <c>n</c> (inv's input cast).</exception>
            /// <exception cref="OpenBlasMissingBackendException">A negative power of a matrix the managed LU cannot take
            ///     with no backend installed.</exception>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html
            ///     <para>
            ///     Computed by binary exponentiation, so the cost is logarithmic in
            ///     <paramref name="n"/> rather than linear — NumPy's own algorithm, and the reason
            ///     the first three powers are special-cased before the loop starts.
            ///     </para>
            ///     <para>
            ///     A negative power is the only route here that needs a matrix backend
            ///     (<c>a**-n</c> is <c>inv(a)**n</c>), so it raises
            ///     <see cref="OpenBlasMissingBackendException"/> while none is installed. Non-negative powers
            ///     work.
            ///     </para>
            /// </remarks>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            [NDScoped] // the class boundary's temporaries; LinalgEngine.MatrixPower scopes the computation's own
            public static NDArray matrix_power(NDArray a, int n)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped((a, n), static s => matrix_power(s.a, s.n));
                return LinalgEngine.MatrixPower(a, n);
            }
        }
    }
}
