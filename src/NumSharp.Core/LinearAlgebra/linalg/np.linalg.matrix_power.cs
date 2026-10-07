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
            /// <exception cref="StringComputationNotSupportedException">A non-empty text matrix with <c>n == 0</c>: NumPy writes
            ///     the identity's numbers into the text dtype as text.</exception>
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
            [NDScoped]
            public static NDArray matrix_power(NDArray a, int n)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped((a, n), static s => matrix_power(s.a, s.n));
                AssertStackedSquare(a);

                var work = a;
                if (n < 0)
                {
                    // Raises without a backend — deliberately BEFORE the identity/short-power cases,
                    // exactly as NumPy inverts first and then exponentiates the inverse.
                    work = inv(a);
                    n = -n;
                }

                if (n == 0)
                    return Identity(work);
                // NumPy's short-cut `if n == 1: return a` hands back the operand itself (or, for n == -1, the fresh
                // inverse) — the same object, so `matrix_power(a, 1) is a` and the result shares a's memory and flags.
                if (n == 1)
                    return work;
                if (n == 2)
                    return np.matmul(work, work);
                if (n == 3)
                    return np.matmul(np.matmul(work, work), work);

                // Binary exponentiation. `result` stays null until the first set bit so the identity
                // multiply is skipped, which is what NumPy's `if result is None` does.
                NDArray result = null;
                var z = work;
                while (n > 0)
                {
                    if ((n & 1) == 1)
                        result = result is null ? z : np.matmul(result, z);

                    n >>= 1;
                    if (n > 0)
                        z = np.matmul(z, z);
                }

                return result;
            }

            /// <summary>
            ///     The identity matching a stack's trailing two axes, broadcast across the batch —
            ///     what <c>matrix_power(a, 0)</c> returns.
            /// </summary>
            private static NDArray Identity(NDArray a)
            {
                long side = a.Shape.dimensions[a.ndim - 1];
                if (side > int.MaxValue)
                    throw new OverflowException($"Matrix dimension {side} exceeds int.MaxValue for np.eye");

                var eye = np.eye((int)side).astype(a.dtype);
                if (a.ndim == 2)
                    return eye;

                return np.broadcast_to(eye, Shape.Adopt(a.Shape.dimensions)).copy();
            }
        }
    }
}
