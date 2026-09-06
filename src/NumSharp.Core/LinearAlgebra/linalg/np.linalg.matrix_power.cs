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
            /// <param name="n">
            ///     Any integer. <c>0</c> gives the identity, a NEGATIVE power inverts first.
            /// </param>
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
            [NDScoped]
            public static NDArray matrix_power(NDArray a, int n)
            {
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
                if (n == 1)
                    return work.copy();
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

                return np.broadcast_to(eye, new Shape(a.shape)).copy();
            }
        }
    }
}
