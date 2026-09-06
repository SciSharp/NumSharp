using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Determinant of a matrix, or of each matrix in a stack.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
            ///     <para>
            ///     Computed from an LU factorisation (<c>getrf</c>), not by expansion — so for a
            ///     large or ill-conditioned matrix the product of the pivots can overflow or
            ///     underflow where <see cref="slogdet"/> stays finite.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            // Scope: reclaims the ToCommon dtype-cast temp (minted only when `a` needs casting;
            // a passthrough input is never tracked), yielding the engine's fresh determinant scalar.
            [NDScoped]
            public static NDArray det(NDArray a)
            {
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Det(ToCommon(a, common));
            }

            /// <summary>
            ///     Sign and natural log of the absolute determinant — the overflow-safe
            ///     <see cref="det"/>.
            /// </summary>
            /// <returns>
            ///     <c>(sign, logabsdet)</c>, such that <c>sign * exp(logabsdet)</c> is the
            ///     determinant. A singular matrix gives <c>(0, -inf)</c>; for a complex operand the
            ///     sign is a unit-modulus complex number rather than ±1.
            /// </returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            // Scope: reclaims the ToCommon dtype-cast temp; the (sign, logabsdet) tuple result is
            // yielded through the weaver's Returns<T1,T2> tuple egress (both engine-fresh).
            [NDScoped]
            public static (NDArray sign, NDArray logabsdet) slogdet(NDArray a)
            {
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Slogdet(ToCommon(a, common));
            }
        }
    }
}
