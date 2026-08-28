using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the roots of a polynomial with coefficients given in <paramref name="p"/>.
        ///     If <c>p</c> has length <c>n+1</c> the polynomial is
        ///     <c>p[0]*x**n + p[1]*x**(n-1) + ... + p[n]</c>.
        /// </summary>
        /// <param name="p">Rank-1 array of polynomial coefficients.</param>
        /// <returns>The roots — the eigenvalues of the companion matrix.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.roots.html</remarks>
        [NDScoped] // reclaims the ravel/nonzero/strip/astype chain and the companion-matrix + eigvals intermediates
        public static NDArray roots(NDArray p)
        {
            p = np.atleast_1d(p);
            if (p.ndim != 1)
                throw new ValueError("Input must be a rank-1 array.");

            // Indices of the non-zero coefficients (over the flattened array).
            NDArray nonZero = np.nonzero(np.ravel(p))[0];
            if (nonZero.size == 0)
                return np.array(new double[0]);   // all zeros -> no roots (empty float64)

            long first = nonZero.GetAtIndex<long>(0);
            long last = nonZero.GetAtIndex<long>((int)(nonZero.size - 1));

            // Trailing zeros are roots at 0; strip both leading and trailing zeros.
            long trailing = p.size - last - 1;
            p = p[$"{first}:{last + 1}"];

            // A non-floating coefficient array is promoted to float (NumPy: `p.astype(float)`).
            if (!IsFloatingOrComplex(p.typecode))
                p = p.astype(NPTypeCode.Double);

            long n = p.size;
            NDArray rootsArr;
            if (n > 1)
            {
                // Companion matrix: ones on the sub-diagonal, first row = -p[1:] / p[0].
                NDArray a = np.diag(np.ones(new Shape(n - 2), p.typecode), -1);
                a["0, :"] = np.negative(p["1:"]) / p["0"];
                rootsArr = np.linalg.eigvals(a);
            }
            else
            {
                rootsArr = np.array(new double[0]);
            }

            // Tack the zero-roots onto the back.
            return np.hstack(rootsArr, np.zeros(new Shape(trailing), rootsArr.typecode));
        }

        /// <summary>NumPy's <c>issubclass(dtype.type, (floating, complexfloating))</c>.</summary>
        internal static bool IsFloatingOrComplex(NPTypeCode t)
            => t == NPTypeCode.Half || t == NPTypeCode.Single || t == NPTypeCode.Double || t == NPTypeCode.Complex;
    }
}
