using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the sum of two polynomials. The shorter input is zero-extended on the high-degree side.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polyadd.html</remarks>
        public static NDArray polyadd(NDArray a1, NDArray a2) => PolyAddSub(a1, a2, subtract: false);

        /// <summary>
        ///     Difference (subtraction) of two polynomials, <c>a1 - a2</c>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polysub.html</remarks>
        public static NDArray polysub(NDArray a1, NDArray a2) => PolyAddSub(a1, a2, subtract: true);

        [NDScoped]
        private static NDArray PolyAddSub(NDArray a1, NDArray a2, bool subtract)
        {
            a1 = np.atleast_1d(a1);
            a2 = np.atleast_1d(a2);
            long diff = a2.size - a1.size;

            if (diff > 0)
                a1 = np.concatenate(new[] { np.zeros(new Shape(diff), a1.typecode), a1 }, 0);
            else if (diff < 0)
                a2 = np.concatenate(new[] { np.zeros(new Shape(-diff), a2.typecode), a2 }, 0);

            return subtract ? a1 - a2 : a1 + a2;
        }

        /// <summary>
        ///     Find the product of two polynomials — the (full-mode) convolution of their coefficients.
        ///     Leading zeros of each input are dropped first (NumPy wraps each in <c>poly1d</c>).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polymul.html</remarks>
        [NDScoped]
        public static NDArray polymul(NDArray a1, NDArray a2)
        {
            // poly1d normalisation trims leading zeros (and maps an all-zero input to [0]).
            NDArray c1 = new poly1d(a1).coeffs;
            NDArray c2 = new poly1d(a2).coeffs;
            return np.convolve(c1, c2, "full");
        }

        /// <summary>
        ///     Returns the quotient and remainder of polynomial division <c>u / v</c>.
        /// </summary>
        /// <param name="u">Dividend polynomial coefficients.</param>
        /// <param name="v">Divisor polynomial coefficients.</param>
        /// <returns><c>(q, r)</c> — quotient and remainder coefficients, floating (or complex) dtype.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.polydiv.html
        ///     <para>
        ///     A <c>poly1d</c> input does NOT force a <c>poly1d</c> return here (a C# tuple slot is a fixed
        ///     type); the <c>poly1d</c> <c>/</c> operator performs that wrapping instead.
        ///     </para>
        /// </remarks>
        public static (NDArray q, NDArray r) polydiv(NDArray u, NDArray v)
        {
            u = np.atleast_1d(u) + 0.0;   // floated, exactly like NumPy's `+ 0.0`
            v = np.atleast_1d(v) + 0.0;

            NPTypeCode w = np.result_type(u["0"], v["0"]);              // common type of `u[0] + v[0]`
            long m = u.size - 1;
            long n = v.size - 1;
            NDArray scale = 1.0 / v["0"];

            long qlen = Math.Max(m - n + 1, 1);
            NDArray q = np.zeros(new Shape(qlen), w);
            NDArray r = u.astype(w);

            for (long k = 0; k < m - n + 1; k++)
            {
                NDArray d = scale * r[k.ToString()];
                q[$"{k}:{k + 1}"] = np.atleast_1d(d);
                // r[k : k+n+1] -= d * v   (an in-place, length-(n+1) vector update)
                NDArray seg = r[$"{k}:{k + n + 1}"];
                np.subtract(seg, d * v, @out: seg);
            }

            // Strip leading near-zero remainder coefficients (NumPy: rtol 1e-14, keep at least one).
            while (r.shape[r.ndim - 1] > 1 && np.allclose(r["0"], NDArray.Scalar(0.0), rtol: 1e-14))
                r = r["1:"];

            return (q, r);
        }
    }
}
