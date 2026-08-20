using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the derivative of the specified order of a polynomial.
        /// </summary>
        /// <param name="p">Polynomial coefficients, highest degree first.</param>
        /// <param name="m">Order of differentiation (default 1).</param>
        /// <returns>A new polynomial representing the derivative.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.polyder.html</remarks>
        public static NDArray polyder(NDArray p, int m = 1)
        {
            if (m < 0)
                throw new ValueError("Order of derivative must be positive (see polyint)");

            p = np.asarray(p);

            // Each order differentiates once: y = p[:-1] * arange(n, 0, -1), where n = len(p) - 1.
            // The int64 arange is what lifts int32 coefficients to int64 (matching NumPy).
            for (int iteration = 0; iteration < m; iteration++)
            {
                long n = p.size - 1;
                NDArray head = p[":-1"];
                NDArray factors = np.arange(n, 0L, -1L);
                p = head * factors;
            }

            return p;
        }
    }
}
