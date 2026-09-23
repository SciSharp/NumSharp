using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        // Window functions — port of NumPy 2.4.2 numpy/lib/_function_base_impl.py
        // (bartlett / blackman / hamming / hanning / kaiser).
        //
        // Every window is a 1-D float64 taper of length M. NumPy forces the
        // output to at least float64 by threading M through np.array([0.0, M]);
        // NumPy's OWN type stub declares `M: _FloatLike_co` (not int), so the
        // public NumSharp surface takes `double M` — an int caller (`np.hanning(5)`)
        // binds via the implicit int→double conversion and gets the identical
        // result, while a non-integer M reproduces NumPy's exact (quirky) float-M
        // behavior: the window length is `len(arange(1-M, M, 2))`, which can be a
        // fractional count, and kaiser can produce a trailing NaN once |(n-α)/α| > 1.
        //
        // Shared edge cases (cosine windows bartlett/blackman/hamming/hanning):
        //   • M < 1  → empty (0,) float64 array.
        //   • M == 1 → ones(1,) float64 array.
        // kaiser differs (see below): it has NO M < 1 guard, so 0 < M < 1 is a
        // one-element array, not empty.
        //
        // Implementation strategy — FUSION:
        //   NumPy builds each window as a chain of ufuncs over
        //   n = arange(1 - M, M, 2) (or arange(0, M) for kaiser), materializing
        //   ~6 intermediate arrays. NumSharp materializes only `n` and fuses the
        //   whole elementwise transform into ONE np.evaluate pass, so the buffer
        //   is read and written once. The fused expression tree is built in
        //   NumPy's exact operation order, which makes the result BIT-IDENTICAL
        //   to NumPy's unfused ufunc chain (verified across 152 (M, beta) cases
        //   and the differential-fuzz oracle): float64 add/sub/mul/div are IEEE
        //   exact, and float64 cos is the same scalar Math.Cos the CRT and NumPy
        //   both call. Measured NPY/NS ≈ 1.8×–9× at 100K/10M; at 1K the window
        //   sits at NumSharp's per-op NDIter-setup floor (window generation is a
        //   compute-once, amortized operation).
        //
        // The M-dependent values (M-1, alpha, beta, i0(beta)) ride as 0-d float64
        // OPERANDS, never as literals: a literal is baked into the fused kernel's
        // IL (and its cache key), so every distinct M — the one parameter a window
        // is called with — would JIT its own kernel (~1 ms) and hold its own
        // program-cache entry. The parameter form serves every M with ONE kernel
        // per window kind; the arithmetic is the same float64 arithmetic on a
        // value loaded from memory instead of an IL constant, so the result is
        // bit-identical (a 0-d float64 operand promotes exactly like a weak float
        // literal meeting a float64 array). The fixed cost of a stride-0 operand
        // (~0.25 µs, NDIter-level) is the price; the JIT it replaces is 4000× it.
        // =====================================================================

        /// <summary>
        ///     Return the Bartlett window — a triangular taper whose end points are zero.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns>
        ///     The triangular window, maximum normalized to one (the value one appears only when
        ///     <paramref name="M"/> is odd), first and last samples equal to zero.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.bartlett.html
        ///     <para>NumPy: <c>where(n &lt;= 0, 1 + n/(M-1), 1 - n/(M-1))</c> with <c>n = arange(1-M, M, 2)</c>.</para>
        /// </remarks>
        [NDScoped] // the arange intermediate `n` is reclaimed; the fused result is yielded
        public static NDArray bartlett(double M)
        {
            if (M < 1) return WindowEmpty();
            if (M == 1) return WindowOnes();

            var n = arange(1.0 - M, (double)M, 2.0, float64);
            var e = (NDExpr)n;
            // M-1 as a 0-d operand (see the header): one kernel for every M. The same instance on both
            // branches dedups to one iterator stream.
            var m1 = (NDExpr)NDArray.Scalar(M - 1);
            return evaluate(NDExpr.Where(NDExpr.LessEqual(e, 0.0),
                                         1.0 + e / m1,
                                         1.0 - e / m1));
        }

        /// <summary>
        ///     Return the Blackman window — a taper formed by the first three terms of a summation of cosines.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns>The window, maximum normalized to one (the value one appears only when <paramref name="M"/> is odd).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.blackman.html
        ///     <para>NumPy: <c>0.42 + 0.5*cos(pi*n/(M-1)) + 0.08*cos(2.0*pi*n/(M-1))</c> with <c>n = arange(1-M, M, 2)</c>.</para>
        /// </remarks>
        [NDScoped]
        public static NDArray blackman(double M)
        {
            if (M < 1) return WindowEmpty();
            if (M == 1) return WindowOnes();

            var n = arange(1.0 - M, (double)M, 2.0, float64);
            // Operation order preserved verbatim from NumPy for bit-parity:
            //   pi * n / (M - 1)   ≡ (pi * n) / (M - 1)      (left-to-right)
            //   2.0 * pi * n / (M-1) ≡ ((2.0 * pi) * n) / (M-1)
            // M-1 as a 0-d operand (see the header): one kernel for every M.
            var m1 = (NDExpr)NDArray.Scalar(M - 1);
            var arg = pi * (NDExpr)n / m1;
            var arg2 = 2.0 * pi * (NDExpr)n / m1;
            return evaluate(0.42 + 0.5 * NDExpr.Cos(arg) + 0.08 * NDExpr.Cos(arg2));
        }

        /// <summary>
        ///     Return the Hamming window — a taper formed by a weighted cosine (raised-cosine, 0.54/0.46).
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns>The window, maximum normalized to one (the value one appears only when <paramref name="M"/> is odd).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.hamming.html
        ///     <para>NumPy: <c>0.54 + 0.46*cos(pi*n/(M-1))</c> with <c>n = arange(1-M, M, 2)</c>.</para>
        /// </remarks>
        public static NDArray hamming(double M) => CosineWindow(M, 0.54, 0.46);

        /// <summary>
        ///     Return the Hanning window — a taper formed by a weighted cosine (raised-cosine, 0.5/0.5).
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns>The window, maximum normalized to one (the value one appears only when <paramref name="M"/> is odd).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.hanning.html
        ///     <para>NumPy: <c>0.5 + 0.5*cos(pi*n/(M-1))</c> with <c>n = arange(1-M, M, 2)</c>.</para>
        /// </remarks>
        public static NDArray hanning(double M) => CosineWindow(M, 0.5, 0.5);

        /// <summary>
        ///     Return the Kaiser window — a taper formed by a Bessel function.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <param name="beta">
        ///     Shape parameter for the window. As <paramref name="beta"/> grows the window narrows;
        ///     <c>beta == 0</c> is a rectangular window. Typical values: 5 (~Hamming), 6 (~Hanning),
        ///     8.6 (~Blackman).
        /// </param>
        /// <returns>The window, maximum normalized to one (the value one appears only when <paramref name="M"/> is odd).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html
        ///     <para>NumPy: <c>i0(beta * sqrt(1 - ((n-alpha)/alpha)**2)) / i0(beta)</c> with
        ///     <c>n = arange(0, M)</c> and <c>alpha = (M-1)/2</c>. The modified Bessel function
        ///     <see cref="BesselI0"/> is the cephes Chebyshev evaluation NumPy's <c>np.i0</c> uses,
        ///     applied per element in the fused pass (bit-identical to NumPy's piecewise result).</para>
        /// </remarks>
        [NDScoped]
        public static NDArray kaiser(double M, double beta)
        {
            // NumPy has NO `M < 1` guard here (unlike the cosine windows): kaiser uses
            // n = arange(0, M), which is EMPTY for M <= 0 (→ empty result) but ONE element
            // for 0 < M < 1 (→ a computed, non-empty window). Only M == 1 is special-cased,
            // to avoid the alpha == 0 division. Reproduced verbatim from NumPy 2.4.2.
            if (M == 1) return WindowOnes();

            var n = arange(0.0, M, 1.0, float64);
            // alpha, beta and i0(beta) as 0-d operands (see the header): one kernel for every (M, beta)
            // pair instead of one JIT per pair. `alpha` appears twice and dedups to one stream.
            var alpha = (NDExpr)NDArray.Scalar((M - 1) / 2.0);
            var betaOp = (NDExpr)NDArray.Scalar(beta);
            var denom = (NDExpr)NDArray.Scalar(BesselI0(beta));
            var inner = ((NDExpr)n - alpha) / alpha;
            // beta * sqrt(1 - ((n-alpha)/alpha)**2), then i0(.) / i0(beta) — per element.
            var arg = betaOp * NDExpr.Sqrt(1.0 - NDExpr.Power(inner, 2.0));
            return evaluate(NDExpr.Call(_besselI0, arg) / denom);
        }

        // ------------------------------------------------------------------
        // Shared helpers
        // ------------------------------------------------------------------

        /// <summary>hanning / hamming share the raised-cosine form <c>a0 + a1*cos(pi*n/(M-1))</c>.</summary>
        [NDScoped]
        private static NDArray CosineWindow(double M, double a0, double a1)
        {
            if (M < 1) return WindowEmpty();
            if (M == 1) return WindowOnes();

            var n = arange(1.0 - M, (double)M, 2.0, float64);
            // M-1 as a 0-d operand (see the header): one kernel per (a0, a1) pair — i.e. one for hanning
            // and one for hamming — for every M.
            var m1 = (NDExpr)NDArray.Scalar(M - 1);
            return evaluate(a0 + a1 * NDExpr.Cos(pi * (NDExpr)n / m1));
        }

        /// <summary>NumPy's <c>array([], dtype=float64)</c> — the M &lt; 1 result.</summary>
        private static NDArray WindowEmpty() => zeros(new int[] { 0 }, float64);

        /// <summary>NumPy's <c>ones(1, dtype=float64)</c> — the M == 1 result.</summary>
        private static NDArray WindowOnes() => ones(new int[] { 1 }, float64);

        // ------------------------------------------------------------------
        // Modified Bessel function of the first kind, order 0 (I_0).
        //
        // Port of the cephes routine NumPy's np.i0 uses (numpy/lib
        // _function_base_impl.py: _i0_1 / _i0_2 / _chbevl). The domain [0, inf)
        // is split at 8.0 with a Chebyshev expansion on each side. Kaiser only
        // needs the scalar float64 form, applied per element in the fused pass;
        // it is bit-identical to NumPy's vectorized piecewise result because the
        // per-element arithmetic (exp / sqrt / the Horner recurrence) is the same
        // sequence of IEEE float64 operations.
        // ------------------------------------------------------------------

        private static readonly Func<double, double> _besselI0 = BesselI0;

        // Chebyshev coefficients for exp(-x) I0(x) on the interval [0, 8].
        private static readonly double[] _i0A =
        {
            -4.41534164647933937950E-18,  3.33079451882223809783E-17, -2.43127984654795469359E-16,
             1.71539128555513303061E-15, -1.16853328779934516808E-14,  7.67618549860493561688E-14,
            -4.85644678311192946090E-13,  2.95505266312963983461E-12, -1.72682629144155570723E-11,
             9.67580903537323691224E-11, -5.18979560163526290666E-10,  2.65982372468238665035E-9,
            -1.30002500998624804212E-8,   6.04699502254191894932E-8,  -2.67079385394061173391E-7,
             1.11738753912010371815E-6,  -4.41673835845875056359E-6,   1.64484480707288970893E-5,
            -5.75419501008210370398E-5,   1.88502885095841655729E-4,  -5.76375574538582365885E-4,
             1.63947561694133579842E-3,  -4.32430999505057594430E-3,   1.05464603945949983183E-2,
            -2.37374148058994688156E-2,   4.93052842396707084878E-2,  -9.49010970480476444210E-2,
             1.71620901522208775349E-1,  -3.04682672343198398683E-1,   6.76795274409476084995E-1
        };

        // Chebyshev coefficients for exp(-x) sqrt(x) I0(x) on the interval (8, inf).
        private static readonly double[] _i0B =
        {
            -7.23318048787475395456E-18, -4.83050448594418207126E-18,  4.46562142029675999901E-17,
             3.46122286769746109310E-17, -2.82762398051658348494E-16, -3.42548561967721913462E-16,
             1.77256013305652638360E-15,  3.81168066935262242075E-15, -9.55484669882830764870E-15,
            -4.15056934728722208663E-14,  1.54008621752140982691E-14,  3.85277838274214270114E-13,
             7.18012445138366623367E-13, -1.79417853150680611778E-12, -1.32158118404477131188E-11,
            -3.14991652796324136454E-11,  1.18891471078464383424E-11,  4.94060238822496958910E-10,
             3.39623202570838634515E-9,   2.26666899049817806459E-8,   2.04891858946906374183E-7,
             2.89137052083475648297E-6,   6.88975834691682398426E-5,   3.36911647825569408990E-3,
             8.04490411014108831608E-1
        };

        /// <summary>Clenshaw evaluation of a Chebyshev series (cephes <c>chbevl</c>).</summary>
        private static double Chbevl(double x, double[] vals)
        {
            double b0 = vals[0], b1 = 0.0, b2 = 0.0;
            for (int i = 1; i < vals.Length; i++)
            {
                b2 = b1;
                b1 = b0;
                b0 = x * b1 - b2 + vals[i];
            }
            return 0.5 * (b0 - b2);
        }

        /// <summary>
        ///     Modified Bessel function of the first kind, order 0 — scalar float64,
        ///     matching NumPy's <c>np.i0</c> element-for-element.
        /// </summary>
        internal static double BesselI0(double x)
        {
            x = Math.Abs(x);
            if (x <= 8.0)
                return Math.Exp(x) * Chbevl(x / 2.0 - 2.0, _i0A);
            return Math.Exp(x) * Chbevl(32.0 / x - 2.0, _i0B) / Math.Sqrt(x);
        }
    }
}
