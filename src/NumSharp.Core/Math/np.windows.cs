using System;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        // Window functions — NumPy 2.4.2 numpy/lib/_function_base_impl.py
        // (bartlett / blackman / hamming / hanning / kaiser).
        //
        // The facades: each window's statements are the TensorEngine member of
        // its name (TensorEngine.FunctionBase.cs), which DefaultEngine runs as one
        // fused pass (Backends/Default/Math/Default.Windows.cs, whose header holds
        // the fusion and edge-case notes). A window takes no array, so no operand
        // can choose the engine: every facade calls the default engine's member.
        //
        // NumPy's OWN type stub declares `M: _FloatLike_co` (not int), so the
        // public surface takes `double M`: an int caller (`np.hanning(5)`) binds
        // through the implicit int→double conversion and gets the identical
        // result, while a non-integer M reproduces NumPy's float-M behaviour (the
        // window length is `len(arange(1-M, M, 2))`).
        //
        // This file also keeps the cephes I_0 routine (BesselI0, Chbevl and the
        // coefficient tables below): np.i0's per-precision kernels build on them,
        // and np.kaiser's engine member applies the float64 one per element
        // through np.i0's delegate np._i0Double.
        // =====================================================================

        /// <summary>
        ///     Return the Bartlett window — a triangular taper whose end points are zero.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns>
        ///     The triangular window, maximum normalized to one (the value one appears only when
        ///     <paramref name="M"/> is odd), first and last samples equal to zero.
        /// </returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN (<c>arange: cannot compute length</c>), +infinity or too
        ///     large for arange to size (<c>Maximum allowed size exceeded</c>).</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.bartlett.html
        ///     <para>NumPy: <c>where(n &lt;= 0, 1 + n/(M-1), 1 - n/(M-1))</c> with <c>n = arange(1-M, M, 2)</c>; the
        ///     statements are <see cref="TensorEngine.Bartlett"/>, called on the default engine (a window has no operand to
        ///     choose another).</para>
        /// </remarks>
        public static NDArray bartlett(double M) => TensorEngine.Default.Bartlett(M);

        /// <summary>
        ///     Return the Blackman window — a taper formed by the first three terms of a summation of cosines.
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns>The window, maximum normalized to one (the value one appears only when <paramref name="M"/> is odd).</returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN, +infinity or too large for arange to size.</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.blackman.html
        ///     <para>NumPy: <c>0.42 + 0.5*cos(pi*n/(M-1)) + 0.08*cos(2.0*pi*n/(M-1))</c> with <c>n = arange(1-M, M, 2)</c>; the
        ///     statements are <see cref="TensorEngine.Blackman"/>, called on the default engine.</para>
        /// </remarks>
        public static NDArray blackman(double M) => TensorEngine.Default.Blackman(M);

        /// <summary>
        ///     Return the Hamming window — a taper formed by a weighted cosine (raised-cosine, 0.54/0.46).
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns>The window, maximum normalized to one (the value one appears only when <paramref name="M"/> is odd).</returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN, +infinity or too large for arange to size.</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.hamming.html
        ///     <para>NumPy: <c>0.54 + 0.46*cos(pi*n/(M-1))</c> with <c>n = arange(1-M, M, 2)</c>; the statements are
        ///     <see cref="TensorEngine.Hamming"/>, called on the default engine.</para>
        /// </remarks>
        public static NDArray hamming(double M) => TensorEngine.Default.Hamming(M);

        /// <summary>
        ///     Return the Hanning window — a taper formed by a weighted cosine (raised-cosine, 0.5/0.5).
        /// </summary>
        /// <param name="M">Number of points in the output window. If zero or less, an empty array is returned.</param>
        /// <returns>The window, maximum normalized to one (the value one appears only when <paramref name="M"/> is odd).</returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN, +infinity or too large for arange to size.</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.hanning.html
        ///     <para>NumPy: <c>0.5 + 0.5*cos(pi*n/(M-1))</c> with <c>n = arange(1-M, M, 2)</c>; the statements are
        ///     <see cref="TensorEngine.Hanning"/>, called on the default engine.</para>
        /// </remarks>
        public static NDArray hanning(double M) => TensorEngine.Default.Hanning(M);

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
        /// <exception cref="ValueError"><paramref name="M"/> is NaN, or an infinity or too large for arange to size.</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html
        ///     <para>NumPy: <c>i0(beta * sqrt(1 - ((n-alpha)/alpha)**2)) / i0(beta)</c> with
        ///     <c>n = arange(0, M)</c> and <c>alpha = (M-1)/2</c>; the statements are <see cref="TensorEngine.Kaiser"/>, called
        ///     on the default engine. The modified Bessel function <see cref="BesselI0"/> is the cephes Chebyshev evaluation
        ///     NumPy's <c>np.i0</c> uses, applied per element in the fused pass (bit-identical to NumPy's piecewise
        ///     result).</para>
        /// </remarks>
        public static NDArray kaiser(double M, double beta) => TensorEngine.Default.Kaiser(M, beta);

        // ------------------------------------------------------------------
        // Modified Bessel function of the first kind, order 0 (I_0).
        //
        // Port of the cephes routine NumPy's np.i0 uses (numpy/lib
        // _function_base_impl.py: _i0_1 / _i0_2 / _chbevl). The domain [0, inf)
        // is split at 8.0 with a Chebyshev expansion on each side. np.i0's
        // float64 kernel is this scalar form (np._i0Double), and np.kaiser's
        // engine member applies it per element in its fused pass; it is
        // bit-identical to NumPy's vectorized piecewise result because the
        // per-element arithmetic (exp / sqrt / the Horner recurrence) is the same
        // sequence of IEEE float64 operations.
        // ------------------------------------------------------------------

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

        /// <summary>Clenshaw evaluation of a Chebyshev series (cephes <c>chbevl</c>), each step one float64 operation in
        ///     NumPy's order, so the sum rounds as NumPy's <c>_chbevl</c> does.</summary>
        /// <param name="x">The recentred evaluation point (<c>x/2 - 2</c> on [0, 8], <c>32/x - 2</c> above).</param>
        /// <param name="vals">The Chebyshev coefficients, highest order first (<see cref="_i0A"/> or <see cref="_i0B"/>).</param>
        /// <returns>The series value.</returns>
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
        ///     matching NumPy's <c>np.i0</c> element-for-element. It is np.i0's float64 kernel
        ///     (<see cref="_i0Double"/>), the one np.kaiser's engine member applies per element.
        /// </summary>
        /// <param name="x">The argument; I₀ is even, so its magnitude is used.</param>
        /// <returns>I₀(<paramref name="x"/>): +infinity once the series' <c>exp</c> overflows (|x| past about 709.78), NaN
        ///     for NaN.</returns>
        internal static double BesselI0(double x)
        {
            x = Math.Abs(x);
            if (x <= 8.0)
                return Math.Exp(x) * Chbevl(x / 2.0 - 2.0, _i0A);
            return Math.Exp(x) * Chbevl(32.0 / x - 2.0, _i0B) / Math.Sqrt(x);
        }
    }
}
