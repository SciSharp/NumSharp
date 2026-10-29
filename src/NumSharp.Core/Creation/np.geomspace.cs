using System;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return numbers spaced evenly on a log scale (a geometric progression). Each output sample is a constant
        ///     multiple of the previous — like <see cref="logspace(double,double,long,bool,double,DType,int)"/> but with the
        ///     endpoints specified directly rather than as exponents.
        /// </summary>
        /// <param name="start">The starting value of the sequence. Must be non-zero (a geometric sequence cannot include zero).</param>
        /// <param name="stop">
        ///     The final value of the sequence, unless <paramref name="endpoint"/> is False. In that case <c>num + 1</c>
        ///     values are spaced over the interval in log-space, of which all but the last are returned. Must be non-zero.
        /// </param>
        /// <param name="num">Number of samples to generate. Default 50. Must be non-negative.</param>
        /// <param name="endpoint">If True (default) <paramref name="stop"/> is the last sample; otherwise it is excluded.</param>
        /// <param name="dtype">
        ///     The dtype of the output — one descriptor parameter, like NumPy's <c>dtype=</c> (Type/NPTypeCode/string/DType
        ///     convert implicitly). If null (default) the result is float64. A real dtype casts a float64 computation
        ///     (integer dtype TRUNCATES toward zero); requesting <see cref="System.Numerics.Complex"/> switches the
        ///     computation itself into the complex128 domain (matching NumPy's <c>dt</c> resolution), so the interior
        ///     samples differ from the real path.
        /// </param>
        /// <param name="axis">
        ///     Kept for NumPy signature parity. It is a no-op for scalar inputs (the output is always 1-D) but is still
        ///     validated: only 0 and -1 are accepted, else <see cref="AxisError"/> with NumPy's verbatim message.
        /// </param>
        /// <returns>A 1-D array of <paramref name="num"/> samples, equally spaced on a log scale.</returns>
        /// <exception cref="ValueError">If <paramref name="start"/> or <paramref name="stop"/> is zero, or <paramref name="num"/> is negative (NumPy's ValueError).</exception>
        /// <exception cref="AxisError">If <paramref name="axis"/> is neither 0 nor -1.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html</remarks>
        public static NDArray geomspace(double start, double stop, int num = 50, bool endpoint = true, DType dtype = null, int axis = 0)
            => geomspace(start, stop, (long)num, endpoint, dtype, axis);

        /// <summary>
        ///     Return numbers spaced evenly on a log scale (<see cref="long"/>-count overload — see the <see cref="int"/>
        ///     overload for the full contract).
        /// </summary>
        /// <param name="start">The starting value of the sequence. Must be non-zero.</param>
        /// <param name="stop">The final value of the sequence (excluded when <paramref name="endpoint"/> is False). Must be non-zero.</param>
        /// <param name="num">Number of samples to generate. Must be non-negative.</param>
        /// <param name="endpoint">If True (default) <paramref name="stop"/> is the last sample; otherwise it is excluded.</param>
        /// <param name="dtype">The dtype of the output (null → float64; complex switches the computation into the complex128 domain).</param>
        /// <param name="axis">Validated against the 1-D output; only 0 and -1 are accepted (no-op for scalar inputs).</param>
        /// <returns>A 1-D array of <paramref name="num"/> samples, equally spaced on a log scale.</returns>
        /// <exception cref="ValueError">If <paramref name="start"/> or <paramref name="stop"/> is zero, or <paramref name="num"/> is negative (NumPy's ValueError).</exception>
        /// <exception cref="AxisError">If <paramref name="axis"/> is neither 0 nor -1.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html</remarks>
        public static NDArray geomspace(double start, double stop, long num, bool endpoint = true, DType dtype = null, int axis = 0)
            => np.EngineOf(null).Geomspace(start, stop, num, endpoint, dtype, axis);

        /// <summary>
        ///     Return numbers spaced evenly on a log scale for COMPLEX endpoints — a logarithmic spiral in the complex
        ///     plane (there are infinitely many spirals through two points; this follows the shortest path, matching NumPy).
        /// </summary>
        /// <param name="start">The starting value of the sequence. Must be non-zero (<c>0+0j</c> is rejected).</param>
        /// <param name="stop">The final value of the sequence (excluded when <paramref name="endpoint"/> is False). Must be non-zero.</param>
        /// <param name="num">Number of samples to generate. Default 50. Must be non-negative.</param>
        /// <param name="endpoint">If True (default) <paramref name="stop"/> is the last sample; otherwise it is excluded.</param>
        /// <param name="dtype">The dtype of the output (null → complex128). A real dtype drops the imaginary part on the final cast, matching NumPy.</param>
        /// <param name="axis">Kept for signature parity; a no-op for scalar inputs, validated to 0 or -1.</param>
        /// <returns>A 1-D array of <paramref name="num"/> complex samples following the shortest logarithmic spiral.</returns>
        /// <exception cref="ValueError">If <paramref name="start"/> or <paramref name="stop"/> is <c>0+0j</c>, or <paramref name="num"/> is negative (NumPy's ValueError).</exception>
        /// <exception cref="AxisError">If <paramref name="axis"/> is neither 0 nor -1.</exception>
        /// <remarks>
        ///     ACCURATE to NumPy within the documented ≤3-ULP complex-unary envelope but NOT byte-reproducible — the
        ///     complex log10/power route through host-specific complex routines. https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html
        /// </remarks>
        public static NDArray geomspace(Complex start, Complex stop, int num = 50, bool endpoint = true, DType dtype = null, int axis = 0)
            => geomspace(start, stop, (long)num, endpoint, dtype, axis);

        /// <summary>
        ///     Return numbers spaced evenly on a log scale for COMPLEX endpoints (<see cref="long"/>-count overload —
        ///     see the <see cref="int"/> overload for the full contract).
        /// </summary>
        /// <param name="start">The starting value of the sequence. Must be non-zero.</param>
        /// <param name="stop">The final value of the sequence (excluded when <paramref name="endpoint"/> is False). Must be non-zero.</param>
        /// <param name="num">Number of samples to generate. Must be non-negative.</param>
        /// <param name="endpoint">If True (default) <paramref name="stop"/> is the last sample; otherwise it is excluded.</param>
        /// <param name="dtype">The dtype of the output (null → complex128).</param>
        /// <param name="axis">Validated to 0 or -1 (no-op for scalar inputs).</param>
        /// <returns>A 1-D array of <paramref name="num"/> complex samples following the shortest logarithmic spiral.</returns>
        /// <exception cref="ValueError">If <paramref name="start"/> or <paramref name="stop"/> is <c>0+0j</c>, or <paramref name="num"/> is negative (NumPy's ValueError).</exception>
        /// <exception cref="AxisError">If <paramref name="axis"/> is neither 0 nor -1.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html</remarks>
        public static NDArray geomspace(Complex start, Complex stop, long num, bool endpoint = true, DType dtype = null, int axis = 0)
            => np.EngineOf(null).Geomspace(start, stop, num, endpoint, dtype, axis);
    }
}
