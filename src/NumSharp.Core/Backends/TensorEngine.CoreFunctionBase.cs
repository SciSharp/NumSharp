// The functions of numpy/_core/function_base.py that compute: linspace, logspace and geomspace (the np-layer dtype
// plan's phase P3, batch 9). NumPy writes them in Python over ufuncs - a ramp of np.arange times the step plus the start,
// then np.power, np.log10 and np.sign for the log scales; here each is an engine member that DefaultEngine implements
// with NumPy's statements. Their facades take no array operand, so np.linspace / np.logspace / np.geomspace call the
// default engine's member (np.EngineOf(null), as np.fft.fftfreq does); an engine that derives from DefaultEngine reaches
// its own override by calling the member on itself. np.arange, which NumPy fills through the dtype's ArrFuncs fill and
// no ufunc, is the array object's (UnmanagedStorage.Ctors.cs), not an engine member (Appendix D). NumSharp's spacing
// functions take scalar ends only: array_like ends and the new axis they would place are not ported.

using System;
using System.Numerics;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.linspace(start, stop, num, endpoint=True, dtype=None)</c> over scalar ends:
        ///     <paramref name="num"/> evenly spaced samples from <paramref name="start"/> to <paramref name="stop"/>. With
        ///     <paramref name="endpoint"/> the step is <c>(stop - start) / (num - 1)</c> and the last sample is
        ///     <paramref name="stop"/> itself (NumPy overwrites <c>y[-1]</c>, so no rounding of <c>start + i * step</c>
        ///     reaches it); without it the step is <c>(stop - start) / num</c> and <paramref name="stop"/> is not a sample.
        /// </summary>
        /// <param name="start">The first sample.</param>
        /// <param name="stop">The last sample when <paramref name="endpoint"/> is true; otherwise the bound the samples
        ///     approach.</param>
        /// <param name="num">The number of samples.</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the last sample.</param>
        /// <param name="dtype">
        ///     The result dtype, or null for float64. The samples are computed in float64 and then converted: an integer
        ///     dtype FLOORS them (NumPy 1.20+'s <c>_nx.floor(y, out=y)</c> before the cast, not a truncation toward zero;
        ///     char follows uint16), bool is <c>sample != 0</c>, complex128 takes the sample as its real part.
        /// </param>
        /// <returns>
        ///     A new 1-D array of <paramref name="num"/> samples. A float64 result of two or more samples, or of one sample
        ///     with <paramref name="endpoint"/> false, does not own its buffer (<c>OWNDATA</c> false: NumPy returns its
        ///     in-place ramp through <c>astype(copy=False)</c>, which is a view); every other result owns its buffer.
        /// </returns>
        /// <exception cref="ValueError">
        ///     A negative <paramref name="num"/>. DefaultEngine allocates the result before it reads the count, so the text
        ///     is the allocation's <c>negative dimensions are not allowed</c>; NumPy's linspace checks the count first
        ///     (<c>Number of samples, -1, must be non-negative.</c>).
        /// </exception>
        /// <exception cref="NotSupportedException">
        ///     <paramref name="dtype"/> has no storage lane (a datetime descriptor), or is a text dtype (<c>S</c>, <c>U</c>,
        ///     <c>StringDType</c>): DefaultEngine allocates the result by storage lane, which carries no string class or
        ///     width.
        /// </exception>
        /// <remarks>
        ///     <c>np.linspace</c> checks its Array-API <c>device=</c> argument, then calls this member of the default
        ///     engine. An override must keep the floor of the integer dtypes and the exact last sample: they are NumPy's
        ///     results, which code ported from NumPy reads (an integer grid's last value, a range's exact end).
        /// </remarks>
        public abstract NDArray Linspace(double start, double stop, long num, bool endpoint = true, DType dtype = null);

        /// <summary>
        ///     NumPy's <c>np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)</c> over scalar
        ///     ends: <c>base ** linspace(start, stop, num, endpoint)</c>, computed in float64 and only then cast to
        ///     <paramref name="dtype"/> (computing in a narrower dtype would round differently).
        /// </summary>
        /// <param name="start"><c>base ** start</c> is the first sample.</param>
        /// <param name="stop"><c>base ** stop</c> is the last sample when <paramref name="endpoint"/> is true.</param>
        /// <param name="num">The number of samples.</param>
        /// <param name="endpoint">Whether <c>base ** stop</c> is the last sample (its exponent is <paramref name="stop"/>
        ///     itself, as NumPy's linspace writes <c>y[-1] = stop</c>).</param>
        /// <param name="base">The base of the log space. A negative base with a fractional exponent gives NaN, as NumPy's
        ///     real power does.</param>
        /// <param name="dtype">The result dtype, or null for float64. The float64 powers are cast, so an integer dtype
        ///     TRUNCATES toward zero (unlike <see cref="Linspace"/>, which floors) and a text dtype formats them.</param>
        /// <param name="axis">The destination of NumPy's trailing <c>moveaxis(y, 0, axis)</c>: a no-op for the 1-D result
        ///     of scalar ends, but only 0 and -1 are in range.</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples that owns its buffer.</returns>
        /// <exception cref="ValueError">A negative <paramref name="num"/> (<c>Number of samples, -1, must be
        ///     non-negative.</c>), checked before <paramref name="axis"/>.</exception>
        /// <exception cref="AxisError">An <paramref name="axis"/> other than 0 or -1 (<c>destination: axis 1 is out of
        ///     bounds for array of dimension 1</c>).</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no storage lane (a datetime descriptor):
        ///     the final cast refuses it.</exception>
        /// <remarks>
        ///     <c>np.logspace</c> calls this member of the default engine. DefaultEngine fills the powers in one pass with
        ///     <see cref="System.Math.Pow(double, double)"/>, the CRT routine NumPy's <c>npy_pow</c> calls on win-amd64, so
        ///     its float64 result is NumPy's byte for byte; an override that computes the powers otherwise loses that.
        /// </remarks>
        public abstract NDArray Logspace(double start, double stop, long num = 50, bool endpoint = true, double @base = 10.0, DType dtype = null, int axis = 0);

        /// <summary>
        ///     NumPy's <c>np.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)</c> over real scalar ends: a
        ///     geometric progression from <paramref name="start"/> to <paramref name="stop"/>, computed as NumPy computes
        ///     it - both ends divided by <c>sign(start)</c>, <c>logspace(log10(start), log10(stop))</c>, the ends written
        ///     back, the whole multiplied by the sign - so the ends are the arguments bit for bit and ends of opposite
        ///     signs give NaN between them (the log of a negative value).
        /// </summary>
        /// <param name="start">The first sample; not zero.</param>
        /// <param name="stop">The last sample when <paramref name="endpoint"/> is true; not zero.</param>
        /// <param name="num">The number of samples.</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the last sample.</param>
        /// <param name="dtype">The result dtype, or null for float64. A complex dtype computes in the complex128 domain,
        ///     as the <see cref="Geomspace(Complex, Complex, long, bool, DType, int)"/> overload does (its interior differs
        ///     from the real path's); any other dtype casts the float64 result (an integer dtype truncates toward
        ///     zero).</param>
        /// <param name="axis">The destination of NumPy's trailing moveaxis: a no-op for the 1-D result, but only 0 and -1
        ///     are in range.</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples that owns its buffer.</returns>
        /// <exception cref="ValueError">A zero end (<c>Geometric sequence cannot include zero</c>, NumPy's first check),
        ///     then a negative <paramref name="num"/> (<c>Number of samples, -1, must be non-negative.</c>).</exception>
        /// <exception cref="AxisError">An <paramref name="axis"/> other than 0 or -1, checked after
        ///     <paramref name="num"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no storage lane (a datetime descriptor):
        ///     the final cast refuses it.</exception>
        /// <remarks>
        ///     <c>np.geomspace</c> calls this member of the default engine. DefaultEngine's real path is NumPy's float64
        ///     result byte for byte (the sign is exactly 1 or -1, and the interior is <see cref="System.Math.Pow(double, double)"/>
        ///     and <see cref="System.Math.Log10(double)"/>, the routines NumPy calls on win-amd64); a NaN start makes every
        ///     sample NaN, as NumPy's <c>result *= out_sign</c> does.
        /// </remarks>
        public abstract NDArray Geomspace(double start, double stop, long num = 50, bool endpoint = true, DType dtype = null, int axis = 0);

        /// <summary>
        ///     NumPy's <c>np.geomspace</c> over complex scalar ends: a logarithmic spiral in the complex plane from
        ///     <paramref name="start"/> to <paramref name="stop"/>, the shortest of the spirals through the two points, as
        ///     NumPy computes it (both ends divided by <c>start / |start|</c>, a complex log-space interior, the ends written
        ///     back, the whole multiplied by that sign).
        /// </summary>
        /// <param name="start">The first sample; not <c>0+0j</c> (a purely imaginary value is allowed).</param>
        /// <param name="stop">The last sample when <paramref name="endpoint"/> is true; not <c>0+0j</c>.</param>
        /// <param name="num">The number of samples.</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the last sample.</param>
        /// <param name="dtype">The result dtype, or null for complex128; a real dtype drops the imaginary parts in the
        ///     final cast, as NumPy's <c>astype</c> does.</param>
        /// <param name="axis">Only 0 and -1 are in range (NumPy's trailing moveaxis of the 1-D result).</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples that owns its buffer.</returns>
        /// <exception cref="ValueError">A zero end (<c>Geometric sequence cannot include zero</c>), then a negative
        ///     <paramref name="num"/>.</exception>
        /// <exception cref="AxisError">An <paramref name="axis"/> other than 0 or -1, checked after
        ///     <paramref name="num"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no storage lane (a datetime descriptor):
        ///     the final cast refuses it.</exception>
        /// <remarks>
        ///     DefaultEngine composes NumSharp's complex power (the <c>npy_cpow</c> port) and complex log10: its values are
        ///     within the complex-unary envelope of NumPy's (about 3 ULP) but not NumPy's bits, and the ends are
        ///     <c>(start / sign) * sign</c>, which complex division and multiplication need not round back to
        ///     <paramref name="start"/>.
        /// </remarks>
        public abstract NDArray Geomspace(Complex start, Complex stop, long num = 50, bool endpoint = true, DType dtype = null, int axis = 0);
    }
}
