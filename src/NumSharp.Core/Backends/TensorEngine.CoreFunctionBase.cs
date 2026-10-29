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
        ///     NumPy's <c>np.linspace(start, stop, num, endpoint=True, dtype=None, *, device=None)</c> over scalar ends:
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
        ///     char follows uint16), bool is <c>sample != 0</c>, complex128 takes the sample as its real part, and a text
        ///     dtype (<c>S</c>, <c>U</c>, <c>StringDType</c>) formats them as NumPy's <c>astype</c> does (<c>'0.25'</c>,
        ///     cut to the width of a fixed-width dtype).
        /// </param>
        /// <param name="device">NumPy's Array-API <c>device=</c>: only <c>"cpu"</c> and null are accepted. NumPy reads it
        ///     after the sample count (it is the ramp's np.arange that checks it), so a negative count is reported
        ///     first.</param>
        /// <returns>
        ///     A new 1-D array of <paramref name="num"/> samples. A float64 result of two or more samples, or of one sample
        ///     with <paramref name="endpoint"/> false, does not own its buffer (<c>OWNDATA</c> false: NumPy returns its
        ///     in-place ramp through <c>astype(copy=False)</c>, which is a view); every other result owns its buffer.
        /// </returns>
        /// <exception cref="ValueError">
        ///     A negative <paramref name="num"/> (<c>Number of samples, -1, must be non-negative.</c>), NumPy's first
        ///     check: before the device and before anything is allocated.
        /// </exception>
        /// <exception cref="ArgumentException"><paramref name="device"/> is neither null nor <c>"cpu"</c> (NumPy's
        ///     <c>Device not understood. Only "cpu" is allowed, but received: ...</c>, a ValueError there).</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no storage lane (a datetime
        ///     descriptor).</exception>
        /// <remarks>
        ///     <c>np.linspace</c> calls this member of the default engine. An override must keep NumPy's arithmetic, which
        ///     code ported from NumPy reads: the floor of the integer dtypes, the exact last sample, one sample as
        ///     <c>0 * (stop - start) + start</c> (a NaN when an end is infinite or NaN, +0.0 for a start of -0.0), and a
        ///     step that underflows to zero replaced by <c>(i / div) * (stop - start) + start</c> (NumPy's gh-5437 branch,
        ///     which keeps a subnormal range spread out).
        /// </remarks>
        public abstract NDArray Linspace(double start, double stop, long num, bool endpoint = true, DType dtype = null, string device = null);

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
        ///     from the real path's); a text dtype is refused, as NumPy refuses it (below); any other dtype casts the
        ///     float64 result (an integer dtype truncates toward zero).</param>
        /// <param name="axis">The destination of NumPy's trailing moveaxis: the same values for the 1-D result, but only 0
        ///     and -1 are in range, and -1 returns NumPy's moveaxis view.</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples. With <paramref name="axis"/> -1 it is NumPy's
        ///     <c>moveaxis</c> view, which does not own its buffer, unless the final cast to <paramref name="dtype"/> copies
        ///     it; otherwise it owns its buffer.</returns>
        /// <exception cref="ValueError">A zero end (<c>Geometric sequence cannot include zero</c>, NumPy's first check),
        ///     then (after the text dtypes) a negative <paramref name="num"/> (<c>Number of samples, -1, must be
        ///     non-negative.</c>).</exception>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c> <paramref name="dtype"/>: NumPy's
        ///     <c>result_type(start, stop, float(num), zeros((), dtype))</c> finds no common DType (its text lists the four
        ///     DType classes), right after the zero check.</exception>
        /// <exception cref="TypeError">An <c>S</c> or <c>U</c> <paramref name="dtype"/>: the ends promote to text and NumPy's
        ///     <c>sign(start)</c> has no string loop (<c>ufunc 'sign' did not contain a loop with signature matching types
        ///     &lt;class 'numpy.dtypes.StrDType'&gt; -&gt; None</c>), right after the zero check.</exception>
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
        ///     final cast, as NumPy's <c>astype</c> does; a text dtype is refused, as NumPy refuses it (below).</param>
        /// <param name="axis">Only 0 and -1 are in range (NumPy's trailing moveaxis of the 1-D result); -1 returns NumPy's
        ///     moveaxis view.</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples. With <paramref name="axis"/> -1 it is NumPy's
        ///     <c>moveaxis</c> view, which does not own its buffer, unless the final cast to <paramref name="dtype"/> copies
        ///     it; otherwise it owns its buffer.</returns>
        /// <exception cref="ValueError">A zero end (<c>Geometric sequence cannot include zero</c>), then (after the text
        ///     dtypes) a negative <paramref name="num"/>.</exception>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c> <paramref name="dtype"/>: no common DType with the
        ///     complex128 ends (NumPy's text, which lists <c>Complex128DType</c> twice), right after the zero
        ///     check.</exception>
        /// <exception cref="TypeError">An <c>S</c> or <c>U</c> <paramref name="dtype"/>: NumPy's <c>sign(start)</c> of the
        ///     ends cast to text has no string loop, right after the zero check.</exception>
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
