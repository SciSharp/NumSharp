// The functions of numpy/_core/function_base.py that compute: linspace, logspace and geomspace (the np-layer dtype
// plan's phase P3, batch 9). NumPy writes them in Python over ufuncs - a ramp of np.arange times the step plus the start,
// then np.power, np.log10 and np.sign for the log scales; here each is an engine member that DefaultEngine implements
// with NumPy's statements. Their facades take no array operand, so np.linspace / np.logspace / np.geomspace call the
// default engine's member (TensorEngine.Default, as np.fft.fftfreq does); an engine that derives from DefaultEngine reaches
// its own override by calling the member on itself. np.arange, which NumPy fills through the dtype's ArrFuncs fill and
// no ufunc, is the array object's (UnmanagedStorage.Ctors.cs), not an engine member (Appendix D). np.linspace takes NumPy's
// array_like ends (the NDArray member: its dtype from result_type, the new sample axis placed by `axis`), reached through the
// start operand's engine, and so does np.geomspace (API-parity batch 10); np.logspace takes scalar ends.

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
        /// <exception cref="ValueError"><paramref name="device"/> is neither null nor <c>"cpu"</c> (NumPy's
        ///     <c>Device not understood. Only "cpu" is allowed, but received: ...</c>).</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no storage lane (a datetime
        ///     descriptor).</exception>
        /// <remarks>
        ///     <c>np.linspace</c> reaches this member through the array-ends member, which hands it two 0-d ends that compute
        ///     in float64 (two C# numbers, NumPy float64 / integer / bool scalars) with the sample axis first: the same
        ///     samples, without the ramp arrays. An override must keep NumPy's arithmetic, which
        ///     code ported from NumPy reads: the floor of the integer dtypes, the exact last sample, one sample as
        ///     <c>0 * (stop - start) + start</c> (a NaN when an end is infinite or NaN, +0.0 for a start of -0.0), and a
        ///     step that underflows to zero replaced by <c>(i / div) * (stop - start) + start</c> (NumPy's gh-5437 branch,
        ///     which keeps a subnormal range spread out).
        /// </remarks>
        public abstract NDArray Linspace(double start, double stop, long num, bool endpoint = true, DType dtype = null, string device = null);

        /// <summary>
        ///     NumPy's <c>np.linspace(start, stop, num, endpoint, dtype=dtype, axis=axis, device=device)</c> over array_like
        ///     ends, statement for statement: the samples are computed in <c>result_type(start, stop, ensure_inexact=True)</c>
        ///     (float32 ends stay float32, integer ends compute in float64, complex ends in complex), along a new axis of
        ///     length <paramref name="num"/> in front of the ends' broadcast shape, which <paramref name="axis"/> then moves.
        /// </summary>
        /// <param name="start">The first sample (a 0-d array for a scalar end). A C# number that reached the
        ///     <see cref="NDArray"/> parameter through its implicit conversion is a WEAK Python number (NEP 50): it takes the
        ///     other end's dtype, so <c>(0, float32 1.5)</c> computes in float32 and two C# numbers in float64.</param>
        /// <param name="stop">The last sample when <paramref name="endpoint"/> is true; otherwise the bound the samples
        ///     approach. Broadcast against <paramref name="start"/>.</param>
        /// <param name="num">The number of samples.</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the last sample (it is then written exactly:
        ///     <c>y[-1, ...] = stop</c>).</param>
        /// <param name="dtype">The result dtype, or null for the computation dtype. An integer dtype FLOORS the samples before
        ///     the cast (NumPy 1.20+), bool casts (nonzero is True), a text dtype formats them as <c>astype</c> does.</param>
        /// <param name="axis">Where the sample axis goes in the result (NumPy's <c>moveaxis(y, 0, axis)</c>); 0 keeps it
        ///     first, -1 makes it last.</param>
        /// <param name="device">NumPy's Array-API <c>device=</c>: only <c>"cpu"</c> and null are accepted, checked where
        ///     NumPy's ramp <c>np.arange</c> checks it (after the count, the promotion and the broadcast).</param>
        /// <returns>The samples: <c>(num, *broadcast(start, stop).shape)</c> with the sample axis moved to
        ///     <paramref name="axis"/>. Ownership follows NumPy's statements: a moved axis or two scalar ends computed in
        ///     float64 give a view whose buffer the array does not own; array ends give a fresh array.</returns>
        /// <exception cref="ValueError">A negative <paramref name="num"/> (<c>Number of samples, -1, must be
        ///     non-negative.</c>), checked first; a <paramref name="device"/> other than <c>"cpu"</c>.</exception>
        /// <exception cref="DTypePromotionError">Ends with no common inexact dtype (text ends: NumPy's text naming the
        ///     DType classes and <c>_PyFloatDType</c>).</exception>
        /// <exception cref="IncorrectShapeException">Ends that do not broadcast (NumPy's ValueError text, <c>operands could
        ///     not be broadcast together with shapes ...</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range for the result's rank (NumPy's
        ///     <c>destination: axis 3 is out of bounds for array of dimension 2</c>).</exception>
        /// <exception cref="TypeError">An integer <paramref name="dtype"/> over complex ends: NumPy's <c>floor</c> has no
        ///     complex loop.</exception>
        /// <remarks>
        ///     <c>np.linspace</c> calls this member of the start operand's engine. An override must keep NumPy's arithmetic,
        ///     which ported code reads bit for bit: <c>arange(num) * step + start</c> in the computation dtype (float32 ends
        ///     round in float32), the exact last sample, <c>y * delta + start</c> for one sample with the endpoint, and NumPy's
        ///     gh-5437 order <c>arange(num) / div * delta + start</c> when ANY step is zero. A subclass operand
        ///     (<c>numpy.matrix</c>) arrives as a plain view and the result is wrapped by its class, as NumPy's
        ///     <c>_array_converter.wrap</c> does.
        /// </remarks>
        [ArrayWrap("start", "stop")]
        public abstract NDArray Linspace(NDArray start, NDArray stop, long num, bool endpoint, DType dtype, int axis, string device);

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
        ///     NumPy's <c>np.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)</c>: numbers spaced evenly on a
        ///     log scale, each end pair its own geometric progression - computed as NumPy computes it: both ends divided by
        ///     <c>sign(start)</c>, <c>logspace(log10(start), log10(stop))</c>, the ends written back exactly, the whole
        ///     multiplied by the sign, the sample axis moved to <paramref name="axis"/> - so ends of opposite signs give NaN
        ///     between them (the log of a negative value) and complex ends a logarithmic spiral.
        /// </summary>
        /// <param name="start">The first sample(s): a 0-d array (a C# number converts to one, read as NumPy reads a Python
        ///     number) or any array that broadcasts with <paramref name="stop"/>; no element may be zero.</param>
        /// <param name="stop">The last sample(s) when <paramref name="endpoint"/> is true; no element may be zero.</param>
        /// <param name="num">The number of samples.</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the last sample.</param>
        /// <param name="dtype">The result dtype, or null for the computation dtype <c>dt = result_type(start, stop,
        ///     float(num), zeros((), dtype))</c> (float64 for real ends unless they are a narrower float array AND the
        ///     dtype asks for it, complex128 for complex ends or a complex dtype); the samples are computed in <c>dt</c>
        ///     and cast at the end (an integer dtype truncates toward zero); a text dtype is refused, as NumPy refuses
        ///     it (below).</param>
        /// <param name="axis">The axis of the result that holds the samples (NumPy's trailing <c>moveaxis(result, 0,
        ///     axis)</c>): in range for the result's rank, one more than the broadcast ends'.</param>
        /// <returns>A new array of shape <c>(num, *broadcast(start, stop).shape)</c> with the sample axis moved to
        ///     <paramref name="axis"/>; a moved axis is NumPy's <c>moveaxis</c> view, which does not own its buffer unless
        ///     the final cast to <paramref name="dtype"/> copies it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="start"/> or <paramref name="stop"/> is null.</exception>
        /// <exception cref="ValueError">A zero element in either end (<c>Geometric sequence cannot include zero</c>, NumPy's
        ///     first check), then (after the text dtypes) a negative <paramref name="num"/> (<c>Number of samples, -1, must
        ///     be non-negative.</c>).</exception>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c> <paramref name="dtype"/>: NumPy's
        ///     <c>result_type(start, stop, float(num), zeros((), dtype))</c> finds no common DType (its text lists the four
        ///     DType classes), right after the zero check.</exception>
        /// <exception cref="TypeError">An <c>S</c> or <c>U</c> <paramref name="dtype"/>: the ends promote to text and NumPy's
        ///     <c>sign(start)</c> has no string loop (<c>ufunc 'sign' did not contain a loop with signature matching types
        ///     &lt;class 'numpy.dtypes.StrDType'&gt; -&gt; None</c>), right after the zero check.</exception>
        /// <exception cref="IncorrectShapeException">Ends that do not broadcast (the division by the start's sign).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range for the result (<c>destination: axis 1 is out of
        ///     bounds for array of dimension 1</c>), checked after <paramref name="num"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no storage lane (a datetime descriptor):
        ///     the final cast refuses it.</exception>
        /// <remarks>
        ///     <para><c>np.geomspace</c> calls this member of the start operand's engine. DefaultEngine follows NumPy's
        ///     statements over arrays (sign, true_divide, log10, linspace, power, the end writes, multiply, moveaxis), so an
        ///     override must keep that arithmetic in <c>dt</c>: float32 ends with a float32 dtype round in float32, as
        ///     NumPy's do. Two 0-d ends that compute in float64 are NumPy's float64 result byte for byte (the sign is exactly
        ///     1 or -1, and the interior is <see cref="System.Math.Pow(double, double)"/> and
        ///     <see cref="System.Math.Log10(double)"/>, the routines NumPy calls on win-amd64); complex ends are within the
        ///     complex-unary envelope of NumPy's values (about 3 ULP), not its bits. A NaN start makes every sample of its
        ///     progression NaN, as NumPy's <c>result *= out_sign</c> does.</para>
        ///     <para>API-parity batch 10: this one member replaces the scalar <c>Geomspace(double, ...)</c> and
        ///     <c>Geomspace(Complex, ...)</c> members; a C# number or <see cref="Complex"/> converts to a 0-d array.</para>
        /// </remarks>
        public abstract NDArray Geomspace(NDArray start, NDArray stop, long num = 50, bool endpoint = true, DType dtype = null, int axis = 0);
    }
}
