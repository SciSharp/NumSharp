using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

// =============================================================================
// SINGLE-PRECISION companion to PocketFFT.Twiddle.cs.
//
// pocketfft is templated on T0 (the arithmetic type). numpy 2.x registers the
// fft ufuncs for BOTH float and double, so a float32/float16 input is computed
// in *single precision* (cfftp<float>/rfftp<float>) and returned as complex64 —
// which is why numpy's float32 fft is NOT the double result rounded (the norm
// scaling and butterfly accumulation all happen in float). NumSharp has no
// complex64 dtype (issue #569), so the single-precision result is UPCAST into
// complex128/float64 by the driver; the VALUES are bit-identical to numpy's
// complex64 (upcast), leaving only the result dtype divergent.
//
// This file ports pocketfft's cmplx<float>. The twiddle TABLE is NOT duplicated:
// sincos_2pibyn<float> keeps Thigh==double (sizeof(float) < sizeof(double)), so
// its v1/v2 tables and their combination are computed in double exactly as the
// double engine's SinCos2PiByN already does — only the final lookup is cast to
// float. The single engine therefore reuses SinCos2PiByN's (double) indexer and
// narrows at the call site: (float)twiddle[idx].r / .i, which is bit-identical to
// pocketfft's operator[] for T==float (`cmplx<T>(T(...), T(...))`; the imaginary
// negation is sign-exact so (float)(-x) == -(float)(x)).
// =============================================================================

namespace NumSharp.Fourier
{
    /// <summary>
    /// Port of pocketfft's <c>cmplx&lt;T0&gt;</c> for the single-precision engine (T0 == float).
    /// Field layout (<c>r</c> then <c>i</c>) is identical to <see cref="Cmplx"/> and
    /// <see cref="System.Numerics.Complex"/>; the driver reinterprets a <c>CmplxF*</c> as an
    /// interleaved <c>float*</c> for the real-transform packing (mirroring the double engine).
    /// All operators reproduce pocketfft's exact real/imag operation order, in <b>float</b>.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CmplxF
    {
        public float r, i;

        [MethodImpl(OptimizeAndInline)]
        public CmplxF(float r_, float i_) { r = r_; i = i_; }

        // cmplx + cmplx  -> {r+o.r, i+o.i}
        [MethodImpl(OptimizeAndInline)]
        public static CmplxF operator +(in CmplxF a, in CmplxF b) => new CmplxF(a.r + b.r, a.i + b.i);

        // cmplx - cmplx  -> {r-o.r, i-o.i}
        [MethodImpl(OptimizeAndInline)]
        public static CmplxF operator -(in CmplxF a, in CmplxF b) => new CmplxF(a.r - b.r, a.i - b.i);

        // cmplx * scalar -> {r*s, i*s}
        [MethodImpl(OptimizeAndInline)]
        public static CmplxF operator *(in CmplxF a, float s) => new CmplxF(a.r * s, a.i * s);

        // cmplx * cmplx  -> {r*o.r - i*o.i, r*o.i + i*o.r}
        [MethodImpl(OptimizeAndInline)]
        public static CmplxF operator *(in CmplxF a, in CmplxF b)
            => new CmplxF(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r);

        /// <summary>
        /// pocketfft <c>special_mul&lt;fwd&gt;</c> (twiddle multiply with direction-dependent
        /// conjugation), in float. fwd: <c>(r*o.r+i*o.i, i*o.r-r*o.i)</c>; bwd: <c>(r*o.r-i*o.i,
        /// r*o.i+i*o.r)</c>.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static CmplxF SpecialMul(bool fwd, in CmplxF v1, in CmplxF v2)
            => fwd
                ? new CmplxF(v1.r * v2.r + v1.i * v2.i, v1.i * v2.r - v1.r * v2.i)
                : new CmplxF(v1.r * v2.r - v1.i * v2.i, v1.r * v2.i + v1.i * v2.r);
    }

    /// <summary>
    /// Narrowing helpers that turn the (double) <see cref="SinCos2PiByN"/> table into the
    /// single-precision twiddles pocketfft's <c>cmplx&lt;float&gt;</c> engine stores. Kept in one
    /// place so the "combine in double, cast to float" rule (pocketfft's <c>operator[]</c> for
    /// T==float) is applied identically everywhere the single engine reads a twiddle.
    /// </summary>
    internal static class TwiddleF
    {
        [MethodImpl(OptimizeAndInline)]
        public static CmplxF At(SinCos2PiByN tw, long idx)
        {
            Cmplx t = tw[idx];                       // combined in double, exactly as pocketfft's Thigh==double
            return new CmplxF((float)t.r, (float)t.i); // (float)(-x) == -(float)(x): sign-exact narrowing
        }
    }
}
