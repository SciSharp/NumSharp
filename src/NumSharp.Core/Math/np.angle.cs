using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the angle (phase) of the complex argument, element-wise — the counterclockwise angle
        ///     from the positive real axis, in the range (-pi, pi]. One of the four basic complex-number
        ///     accessors (with <see cref="real"/>, <see cref="imag"/> and <see cref="conjugate"/>) — the
        ///     standard post-FFT PHASE spectrum: for <c>A = np.fft.fft(a)</c>, <c>np.angle(A)</c> is the
        ///     phase spectrum.
        /// </summary>
        /// <param name="z">A complex number or sequence of complex numbers (any real dtype is also accepted).</param>
        /// <param name="deg">Return the angle in degrees if <c>true</c>, radians (default) if <c>false</c>.</param>
        /// <returns>
        ///     The phase angle, computed as <c>arctan2(imag, real)</c> element-wise (so it follows the
        ///     <see cref="arctan2(NDArray, NDArray, NDArray, NDArray, System.Nullable{NPTypeCode})"/> IEEE
        ///     conventions at magnitude zero and at the infinities). complex128 -&gt; float64. A REAL input
        ///     yields <c>0</c> for a positive value and <c>pi</c> for a negative one (<c>arctan2(0, z)</c>),
        ///     with NumPy's per-dtype float tier: bool/int32+/float64/complex -&gt; float64, int8/uint8/float16
        ///     -&gt; float16, int16/uint16/float32 -&gt; float32.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.angle.html</remarks>
        public static NDArray angle(NDArray z, bool deg = false)
        {
            NDArray a;
            if (z.typecode == NPTypeCode.Complex)
            {
                // NumPy: a = arctan2(z.imag, z.real). Both lanes are float64 VIEWS onto the Complex
                // storage (the FFT driver's strided complex->real read), so the product is float64 and
                // bit-identical to NumPy's arctan2 (verified against NumPy 2.4.2).
                var zimag = new NDArray(z.Storage.AliasComplexLane(imaginary: true));
                var zreal = new NDArray(z.Storage.AliasComplexLane(imaginary: false));
                a = np.arctan2(zimag, zreal);
            }
            else
            {
                // NumPy: a = arctan2(0, z) with a WEAK Python int 0. Its only possible values are
                // {+0, pi, nan} (the sign of z decides), so computing arctan2 in float64 then casting to
                // NumPy's angle tier is bit-exact with NumPy computing directly at that tier (verified);
                // this also reuses the existing arctan2 kernel per NumPy's own composition.
                a = np.arctan2(NDArray.Scalar(0.0), z.astype(NPTypeCode.Double));
                NPTypeCode tier = AngleRealTier(z.typecode);
                if (tier != NPTypeCode.Double)
                    a = a.astype(tier);
            }

            if (deg)
                // NumPy: `a *= 180 / pi`, an in-place multiply at a's own dtype by the WEAK double 180/pi.
                // NumSharp's `NDArray * double` keeps a's tier and computes at that precision, which is
                // bit-identical to NumPy (verified at float16/float32/float64 — e.g. float16 pi -> 179.9,
                // NOT 180, because the multiply rounds in float16).
                a = a * (180.0 / Math.PI);

            return a;
        }

        /// <summary>
        ///     NumPy's per-dtype float tier for the REAL-input angle path, i.e. the dtype of
        ///     <c>arctan2(0, z)</c> where <c>0</c> is a weak Python int (probed against NumPy 2.4.2).
        ///     Note <c>bool</c> resolves to float64 (unlike the ordinary unary-math tier's float16).
        ///     Char (uint16-like) follows uint16 -&gt; float32; Decimal (no NumPy analog) -&gt; float64.
        /// </summary>
        private static NPTypeCode AngleRealTier(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Half:
                    return NPTypeCode.Half;
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Single:
                    return NPTypeCode.Single;
                default:
                    // Boolean, Int32/UInt32/Int64/UInt64, Double, Decimal.
                    return NPTypeCode.Double;
            }
        }
    }
}
