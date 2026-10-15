using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        // np.geomspace — numbers spaced evenly on a log scale (a GEOMETRIC
        // progression), port of NumPy 2.4.2 numpy/_core/function_base.py::geomspace.
        //
        // NumPy's algorithm, followed here operation-for-operation:
        //     if any(start==0) or any(stop==0): raise "Geometric sequence cannot include zero"
        //     out_sign = sign(start)                 # rotate start onto the positive real axis
        //     start /= out_sign;  stop /= out_sign   # so log10 is real for real inputs
        //     result = logspace(log10(start), log10(stop), num, endpoint, base=10)
        //     if num>0: result[0]=start; if num>1 and endpoint: result[-1]=stop   # exact endpoints
        //     result *= out_sign                     # undo the rotation
        //
        // REAL path (this file's double overload) is BIT-EXACT vs NumPy: out_sign is
        // ±1.0, so the rotate (start/out_sign) and the final multiply are exact, and
        // result[0]=start/result[-1]=stop land the ORIGINAL endpoints byte-for-byte
        // (start/out_sign*out_sign == start). The interior rides logspace's bit-exact
        // Math.Pow/Math.Log10 == npy_pow/npy_log10 (win-amd64). Mixed-sign real inputs
        // (e.g. geomspace(-1,1)) reproduce NumPy's [-1, nan, nan, 1] automatically:
        // rotating makes stop negative, log10(negative)=NaN poisons the interior, and
        // the endpoints are overwritten.
        //
        // COMPLEX path (dtype=complex OR the Complex-input overload) computes in the
        // complex128 domain — the interior differs from the real path (NumPy's
        // geomspace(1,8,4,dtype=complex)[1] == 1.9999999999999998, not the real path's
        // exact 2.0), so the domain genuinely matters. It composes NumSharp's
        // NumPy-tuned complex power (npy_cpow port) and complex log10, so it is
        // ACCURATE to NumPy within the documented ≤3-ULP complex-unary envelope but
        // NOT byte-reproducible (like np.sinc's complex path) — allclose, not bit-exact.
        //
        // SCOPE: scalar start/stop only (NumSharp's linspace family is scalar-only;
        // array-like start/stop and the `axis` placement of a new sample axis are a
        // separate, larger feature). `axis` is kept for signature parity and validated
        // against the 1-D output, but is a no-op for scalar inputs.
        // =====================================================================

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
        {
            // NumPy order: the zero-check runs FIRST (before the num check inside logspace and before the trailing
            // axis moveaxis), so a zero endpoint wins over a negative num which wins over a bad axis.
            if (start == 0.0 || stop == 0.0)
                throw new ValueError("Geometric sequence cannot include zero");
            ValidateSampleCount(num);
            ValidateSpacingAxis(axis);

            // NumPy's dt = result_type(start, stop, float(num), zeros((), dtype)). For scalar-double inputs that is
            // float64 UNLESS a complex dtype forces complex128 — the only way a real-input geomspace enters the
            // complex domain. Every other dtype (float32/float16/int) still computes float64 then casts at the end.
            if (dtype != null && dtype.GetTypeCode() == NPTypeCode.Complex)
                return GeomspaceComplexCore(new Complex(start, 0.0), new Complex(stop, 0.0), num, endpoint, dtype);

            NDArray ret = GeomspaceRealCore(start, stop, num, endpoint);
            return CastSpacingResult(ret, dtype);
        }

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
        {
            // NumPy checks each endpoint against 0+0j (a purely-imaginary value is NOT zero and is allowed).
            if (start == Complex.Zero || stop == Complex.Zero)
                throw new ValueError("Geometric sequence cannot include zero");
            ValidateSampleCount(num);
            ValidateSpacingAxis(axis);

            return GeomspaceComplexCore(start, stop, num, endpoint, dtype);
        }

        /// <summary>
        ///     The bit-exact float64 core: fills a fresh 1-D buffer following NumPy's geomspace fused into ONE pass —
        ///     the sign rotation, the log-space interior (<c>out_sign * 10 ** (log_start + i*step)</c>), and the exact
        ///     endpoint overwrites — with NO intermediate arrays.
        /// </summary>
        /// <param name="start">The (non-zero) starting value.</param>
        /// <param name="stop">The (non-zero) final value.</param>
        /// <param name="num">Number of samples (already validated non-negative).</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the final sample.</param>
        /// <returns>A fresh, C-contiguous, owning float64 array of length <paramref name="num"/>.</returns>
        /// <remarks>
        ///     out_sign is ±1.0 (sign of a non-zero real), so writing the ORIGINAL <paramref name="start"/> /
        ///     <paramref name="stop"/> at the endpoints equals NumPy's <c>(x/out_sign)*out_sign</c> byte-for-byte, and
        ///     <c>out_sign * Math.Pow(10, log_start + i*step)</c> equals NumPy's <c>power(10, linspace_val) * out_sign</c>
        ///     (multiplication commutes; ±1 is exact). Mixed-sign inputs give NaN interiors exactly as NumPy does.
        /// </remarks>
        private static NDArray GeomspaceRealCore(double start, double stop, long num, bool endpoint)
        {
            // out_sign = np.sign(start): ±1 for a finite/inf non-zero, but NaN for a NaN start (start != 0 is
            // guaranteed and -0.0 was rejected by the ==0.0 check). Propagating NaN here is load-bearing — NumPy's
            // trailing `result *= out_sign` turns EVERY element (endpoints included) into NaN for a NaN start.
            double outSign = double.IsNaN(start) ? start : start < 0.0 ? -1.0 : 1.0;
            double startR = start / outSign;                 // |start| on the positive real axis (NaN if start is NaN)
            double stopR = stop / outSign;                   // may be negative if start,stop have opposite signs
            double logStart = Math.Log10(startR);
            double logStop = Math.Log10(stopR);              // NaN when stopR < 0 (mixed-sign inputs)

            NDArray ret = new NDArray(NPTypeCode.Double, new Shape(num), false);
            if (num == 0)
                return ret;

            // step is undefined for num==1 (div==0); the loop never reads it there (only the i==0 endpoint runs).
            double step = num > 1 ? (logStop - logStart) / (endpoint ? num - 1.0 : num) : 0.0;

            unsafe
            {
                double* addr = (double*)ret.Address;
                for (long i = 0; i < num; i++)
                {
                    // NumPy computes the ROTATED sequence (startR at 0, stopR at the endpoint, 10**exponent in
                    // between) and only THEN does `result *= out_sign` — so the multiply is applied UNIFORMLY,
                    // including the endpoints. For out_sign = ±1 this is bit-identical to writing the original
                    // start/stop (startR*out_sign == start exactly — double negation is exact), and for out_sign =
                    // NaN it correctly yields NaN everywhere, matching NumPy's NaN-start behaviour.
                    double baseV;
                    if (i == 0)
                        baseV = startR;
                    else if (endpoint && i == num - 1)
                        baseV = stopR;
                    else
                        baseV = Math.Pow(10.0, logStart + i * step);
                    addr[i] = baseV * outSign;
                }
            }

            return ret;
        }

        /// <summary>
        ///     The complex128 core (spiral): composes NumSharp's NumPy-tuned complex power over a complex linspace of
        ///     log-space exponents, then applies the exact endpoints and the sign rotation in one in-place pass.
        /// </summary>
        /// <param name="start">The (non-zero) complex starting value.</param>
        /// <param name="stop">The (non-zero) complex final value.</param>
        /// <param name="num">Number of samples (already validated non-negative).</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the final sample.</param>
        /// <param name="dtype">The requested output dtype (null → complex128); applied as the final cast.</param>
        /// <returns>A fresh, owning array of length <paramref name="num"/> in the requested dtype.</returns>
        /// <remarks>
        ///     Accurate but not byte-reproducible (see the class comment). Follows NumPy's exact operation order:
        ///     result[0]=start_r / result[-1]=stop_r are set, THEN the whole array is multiplied by out_sign — so the
        ///     endpoints become <c>start_r*out_sign</c> / <c>stop_r*out_sign</c> (not necessarily the original inputs,
        ///     since complex divide-then-multiply is not exact), matching NumPy.
        /// </remarks>
        private static NDArray GeomspaceComplexCore(Complex start, Complex stop, long num, bool endpoint, DType dtype)
        {
            // sign(z) = z / |z| (NumPy 2.x complex sign); z != 0 guaranteed by the caller.
            Complex outSign = start / Complex.Abs(start);
            Complex startR = start / outSign;
            Complex stopR = stop / outSign;
            Complex logStart = NDComplexMath.Log10(startR);
            Complex logStop = NDComplexMath.Log10(stopR);

            // Complex linspace of the log-space exponents (a straight line between the two log10 values).
            NDArray cLin = ComplexLinspaceCore(logStart, logStop, num, endpoint);
            // 10 ** exponent through NumSharp's npy_cpow port (best parity with NumPy's own complex power).
            NDArray base10 = NDArray.Scalar(new Complex(10.0, 0.0));
            NDArray result = np.power(base10, cLin);
            // cLin/base10 were pure inputs to power (which materialises a fresh result); reclaim them now.
            base10.Dispose();
            cLin.Dispose();

            if (num > 0)
            {
                unsafe
                {
                    Complex* addr = (Complex*)result.Address;
                    for (long i = 0; i < num; i++)
                    {
                        // Endpoints take the ROTATED start/stop; the sign multiply below then un-rotates them, exactly
                        // as NumPy's `result[0]=start; result[-1]=stop; result *= out_sign` does.
                        Complex v = i == 0 ? startR
                                  : endpoint && i == num - 1 ? stopR
                                  : addr[i];
                        addr[i] = v * outSign;
                    }
                }
            }

            return CastSpacingResult(result, dtype);
        }

        /// <summary>
        ///     A complex128 <c>linspace(start, stop, num, endpoint)</c> fill loop (used only by the geomspace spiral):
        ///     <c>start + i*step</c> with the last sample pinned to <paramref name="stop"/> under <paramref name="endpoint"/>.
        /// </summary>
        /// <param name="start">The first sample.</param>
        /// <param name="stop">The last sample when <paramref name="endpoint"/> is True.</param>
        /// <param name="num">Number of samples (already validated non-negative).</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the final sample.</param>
        /// <returns>A fresh, C-contiguous, owning complex128 array of length <paramref name="num"/>.</returns>
        private static NDArray ComplexLinspaceCore(Complex start, Complex stop, long num, bool endpoint)
        {
            NDArray ret = new NDArray(NPTypeCode.Complex, new Shape(num), false);
            if (num == 0)
                return ret;
            if (num == 1)
            {
                // linspace(a, b, 1) == [a] for either endpoint (mirrors the real linspace core).
                ret.SetAtIndex(start, 0);
                return ret;
            }

            Complex step = (stop - start) / (endpoint ? num - 1.0 : num);
            unsafe
            {
                Complex* addr = (Complex*)ret.Address;
                for (long i = 0; i < num; i++)
                    addr[i] = endpoint && i == num - 1 ? stop : start + (double)i * step;
            }

            return ret;
        }
    }
}
