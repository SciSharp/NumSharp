// NumPy's numpy/_core/function_base.py functions that compute, as DefaultEngine's members: np.linspace, np.logspace and
// np.geomspace (real and complex ends), each function's statements as its np.* facade held them, with their helpers.
// The file sits beside the facades while its statements call np.* (geomspace's complex path raises 10 to its exponents
// with np.power): ratchet R2 counts np.* calls against every file under Backends/ (the np-layer dtype plan, C3); it moves
// under Backends/Default/ in P6.

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        public override NDArray Linspace(double start, double stop, long num, bool endpoint = true, DType dtype = null, string device = null)
        {
            // NumPy's first statements are `num = operator.index(num); if num < 0: raise ValueError(...)`; the device is
            // read later, by the np.arange that builds the ramp. So a negative count wins over a bad device, and neither
            // reaches the allocation (whose own text would be `negative dimensions are not allowed`).
            ValidateSampleCount(num);
            np.ValidateDevice(device);
            // A text dtype (S, U or StringDType: the String storage lane) has no typed fill. NumPy computes the float64
            // samples (issubdtype(dtype, integer) is false: no floor) and its final astype formats them, so
            // np.linspace(0, 1, 5, dtype='U5') is ['0.0', '0.25', '0.5', '0.75', '1.0'].
            if (dtype != null && dtype.GetTypeCode() == NPTypeCode.String)
            {
                // The cast to text always copies (float64 is never a text dtype), so the float64 samples - a view of
                // the fill's buffer for two or more samples - are released here, whatever astype does.
                using NDArray samples = LinspaceCore(start, stop, num, endpoint, NPTypeCode.Double);
                return samples.astype(dtype, false);
            }
            return LinspaceCore(start, stop, num, endpoint, dtype?.GetTypeCode() ?? NPTypeCode.Double);
        }

        /// <summary>
        ///     The storage-lane core behind every public <c>linspace(..., dtype)</c> overload (which all take the single
        ///     <see cref="DType"/> descriptor parameter, like NumPy's <c>dtype=</c>).
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint is False.</param>
        /// <param name="num">Number of samples to generate. Default is 50. Must be non-negative.</param>
        /// <param name="endpoint">If True, stop is the last sample. Otherwise, it is not included. Default is True.</param>
        /// <param name="typeCode">The type of the output array.</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples; a float64 result whose NumPy counterpart is a view
        ///     (<see cref="TensorEngine.Linspace"/>) is an alias that does not own its buffer.</returns>
        /// <exception cref="ArgumentException"><paramref name="typeCode"/> is <see cref="NPTypeCode.Empty"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="typeCode"/> has no fixed-size storage lane (the text
        ///     lane), or no fill below.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        ///     <para>Each sample is NumPy's float64 ramp value (<see cref="LinspaceValue"/>), floored for an integer type,
        ///     then converted in one typed loop per storage lane.</para></remarks>
        private static NDArray LinspaceCore(double start, double stop, long num, bool endpoint, NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentException("Invalid typeCode", nameof(typeCode));

            // NumPy 1.20+ floors the inexact linspace values before casting to an explicitly
            // requested integer dtype (rather than truncating toward zero). Char follows UInt16.
            bool floorInteger = typeCode == NPTypeCode.SByte || typeCode == NPTypeCode.Byte
                                || typeCode == NPTypeCode.Int16 || typeCode == NPTypeCode.UInt16
                                || typeCode == NPTypeCode.Int32 || typeCode == NPTypeCode.UInt32
                                || typeCode == NPTypeCode.Int64 || typeCode == NPTypeCode.UInt64
                                || typeCode == NPTypeCode.Char;

            NDArray ret = new NDArray(typeCode, new Shape(num), false);

            if (num == 0)
                return ret;

            // NumPy's quantities: delta = stop - start, and div = num - 1 with the endpoint, num without. One sample with
            // the endpoint has div 0 and no step; LinspaceValue takes NumPy's branch for it (0 * delta + start), so the
            // division here may give an infinity or a NaN that is never read.
            double delta = stop - start;
            double div = endpoint ? num - 1.0 : num;
            double step = delta / div;

            double Sample(long i)
            {
                // NumPy's float64 ramp value, then the floor NumPy applies before an integer cast.
                double value = LinspaceValue(i, num, endpoint, start, stop, delta, div, step);
                return floorInteger ? Math.Floor(value) : value;
            }

            switch (ret.GetTypeCode)
            {
	            // case NPTypeCode.Boolean:
	            // {
                    // unsafe
                    // {
                        // var addr = (bool*)ret.Address;
                        // for (long i = 0; i < num; i++) addr[i] = (start + i * step) != 0;
                    // }

                    // return ret;
	            // }
	            // %foreach except(supported_dtypes, "Boolean"),except(supported_dtypes_lowercase, "bool")%
	            // case NPTypeCode.#1:
	            // {
                    // unsafe
                    // {
                        // var addr = (#2*)ret.Address;
                         // for (long i = 0; i < num; i++) addr[i] = Converts.To#1(start + i * step);
                    // }

                    // return ret;
	            // }
	            // %
	            // default:
		            // throw new NotSupportedException();
	            case NPTypeCode.Boolean:
	            {
                    unsafe
                    {
                        var addr = (bool*)ret.Address;
                        for (long i = 0; i < num; i++) addr[i] = Sample(i) != 0;
                    }

                    return ret;
	            }
	            case NPTypeCode.Byte:
	            {
                    unsafe
                    {
                        var addr = (byte*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToByte(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.SByte:
	            {
                    unsafe
                    {
                        var addr = (sbyte*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToSByte(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Int16:
	            {
                    unsafe
                    {
                        var addr = (short*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToInt16(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.UInt16:
	            {
                    unsafe
                    {
                        var addr = (ushort*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToUInt16(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Int32:
	            {
                    unsafe
                    {
                        var addr = (int*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToInt32(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.UInt32:
	            {
                    unsafe
                    {
                        var addr = (uint*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToUInt32(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Int64:
	            {
                    unsafe
                    {
                        var addr = (long*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToInt64(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.UInt64:
	            {
                    unsafe
                    {
                        var addr = (ulong*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToUInt64(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Char:
	            {
                    unsafe
                    {
                        var addr = (char*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToChar(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Half:
	            {
                    unsafe
                    {
                        var addr = (Half*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = (Half)Sample(i);
                    }

                    return ret;
	            }
	            case NPTypeCode.Double:
	            {
                    unsafe
                    {
                        var addr = (double*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToDouble(Sample(i));
                    }

                    // One sample with the endpoint: NumPy's div is 0, so it takes the rebinding `y = y * delta` branch
                    // and its result owns its buffer (probed 2.4.2).
                    if (div == 0)
                        return ret;

                    // NumPy's float64 linspace is a reshape-VIEW of its internal arange buffer
                    // mutated in place (y *= step; y += start), and astype(float64, copy=False)
                    // returns that view verbatim — so the result reports OWNDATA=False (probed
                    // 2.4.2; div > 0 here — two or more samples, or one without the endpoint — so the
                    // in-place branch is the one NumPy took). Every other dtype rebinds through a real
                    // astype copy and owns.
                    var view = new NDArray(ret.Storage.Alias()) { TensorEngine = ret.TensorEngine };
                    ret.Dispose();
                    return view;
	            }
	            case NPTypeCode.Single:
	            {
                    unsafe
                    {
                        var addr = (float*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToSingle(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Decimal:
	            {
                    unsafe
                    {
                        var addr = (decimal*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = Converts.ToDecimal(Sample(i));
                    }

                    return ret;
	            }
	            case NPTypeCode.Complex:
	            {
                    unsafe
                    {
                        var addr = (System.Numerics.Complex*)ret.Address;
                         for (long i = 0; i < num; i++) addr[i] = new System.Numerics.Complex(Sample(i), 0);
                    }

                    return ret;
	            }
	            default:
		            throw new NotSupportedException();
            }
        }

        /// <inheritdoc />
        public override NDArray Logspace(double start, double stop, long num = 50, bool endpoint = true, double @base = 10.0, DType dtype = null, int axis = 0)
        {
            // NumPy order: linspace validates `num` (ValueError) BEFORE the trailing moveaxis validates
            // `axis`, so the sample-count error wins when both are bad.
            ValidateSampleCount(num);
            ValidateSpacingAxis(axis);

            // NumPy computes `power(base, y)` in the inferred (float64 for scalar-double inputs) dtype and
            // ONLY THEN casts to `dtype` — computing directly in a narrow dtype would round differently.
            NDArray result = LogspaceCore(start, stop, num, endpoint, @base);
            return CastSpacingResult(result, dtype);
        }

        /// <summary>
        ///     The float64 core behind every <c>logspace</c> overload: fills a fresh 1-D buffer with
        ///     <c>base ** (start + i*step)</c> in a single fused pass (no intermediate <c>linspace</c> array),
        ///     bit-identical to NumPy's <c>power(base, linspace(start, stop, num, endpoint))</c>.
        /// </summary>
        /// <param name="start">Log-space start exponent.</param>
        /// <param name="stop">Log-space stop exponent (the last sample when <paramref name="endpoint"/> is True).</param>
        /// <param name="num">Number of samples (already validated non-negative by the caller).</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is the final exponent.</param>
        /// <param name="base">The base raised to each exponent.</param>
        /// <returns>A fresh, C-contiguous, owning float64 array of length <paramref name="num"/>.</returns>
        /// <remarks>
        ///     The exponents are NumPy's linspace values (<see cref="LinspaceValue"/>): the endpoint is <c>stop</c>
        ///     itself (NumPy overwrites <c>y[-1] = stop</c> before the power), a single exponent is <c>0 * delta + start</c>
        ///     (NaN for an infinite or NaN stop, so <c>base ** y</c> is NaN too), and a step that underflows to zero takes
        ///     NumPy's <c>(i / div) * delta + start</c>. Addition/multiplication commute bit-for-bit with NumPy's
        ///     <c>arange*step + start</c>, and <see cref="Math.Pow(double,double)"/> is the same <c>ucrtbase</c> routine
        ///     NumPy's <c>npy_pow</c> calls on win-amd64, so the float64 result is byte-exact.
        /// </remarks>
        private static NDArray LogspaceCore(double start, double stop, long num, bool endpoint, double @base)
        {
            // fillZeros:false — every element is written below, so zeroing the buffer first would be wasted work.
            NDArray ret = new NDArray(NPTypeCode.Double, new Shape(num), false);

            if (num == 0)
                return ret;

            // NumPy's logspace is power(base, linspace(start, stop, num, endpoint)): the linspace quantities, with div 0
            // (one exponent and the endpoint) handled by LinspaceValue before the step is read.
            double delta = stop - start;
            double div = endpoint ? num - 1.0 : num;
            double step = delta / div;

            unsafe
            {
                double* addr = (double*)ret.Address;
                for (long i = 0; i < num; i++)
                    addr[i] = Math.Pow(@base, LinspaceValue(i, num, endpoint, start, stop, delta, div, step));
            }

            return ret;
        }

        /// <summary>
        ///     Sample <paramref name="i"/> of NumPy's float64 linspace ramp, the value NumPy's statements leave in
        ///     <c>y[i]</c> before an integer dtype's floor and the final cast. Shared by <see cref="LinspaceCore"/> and
        ///     <see cref="LogspaceCore"/> (whose exponents are NumPy's linspace).
        /// </summary>
        /// <param name="i">The sample's index, in <c>[0, num)</c>.</param>
        /// <param name="num">The number of samples, at least 1.</param>
        /// <param name="endpoint">Whether the last of two or more samples is <paramref name="stop"/>.</param>
        /// <param name="start">The first end.</param>
        /// <param name="stop">The second end.</param>
        /// <param name="delta"><c>stop - start</c>.</param>
        /// <param name="div"><c>num - 1</c> with the endpoint, <c>num</c> without.</param>
        /// <param name="step"><c>delta / div</c>; not read when <paramref name="div"/> is 0.</param>
        /// <returns>The sample as NumPy computes it.</returns>
        /// <remarks>
        ///     NumPy's branches, each observable in the result:
        ///     <list type="bullet">
        ///       <item><description>two or more samples with the endpoint: the last is <paramref name="stop"/> itself
        ///       (<c>y[-1] = stop</c>), so no rounding of the ramp reaches it;</description></item>
        ///       <item><description>one sample with the endpoint (<paramref name="div"/> 0, no step): <c>y = y * delta;
        ///       y += start</c>, so <c>0 * delta + start</c> - a NaN when an end is infinite or NaN, and +0.0 for a start of
        ///       -0.0 (one sample without the endpoint gives the same value through the step, which is delta);</description></item>
        ///       <item><description>a step that underflows to zero (gh-5437: a subnormal delta over many samples):
        ///       <c>y /= div; y *= delta</c>, so <c>(i / div) * delta + start</c>, which keeps the spread
        ///       <c>i * step</c> loses;</description></item>
        ///       <item><description>otherwise <c>i * step + start</c>.</description></item>
        ///     </list>
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)] // called once per sample from the fill loops
        private static double LinspaceValue(long i, long num, bool endpoint, double start, double stop, double delta, double div, double step)
        {
            if (endpoint && num > 1 && i == num - 1)
                return stop;
            if (div == 0)
                return i * delta + start;
            if (step == 0)
                return i / div * delta + start;
            return start + i * step;
        }

        /// <summary>
        ///     Validates a sample count for the spacing generators (<c>logspace</c>/<c>geomspace</c>), matching NumPy's
        ///     <c>linspace</c> guard verbatim so a negative count fails before any allocation.
        /// </summary>
        /// <param name="num">The requested number of samples.</param>
        /// <exception cref="ValueError">If <paramref name="num"/> is negative (NumPy's <c>ValueError</c>, verbatim message).</exception>
        internal static void ValidateSampleCount(long num)
        {
            // No paramName: NumPy's ValueError message has no ".NET (Parameter 'num')" suffix, so keep it verbatim.
            if (num < 0)
                throw new ValueError($"Number of samples, {num}, must be non-negative.");
        }

        /// <summary>
        ///     Validates the <c>axis</c> argument for the scalar spacing generators. Their output is always 1-D, so —
        ///     like NumPy's trailing <c>moveaxis(y, 0, axis)</c> — only destinations 0 and -1 are in range; anything
        ///     else raises with NumPy's verbatim "destination:" message reporting the ORIGINAL (un-normalized) axis.
        /// </summary>
        /// <param name="axis">The requested destination axis.</param>
        /// <exception cref="AxisError">If <paramref name="axis"/> is neither 0 nor -1.</exception>
        internal static void ValidateSpacingAxis(int axis)
        {
            if (axis != 0 && axis != -1)
                throw new AxisError($"destination: axis {axis} is out of bounds for array of dimension 1");
        }

        /// <summary>
        ///     Applies NumPy's final <c>result.astype(dtype, copy=False)</c> to a freshly-computed spacing result:
        ///     returns it unchanged when no dtype is requested or the result is already that dtype, otherwise casts
        ///     (float→integer TRUNCATES toward zero; complex→float drops the imaginary part, matching NumPy).
        /// </summary>
        /// <param name="result">
        ///     The freshly-computed array in the computation domain (float64 or complex128). OWNERSHIP TRANSFERS in:
        ///     it is returned as-is when no cast is needed, and DISPOSED here when a cast replaces it, so the caller
        ///     must not touch it afterwards.
        /// </param>
        /// <param name="dtype">The requested output dtype, or null to keep the computation domain.</param>
        /// <returns>The result in the requested dtype (the same instance when the dtype already matches or is null).</returns>
        /// <exception cref="NotSupportedException">The descriptor's class has no storage lane (propagated from <see cref="NDArray.astype(DType, bool)"/>).</exception>
        /// <remarks>
        ///     Domain-agnostic on purpose: <c>logspace</c> feeds a float64 result while <c>geomspace</c>'s complex path
        ///     feeds a complex128 one, and both want "cast only if the dtype actually differs" — so it compares the
        ///     result's own type code, not a hard-coded float64.
        /// </remarks>
        private static NDArray CastSpacingResult(NDArray result, DType dtype)
        {
            // Keep the computation-domain array when no cast is needed (NumPy's astype(..., copy=False) no-op).
            if (dtype == null || result.GetTypeCode == dtype.GetTypeCode())
                return result;
            // A genuine dtype change always allocates fresh storage; copy:false mirrors NumPy's copy=False.
            NDArray cast = result.astype(dtype, false);
            // The computation-domain array is now a dead intermediate: every caller passes a fresh array
            // it owns and returns only what this method hands back. Release it now, or its pooled buffer
            // waits for the finalizer (the FuzzMatrix scope audit caught exactly one escaped buffer per
            // non-float64 logspace/geomspace call). The guard keeps a same-instance astype from being
            // disposed out from under the caller.
            if (!ReferenceEquals(cast, result))
                result.Dispose();
            return cast;
        }

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
        // REAL path (the double overload) is BIT-EXACT vs NumPy: out_sign is
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

        /// <inheritdoc />
        public override NDArray Geomspace(double start, double stop, long num = 50, bool endpoint = true, DType dtype = null, int axis = 0)
        {
            // NumPy order: the zero-check runs FIRST (before the num check inside logspace and before the trailing
            // axis moveaxis), so a zero endpoint wins over a negative num which wins over a bad axis.
            if (start == 0.0 || stop == 0.0)
                throw new ValueError("Geometric sequence cannot include zero");
            // A text dtype fails NumPy's next statements, before the count and the axis are read.
            if (dtype != null && dtype.GetTypeCode() == NPTypeCode.String)
                RefuseGeomspaceTextDtype(NDArray.Scalar(start), num, dtype);
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

        /// <inheritdoc />
        public override NDArray Geomspace(Complex start, Complex stop, long num = 50, bool endpoint = true, DType dtype = null, int axis = 0)
        {
            // NumPy checks each endpoint against 0+0j (a purely-imaginary value is NOT zero and is allowed).
            if (start == Complex.Zero || stop == Complex.Zero)
                throw new ValueError("Geometric sequence cannot include zero");
            // A text dtype fails NumPy's next statements, before the count and the axis are read.
            if (dtype != null && dtype.GetTypeCode() == NPTypeCode.String)
                RefuseGeomspaceTextDtype(NDArray.Scalar(start), num, dtype);
            ValidateSampleCount(num);
            ValidateSpacingAxis(axis);

            return GeomspaceComplexCore(start, stop, num, endpoint, dtype);
        }

        /// <summary>
        ///     NumPy's geomspace statements after the zero check, for a text dtype - they always raise, before the sample
        ///     count and the axis are read: <c>dt = result_type(start, stop, float(num), zeros((), dtype))</c> finds no
        ///     common DType for <c>StringDType</c>; <c>S</c> and <c>U</c> promote to text, NumPy casts the ends to it
        ///     (<c>start.astype(dt, copy=True)</c>), and <c>out_sign = sign(start)</c> has no string loop.
        /// </summary>
        /// <param name="start">The first end as a 0-d array of the ends' dtype (float64, or complex128 for complex ends);
        ///     consumed (disposed) here.</param>
        /// <param name="num">The sample count, which NumPy promotes as <c>float(num)</c>, a weak Python float.</param>
        /// <param name="dtype">The text dtype (the String storage lane).</param>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c>: NumPy's text, which lists the four DType classes
        ///     (<c>The DType &lt;class 'numpy.dtypes.StringDType'&gt; could not be promoted by &lt;class
        ///     'numpy.dtypes.Float64DType'&gt;. ...</c>).</exception>
        /// <exception cref="TypeError">An <c>S</c> or <c>U</c> dtype: sign's refusal (<c>ufunc 'sign' did not contain a loop
        ///     with signature matching types &lt;class 'numpy.dtypes.StrDType'&gt; -&gt; None</c>, a UFuncTypeError in
        ///     NumPy).</exception>
        /// <exception cref="InvalidOperationException">Never: np.sign computing text would be a library bug.</exception>
        private void RefuseGeomspaceTextDtype(NDArray start, long num, DType dtype)
        {
            using (start)
            {
                // The ends' arrays (start, stop: one dtype) are strong, float(num) is weak, the dtype's 0-d zeros strong:
                // DTypePromotion.ResultType takes the same operands as NumPy's call, in its order (the DTypePromotionError
                // text lists them).
                DType dt = DTypePromotion.ResultType(start.dtype, start.dtype, (double)num, dtype);
                using NDArray text = start.astype(dt);
                using NDArray sign = Sign(text);
            }
            throw new InvalidOperationException("np.sign computed the sign of text; NumPy has no string loop for it.");
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
