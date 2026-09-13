using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return numbers spaced evenly on a log scale.<br></br>
        ///     In linear space, the sequence starts at <c>base ** start</c> and ends with <c>base ** stop</c>
        ///     (see <paramref name="endpoint"/>). Equivalent to <c>power(base, linspace(start, stop, num, endpoint))</c>.
        /// </summary>
        /// <param name="start"><c>base ** start</c> is the starting value of the sequence.</param>
        /// <param name="stop">
        ///     <c>base ** stop</c> is the final value of the sequence, unless <paramref name="endpoint"/> is False.
        ///     In that case <c>num + 1</c> values are spaced over the interval in log-space, of which all but the
        ///     last are returned (so the step size changes when <paramref name="endpoint"/> is False).
        /// </param>
        /// <param name="num">Number of samples to generate. Default 50. Must be non-negative.</param>
        /// <param name="endpoint">If True (default) <paramref name="stop"/> is the last sample; otherwise it is excluded.</param>
        /// <param name="base">
        ///     The base of the log space (default 10.0). The step between the elements in <c>log_base(samples)</c>
        ///     is uniform. A negative base with fractional exponents yields NaN (real power of a negative base),
        ///     matching NumPy.
        /// </param>
        /// <param name="dtype">
        ///     The dtype of the output — one descriptor parameter, like NumPy's <c>dtype=</c>: a C# <see cref="Type"/>,
        ///     an <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert
        ///     implicitly. If null (default) the result is float64. The computation ALWAYS runs in float64 and is then
        ///     cast to <paramref name="dtype"/> — so an integer dtype TRUNCATES toward zero (NumPy's
        ///     <c>power(...).astype(dtype)</c>), unlike <see cref="linspace(double,double,long,bool,DType,string)"/> which floors.
        /// </param>
        /// <param name="axis">
        ///     The axis in the result to store the samples. Relevant only for array-like inputs, which NumSharp's scalar
        ///     <c>logspace</c> does not take, so it is a no-op here — but it is still validated against the 1-D output:
        ///     only <c>0</c> and <c>-1</c> are accepted (NumPy <c>moveaxis</c> destination), anything else raises
        ///     <see cref="AxisError"/> with NumPy's verbatim "destination:" message.
        /// </param>
        /// <returns>A 1-D array of <paramref name="num"/> samples, equally spaced on a log scale.</returns>
        /// <exception cref="ValueError">If <paramref name="num"/> is negative (NumPy's ValueError).</exception>
        /// <exception cref="AxisError">If <paramref name="axis"/> is neither 0 nor -1 (out of bounds for the 1-D output).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logspace.html</remarks>
        public static NDArray logspace(double start, double stop, int num = 50, bool endpoint = true, double @base = 10.0, DType dtype = null, int axis = 0)
            => logspace(start, stop, (long)num, endpoint, @base, dtype, axis);

        /// <summary>
        ///     Return numbers spaced evenly on a log scale (<see cref="long"/>-count overload — see the <see cref="int"/>
        ///     overload for the full contract). Equivalent to <c>power(base, linspace(start, stop, num, endpoint))</c>.
        /// </summary>
        /// <param name="start"><c>base ** start</c> is the starting value of the sequence.</param>
        /// <param name="stop"><c>base ** stop</c> is the final value of the sequence, unless <paramref name="endpoint"/> is False.</param>
        /// <param name="num">Number of samples to generate. Must be non-negative.</param>
        /// <param name="endpoint">If True (default) <paramref name="stop"/> is the last sample; otherwise it is excluded.</param>
        /// <param name="base">The base of the log space (default 10.0).</param>
        /// <param name="dtype">The dtype of the output (null → float64). Computation runs in float64 and is then cast (integer dtype truncates toward zero).</param>
        /// <param name="axis">Validated against the 1-D output; only 0 and -1 are accepted (no-op for scalar inputs).</param>
        /// <returns>A 1-D array of <paramref name="num"/> samples, equally spaced on a log scale.</returns>
        /// <exception cref="ValueError">If <paramref name="num"/> is negative (NumPy's ValueError).</exception>
        /// <exception cref="AxisError">If <paramref name="axis"/> is neither 0 nor -1.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logspace.html</remarks>
        public static NDArray logspace(double start, double stop, long num, bool endpoint = true, double @base = 10.0, DType dtype = null, int axis = 0)
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
        ///     The endpoint is written as <c>base ** stop</c> directly (NumPy overwrites <c>y[-1] = stop</c> before the
        ///     power), avoiding accumulated <c>start + i*step</c> drift. Addition/multiplication commute bit-for-bit
        ///     with NumPy's <c>arange*step + start</c>, and <see cref="Math.Pow(double,double)"/> is the same
        ///     <c>ucrtbase</c> routine NumPy's <c>npy_pow</c> calls on win-amd64, so the float64 result is byte-exact.
        /// </remarks>
        private static NDArray LogspaceCore(double start, double stop, long num, bool endpoint, double @base)
        {
            // fillZeros:false — every element is written below, so zeroing the buffer first would be wasted work.
            NDArray ret = new NDArray(NPTypeCode.Double, new Shape(num), false);

            if (num == 0)
                return ret;

            if (num == 1)
            {
                // NumPy: linspace(start, stop, 1) == [start] for either endpoint, so logspace == [base ** start].
                ret.SetAtIndex(Math.Pow(@base, start), 0);
                return ret;
            }

            // div is num-1 when the endpoint is included (so stop lands exactly on the last sample), else num.
            double step = (stop - start) / (endpoint ? num - 1.0 : num);

            unsafe
            {
                double* addr = (double*)ret.Address;
                for (long i = 0; i < num; i++)
                {
                    // Exponent: `stop` verbatim on the last sample under endpoint (matches NumPy's y[-1]=stop),
                    // else the evenly-spaced `start + i*step`.
                    double y = endpoint && i == num - 1 ? stop : start + i * step;
                    addr[i] = Math.Pow(@base, y);
                }
            }

            return ret;
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
        /// <param name="result">The freshly-computed array in the computation domain (float64 or complex128; owned, may be returned as-is).</param>
        /// <param name="dtype">The requested output dtype, or null to keep the computation domain.</param>
        /// <returns>The result in the requested dtype (the same instance when the dtype already matches or is null).</returns>
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
            return result.astype(dtype, false);
        }
    }
}
