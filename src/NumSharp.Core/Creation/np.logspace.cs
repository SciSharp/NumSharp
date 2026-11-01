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
            => TensorEngine.Default.Logspace(start, stop, num, endpoint, @base, dtype, axis);
    }
}
