using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return evenly spaced numbers over a specified interval.<br></br>
        ///     Returns num evenly spaced samples, calculated over the interval[start, stop].<br></br>
        ///     The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint is False.</param>
        /// <param name="num">Number of samples to generate. Default is 50. Must be non-negative.</param>
        /// <param name="endpoint">If True, stop is the last sample. Otherwise, it is not included. Default is True.</param>
        /// <param name="dtype">
        ///     The dtype of the output array — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     If not given (null), the output is float64 (NumPy infers float from the other input arguments).
        /// </param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples (<see cref="TensorEngine.Linspace"/> describes its
        ///     dtype and which results are views).</returns>
        /// <exception cref="ValueError">A negative <paramref name="num"/> (<c>Number of samples, -1, must be
        ///     non-negative.</c>), checked first, as NumPy checks it.</exception>
        /// <exception cref="ValueError">A <paramref name="device"/> other than <c>"cpu"</c> (NumPy's <c>Device not understood.
        ///     Only "cpu" is allowed, but received: ...</c>), checked after the count.</exception>
        /// <exception cref="NotSupportedException">A <paramref name="dtype"/> without a storage lane (a datetime
        ///     descriptor); a text dtype formats the float64 samples, as NumPy's astype does.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        ///     <para>Computed by the default engine's <see cref="TensorEngine.Linspace"/>: there is no operand to choose
        ///     another.</para></remarks>
        public static NDArray linspace(double start, double stop, long num, bool endpoint = true, DType dtype = null, string device = null)
            => TensorEngine.Default.Linspace(start, stop, num, endpoint, dtype, device);

        /// <summary>
        ///     Return evenly spaced numbers over a specified interval.<br></br>
        ///     Returns num evenly spaced samples, calculated over the interval[start, stop].<br></br>
        ///     The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint is False.</param>
        /// <param name="num">Number of samples to generate. Default is 50. Must be non-negative.</param>
        /// <param name="endpoint">If True, stop is the last sample. Otherwise, it is not included. Default is True.</param>
        /// <param name="dtype">The type of the output array. If dtype is not given, infer the data type from the other input arguments.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples, exactly as the
        ///     <see cref="linspace(double, double, long, bool, DType, string)"/> overload returns it.</returns>
        /// <exception cref="ValueError">A negative <paramref name="num"/>, or a <paramref name="device"/> other than
        ///     <c>"cpu"</c>.</exception>
        /// <exception cref="NotSupportedException">A <paramref name="dtype"/> without a storage lane (a datetime
        ///     descriptor).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        ///     <para>The <see cref="int"/> count only widens to <see cref="long"/>; it exists so an <c>int</c> literal binds
        ///     without a cast.</para></remarks>
        public static NDArray linspace(double start, double stop, int num, bool endpoint = true, DType dtype = null, string device = null)
            => linspace(start, stop, (long)num, endpoint, dtype, device);

        /// <summary>
        ///     Return evenly spaced numbers over a specified interval.<br></br>
        ///     Returns num evenly spaced samples, calculated over the interval[start, stop].<br></br>
        ///     The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint is False.</param>
        /// <param name="num">Number of samples to generate. Default is 50. Must be non-negative.</param>
        /// <param name="endpoint">If True, stop is the last sample. Otherwise, it is not included. Default is True.</param>
        /// <param name="dtype">The type of the output array. If dtype is not given, infer the data type from the other input arguments.</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples: float64 unless <paramref name="dtype"/> names another,
        ///     even though the bounds are float32 (NumPy's default).</returns>
        /// <exception cref="ValueError">A negative <paramref name="num"/>.</exception>
        /// <exception cref="NotSupportedException">A <paramref name="dtype"/> without a storage lane (a datetime
        ///     descriptor).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        ///     <para>The float32 bounds widen to float64 before the samples are computed, so pass
        ///     <c>dtype: np.float32</c> for a float32 result.</para></remarks>
        public static NDArray linspace(float start, float stop, long num, bool endpoint = true, DType dtype = null)
        {
            // NumPy: linspace always returns float64 by default, regardless of input types
            return linspace((double)start, (double)stop, num, endpoint, dtype);
        }

        /// <summary>
        ///     Return evenly spaced numbers over a specified interval.<br></br>
        ///     Returns num evenly spaced samples, calculated over the interval[start, stop].<br></br>
        ///     The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">The starting value of the sequence.</param>
        /// <param name="stop">The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint is False.</param>
        /// <param name="num">Number of samples to generate. Default is 50. Must be non-negative.</param>
        /// <param name="endpoint">If True, stop is the last sample. Otherwise, it is not included. Default is True.</param>
        /// <param name="dtype">The type of the output array. If dtype is not given, infer the data type from the other input arguments.</param>
        /// <returns>A new 1-D array of <paramref name="num"/> samples: float64 unless <paramref name="dtype"/> names another,
        ///     even though the bounds are float32 (NumPy's default).</returns>
        /// <exception cref="ValueError">A negative <paramref name="num"/>.</exception>
        /// <exception cref="NotSupportedException">A <paramref name="dtype"/> without a storage lane (a datetime
        ///     descriptor).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        ///     <para>The float32 bounds widen to float64 and the <see cref="int"/> count to <see cref="long"/>, so pass
        ///     <c>dtype: np.float32</c> for a float32 result.</para></remarks>
        public static NDArray linspace(float start, float stop, int num, bool endpoint = true, DType dtype = null)
            => linspace(start, stop, (long)num, endpoint, dtype);
    }
}
