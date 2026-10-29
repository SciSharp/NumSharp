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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        public static NDArray linspace(double start, double stop, long num, bool endpoint = true, DType dtype = null, string device = null)
        {
            ValidateDevice(device);
            return np.EngineOf(null).Linspace(start, stop, num, endpoint, dtype);
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
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
        public static NDArray linspace(float start, float stop, int num, bool endpoint = true, DType dtype = null)
            => linspace(start, stop, (long)num, endpoint, dtype);
    }
}
