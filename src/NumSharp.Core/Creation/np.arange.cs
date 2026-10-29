using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        #region Core implementation with dtype support

        /// <summary>
        /// Return evenly spaced values within a given interval.
        ///
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// </summary>
        /// <param name="start">Start of interval, a Python float. The interval includes this value.</param>
        /// <param name="stop">End of interval, a Python float. The interval does not include this value.</param>
        /// <param name="step">Spacing between values, a Python float. Default is 1.</param>
        /// <param name="dtype">
        ///     The dtype of the output array — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     If null, NumPy's discovery from Python floats: float64.
        /// </param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of evenly spaced values.</returns>
        /// <exception cref="ValueError"><paramref name="device"/> is not <c>"cpu"</c>; the length is NaN (<c>arange: cannot
        ///     compute length</c>) or past npy_intp (<c>Maximum allowed size exceeded</c>); the array is past the byte limit; or
        ///     a NaN is stored into an integer dtype.</exception>
        /// <exception cref="DivideByZeroException"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: float
        ///     division by zero</c>).</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a text dtype (<c>arange() not supported for inputs with
        ///     DType ...</c>), or bool with more than two elements.</exception>
        /// <exception cref="OverflowException"><paramref name="start"/> or <paramref name="start"/> + <paramref name="step"/>
        ///     does not fit an integer dtype (<c>Python integer 300 out of bounds for int8</c>).</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage yet (datetime64,
        ///     timedelta64).</exception>
        /// <remarks>
        ///     NumPy's <c>PyArray_ArangeObj</c> (<see cref="UnmanagedStorage.ArangeObj(double, double, double, DType)"/>): the
        ///     length is <c>ceil((stop - start) / step)</c> in float arithmetic, the first two elements are <c>start</c> and
        ///     <c>start + step</c> stored into the dtype, and the rest are <c>start + i * delta</c> computed in the dtype's
        ///     own arithmetic, with <c>delta</c> the difference of the two stored elements. So <c>np.arange(0.1, 2, 0.3)[3]</c>
        ///     is <c>1.0000000000000002</c>, a float32 ramp is computed in float32 and a float16 ramp in float32 rounded per
        ///     element, as in NumPy; an integer ramp wraps in the dtype past its first two elements.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        /// </remarks>
        public static NDArray arange(double start, double stop, double step, DType dtype, string device = null)
        {
            ValidateDevice(device);
            return UnmanagedStorage.ArangeObj(start, stop, step, dtype);
        }

        #endregion

        #region Overloads with dtype parameter

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="dtype">The dtype of the output array (a <see cref="Type"/>, <see cref="NPTypeCode"/>, dtype string or <see cref="DType"/> — all convert implicitly).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of evenly spaced values from 0 to stop-1.</returns>
        /// <exception cref="ValueError"><paramref name="device"/> is not <c>"cpu"</c>; the length is NaN or past npy_intp; or
        ///     the array is past the byte limit.</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a text dtype, or bool with more than two elements.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage yet (datetime64,
        ///     timedelta64).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arange.html</remarks>
        public static NDArray arange(double stop, DType dtype, string device = null)
            => arange(0, stop, 1, dtype, device);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="dtype">The dtype of the output array (a <see cref="Type"/>, <see cref="NPTypeCode"/>, dtype string or <see cref="DType"/> — all convert implicitly).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of evenly spaced values.</returns>
        /// <exception cref="ValueError"><paramref name="device"/> is not <c>"cpu"</c>; the length is NaN or past npy_intp; or
        ///     the array is past the byte limit.</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a text dtype, or bool with more than two elements.</exception>
        /// <exception cref="OverflowException"><paramref name="start"/> or <paramref name="start"/> + 1 does not fit an integer
        ///     dtype.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage yet (datetime64,
        ///     timedelta64).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arange.html</remarks>
        public static NDArray arange(double start, double stop, DType dtype, string device = null)
            => arange(start, stop, 1, dtype, device);

        #endregion

        #region Original overloads (type inferred from C# parameter types)

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <returns>Array of evenly spaced float values.</returns>
        /// <exception cref="ValueError">The length is NaN or past npy_intp, or the array is past the byte limit.</exception>
        public static NDArray arange(float stop)
            => arange(0f, stop, 1f);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <returns>Array of evenly spaced double values.</returns>
        /// <exception cref="ValueError">The length is NaN or past npy_intp, or the array is past the byte limit.</exception>
        public static NDArray arange(double stop)
            => arange(0d, stop, 1d);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="step">Spacing between values. Default is 1.</param>
        /// <returns>Array of evenly spaced float values.</returns>
        /// <exception cref="ValueError">The length is NaN or past npy_intp, or the array is past the byte limit.</exception>
        /// <exception cref="DivideByZeroException"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: float
        ///     division by zero</c>).</exception>
        public static NDArray arange(float start, float stop, float step = 1)
            => arange(start, stop, step, NPTypeCode.Single);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="step">Spacing between values. Default is 1.</param>
        /// <returns>Array of evenly spaced double values.</returns>
        /// <exception cref="ValueError">The length is NaN or past npy_intp, or the array is past the byte limit.</exception>
        /// <exception cref="DivideByZeroException"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: float
        ///     division by zero</c>).</exception>
        public static NDArray arange(double start, double stop, double step = 1)
            => arange(start, stop, step, NPTypeCode.Double);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <returns>Array of evenly spaced int64 values.</returns>
        /// <exception cref="ValueError">The array is past the byte limit (<c>array is too big</c>).</exception>
        /// <remarks>NumPy 2.x returns int64 for integer arange.</remarks>
        public static NDArray arange(int stop)
            => arange(0L, (long)stop, 1L);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="step">Spacing between values. Default is 1.</param>
        /// <returns>Array of evenly spaced int64 values.</returns>
        /// <exception cref="ValueError">The array is past the byte limit (<c>array is too big</c>).</exception>
        /// <exception cref="DivideByZeroException"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: division by
        ///     zero</c>).</exception>
        /// <remarks>NumPy 2.x returns int64 for integer arange.</remarks>
        public static NDArray arange(int start, int stop, int step = 1)
            => arange((long)start, (long)stop, (long)step);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <returns>Array of evenly spaced int64 values.</returns>
        /// <exception cref="ValueError">The array is past the byte limit (<c>array is too big</c>).</exception>
        /// <remarks>NumPy 2.x returns int64 for integer arange.</remarks>
        public static NDArray arange(long stop)
            => arange(0L, stop, 1L);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive), a Python int.</param>
        /// <param name="stop">End of interval (exclusive), a Python int.</param>
        /// <param name="step">Spacing between values, a Python int. Default is 1.</param>
        /// <returns>Array of evenly spaced int64 values.</returns>
        /// <exception cref="DivideByZeroException"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: division by
        ///     zero</c>).</exception>
        /// <exception cref="ValueError">The array is past the byte limit (<c>array is too big</c>).</exception>
        /// <remarks>
        ///     NumPy 2.x returns int64 for integer arange. NumPy's <c>PyArray_ArangeObj</c> over Python ints
        ///     (<see cref="UnmanagedStorage.ArangeObj(long, long, long, DType)"/>): the length is the ceiling of Python's
        ///     correctly rounded <c>(stop - start) / step</c> and the elements are exact at every magnitude
        ///     (<c>np.arange(2**60, 2**60 + 3)</c> has three); a quotient whose ceiling is exactly <c>2**63</c> gives an empty
        ///     array, as NumPy's on x86.
        /// </remarks>
        public static NDArray arange(long start, long stop, long step = 1)
            => UnmanagedStorage.ArangeObj(start, stop, step, null);

        #endregion
    }
}
