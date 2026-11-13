using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        #region NumPy's signature

        /// <summary>
        ///     NumPy's <c>np.arange([start,] stop[, step], dtype=None, *, device=None)</c>, reading its arguments as NumPy
        ///     does: evenly spaced values in the half-open interval <c>[start, stop)</c>, in the dtype NumPy discovers from
        ///     the arguments themselves.
        /// </summary>
        /// <param name="start">The start, or with no <paramref name="stop"/> the stop (<c>np.arange(5)</c> is
        ///     <c>[0, 5)</c>). A C# number is the Python number it is written as (a <c>float</c> is a Python float, a
        ///     <c>ulong</c> a Python int by its value); a NumPy scalar (<c>np.float32(0.1)</c>) or a 0-d array is itself.</param>
        /// <param name="stop">The stop (excluded), or null: then <paramref name="start"/> is the stop and 0 the start.</param>
        /// <param name="step">The spacing, or null for 1 (<c>np.arange(5, step: 2)</c> is <c>[0, 2, 4]</c>).</param>
        /// <param name="dtype">The result dtype, or null for NumPy's discovery: int64 promoted with each argument's own
        ///     dtype - int64 for Python ints, float64 for a Python float, a float32 scalar or a uint64 value, complex128
        ///     for a complex.</param>
        /// <param name="device">Only <c>"cpu"</c> and null are accepted (Array-API parity).</param>
        /// <returns>A new 1-D array; empty when the interval holds no value.</returns>
        /// <exception cref="TypeError">Neither <paramref name="start"/> nor <paramref name="stop"/> given (<c>arange()
        ///     requires stop to be specified.</c>); a text dtype or a text argument (<c>arange() not supported for inputs
        ///     with DType ...</c>); bool with more than two elements; a complex interval into a real dtype (<c>must be real
        ///     number, not complex</c>); an argument with dimensions (<c>only 0-dimensional arrays can be converted to
        ///     Python scalars</c>).</exception>
        /// <exception cref="ZeroDivisionError">A zero Python step (<c>division by zero</c>, <c>float division by
        ///     zero</c>, <c>complex division by zero</c>).</exception>
        /// <exception cref="ValueError"><paramref name="device"/> is not <c>"cpu"</c>; a NaN length (<c>arange: cannot
        ///     compute length</c>); a length past npy_intp, a NumPy scalar's zero step or a Python int out of a NumPy
        ///     integer's bounds (<c>Maximum allowed size exceeded</c>); an argument of more than one element or none (NumPy's
        ///     truth-value texts); an allocation past the byte limit.</exception>
        /// <exception cref="OverflowError">The first or second element does not fit <paramref name="dtype"/>
        ///     (<c>Python integer 1000 out of bounds for int8</c>).</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> has no NumSharp storage yet (datetime64,
        ///     timedelta64).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's <c>array_arange</c> then <c>PyArray_ArangeObj</c>
        ///     (<see cref="UnmanagedStorage.ArangeObj(NDArray, NDArray, NDArray, DType)"/>): the length is
        ///     <c>ceil((stop - start) / step)</c> computed in the arguments' own arithmetic - exact for Python ints, NEP 50
        ///     for a NumPy scalar (<c>np.arange(0, 1, np.float32(0.1))</c> divides in float32) -, the first two elements
        ///     are <c>start</c> and <c>start + step</c> stored into the dtype, and the rest <c>start + i * delta</c> in the
        ///     dtype's own arithmetic. A complex dtype takes the smaller ceiling of the quotient's two parts
        ///     (<c>np.arange(new Complex(1, 2))</c> is <c>[0j]</c>).
        ///     </para>
        ///     <para>
        ///     The typed overloads stay beside this one and win for plain C# numbers (<c>np.arange(5)</c>,
        ///     <c>np.arange(0.0, 1.0, 0.1)</c>, <c>np.arange(5, np.float32)</c>), with the same answers for those numbers;
        ///     this signature is the one NumPy's keywords reach (<c>step:</c>, <c>device:</c> alone), and the one a NumPy
        ///     scalar, a complex or a 0-d array binds. C# cannot tell <c>np.arange(start: 5)</c> from <c>np.arange(5)</c>:
        ///     both read 5 as the stop, where NumPy raises the TypeError for the keyword form. A C# number argument becomes a
        ///     small 0-d array per call.
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        /// </remarks>
        public static NDArray arange(NDArray start = null, NDArray stop = null, NDArray step = null, DType dtype = null,
                                     string device = null)
        {
            // npy_parse_arguments converts the device first: a bad device fails before any argument is read.
            DeviceArguments.ValidateDevice(device);
            // array_arange: a lone stop becomes the start (np.arange(stop=5) is np.arange(5)), and with neither there is
            // nothing to count to.
            if (stop is null)
            {
                if (start is null)
                    throw new TypeError("arange() requires stop to be specified.");
            }
            else if (start is null)
            {
                start = stop;
                stop = null;
            }
            return UnmanagedStorage.ArangeObj(start, stop, step, dtype);
        }

        #endregion

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
        /// <exception cref="ZeroDivisionError"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: float
        ///     division by zero</c>).</exception>
        /// <exception cref="TypeError"><paramref name="dtype"/> is a text dtype (<c>arange() not supported for inputs with
        ///     DType ...</c>), or bool with more than two elements.</exception>
        /// <exception cref="OverflowError"><paramref name="start"/> or <paramref name="start"/> + <paramref name="step"/>
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
            DeviceArguments.ValidateDevice(device);
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
        /// <exception cref="OverflowError"><paramref name="start"/> or <paramref name="start"/> + 1 does not fit an integer
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
        /// <param name="stop">End of interval (exclusive), a Python float (its exact float32 value).</param>
        /// <returns>Array of evenly spaced float64 values (NumPy's dtype for a float argument).</returns>
        /// <exception cref="ValueError">The length is NaN or past npy_intp, or the array is past the byte limit.</exception>
        /// <remarks>float64 since API-parity batch 13 (parity QA-02), as <see cref="arange(float, float, float)"/>.</remarks>
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
        /// <param name="start">Start of interval (inclusive), a Python float (its exact float32 value).</param>
        /// <param name="stop">End of interval (exclusive), a Python float.</param>
        /// <param name="step">Spacing between values, a Python float. Default is 1.</param>
        /// <returns>Array of evenly spaced float64 values: <c>np.arange(0f, 1f, 0.1f)[3]</c> is
        ///     <c>0.30000000447034836</c>, NumPy's <c>np.arange(np.float32(0), np.float32(1), np.float32(0.1))[3]</c>.</returns>
        /// <exception cref="ValueError">The length is NaN or past npy_intp, or the array is past the byte limit.</exception>
        /// <exception cref="ZeroDivisionError"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: float
        ///     division by zero</c>).</exception>
        /// <remarks>
        ///     A C# <c>float</c> is a Python float in NumSharp's boundary map, and NumPy's dtype for Python floats is float64:
        ///     the bounds widen exactly, the length and the elements are the float form's
        ///     (<see cref="UnmanagedStorage.ArangeObj(double, double, double, DType)"/>). Until API-parity batch 13 (parity
        ///     QA-02) this overload returned float32 computed in float32. For a float32 result pass the dtype:
        ///     <c>np.arange(0, 1, 0.1, np.float32)</c>.
        /// </remarks>
        public static NDArray arange(float start, float stop, float step = 1)
            => UnmanagedStorage.ArangeObj((double)start, stop, step, null);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="step">Spacing between values. Default is 1.</param>
        /// <returns>Array of evenly spaced double values.</returns>
        /// <exception cref="ValueError">The length is NaN or past npy_intp, or the array is past the byte limit.</exception>
        /// <exception cref="ZeroDivisionError"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: float
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
        /// <exception cref="ZeroDivisionError"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: division by
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
        /// <exception cref="ZeroDivisionError"><paramref name="step"/> is zero (NumPy's <c>ZeroDivisionError: division by
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

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive), a Python int (its exact value).</param>
        /// <returns>Array of evenly spaced values: int64 while <paramref name="stop"/> fits int64, float64 past it (NumPy's
        ///     dtype for a Python int beyond int64 promoted with int64).</returns>
        /// <exception cref="ValueError">A length past npy_intp (<c>Maximum allowed size exceeded</c>), or the array is past
        ///     the byte limit.</exception>
        /// <remarks>Added in API-parity batch 13 so a C# <c>ulong</c> stays a Python int: it bound the float overload, which
        ///     rounded it to float32 first.</remarks>
        [NDScoped /* the literal arrays of the bounds are this call's own */]
        public static NDArray arange(ulong stop)
            => UnmanagedStorage.ArangeObj(NDArray.PythonLiteral(stop), null, null, null);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive), a Python int.</param>
        /// <param name="stop">End of interval (exclusive), a Python int.</param>
        /// <param name="step">Spacing between values, a Python int. Default is 1.</param>
        /// <returns>Array of evenly spaced values: int64 while every bound fits int64, else float64 (NumPy:
        ///     <c>np.arange(2**64 - 6, 2**64 - 1, 2)</c> is three float64 elements).</returns>
        /// <exception cref="ZeroDivisionError"><paramref name="step"/> is zero (<c>division by zero</c>).</exception>
        /// <exception cref="ValueError">A length past npy_intp (<c>Maximum allowed size exceeded</c>), or the array is past
        ///     the byte limit.</exception>
        /// <remarks>
        ///     NumPy's <c>PyArray_ArangeObj</c> over Python ints
        ///     (<see cref="UnmanagedStorage.ArangeObj(NDArray, NDArray, NDArray, DType)"/>): the length from exact integer
        ///     arithmetic, the dtype from the values (a value past <c>long.MaxValue</c> is uint64 in NumPy's discovery, and
        ///     uint64 with int64 is float64). Added in API-parity batch 13 so a C# <c>ulong</c> stays a Python int (it bound
        ///     the float overload, whose float32 rounding made <c>np.arange(18446744073709551610UL, 18446744073709551615UL,
        ///     2UL)</c> empty). A negative step needs a signed overload: <c>-1</c> is no <c>ulong</c>.
        /// </remarks>
        [NDScoped /* the literal arrays of the bounds are this call's own */]
        public static NDArray arange(ulong start, ulong stop, ulong step = 1)
            => UnmanagedStorage.ArangeObj(NDArray.PythonLiteral(start), NDArray.PythonLiteral(stop), NDArray.PythonLiteral(step), null);

        #endregion
    }
}
