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
        /// <param name="start">Start of interval. The interval includes this value.</param>
        /// <param name="stop">End of interval. The interval does not include this value.</param>
        /// <param name="step">Spacing between values. Default is 1.</param>
        /// <param name="dtype">The type of the output array. If null, infers from inputs (int64 for integers, float64 for floats).</param>
        /// <returns>Array of evenly spaced values.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arange.html</remarks>
        public static NDArray arange(double start, double stop, double step, Type dtype)
        {
            return arange(start, stop, step, dtype?.GetTypeCode() ?? NPTypeCode.Empty);
        }

        /// <summary>
        /// Return evenly spaced values within a given interval.
        ///
        /// Values are generated within the half-open interval [start, stop)
        /// (in other words, the interval including start but excluding stop).
        /// </summary>
        /// <param name="start">Start of interval. The interval includes this value.</param>
        /// <param name="stop">End of interval. The interval does not include this value.</param>
        /// <param name="step">Spacing between values. Default is 1.</param>
        /// <param name="dtype">The type of the output array. If Empty, infers from inputs (int64 for integers, float64 for floats).</param>
        /// <returns>Array of evenly spaced values.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arange.html</remarks>
        public static NDArray arange(double start, double stop, double step, NPTypeCode dtype)
        {
            if (Math.Abs(step) < 1e-15)
                throw new ArgumentException("step can't be 0", nameof(step));

            // NumPy formula: length = ceil((stop - start) / step)
            // This works for both positive and negative step
            double tmp_len = (stop - start) / step;

            // Handle empty array cases:
            // - Positive step with start >= stop
            // - Negative step with start <= stop
            // - Division resulting in negative or zero
            if (tmp_len <= 0)
            {
                var emptyType = dtype == NPTypeCode.Empty ? NPTypeCode.Double : dtype;
                return new NDArray(emptyType, Shape.Vector(0), false);
            }

            long length = (long)Math.Ceiling(tmp_len);

            // Infer dtype if not specified - default to float64 for this overload
            if (dtype == NPTypeCode.Empty)
                dtype = NPTypeCode.Double;

            var nd = new NDArray(dtype, Shape.Vector(length), false);

            // Fill the array: values[i] = start + i * step
            // This works for both positive and negative step (NumPy's approach)
            unsafe
            {
                switch (dtype)
                {
                    case NPTypeCode.Boolean:
                    {
                        var addr = (bool*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (start + i * step) != 0;
                        break;
                    }
                    case NPTypeCode.Byte:
                    {
                        var addr = (byte*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (byte)(start + i * step);
                        break;
                    }
                    case NPTypeCode.Int16:
                    {
                        var addr = (short*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (short)(start + i * step);
                        break;
                    }
                    case NPTypeCode.UInt16:
                    {
                        var addr = (ushort*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (ushort)(start + i * step);
                        break;
                    }
                    case NPTypeCode.Int32:
                    {
                        var addr = (int*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (int)(start + i * step);
                        break;
                    }
                    case NPTypeCode.UInt32:
                    {
                        var addr = (uint*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (uint)(start + i * step);
                        break;
                    }
                    case NPTypeCode.Int64:
                    {
                        var addr = (long*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (long)(start + i * step);
                        break;
                    }
                    case NPTypeCode.UInt64:
                    {
                        var addr = (ulong*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (ulong)(start + i * step);
                        break;
                    }
                    case NPTypeCode.Char:
                    {
                        var addr = (char*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (char)(start + i * step);
                        break;
                    }
                    case NPTypeCode.Single:
                    {
                        var addr = (float*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (float)(start + i * step);
                        break;
                    }
                    case NPTypeCode.Double:
                    {
                        var addr = (double*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = start + i * step;
                        break;
                    }
                    case NPTypeCode.Decimal:
                    {
                        var addr = (decimal*)nd.Unsafe.Address;
                        for (long i = 0; i < length; i++)
                            addr[i] = (decimal)(start + i * step);
                        break;
                    }
                    default:
                        throw new NotSupportedException($"dtype {dtype} is not supported");
                }
            }

            return nd;
        }

        #endregion

        #region Overloads with dtype parameter

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="dtype">The type of the output array.</param>
        /// <returns>Array of evenly spaced values from 0 to stop-1.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arange.html</remarks>
        public static NDArray arange(double stop, Type dtype)
            => arange(0, stop, 1, dtype);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="dtype">The type of the output array.</param>
        /// <returns>Array of evenly spaced values from 0 to stop-1.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arange.html</remarks>
        public static NDArray arange(double stop, NPTypeCode dtype)
            => arange(0, stop, 1, dtype);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="dtype">The type of the output array.</param>
        /// <returns>Array of evenly spaced values.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arange.html</remarks>
        public static NDArray arange(double start, double stop, Type dtype)
            => arange(start, stop, 1, dtype);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="dtype">The type of the output array.</param>
        /// <returns>Array of evenly spaced values.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.arange.html</remarks>
        public static NDArray arange(double start, double stop, NPTypeCode dtype)
            => arange(start, stop, 1, dtype);

        #endregion

        #region Original overloads (type inferred from C# parameter types)

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <returns>Array of evenly spaced float values.</returns>
        public static NDArray arange(float stop)
            => arange(0f, stop, 1f);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <returns>Array of evenly spaced double values.</returns>
        public static NDArray arange(double stop)
            => arange(0d, stop, 1d);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="step">Spacing between values. Default is 1.</param>
        /// <returns>Array of evenly spaced float values.</returns>
        public static NDArray arange(float start, float stop, float step = 1)
            => arange(start, stop, step, NPTypeCode.Single);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="step">Spacing between values. Default is 1.</param>
        /// <returns>Array of evenly spaced double values.</returns>
        public static NDArray arange(double start, double stop, double step = 1)
            => arange(start, stop, step, NPTypeCode.Double);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <returns>Array of evenly spaced int64 values.</returns>
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
        /// <remarks>NumPy 2.x returns int64 for integer arange.</remarks>
        public static NDArray arange(int start, int stop, int step = 1)
            => arange((long)start, (long)stop, (long)step);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <returns>Array of evenly spaced int64 values.</returns>
        /// <remarks>NumPy 2.x returns int64 for integer arange.</remarks>
        public static NDArray arange(long stop)
            => arange(0L, stop, 1L);

        /// <summary>
        /// Return evenly spaced values within a given interval.
        /// </summary>
        /// <param name="start">Start of interval (inclusive).</param>
        /// <param name="stop">End of interval (exclusive).</param>
        /// <param name="step">Spacing between values. Default is 1.</param>
        /// <returns>Array of evenly spaced int64 values.</returns>
        /// <remarks>NumPy 2.x returns int64 for integer arange.</remarks>
        public static NDArray arange(long start, long stop, long step = 1)
            => arange(start, stop, step, NPTypeCode.Int64);

        #endregion
    }
}
