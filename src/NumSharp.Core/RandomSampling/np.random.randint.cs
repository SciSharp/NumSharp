using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Return random integers from the "discrete uniform" distribution in the half-open interval [low, high).
        /// </summary>
        /// <param name="low">Lowest (signed) integer to be drawn from the distribution (unless high is not provided, in which case this parameter is one above the highest such integer).</param>
        /// <param name="high">If provided, one above the largest (signed) integer to be drawn from the distribution. If not provided (-1), results are from [0, low).</param>
        /// <param name="size">Output shape. If None, a single value is returned.</param>
        /// <param name="dtype">Desired dtype of the result. Default is np.int32.</param>
        /// <returns>Random integers from the appropriate distribution, or a single such random int if size not provided.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
        /// </remarks>
        public NDArray randint(long low, long high = -1, Shape size = default, Type dtype = null)
        {
            dtype = dtype ?? np.int32;
            var typecode = dtype.GetTypeCode();
            if (high == -1)
            {
                high = low;
                low = 0;
            }

            // Validate bounds against dtype (NumPy behavior)
            ValidateRandintBounds(low, high, typecode);

            // Determine if we need int64 range
            bool needsLongRange = high > int.MaxValue || low < int.MinValue || (high - low) > int.MaxValue;

            if (size.IsEmpty || size.IsScalar)
            {
                var value = needsLongRange
                    ? randomizer.NextLong(low, high)
                    : randomizer.Next((int)low, (int)high);
                return NDArray.Scalar(value, typecode);
            }

            var nd = new NDArray(dtype, size);

            if (needsLongRange)
            {
                // Use NextLong for large ranges
                FillRandintLong(nd, low, high, typecode);
            }
            else
            {
                // Use Next for int32 ranges (faster)
                FillRandintInt(nd, (int)low, (int)high, typecode);
            }

            return nd;
        }

        /// <summary>
        ///     Validates that low/high are within bounds for the specified dtype.
        /// </summary>
        private static void ValidateRandintBounds(long low, long high, NPTypeCode typecode)
        {
            // Get min/max for the dtype, and the max allowed high value
            // high is exclusive, so high can be at most max+1 (but watch for overflow)
            (long min, long max, long maxHigh) = typecode switch
            {
                NPTypeCode.Byte => (byte.MinValue, byte.MaxValue, byte.MaxValue + 1L),
                NPTypeCode.Int16 => (short.MinValue, short.MaxValue, short.MaxValue + 1L),
                NPTypeCode.UInt16 => (ushort.MinValue, ushort.MaxValue, ushort.MaxValue + 1L),
                NPTypeCode.Int32 => (int.MinValue, int.MaxValue, (long)int.MaxValue + 1L),
                NPTypeCode.UInt32 => (uint.MinValue, uint.MaxValue, (long)uint.MaxValue + 1L),
                // For int64/uint64, we can't represent max+1 in long, so use long.MaxValue
                // NumPy actually allows high up to 2^63 for int64 (which we represent as high=long.MinValue due to overflow)
                NPTypeCode.Int64 => (long.MinValue, long.MaxValue, long.MaxValue),
                NPTypeCode.UInt64 => (0, long.MaxValue, long.MaxValue),
                _ => (long.MinValue, long.MaxValue, long.MaxValue)
            };

            // NumPy error: "high is out of bounds for {dtype}"
            // For int64/uint64, we allow any valid high since we can't overflow check properly
            if (typecode != NPTypeCode.Int64 && typecode != NPTypeCode.UInt64)
            {
                if (high > maxHigh)
                    throw new ValueError("high is out of bounds for " + typecode.AsNumpyDtypeName());
            }

            // NumPy error: "low is out of bounds for {dtype}"
            if (low < min)
                throw new ValueError("low is out of bounds for " + typecode.AsNumpyDtypeName());

            // NumPy error: "low >= high"
            if (low >= high)
                throw new ValueError("low >= high");
        }

        private void FillRandintInt(NDArray nd, int low, int high, NPTypeCode typecode)
        {
            switch (typecode)
            {
                case NPTypeCode.Byte:
                {
                    var data = (ArraySlice<byte>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (byte)randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.Int16:
                {
                    var data = (ArraySlice<short>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (short)randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var data = (ArraySlice<ushort>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (ushort)randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var data = (ArraySlice<int>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    var data = (ArraySlice<uint>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (uint)randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var data = (ArraySlice<long>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    var data = (ArraySlice<ulong>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (ulong)randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.Char:
                {
                    var data = (ArraySlice<char>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (char)randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.Double:
                {
                    var data = (ArraySlice<double>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.Single:
                {
                    var data = (ArraySlice<float>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = randomizer.Next(low, high);
                    break;
                }
                case NPTypeCode.Decimal:
                {
                    var data = (ArraySlice<decimal>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = randomizer.Next(low, high);
                    break;
                }
            }
        }

        private void FillRandintLong(NDArray nd, long low, long high, NPTypeCode typecode)
        {
            // Use NextLong for all types when range exceeds int32
            // Then cast the result to the target type
            switch (typecode)
            {
                case NPTypeCode.Byte:
                {
                    var data = (ArraySlice<byte>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (byte)randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.Int16:
                {
                    var data = (ArraySlice<short>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (short)randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var data = (ArraySlice<ushort>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (ushort)randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var data = (ArraySlice<int>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (int)randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    var data = (ArraySlice<uint>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (uint)randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var data = (ArraySlice<long>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    var data = (ArraySlice<ulong>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (ulong)randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.Char:
                {
                    var data = (ArraySlice<char>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = (char)randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.Double:
                {
                    var data = (ArraySlice<double>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.Single:
                {
                    var data = (ArraySlice<float>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = randomizer.NextLong(low, high);
                    break;
                }
                case NPTypeCode.Decimal:
                {
                    var data = (ArraySlice<decimal>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = randomizer.NextLong(low, high);
                    break;
                }
            }
        }
    }
}
