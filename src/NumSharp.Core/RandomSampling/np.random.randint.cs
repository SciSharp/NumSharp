using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Return random integers from the “discrete uniform” distribution in the half-open interval [low, high).
        /// </summary>
        /// <param name=”low”>Lowest (signed) integer to be drawn from the distribution (unless high is not provided, in which case this parameter is one above the highest such integer).</param>
        /// <param name=”high”>If provided, one above the largest (signed) integer to be drawn from the distribution. If not provided (-1), results are from [0, low).</param>
        /// <param name=”size”>Output shape. If None, a single value is returned.</param>
        /// <param name=”dtype”>Desired dtype of the result. Default is np.int32.</param>
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

            if (size.IsEmpty || size.IsScalar)
                return NDArray.Scalar(randomizer.NextLong(low, high), typecode);

            var nd = new NDArray(dtype, size); //allocation called inside.
            switch (typecode)
            {
#if _REGEN
                %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1:
                {
                    var data = (ArraySlice<#2>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.To#1(randomizer.NextLong(low, high));
                    
                    break;
                }
                %
#else
                case NPTypeCode.Byte:
                {
                    var data = (ArraySlice<byte>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToByte(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Int16:
                {
                    var data = (ArraySlice<short>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToInt16(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var data = (ArraySlice<ushort>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToUInt16(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var data = (ArraySlice<int>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToInt32(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    var data = (ArraySlice<uint>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToUInt32(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var data = (ArraySlice<long>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToInt64(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    var data = (ArraySlice<ulong>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToUInt64(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Char:
                {
                    var data = (ArraySlice<char>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToChar(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Double:
                {
                    var data = (ArraySlice<double>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToDouble(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Single:
                {
                    var data = (ArraySlice<float>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToSingle(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Decimal:
                {
                    var data = (ArraySlice<decimal>)nd.Array;
                    for (long i = 0; i < data.Count; i++)
                        data[i] = Converts.ToDecimal(randomizer.NextLong(low, high));
                    
                    break;
                }
#endif
            }

            return nd;
        }
    }
}
