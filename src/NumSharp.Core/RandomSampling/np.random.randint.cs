using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” interval [low, high). If high is None (the default), then results are from [0, low).
        /// </summary>
        /// <param name="low">Lowest (signed) integer to be drawn from the distribution (unless high=-1, in which case this parameter is one above the highest such integer).</param>
        /// <param name="high">If provided, one above the largest (signed) integer to be drawn from the distribution (see above for behavior if high=-1).</param>
        /// <param name="size">The shape of the array.</param>
        /// <param name="dtype">Desired dtype of the result. All dtypes are determined by their name, i.e., ‘int64’, ‘int’, etc, so byteorder is not available and a specific precision may have different C types depending on the platform. The default value is ‘np.int’.</param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html</remarks>
        public NDArray randint(long low, long high = -1, Shape size = default, Type dtype = null)
        {
            dtype = dtype ?? np.int32;
            var typecode = dtype.GetTypeCode();
            if (high == -1)
            {
                high = low;
                low = 0;
            }

            if (size.IsScalar)
                return NDArray.Scalar(randomizer.NextLong(low, high), typecode);

            var nd = new NDArray(dtype, size); //allocation called inside.
            switch (typecode)
            {
#if _REGEN
                %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1:
                {
                    var data = (ArraySlice<#2>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.To#1(randomizer.NextLong(low, high));
                    
                    break;
                }
                %
#else
                case NPTypeCode.Byte:
                {
                    var data = (ArraySlice<byte>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToByte(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.Int16:
                {
                    var data = (ArraySlice<short>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToInt16(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.UInt16:
                {
                    var data = (ArraySlice<ushort>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToUInt16(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.Int32:
                {
                    var data = (ArraySlice<int>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToInt32(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.UInt32:
                {
                    var data = (ArraySlice<uint>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToUInt32(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.Int64:
                {
                    var data = (ArraySlice<long>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToInt64(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.UInt64:
                {
                    var data = (ArraySlice<ulong>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToUInt64(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.Char:
                {
                    var data = (ArraySlice<char>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToChar(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.Double:
                {
                    var data = (ArraySlice<double>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToDouble(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.Single:
                {
                    var data = (ArraySlice<float>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToSingle(randomizer.NextLong(low, high));

                    break;
                }

                case NPTypeCode.Decimal:
                {
                    var data = (ArraySlice<decimal>)nd.Array;
                    for (int i = 0; i < data.Length; i++)
                        data[i] = Convert.ToDecimal(randomizer.NextLong(low, high));

                    break;
                }
#endif
            }

            return nd;
        }
    }
}
