using System;
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

            if (size.IsEmpty || size.IsScalar)
                return NDArray.Scalar(randomizer.NextLong(low, high), typecode);

            var nd = new NDArray(dtype, size); //allocation called inside.
            switch (typecode)
            {
#if _REGEN1
                %foreach supported_numericals,supported_numericals_lowercase%
                case NPTypeCode.#1:
                {
                    var data = (ArraySlice<#2>)nd.Array;
                    for (int i = 0; i < data.Count; i++)
                        data[i] = Converts.To#1(randomizer.NextLong(low, high));
                    
                    break;
                }
                %
#else
                case NPTypeCode.Byte:
                {
                    var data = (ArraySlice<byte>)nd.Array;
                    for (int i = 0; i < data.Count; i++)
                        data[i] = Converts.ToByte(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var data = (ArraySlice<int>)nd.Array;
                    for (int i = 0; i < data.Count; i++)
                        data[i] = Converts.ToInt32(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var data = (ArraySlice<long>)nd.Array;
                    for (int i = 0; i < data.Count; i++)
                        data[i] = Converts.ToInt64(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Single:
                {
                    var data = (ArraySlice<float>)nd.Array;
                    for (int i = 0; i < data.Count; i++)
                        data[i] = Converts.ToSingle(randomizer.NextLong(low, high));
                    
                    break;
                }
                case NPTypeCode.Double:
                {
                    var data = (ArraySlice<double>)nd.Array;
                    for (int i = 0; i < data.Count; i++)
                        data[i] = Converts.ToDouble(randomizer.NextLong(low, high));
                    
                    break;
                }
#endif
            }

            return nd;
        }
    }
}
