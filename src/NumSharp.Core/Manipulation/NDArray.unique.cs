using System;
using System.Linq;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray unique()
        {
            switch (typecode)
            {
#if _REGEN
	        %foreach supported_dtypes,supported_dtypes_lowercase%
	        case NPTypeCode.#1: return unique<#2>();
            %
            default: throw new NotSupportedException();
#else
                case NPTypeCode.Boolean: return unique<bool>();
                case NPTypeCode.Byte: return unique<byte>();
                case NPTypeCode.Int16: return unique<short>();
                case NPTypeCode.UInt16: return unique<ushort>();
                case NPTypeCode.Int32: return unique<int>();
                case NPTypeCode.UInt32: return unique<uint>();
                case NPTypeCode.Int64: return unique<long>();
                case NPTypeCode.UInt64: return unique<ulong>();
                case NPTypeCode.Char: return unique<char>();
                case NPTypeCode.Double: return unique<double>();
                case NPTypeCode.Single: return unique<float>();
                case NPTypeCode.Decimal: return unique<decimal>();
                default: throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        /// Find the unique elements of an array.
        /// 
        /// Returns the sorted unique elements of an array.There are three optional outputs in addition to the unique elements:
        /// * the indices of the input array that give the unique values
        /// * the indices of the unique array that reconstruct the input array
        /// * the number of times each unique value comes up in the input array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        protected NDArray unique<T>() where T : unmanaged
        {
            unsafe
            {
                var hashset = new Hashset<T>();
                if (Shape.IsContiguous)
                {
                    var src = (T*)this.Address;
                    int len = this.size;
                    for (int i = 0; i < len; i++) //we do not use Parellel.For because of the internal lock
                        hashset.Add(src[i]);

                    var ret = new NDArray(typeof(T), Shape.Vector(hashset.Count));
                    Hashset<T>.CopyTo(hashset, (ArraySlice<T>)ret.Array);
                    return ret;
                }
                else
                {
                    int len = this.size;
                    var flat = this.flat;
                    var src = (T*)flat.Address;
                    Func<int, int> getOffset = flat.Shape.GetOffset_1D;
                    Parallel.For(0, len, i => hashset.Add(src[getOffset(i)])); //we use parallel to speed up offset computation

                    var dst = new NDArray(typeof(T), Shape.Vector(hashset.Count));
                    Hashset<T>.CopyTo(hashset, (ArraySlice<T>)dst.Array);
                    return dst;
                }
            }
        }
    }
}
