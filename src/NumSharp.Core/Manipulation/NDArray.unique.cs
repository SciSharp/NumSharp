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
        /// <summary>
        ///     Find the unique elements of an array.<br></br>
        ///     
        ///     Returns the sorted unique elements of an array.There are three optional outputs in addition to the unique elements:<br></br>
        ///     * the indices of the input array that give the unique values<br></br>
        ///     * the indices of the unique array that reconstruct the input array<br></br>
        ///     * the number of times each unique value comes up in the input array<br></br>
        /// </summary>
        /// <returns>The sorted unique values.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html</remarks>
        public NDArray unique()
        {
            switch (typecode)
            {
#if _REGEN1
	        %foreach supported_dtypes,supported_dtypes_lowercase%
	        case NPTypeCode.#1: return unique<#2>();
            %
            default: throw new NotSupportedException();
#else
	        case NPTypeCode.Boolean: return unique<bool>();
	        case NPTypeCode.Byte: return unique<byte>();
	        case NPTypeCode.Int32: return unique<int>();
	        case NPTypeCode.Int64: return unique<long>();
	        case NPTypeCode.Single: return unique<float>();
	        case NPTypeCode.Double: return unique<double>();
            default: throw new NotSupportedException();
#endif
            }
        }

        /// <summary>
        ///     Find the unique elements of an array.<br></br>
        ///     
        ///     Returns the sorted unique elements of an array.There are three optional outputs in addition to the unique elements:<br></br>
        ///     * the indices of the input array that give the unique values<br></br>
        ///     * the indices of the unique array that reconstruct the input array<br></br>
        ///     * the number of times each unique value comes up in the input array<br></br>
        /// </summary>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html</remarks>
        protected NDArray unique<T>() where T : unmanaged
        {
            unsafe
            {
                var hashset = new Hashset<T>();
                if (Shape.IsContiguous)
                {
                    var src = (T*)this.Address;
                    int len = this.size;
                    for (int i = 0; i < len; i++) //we do not use Parellel.For to retain order like numpy does.
                        hashset.Add(src[i]);

                    var dst = new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(hashset.Count));
                    Hashset<T>.CopyTo(hashset, (ArraySlice<T>)dst.Array);
                    return dst;
                }
                else
                {
                    int len = this.size;
                    var flat = this.flat;
                    var src = (T*)flat.Address;
                    Func<int, int> getOffset = flat.Shape.GetOffset_1D;
                    for (int i = 0; i < len; i++) //we do not use Parellel.For to retain order like numpy does.
                        hashset.Add(src[getOffset(i)]);

                    var dst = new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(hashset.Count));
                    Hashset<T>.CopyTo(hashset, (ArraySlice<T>)dst.Array);
                    return dst;
                }
            }
        }
    }
}
