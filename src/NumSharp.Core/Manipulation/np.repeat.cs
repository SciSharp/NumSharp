using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Repeat elements of an array.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.</param>
        /// <param name="axis"></param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html</remarks>
        public static NDArray repeat(NDArray a, int repeats) //TODO! , int axis = -1
        {
            int size = a.size * repeats;
            a = a.ravel();
            switch (a.GetTypeCode)
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1:
	            {
		            var ret = new NDArray(NPTypeCode.#1, Shape.Vector(size));
                    var data = a.Data<#2>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean:
	            {
		            var ret = new NDArray(NPTypeCode.Boolean, Shape.Vector(size));
                    var data = a.Data<bool>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.Byte:
	            {
		            var ret = new NDArray(NPTypeCode.Byte, Shape.Vector(size));
                    var data = a.Data<byte>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.Int16:
	            {
		            var ret = new NDArray(NPTypeCode.Int16, Shape.Vector(size));
                    var data = a.Data<short>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.UInt16:
	            {
		            var ret = new NDArray(NPTypeCode.UInt16, Shape.Vector(size));
                    var data = a.Data<ushort>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.Int32:
	            {
		            var ret = new NDArray(NPTypeCode.Int32, Shape.Vector(size));
                    var data = a.Data<int>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.UInt32:
	            {
		            var ret = new NDArray(NPTypeCode.UInt32, Shape.Vector(size));
                    var data = a.Data<uint>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.Int64:
	            {
		            var ret = new NDArray(NPTypeCode.Int64, Shape.Vector(size));
                    var data = a.Data<long>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.UInt64:
	            {
		            var ret = new NDArray(NPTypeCode.UInt64, Shape.Vector(size));
                    var data = a.Data<ulong>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.Char:
	            {
		            var ret = new NDArray(NPTypeCode.Char, Shape.Vector(size));
                    var data = a.Data<char>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.Double:
	            {
		            var ret = new NDArray(NPTypeCode.Double, Shape.Vector(size));
                    var data = a.Data<double>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.Single:
	            {
		            var ret = new NDArray(NPTypeCode.Single, Shape.Vector(size));
                    var data = a.Data<float>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            case NPTypeCode.Decimal:
	            {
		            var ret = new NDArray(NPTypeCode.Decimal, Shape.Vector(size));
                    var data = a.Data<decimal>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(i * repeats + j, data[i]);
                    return ret;
	            }
	            default:
		            throw new NotSupportedException();
#endif
            }
        } 
        
        /// <summary>
        ///     Repeat a scalar.
        /// </summary>
        /// <param name="a">Input scalar.</param>
        /// <param name="repeats">The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.</param>
        /// <param name="axis"></param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html</remarks>
        public static NDArray repeat<T>(T a, int repeats) //TODO! , int axis = -1
        {
            int size = repeats;
            switch (typeof(T).GetTypeCode())
            {
#if _REGEN
	            %foreach supported_currently_supported,supported_currently_supported_lowercase%
	            case NPTypeCode.#1:
	            {
		            var ret = new NDArray(NPTypeCode.#1, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean:
	            {
		            var ret = new NDArray(NPTypeCode.Boolean, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.Byte:
	            {
		            var ret = new NDArray(NPTypeCode.Byte, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.Int16:
	            {
		            var ret = new NDArray(NPTypeCode.Int16, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.UInt16:
	            {
		            var ret = new NDArray(NPTypeCode.UInt16, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.Int32:
	            {
		            var ret = new NDArray(NPTypeCode.Int32, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.UInt32:
	            {
		            var ret = new NDArray(NPTypeCode.UInt32, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.Int64:
	            {
		            var ret = new NDArray(NPTypeCode.Int64, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.UInt64:
	            {
		            var ret = new NDArray(NPTypeCode.UInt64, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.Char:
	            {
		            var ret = new NDArray(NPTypeCode.Char, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.Double:
	            {
		            var ret = new NDArray(NPTypeCode.Double, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.Single:
	            {
		            var ret = new NDArray(NPTypeCode.Single, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            case NPTypeCode.Decimal:
	            {
		            var ret = new NDArray(NPTypeCode.Decimal, Shape.Vector(size));
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(j, a);
                    return ret;
	            }
	            default:
		            throw new NotSupportedException();
#endif
            }
        }
    }
}
