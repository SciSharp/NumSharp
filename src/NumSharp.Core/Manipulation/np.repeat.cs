using System;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Utilities;

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
#if _REGEN1
	            %foreach supported_dtypes,supported_dtypes_lowercase%
	            case NPTypeCode.#1:
	            {
		            var ret = new NDArray(NPTypeCode.#1, Shape.Vector(size));
                    var data = a.MakeGeneric<#2>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
                    return ret;
	            }
	            %
	            default:
		            throw new NotSupportedException();
#else
	            case NPTypeCode.Boolean:
	            {
		            var ret = new NDArray(NPTypeCode.Boolean, Shape.Vector(size));
                    var data = a.MakeGeneric<bool>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
                    return ret;
	            }
	            case NPTypeCode.Byte:
	            {
		            var ret = new NDArray(NPTypeCode.Byte, Shape.Vector(size));
                    var data = a.MakeGeneric<byte>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
                    return ret;
	            }
	            case NPTypeCode.Int32:
	            {
		            var ret = new NDArray(NPTypeCode.Int32, Shape.Vector(size));
                    var data = a.MakeGeneric<int>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
                    return ret;
	            }
	            case NPTypeCode.Int64:
	            {
		            var ret = new NDArray(NPTypeCode.Int64, Shape.Vector(size));
                    var data = a.MakeGeneric<long>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
                    return ret;
	            }
	            case NPTypeCode.Single:
	            {
		            var ret = new NDArray(NPTypeCode.Single, Shape.Vector(size));
                    var data = a.MakeGeneric<float>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
                    return ret;
	            }
	            case NPTypeCode.Double:
	            {
		            var ret = new NDArray(NPTypeCode.Double, Shape.Vector(size));
                    var data = a.MakeGeneric<double>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
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
        public static NDArray repeat<T>(T a, int repeats) where T : unmanaged //TODO! , int axis = -1
        {
            var ret = new NDArray(InfoOf<T>.NPTypeCode, Shape.Vector(repeats));
            Parallel.For(0, repeats, j => ret.itemset<T>(new int[1] {j}, a));
            return ret;
        }
    }
}
