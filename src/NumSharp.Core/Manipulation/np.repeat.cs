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
#if _REGEN
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
	            case NPTypeCode.Int16:
	            {
		            var ret = new NDArray(NPTypeCode.Int16, Shape.Vector(size));
                    var data = a.MakeGeneric<short>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
                    return ret;
	            }
	            case NPTypeCode.UInt16:
	            {
		            var ret = new NDArray(NPTypeCode.UInt16, Shape.Vector(size));
                    var data = a.MakeGeneric<ushort>();
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
	            case NPTypeCode.UInt32:
	            {
		            var ret = new NDArray(NPTypeCode.UInt32, Shape.Vector(size));
                    var data = a.MakeGeneric<uint>();
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
	            case NPTypeCode.UInt64:
	            {
		            var ret = new NDArray(NPTypeCode.UInt64, Shape.Vector(size));
                    var data = a.MakeGeneric<ulong>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
                    return ret;
	            }
	            case NPTypeCode.Char:
	            {
		            var ret = new NDArray(NPTypeCode.Char, Shape.Vector(size));
                    var data = a.MakeGeneric<char>();
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
	            case NPTypeCode.Single:
	            {
		            var ret = new NDArray(NPTypeCode.Single, Shape.Vector(size));
                    var data = a.MakeGeneric<float>();
                    for (int i = 0; i < a.size; i++)
                    for (int j = 0; j < repeats; j++)
                        ret.itemset(new int[1] {i * repeats + j}, data.GetAtIndex(i));
                    return ret;
	            }
	            case NPTypeCode.Decimal:
	            {
		            var ret = new NDArray(NPTypeCode.Decimal, Shape.Vector(size));
                    var data = a.MakeGeneric<decimal>();
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
        ///     Repeat elements of an array with per-element repeat counts.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="repeats">Array of repeat counts for each element. Must have the same size as the flattened input array.</param>
        /// <returns>A new array with each element repeated according to the corresponding count in repeats.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html</remarks>
        public static NDArray repeat(NDArray a, NDArray repeats)
        {
            a = a.ravel();
            var repeatsFlat = repeats.ravel();

            if (a.size != repeatsFlat.size)
                throw new ArgumentException($"repeats array size ({repeatsFlat.size}) must match input array size ({a.size})");

            // Calculate total output size by summing all repeat counts
            int totalSize = 0;
            for (int i = 0; i < repeatsFlat.size; i++)
            {
                int count = repeatsFlat.GetInt32(i);
                if (count < 0)
                    throw new ArgumentException("repeats may not contain negative values");
                totalSize += count;
            }

            // Handle empty result
            if (totalSize == 0)
                return new NDArray(a.GetTypeCode, Shape.Vector(0));

            switch (a.GetTypeCode)
            {
                case NPTypeCode.Boolean:
                {
                    var ret = new NDArray(NPTypeCode.Boolean, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<bool>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.Byte:
                {
                    var ret = new NDArray(NPTypeCode.Byte, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<byte>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.Int16:
                {
                    var ret = new NDArray(NPTypeCode.Int16, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<short>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.UInt16:
                {
                    var ret = new NDArray(NPTypeCode.UInt16, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<ushort>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.Int32:
                {
                    var ret = new NDArray(NPTypeCode.Int32, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<int>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.UInt32:
                {
                    var ret = new NDArray(NPTypeCode.UInt32, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<uint>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.Int64:
                {
                    var ret = new NDArray(NPTypeCode.Int64, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<long>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.UInt64:
                {
                    var ret = new NDArray(NPTypeCode.UInt64, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<ulong>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.Char:
                {
                    var ret = new NDArray(NPTypeCode.Char, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<char>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.Double:
                {
                    var ret = new NDArray(NPTypeCode.Double, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<double>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.Single:
                {
                    var ret = new NDArray(NPTypeCode.Single, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<float>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                case NPTypeCode.Decimal:
                {
                    var ret = new NDArray(NPTypeCode.Decimal, Shape.Vector(totalSize));
                    var data = a.MakeGeneric<decimal>();
                    int outIdx = 0;
                    for (int i = 0; i < a.size; i++)
                    {
                        int count = repeatsFlat.GetInt32(i);
                        var val = data.GetAtIndex(i);
                        for (int j = 0; j < count; j++)
                            ret.itemset(new int[1] { outIdx++ }, val);
                    }
                    return ret;
                }
                default:
                    throw new NotSupportedException();
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
            for (int j = 0; j < repeats; j++)
                ret.itemset<T>(new int[1] {j}, a);
            return ret;
        }
    }
}
