using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">Input data, in any form that can be converted to an array. This includes scalars, lists, lists of tuples, tuples, tuples of tuples, tuples of lists, and ndarrays.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input data.</param>
        /// <returns>Array interpretation of a. If a is an ndarray or a subclass of ndarray, it is returned as-is and no copy is performed.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html</remarks>
        public static NDArray asanyarray(in object a, Type dtype = null) //todo support order
        {
            NDArray ret;
            switch (a) {
                case null:
                    throw new ArgumentNullException(nameof(a));
                case NDArray nd:
                    if (dtype == null || Equals(nd.dtype, dtype))
                        return nd;
                    return nd.astype(dtype, true);
                case Array array:
                    ret = new NDArray(array);
                    break;
                case string str:
                    ret = str; //implicit cast located in NDArray.Implicit.Array
                    break;

                // Handle typed IEnumerable<T> for all 12 NumSharp-supported types
                case IEnumerable<bool> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<byte> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<short> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<ushort> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<int> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<uint> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<long> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<ulong> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<char> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<float> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<double> e: ret = np.array(e.ToArray()); break;
                case IEnumerable<decimal> e: ret = np.array(e.ToArray()); break;

                default:
                    var type = a.GetType();
                    // Check if it's a scalar (primitive or decimal)
                    if (type.IsPrimitive || type == typeof(decimal))
                    {
                        ret = NDArray.Scalar(a);
                        break;
                    }

                    // Handle Memory<T> and ReadOnlyMemory<T> - they don't implement IEnumerable<T>
                    if (type.IsGenericType)
                    {
                        var genericDef = type.GetGenericTypeDefinition();
                        if (genericDef == typeof(Memory<>) || genericDef == typeof(ReadOnlyMemory<>))
                        {
                            ret = ConvertMemory(a, type);
                            if (ret is not null)
                                break;
                        }
                    }

                    throw new NotSupportedException($"Unable to resolve asanyarray for type {type.Name}");
            }

            if (dtype != null && !Equals(ret.dtype, dtype))
                return ret.astype(dtype, true);

            return ret;
        }

        /// <summary>
        ///     Converts Memory&lt;T&gt; or ReadOnlyMemory&lt;T&gt; to an NDArray.
        ///     These types don't implement IEnumerable&lt;T&gt;, so we handle them specially.
        /// </summary>
        private static NDArray ConvertMemory(object a, Type type)
        {
            var genericDef = type.GetGenericTypeDefinition();
            var elementType = type.GetGenericArguments()[0];

            // Handle ReadOnlyMemory<T> first (it cannot be cast to Memory<T>)
            if (genericDef == typeof(ReadOnlyMemory<>))
            {
                if (elementType == typeof(bool)) return np.array(((ReadOnlyMemory<bool>)a).ToArray());
                if (elementType == typeof(byte)) return np.array(((ReadOnlyMemory<byte>)a).ToArray());
                if (elementType == typeof(short)) return np.array(((ReadOnlyMemory<short>)a).ToArray());
                if (elementType == typeof(ushort)) return np.array(((ReadOnlyMemory<ushort>)a).ToArray());
                if (elementType == typeof(int)) return np.array(((ReadOnlyMemory<int>)a).ToArray());
                if (elementType == typeof(uint)) return np.array(((ReadOnlyMemory<uint>)a).ToArray());
                if (elementType == typeof(long)) return np.array(((ReadOnlyMemory<long>)a).ToArray());
                if (elementType == typeof(ulong)) return np.array(((ReadOnlyMemory<ulong>)a).ToArray());
                if (elementType == typeof(char)) return np.array(((ReadOnlyMemory<char>)a).ToArray());
                if (elementType == typeof(float)) return np.array(((ReadOnlyMemory<float>)a).ToArray());
                if (elementType == typeof(double)) return np.array(((ReadOnlyMemory<double>)a).ToArray());
                if (elementType == typeof(decimal)) return np.array(((ReadOnlyMemory<decimal>)a).ToArray());
            }

            // Handle Memory<T>
            if (genericDef == typeof(Memory<>))
            {
                if (elementType == typeof(bool)) return np.array(((Memory<bool>)a).ToArray());
                if (elementType == typeof(byte)) return np.array(((Memory<byte>)a).ToArray());
                if (elementType == typeof(short)) return np.array(((Memory<short>)a).ToArray());
                if (elementType == typeof(ushort)) return np.array(((Memory<ushort>)a).ToArray());
                if (elementType == typeof(int)) return np.array(((Memory<int>)a).ToArray());
                if (elementType == typeof(uint)) return np.array(((Memory<uint>)a).ToArray());
                if (elementType == typeof(long)) return np.array(((Memory<long>)a).ToArray());
                if (elementType == typeof(ulong)) return np.array(((Memory<ulong>)a).ToArray());
                if (elementType == typeof(char)) return np.array(((Memory<char>)a).ToArray());
                if (elementType == typeof(float)) return np.array(((Memory<float>)a).ToArray());
                if (elementType == typeof(double)) return np.array(((Memory<double>)a).ToArray());
                if (elementType == typeof(decimal)) return np.array(((Memory<decimal>)a).ToArray());
            }

            return null;
        }
    }
}
