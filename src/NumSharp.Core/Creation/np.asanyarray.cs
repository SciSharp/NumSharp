using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

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

                    // Handle Tuple<> and ValueTuple<> - they implement ITuple
                    if (a is ITuple tuple)
                    {
                        ret = ConvertTuple(tuple);
                        if (ret is not null)
                            break;
                    }

                    // Fallback: non-generic IEnumerable (element type detected from first item)
                    if (a is IEnumerable enumerable)
                    {
                        ret = ConvertNonGenericEnumerable(enumerable);
                        if (ret is not null)
                            break;
                    }

                    // Fallback: non-generic IEnumerator
                    if (a is IEnumerator enumerator)
                    {
                        ret = ConvertEnumerator(enumerator);
                        if (ret is not null)
                            break;
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
            var elementType = type.GetGenericArguments()[0];
            var isReadOnly = type.GetGenericTypeDefinition() == typeof(ReadOnlyMemory<>);

            // Single type switch - extract array via the appropriate cast
            if (elementType == typeof(bool)) return np.array(isReadOnly ? ((ReadOnlyMemory<bool>)a).ToArray() : ((Memory<bool>)a).ToArray());
            if (elementType == typeof(byte)) return np.array(isReadOnly ? ((ReadOnlyMemory<byte>)a).ToArray() : ((Memory<byte>)a).ToArray());
            if (elementType == typeof(short)) return np.array(isReadOnly ? ((ReadOnlyMemory<short>)a).ToArray() : ((Memory<short>)a).ToArray());
            if (elementType == typeof(ushort)) return np.array(isReadOnly ? ((ReadOnlyMemory<ushort>)a).ToArray() : ((Memory<ushort>)a).ToArray());
            if (elementType == typeof(int)) return np.array(isReadOnly ? ((ReadOnlyMemory<int>)a).ToArray() : ((Memory<int>)a).ToArray());
            if (elementType == typeof(uint)) return np.array(isReadOnly ? ((ReadOnlyMemory<uint>)a).ToArray() : ((Memory<uint>)a).ToArray());
            if (elementType == typeof(long)) return np.array(isReadOnly ? ((ReadOnlyMemory<long>)a).ToArray() : ((Memory<long>)a).ToArray());
            if (elementType == typeof(ulong)) return np.array(isReadOnly ? ((ReadOnlyMemory<ulong>)a).ToArray() : ((Memory<ulong>)a).ToArray());
            if (elementType == typeof(char)) return np.array(isReadOnly ? ((ReadOnlyMemory<char>)a).ToArray() : ((Memory<char>)a).ToArray());
            if (elementType == typeof(float)) return np.array(isReadOnly ? ((ReadOnlyMemory<float>)a).ToArray() : ((Memory<float>)a).ToArray());
            if (elementType == typeof(double)) return np.array(isReadOnly ? ((ReadOnlyMemory<double>)a).ToArray() : ((Memory<double>)a).ToArray());
            if (elementType == typeof(decimal)) return np.array(isReadOnly ? ((ReadOnlyMemory<decimal>)a).ToArray() : ((Memory<decimal>)a).ToArray());

            return null;
        }

        /// <summary>
        ///     Converts a non-generic IEnumerable to an NDArray.
        ///     Element type is detected from the first item.
        /// </summary>
        private static NDArray ConvertNonGenericEnumerable(IEnumerable enumerable)
            => ConvertEnumerator(enumerable.GetEnumerator());

        /// <summary>
        ///     Converts a non-generic IEnumerator to an NDArray.
        ///     Element type is detected from items with NumPy-like type promotion.
        ///     Empty collections return empty double[] to match NumPy's behavior.
        /// </summary>
        private static NDArray ConvertEnumerator(IEnumerator enumerator)
        {
            // Collect items
            var items = new List<object>();

            while (enumerator.MoveNext())
            {
                var item = enumerator.Current;
                if (item != null)
                    items.Add(item);
            }

            // Empty collection: return empty double[] (NumPy defaults to float64)
            if (items.Count == 0)
                return np.array(Array.Empty<double>());

            var elementType = FindCommonNumericType(items);
            return ConvertObjectListToNDArray(items, elementType);
        }

        /// <summary>
        ///     Finds the common numeric type for a list of objects (NumPy-like promotion).
        ///     Promotes to the widest type: bool -> int -> long -> float -> double -> decimal
        /// </summary>
        private static Type FindCommonNumericType(List<object> items)
        {
            // NumPy type promotion priority (simplified):
            // bool < byte < short < ushort < int < uint < long < ulong < float < double
            // If any float/double is present, result is float/double
            // decimal is separate (highest priority if present)

            bool hasDecimal = false;
            bool hasDouble = false;
            bool hasFloat = false;
            bool hasULong = false;
            bool hasLong = false;
            bool hasUInt = false;
            bool hasInt = false;
            bool hasUShort = false;
            bool hasShort = false;
            bool hasByte = false;
            bool hasBool = false;
            bool hasChar = false;
            Type firstType = null;

            foreach (var item in items)
            {
                var t = item.GetType();
                firstType ??= t;

                if (t == typeof(decimal)) hasDecimal = true;
                else if (t == typeof(double)) hasDouble = true;
                else if (t == typeof(float)) hasFloat = true;
                else if (t == typeof(ulong)) hasULong = true;
                else if (t == typeof(long)) hasLong = true;
                else if (t == typeof(uint)) hasUInt = true;
                else if (t == typeof(int)) hasInt = true;
                else if (t == typeof(ushort)) hasUShort = true;
                else if (t == typeof(short)) hasShort = true;
                else if (t == typeof(byte)) hasByte = true;
                else if (t == typeof(bool)) hasBool = true;
                else if (t == typeof(char)) hasChar = true;
            }

            // Promotion rules (NumPy-like):
            // decimal wins if present
            if (hasDecimal) return typeof(decimal);

            // Any floating point promotes to double (NumPy uses float64 for mixed int+float)
            if (hasDouble || hasFloat) return typeof(double);

            // Integer promotion
            if (hasULong) return typeof(ulong);
            if (hasLong || hasUInt) return typeof(long); // uint + anything signed -> long
            if (hasUInt) return typeof(uint);
            if (hasInt || hasUShort) return typeof(int); // ushort + anything signed -> int
            if (hasUShort) return typeof(ushort);
            if (hasShort || hasByte) return typeof(int); // byte + short -> int (safe promotion)
            if (hasByte) return typeof(byte);
            if (hasChar) return typeof(char);
            if (hasBool) return typeof(bool);

            // Fallback to first type
            return firstType ?? typeof(double);
        }

        /// <summary>
        ///     Converts a Tuple or ValueTuple to an NDArray.
        ///     Uses ITuple interface available in .NET Core 2.0+.
        /// </summary>
        private static NDArray ConvertTuple(ITuple tuple)
        {
            if (tuple.Length == 0)
                return np.array(Array.Empty<double>());

            // Collect items and find common type (NumPy-like promotion)
            var items = new List<object>(tuple.Length);

            for (int i = 0; i < tuple.Length; i++)
            {
                var item = tuple[i];
                if (item != null)
                    items.Add(item);
            }

            if (items.Count == 0)
                return np.array(Array.Empty<double>());

            var elementType = FindCommonNumericType(items);
            return ConvertObjectListToNDArray(items, elementType);
        }

        /// <summary>
        ///     Converts a list of objects to an NDArray of the specified element type.
        ///     Uses pattern matching for fast direct cast when types match, with Convert fallback.
        ///     This is ~4x faster than always using Convert for homogeneous collections.
        /// </summary>
        private static NDArray ConvertObjectListToNDArray(List<object> items, Type elementType)
        {
            // Pattern: `is T v ? v : Convert.ToT(item)` gives direct cast speed for homogeneous
            // collections while still handling mixed types correctly
            if (elementType == typeof(bool))
            {
                var arr = new bool[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is bool v ? v : Convert.ToBoolean(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(byte))
            {
                var arr = new byte[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is byte v ? v : Convert.ToByte(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(short))
            {
                var arr = new short[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is short v ? v : Convert.ToInt16(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(ushort))
            {
                var arr = new ushort[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is ushort v ? v : Convert.ToUInt16(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(int))
            {
                var arr = new int[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is int v ? v : Convert.ToInt32(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(uint))
            {
                var arr = new uint[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is uint v ? v : Convert.ToUInt32(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(long))
            {
                var arr = new long[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is long v ? v : Convert.ToInt64(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(ulong))
            {
                var arr = new ulong[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is ulong v ? v : Convert.ToUInt64(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(char))
            {
                var arr = new char[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is char v ? v : Convert.ToChar(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(float))
            {
                var arr = new float[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is float v ? v : Convert.ToSingle(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(double))
            {
                var arr = new double[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is double v ? v : Convert.ToDouble(item);
                }
                return np.array(arr);
            }
            if (elementType == typeof(decimal))
            {
                var arr = new decimal[items.Count];
                for (int i = 0; i < items.Count; i++)
                {
                    var item = items[i];
                    arr[i] = item is decimal v ? v : Convert.ToDecimal(item);
                }
                return np.array(arr);
            }

            return null; // Unsupported element type
        }
    }
}
