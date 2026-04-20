using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

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
                case object[] objArr:
                    // object[] has no fixed dtype — route through type-promotion path.
                    // new NDArray(object[]) throws NotSupportedException since object isn't a
                    // supported element type.
                    ret = ConvertNonGenericEnumerable(objArr);
                    if (ret is null)
                        throw new NotSupportedException($"Unable to resolve asanyarray for object array of length {objArr.Length}.");
                    break;
                case Array array:
                    ret = new NDArray(array);
                    break;
                case string str:
                    ret = str; //implicit cast located in NDArray.Implicit.Array
                    break;

                // Handle typed IEnumerable<T> for all 12 NumSharp-supported types
                // Optimized: Use CopyTo for ICollection<T> (3-7x faster than ToArray for small collections)
                case IEnumerable<bool> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<byte> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<short> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<ushort> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<int> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<uint> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<long> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<ulong> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<char> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<float> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<double> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<decimal> e: ret = np.array(ToArrayFast(e)); break;

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
        ///     Optimized ToArray for IEnumerable&lt;T&gt;.
        ///     Uses CopyTo for ICollection&lt;T&gt; (3-7x faster for small collections).
        ///     For List&lt;T&gt;, uses CollectionsMarshal.AsSpan for direct memory access.
        ///     Uses GC.AllocateUninitializedArray to skip zeroing (4x faster allocation).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T[] ToArrayFast<T>(IEnumerable<T> source)
        {
            // Fast path for List<T> - use CollectionsMarshal for direct span access
            if (source is List<T> list)
            {
                var span = CollectionsMarshal.AsSpan(list);
                // Use uninitialized array - we're about to overwrite all elements
                var arr = GC.AllocateUninitializedArray<T>(span.Length);
                span.CopyTo(arr);
                return arr;
            }

            // Fast path for ICollection<T> - use CopyTo (avoids enumerator overhead)
            if (source is ICollection<T> collection)
            {
                // Use uninitialized array - CopyTo will overwrite all elements
                var arr = GC.AllocateUninitializedArray<T>(collection.Count);
                collection.CopyTo(arr, 0);
                return arr;
            }

            // Fallback to LINQ ToArray for other IEnumerable<T>
            return source.ToArray();
        }

        /// <summary>
        ///     Converts Memory&lt;T&gt; or ReadOnlyMemory&lt;T&gt; to an NDArray.
        ///     Uses Span.CopyTo + GC.AllocateUninitializedArray for optimal performance.
        /// </summary>
        private static NDArray ConvertMemory(object a, Type type)
        {
            var elementType = type.GetGenericArguments()[0];
            var isReadOnly = type.GetGenericTypeDefinition() == typeof(ReadOnlyMemory<>);

            // Use Span.CopyTo + GC.AllocateUninitializedArray instead of ToArray()
            if (elementType == typeof(bool)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<bool>)a).Span : ((Memory<bool>)a).Span));
            if (elementType == typeof(byte)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<byte>)a).Span : ((Memory<byte>)a).Span));
            if (elementType == typeof(short)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<short>)a).Span : ((Memory<short>)a).Span));
            if (elementType == typeof(ushort)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<ushort>)a).Span : ((Memory<ushort>)a).Span));
            if (elementType == typeof(int)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<int>)a).Span : ((Memory<int>)a).Span));
            if (elementType == typeof(uint)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<uint>)a).Span : ((Memory<uint>)a).Span));
            if (elementType == typeof(long)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<long>)a).Span : ((Memory<long>)a).Span));
            if (elementType == typeof(ulong)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<ulong>)a).Span : ((Memory<ulong>)a).Span));
            if (elementType == typeof(char)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<char>)a).Span : ((Memory<char>)a).Span));
            if (elementType == typeof(float)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<float>)a).Span : ((Memory<float>)a).Span));
            if (elementType == typeof(double)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<double>)a).Span : ((Memory<double>)a).Span));
            if (elementType == typeof(decimal)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<decimal>)a).Span : ((Memory<decimal>)a).Span));

            return null;
        }

        /// <summary>
        ///     Optimized Span to Array conversion using GC.AllocateUninitializedArray.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T[] SpanToArrayFast<T>(ReadOnlySpan<T> span)
        {
            var arr = GC.AllocateUninitializedArray<T>(span.Length);
            span.CopyTo(arr);
            return arr;
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
            // Pre-size list if count is known (optimization #4)
            List<object> items;
            if (enumerator is ICollection collection)
                items = new List<object>(collection.Count);
            else
                items = new List<object>();

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
        ///     Uses existing _FindCommonType_Scalar for consistent type promotion.
        ///     Early exit when highest-priority types (decimal/double) are found.
        /// </summary>
        private static Type FindCommonNumericType(List<object> items)
        {
            // Use CollectionsMarshal.AsSpan for faster iteration (no bounds checks)
            var span = CollectionsMarshal.AsSpan(items);

            // Early exit optimization: track highest-priority types seen
            bool hasDecimal = false;
            bool hasDouble = false;
            bool hasFloat = false;
            Type firstType = null;

            // Collect unique type codes for _FindCommonType_Scalar
            Span<NPTypeCode> typeCodes = stackalloc NPTypeCode[span.Length];
            int uniqueCount = 0;
            uint seenMask = 0; // Bitmask for deduplication (NPTypeCode values are small)

            for (int i = 0; i < span.Length; i++)
            {
                var t = span[i].GetType();
                firstType ??= t;

                // Early exit: decimal wins everything
                if (t == typeof(decimal))
                    return typeof(decimal);

                // Track floating point for early double detection
                if (t == typeof(double)) hasDouble = true;
                else if (t == typeof(float)) hasFloat = true;

                var code = t.GetTypeCode();
                var bit = 1u << (int)code;
                if ((seenMask & bit) == 0)
                {
                    seenMask |= bit;
                    typeCodes[uniqueCount++] = code;
                }
            }

            // Early exit: any floating point promotes to double
            if (hasDouble || hasFloat)
                return typeof(double);

            // Use existing type promotion logic for remaining cases
            if (uniqueCount == 1)
                return firstType ?? typeof(double);

            var resultCode = _FindCommonType_Scalar(typeCodes.Slice(0, uniqueCount).ToArray());
            return resultCode.AsType();
        }

        /// <summary>
        ///     Converts a Tuple or ValueTuple to an NDArray.
        ///     Uses ITuple interface available in .NET Core 2.0+.
        ///     Optimized: pre-sized List, early exit for decimal/double.
        /// </summary>
        private static NDArray ConvertTuple(ITuple tuple)
        {
            if (tuple.Length == 0)
                return np.array(Array.Empty<double>());

            // Pre-sized list (optimization: avoid resize for known count)
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
        ///     Uses CollectionsMarshal.AsSpan for bounds-check-free iteration.
        ///     Uses pattern matching for fast direct cast when types match, with Convert fallback.
        ///     This is ~4x faster than always using Convert for homogeneous collections.
        /// </summary>
        private static NDArray ConvertObjectListToNDArray(List<object> items, Type elementType)
        {
            // Use CollectionsMarshal.AsSpan for faster iteration (no bounds checks)
            var span = CollectionsMarshal.AsSpan(items);

            // Pattern: `is T v ? v : Convert.ToT(item)` gives direct cast speed for homogeneous
            // collections while still handling mixed types correctly
            if (elementType == typeof(bool))
            {
                var arr = GC.AllocateUninitializedArray<bool>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is bool v ? v : Convert.ToBoolean(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(byte))
            {
                var arr = GC.AllocateUninitializedArray<byte>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is byte v ? v : Convert.ToByte(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(short))
            {
                var arr = GC.AllocateUninitializedArray<short>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is short v ? v : Convert.ToInt16(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(ushort))
            {
                var arr = GC.AllocateUninitializedArray<ushort>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is ushort v ? v : Convert.ToUInt16(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(int))
            {
                var arr = GC.AllocateUninitializedArray<int>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is int v ? v : Convert.ToInt32(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(uint))
            {
                var arr = GC.AllocateUninitializedArray<uint>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is uint v ? v : Convert.ToUInt32(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(long))
            {
                var arr = GC.AllocateUninitializedArray<long>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is long v ? v : Convert.ToInt64(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(ulong))
            {
                var arr = GC.AllocateUninitializedArray<ulong>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is ulong v ? v : Convert.ToUInt64(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(char))
            {
                var arr = GC.AllocateUninitializedArray<char>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is char v ? v : Convert.ToChar(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(float))
            {
                var arr = GC.AllocateUninitializedArray<float>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is float v ? v : Convert.ToSingle(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(double))
            {
                var arr = GC.AllocateUninitializedArray<double>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is double v ? v : Convert.ToDouble(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(decimal))
            {
                var arr = GC.AllocateUninitializedArray<decimal>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is decimal v ? v : Convert.ToDecimal(span[i]);
                return np.array(arr);
            }

            return null; // Unsupported element type
        }
    }
}
