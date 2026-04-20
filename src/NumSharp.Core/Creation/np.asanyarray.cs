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
                    if (type.IsPrimitive || type == typeof(decimal))
                    {
                        ret = NDArray.Scalar(a);
                        break;
                    }

                    // Memory<T>/ReadOnlyMemory<T> do not implement IEnumerable<T>.
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

                    if (a is ITuple tuple)
                    {
                        ret = ConvertTuple(tuple);
                        if (ret is not null)
                            break;
                    }

                    if (a is IEnumerable enumerable)
                    {
                        ret = ConvertNonGenericEnumerable(enumerable);
                        if (ret is not null)
                            break;
                    }

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
        ///     Copies an <see cref="IEnumerable{T}"/> into a freshly allocated <typeparamref name="T"/>[].
        ///     Specialised for List&lt;T&gt; and ICollection&lt;T&gt; to skip the enumerator and to
        ///     use <see cref="GC.AllocateUninitializedArray{T}(int, bool)"/> since we overwrite every slot.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T[] ToArrayFast<T>(IEnumerable<T> source)
        {
            if (source is List<T> list)
            {
                var span = CollectionsMarshal.AsSpan(list);
                var arr = GC.AllocateUninitializedArray<T>(span.Length);
                span.CopyTo(arr);
                return arr;
            }

            if (source is ICollection<T> collection)
            {
                var arr = GC.AllocateUninitializedArray<T>(collection.Count);
                collection.CopyTo(arr, 0);
                return arr;
            }

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
        ///     Empty collections return empty double[] to match NumPy's float64 default.
        /// </summary>
        private static NDArray ConvertEnumerator(IEnumerator enumerator)
        {
            List<object> items = enumerator is ICollection collection
                ? new List<object>(collection.Count)
                : new List<object>();

            while (enumerator.MoveNext())
            {
                var item = enumerator.Current;
                if (item != null)
                    items.Add(item);
            }

            if (items.Count == 0)
                return np.array(Array.Empty<double>());

            var elementType = FindCommonNumericType(items);
            return ConvertObjectListToNDArray(items, elementType);
        }

        /// <summary>
        ///     Finds the common numeric type for a list of objects (NumPy-like promotion).
        ///     Uses existing _FindCommonType_Scalar for consistent type promotion.
        /// </summary>
        private static Type FindCommonNumericType(List<object> items)
        {
            var span = CollectionsMarshal.AsSpan(items);

            bool hasDouble = false;
            bool hasFloat = false;
            Type firstType = null;

            // At most 12 unique NPTypeCode values exist; bound the stackalloc accordingly
            // (otherwise large user lists could blow the stack).
            Span<NPTypeCode> typeCodes = stackalloc NPTypeCode[12];
            int uniqueCount = 0;
            uint seenMask = 0;

            for (int i = 0; i < span.Length; i++)
            {
                var t = span[i].GetType();
                firstType ??= t;

                if (t == typeof(decimal))
                    return typeof(decimal);

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

            if (hasDouble || hasFloat)
                return typeof(double);

            if (uniqueCount == 1)
                return firstType ?? typeof(double);

            var resultCode = _FindCommonType_Scalar(typeCodes.Slice(0, uniqueCount).ToArray());
            return resultCode.AsType();
        }

        /// <summary>
        ///     Converts a Tuple or ValueTuple to an NDArray via the ITuple interface.
        /// </summary>
        private static NDArray ConvertTuple(ITuple tuple)
        {
            if (tuple.Length == 0)
                return np.array(Array.Empty<double>());

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
        ///     The pattern <c>is T v ? v : Convert.ToT(item)</c> takes the direct-cast fast path for
        ///     homogeneous collections while still handling mixed-type promotion via Convert.
        /// </summary>
        private static NDArray ConvertObjectListToNDArray(List<object> items, Type elementType)
        {
            var span = CollectionsMarshal.AsSpan(items);

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
