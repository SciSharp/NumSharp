using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Utilities;

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
        public static NDArray asanyarray(in object a, DType dtype = null, string device = null)
            => asanyarray(in a, dtype, 'K', device);

        /// <summary>
        ///     Convert the input to an ndarray with a specified memory layout.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input data.</param>
        /// <param name="order">'C', 'F', 'A' or 'K' (default — resolved against a).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array interpretation of a in the requested layout.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html
        ///     <para>A null <paramref name="a"/> is Python's <c>None</c>: with a dtype it is the 0-d array NumPy's
        ///     <c>np.asanyarray(None, dtype)</c> gives (NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>, a
        ///     <c>StringDType</c> NA object); without one NumPy builds an object array, which NumSharp refuses.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> is null and <paramref name="dtype"/> is null (NumPy's
        ///     object array).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is null and <paramref name="dtype"/> is an integer (or
        ///     <c>decimal</c>) dtype: Python's <c>int(None)</c>.</exception>
        public static NDArray asanyarray(in object a, DType dtype, char order, string device = null)
        {
            ValidateDevice(device);
            NDArray ret;
            switch (a) {
                case null:
                    // Python's None: np.array(None, dtype) packs it with the dtype's setitem; without a dtype NumPy's
                    // discovery gives the object dtype.
                    if (dtype is null)
                        throw NDArray.NoneWithoutDtype("np.asanyarray");
                    return NDArray.NoneForStore(dtype);
                case NDArray nd:
                    // a numpy.matrix passes through as a matrix (np.asarray, which this calls, strips the class)
                    if (NDMatrixBoundary.Sees(nd))
                        return NDMatrixBoundary.Wrapped((nd, dtype, order), static s => asarray(s.nd, s.dtype, s.order));
                    return asarray(nd, dtype, order);
                case Array textArr when Backends.Strings.NDStringArrays.IsTextInput(textArr, dtype):
                    // Text (string[], an object[] holding str/bytes, or any array under a string dtype): NumPy's
                    // np.asanyarray of a list of str — a fixed-width string array.
                    ret = Backends.Strings.NDStringArrays.FromManaged(textArr, dtype, 0, 'C');
                    break;
                case IEnumerable<string> texts:
                    ret = Backends.Strings.NDStringArrays.FromManaged(texts.ToArray(), dtype, 0, 'C');
                    break;
                case object[] objArr:
                    // object[] has no fixed dtype — route through type-promotion path.
                    // new NDArray(object[]) throws NotSupportedException since object isn't a
                    // supported element type.
                    ret = ConvertNonGenericEnumerable(objArr);
                    if (ret is null)
                        throw new NotSupportedException($"Unable to resolve asanyarray for object[] (length {objArr.Length}): element type is not a supported NumSharp dtype.");
                    break;
                case Array array:
                    ret = new NDArray(array);
                    break;
                case string str:
                    ret = str; //implicit cast located in NDArray.Implicit.Array
                    break;

                case IEnumerable<bool> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<byte> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<sbyte> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<short> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<ushort> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<int> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<uint> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<long> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<ulong> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<char> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<Half> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<float> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<double> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<decimal> e: ret = np.array(ToArrayFast(e)); break;
                case IEnumerable<Complex> e: ret = np.array(ToArrayFast(e)); break;

                default:
                    var type = a.GetType();
                    //is it a scalar
                    if (type.IsPrimitive || type == typeof(decimal) || type == typeof(Half) || type == typeof(System.Numerics.Complex))
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

            if (dtype != null && !Backends.Strings.NDStringArrays.SatisfiesRequest(ret, dtype))
                ret = ret.astype(dtype, true);

            // Apply requested order (no-op for scalars / 1-D / already-matching layouts).
            char physical = OrderResolver.Resolve(order, ret.Shape);
            if (ret.Shape.NDim > 1 && ret.size > 1)
            {
                bool layoutMatches = physical == 'C' ? ret.Shape.IsContiguous : ret.Shape.IsFContiguous;
                if (!layoutMatches)
                    ret = ret.copy(physical);
            }
            return ret;
        }

        /// <summary>
        ///     Convert a <see cref="MemoryView"/> (obtained from <see cref="NDArray.data"/>) to an ndarray,
        ///     passing ndarray subclasses through — the consumer round-trip of <c>ndarray.data</c>. Matches
        ///     NumPy's <c>np.asanyarray(a.data)</c> (a zero-copy VIEW sharing memory, copying only if a
        ///     dtype/layout change forces it). A dedicated overload is required because a <see cref="MemoryView"/>
        ///     is not otherwise resolvable by the <c>object</c> converter above. Equivalent to
        ///     <c>np.asanyarray(buffer.obj, …)</c>.
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over an array. <c>null</c> is Python's <c>None</c>, as for the
        ///     <c>object</c> overload: C# binds a bare <c>np.asanyarray(null, dtype)</c> HERE (a <see cref="MemoryView"/> is
        ///     the more specific parameter), so the port of NumPy's <c>np.asanyarray(None, dtype)</c> must mean the same.</param>
        /// <param name="dtype">By default, the data-type is inferred from the source.</param>
        /// <param name="order">'C', 'F', 'A' or 'K' (default).</param>
        /// <param name="device">Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html</remarks>
        /// <exception cref="NotSupportedException"><paramref name="buffer"/> is null and <paramref name="dtype"/> is null
        ///     (NumPy's object array).</exception>
        /// <exception cref="TypeError"><paramref name="buffer"/> is null and <paramref name="dtype"/> is an integer (or
        ///     <c>decimal</c>) dtype.</exception>
        public static NDArray asanyarray(MemoryView buffer, DType dtype = null, char order = 'K', string device = null)
        {
            // None: the object overload's rule (its 0-d setitem value in dtype, or the object-dtype refusal).
            if (buffer is null)
                return asanyarray((object)null, dtype, order, device);
            return asanyarray(buffer.obj, dtype, order, device);
        }

        /// <summary>
        ///     Copies an <see cref="IEnumerable{T}"/> into a freshly allocated <typeparamref name="T"/>[].
        ///     Specialised for List&lt;T&gt; and ICollection&lt;T&gt; to skip the enumerator and to
        ///     use <see cref="GC.AllocateUninitializedArray{T}(int, bool)"/> since we overwrite every slot.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
            if (elementType == typeof(sbyte)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<sbyte>)a).Span : ((Memory<sbyte>)a).Span));
            if (elementType == typeof(short)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<short>)a).Span : ((Memory<short>)a).Span));
            if (elementType == typeof(ushort)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<ushort>)a).Span : ((Memory<ushort>)a).Span));
            if (elementType == typeof(int)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<int>)a).Span : ((Memory<int>)a).Span));
            if (elementType == typeof(uint)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<uint>)a).Span : ((Memory<uint>)a).Span));
            if (elementType == typeof(long)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<long>)a).Span : ((Memory<long>)a).Span));
            if (elementType == typeof(ulong)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<ulong>)a).Span : ((Memory<ulong>)a).Span));
            if (elementType == typeof(char)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<char>)a).Span : ((Memory<char>)a).Span));
            if (elementType == typeof(Half)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<Half>)a).Span : ((Memory<Half>)a).Span));
            if (elementType == typeof(float)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<float>)a).Span : ((Memory<float>)a).Span));
            if (elementType == typeof(double)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<double>)a).Span : ((Memory<double>)a).Span));
            if (elementType == typeof(decimal)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<decimal>)a).Span : ((Memory<decimal>)a).Span));
            if (elementType == typeof(Complex)) return np.array(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<Complex>)a).Span : ((Memory<Complex>)a).Span));

            return null;
        }

        /// <summary>
        ///     Optimized Span to Array conversion using GC.AllocateUninitializedArray.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

            Type firstType = null;

            // 15 NPTypeCode dtypes (the +String slot is unused for numeric promotion).
            // Bounded stackalloc keeps the worst-case stack size predictable for large
            // user lists. SeenMask uses (int)code & 31 to avoid out-of-range shifts;
            // NPTypeCode.Complex == 128 would otherwise alias bit 0.
            Span<NPTypeCode> typeCodes = stackalloc NPTypeCode[15];
            int uniqueCount = 0;
            uint seenMask = 0;

            for (int i = 0; i < span.Length; i++)
            {
                var t = span[i].GetType();
                firstType ??= t;

                // decimal wins everything in NumPy promotion
                if (t == typeof(decimal))
                    return typeof(decimal);

                var code = t.GetTypeCode();
                var bit = 1u << ((int)code & 31);
                if ((seenMask & bit) == 0)
                {
                    seenMask |= bit;
                    if (uniqueCount < typeCodes.Length)
                        typeCodes[uniqueCount++] = code;
                }
            }

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
        ///     The pattern <c>is T v ? v : Converts.ToT(item)</c> takes the direct-cast fast path for
        ///     homogeneous collections while still handling mixed-type promotion via <see cref="Converts"/>.
        /// </summary>
        /// <param name="items">The non-null items, in order.</param>
        /// <param name="elementType">The promoted element type (<see cref="FindCommonNumericType"/>).</param>
        /// <returns>The 1-D array, or null when <paramref name="elementType"/> is not a NumSharp dtype.</returns>
        /// <remarks>
        ///     The element type is the promotion of every item's type, so a numeric item converts into a type at least
        ///     as wide as its own and the conversion is exact. It goes through <see cref="Converts"/>, not
        ///     <see cref="Convert"/>: <see cref="Half"/> and <see cref="Complex"/> are not <see cref="IConvertible"/>,
        ///     so <c>Convert.ToDouble</c> threw for a float16 item beside a float64 one (<c>new object[] { (Half)1, 2.5 }</c>),
        ///     and a <see cref="Complex"/> target read every non-complex item through that same call (a float16 item threw
        ///     there too).
        /// </remarks>
        private static NDArray ConvertObjectListToNDArray(List<object> items, Type elementType)
        {
            var span = CollectionsMarshal.AsSpan(items);

            if (elementType == typeof(bool))
            {
                var arr = GC.AllocateUninitializedArray<bool>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is bool v ? v : Converts.ToBoolean(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(byte))
            {
                var arr = GC.AllocateUninitializedArray<byte>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is byte v ? v : Converts.ToByte(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(short))
            {
                var arr = GC.AllocateUninitializedArray<short>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is short v ? v : Converts.ToInt16(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(ushort))
            {
                var arr = GC.AllocateUninitializedArray<ushort>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is ushort v ? v : Converts.ToUInt16(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(int))
            {
                var arr = GC.AllocateUninitializedArray<int>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is int v ? v : Converts.ToInt32(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(uint))
            {
                var arr = GC.AllocateUninitializedArray<uint>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is uint v ? v : Converts.ToUInt32(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(long))
            {
                var arr = GC.AllocateUninitializedArray<long>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is long v ? v : Converts.ToInt64(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(ulong))
            {
                var arr = GC.AllocateUninitializedArray<ulong>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is ulong v ? v : Converts.ToUInt64(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(char))
            {
                var arr = GC.AllocateUninitializedArray<char>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is char v ? v : Converts.ToChar(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(float))
            {
                var arr = GC.AllocateUninitializedArray<float>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is float v ? v : Converts.ToSingle(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(double))
            {
                var arr = GC.AllocateUninitializedArray<double>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is double v ? v : Converts.ToDouble(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(decimal))
            {
                var arr = GC.AllocateUninitializedArray<decimal>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is decimal v ? v : Converts.ToDecimal(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(sbyte))
            {
                var arr = GC.AllocateUninitializedArray<sbyte>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is sbyte v ? v : Converts.ToSByte(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(Half))
            {
                var arr = GC.AllocateUninitializedArray<Half>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is Half v ? v : Converts.ToHalf(span[i]);
                return np.array(arr);
            }
            if (elementType == typeof(Complex))
            {
                var arr = GC.AllocateUninitializedArray<Complex>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is Complex v ? v : Converts.ToComplex(span[i]);
                return np.array(arr);
            }

            return null; // Unsupported element type
        }
    }
}
