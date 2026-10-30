using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    // ---------------------------------------------------------------------------------------------------------------------
    // A C# value into a new array: the side of NumPy's array_coercion.c (PyArray_DiscoverDTypeAndShape, the fill) and of
    // PyArray_FromAny that NumSharp's C# boundary needs - a Python scalar (a C# number, a str, bytes, None) into a 0-d array,
    // a C# array (typed, jagged, of objects) into an array of its shape, and the sequences np.asanyarray reads (object
    // lists and grids, tuples, Memory<T>, enumerables). The array object's own members (the np-layer dtype plan's D1);
    // np.array, np.asanyarray, NDArray.Scalar and the implicit conversion are facades over them.
    //
    // The plan's phase P6 made its np.array calls the statement np.array<T>(params T[]) runs (VectorOf, below) and
    // moved the file from beside its facades (Creation/) to Backends/Unmanaged/. np._FindCommonType_Scalar stays: a
    // promotion helper of np's, which ratchet R2 pins until the helpers move into the dtype layer.
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class UnmanagedStorage
    {
        /// <summary>
        ///     <c>np.array(values)</c> for a C# array of one of NumSharp's element types, np.array&lt;T&gt;(params T[])'s
        ///     statement: a new 1-D array over a copy of <paramref name="values"/>.
        /// </summary>
        /// <typeparam name="T">The element type (one of the 15 storage types).</typeparam>
        /// <param name="values">The values (not null; may be empty).</param>
        /// <returns>A new C-contiguous <c>(values.Length,)</c> array the caller owns; later writes to
        ///     <paramref name="values"/> do not reach it.</returns>
        private static NDArray VectorOf<T>(T[] values) where T : unmanaged
            => new NDArray(ArraySlice.FromArray(values, true), Shape.Vector(values.Length));

        /// <summary>
        ///     <see cref="NDArray.Scalar(object, DType)"/>'s body once its null check ran: one value into a new 0-d array of
        ///     <paramref name="dtype"/> - a <c>StringDType</c> stores it through its setitem (its <c>str()</c>, the NA object,
        ///     <c>coerce=False</c>'s refusal), a fixed-width string dtype encodes it as text, any other dtype converts it
        ///     with <see cref="Converts"/> (C#'s conversions, not NumPy's <c>PyArray_Pack</c>).
        /// </summary>
        /// <param name="value">The value (a C# number, a str, bytes, or null for None).</param>
        /// <param name="dtype">The array's dtype; not null.</param>
        /// <returns>The new 0-d array.</returns>
        /// <exception cref="ValueError">A non-text value into a <c>coerce=False</c> <c>StringDType</c>.</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8 into a <c>StringDType</c>.</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate into a <c>StringDType</c>, or non-ASCII text
        ///     into <c>S</c>.</exception>
        /// <exception cref="InvalidCastException">A value <see cref="Converts"/> cannot convert to the dtype.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NDArray ScalarFromObject(object value, DType dtype)
        {
            // The text lane (a fixed-width string class or StringDType) stores the value through its class's setitem: NumPy's
            // PyArray_Pack (its str() or its encoding, the NA object, coerce=False's refusal), never IConvertible. The numeric
            // lanes keep NDArray.Scalar's documented C# conversion below.
            if (dtype.typecode == NPTypeCode.String)
                return FromAnyScalar(value, dtype);
            return new NDArray(UnmanagedStorage.Scalar(value, dtype.GetTypeCode()));
        }

        /// <summary>
        ///     <see cref="NDArray.Scalar(object)"/>'s body: one value into a new 0-d array of its own dtype - a str is a
        ///     <c>U</c> array and a <c>byte[]</c> an <c>S</c> array (<c>np.array('ab')</c>, <c>np.array(b'ab')</c>), a C#
        ///     number the array of its type.
        /// </summary>
        /// <param name="value">The value: a C# number, a <see cref="string"/> or a <c>byte[]</c>.</param>
        /// <returns>The new 0-d array.</returns>
        /// <exception cref="NotSupportedException">A value of a type NumSharp has no dtype for.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NDArray ScalarFromObject(object value)
        {
            // Text values are 0-d fixed-width strings: a string is U (np.array('ab')), a byte[] is S (np.array(b'ab')).
            if (value is string || value is byte[])
                return FromAnyScalar(value, null);
            return new NDArray(UnmanagedStorage.Scalar(value));
        }

        /// <summary>
        ///     NumPy's <c>PyArray_FromAny</c> of ONE Python object with an optional requested descriptor
        ///     (<c>np.array('abc')</c>, <c>np.array('12', dtype=int)</c>): a new 0-d array - a text value discovers its string
        ///     class and width (<c>&lt;U3</c>), a requested string dtype holds it (truncated, or the width discovered for an
        ///     unsized class), a <c>StringDType</c> stores it through its setitem, and a numeric dtype packs it with its
        ///     setitem (CPython's <c>int()</c> / <c>float()</c> / <c>complex()</c> parse a text).
        /// </summary>
        /// <param name="op">The value: a <see cref="string"/>, a <c>byte[]</c>, a <see cref="char"/>, a number, or null for
        ///     None.</param>
        /// <param name="dtype">The requested dtype, or null to discover it.</param>
        /// <returns>The new 0-d array.</returns>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        /// <exception cref="ValueError">A text that does not parse as the numeric dtype's number.</exception>
        /// <exception cref="OverflowException">An integer text outside an integer dtype's range.</exception>
        /// <exception cref="NotSupportedException">None without a dtype (NumPy's object array).</exception>
        internal static NDArray FromAnyScalar(object op, DType dtype)
            => Build(Array.Empty<long>(), new List<object>(1) { op }, dtype, 'C');

        /// <summary>
        ///     NumPy's <c>discover_dtype_from_pyobject</c> (<c>array_coercion.c</c>) over the C# boundary map: the DType
        ///     CLASS a Python value is discovered as. A requested class that claims the value's type (its own scalar type,
        ///     or its <see cref="DTypeMeta.is_known_scalar_type"/>: the string classes claim Python's str, bytes, int,
        ///     float, complex and bool) discovers the value itself. Otherwise the value's own class
        ///     (<c>npy_discover_dtype_from_pytype</c>): a str or a C# char (Python's one-character str) the <c>U</c>
        ///     class, a <c>byte[]</c> the <c>S</c> class, a C# bool the bool class, a C# integer or
        ///     <see cref="BigInteger"/> NumPy's abstract <c>_PyLongDType</c>, a float or double <c>_PyFloatDType</c>, a
        ///     <see cref="Complex"/> <c>_PyComplexDType</c>, a NumPy scalar (<see cref="Half"/>, NumSharp's
        ///     <see cref="decimal"/>) its dtype's class.
        /// </summary>
        /// <param name="obj">The value in the C# boundary map; null is Python's None.</param>
        /// <param name="fixedDType">The requested class, or null.</param>
        /// <returns>The class, or null where NumPy maps the value to no class (None, or an object NumSharp has no class
        ///     for): <see cref="FindScalarDescriptor"/> then gives a requested class its second chance, and makes anything
        ///     else NumPy's object descriptor, which NumSharp refuses.</returns>
        /// <remarks>
        ///     The class only, never a width or a value range: those are the class's
        ///     <see cref="DTypeMeta.discover_descr_from_pyobject"/>, which <see cref="FindScalarDescriptor"/> asks next
        ///     (a Python int is int64 or uint64 by its value there).
        /// </remarks>
        internal static DTypeMeta DiscoverDTypeFromPyObject(object obj, DTypeMeta fixedDType)
        {
            // None: NoneType maps to no DType class, with or without a request (a requested class still discovers it, as
            // the second chance of find_scalar_descriptor).
            if (obj is null)
                return null;
            Type type = obj.GetType();
            // Let the requested class handle the discovery when the scalar type matches exactly or the class claims it.
            if (fixedDType != null && (type == fixedDType.type || fixedDType.is_known_scalar_type(type)))
                return fixedDType;
            return obj switch
            {
                string or char => DTypeRegistry.Str,
                byte[] => DTypeRegistry.Bytes,
                bool => DTypeRegistry.FromTypeCode(NPTypeCode.Boolean),
                sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger => DTypeRegistry.PyLong,
                float or double => DTypeRegistry.PyFloat,
                Complex => DTypeRegistry.PyComplex,
                Half => DTypeRegistry.FromTypeCode(NPTypeCode.Half),
                decimal => DTypeRegistry.FromTypeCode(NPTypeCode.Decimal),
                _ => null,
            };
        }

        /// <summary>
        ///     NumPy's <c>find_scalar_descriptor</c> (<c>array_coercion.c</c>): the descriptor of one Python value - its
        ///     discovered class's <see cref="DTypeMeta.discover_descr_from_pyobject"/>, cast to the requested class when
        ///     there is one (<see cref="DTypePromotion.CastDescrToDType"/>: a Python int under an unsized <c>U</c> request is
        ///     <c>&lt;U21</c> through the int64-to-str cast). A value with no class of its own gets the requested class's
        ///     discovery as a second chance (a string class takes the length of its <c>str()</c>: None is
        ///     <c>'None'</c>).
        /// </summary>
        /// <param name="fixedDType">The requested class, or null.</param>
        /// <param name="discovered">The value's class from <see cref="DiscoverDTypeFromPyObject"/>, or null.</param>
        /// <param name="obj">The value; null is Python's None.</param>
        /// <returns>The value's descriptor.</returns>
        /// <exception cref="NotSupportedException">The value has no class and nothing is requested: NumPy's object
        ///     descriptor, which NumSharp does not have (None, an object of no NumPy type); or a Python int past 64 bits
        ///     (<c>_PyLongDType</c>'s object descriptor).</exception>
        /// <exception cref="TypeError">A text too long for its string class (<c>string too large to store inside
        ///     array.</c>), or a descriptor the requested class cannot hold (<c>cannot cast dtype %S to %S.</c>).</exception>
        /// <exception cref="ValueError">A Python int of more than 4300 decimal digits under a string class (CPython
        ///     refuses its <c>str()</c>).</exception>
        internal static DType FindScalarDescriptor(DTypeMeta fixedDType, DTypeMeta discovered, object obj)
        {
            DType descr;
            if (discovered == null && fixedDType == null)
            {
                // No known class and no request: NumPy goes to object.
                throw new NotSupportedException(obj is null
                    ? "None makes NumPy build an object array, which NumSharp does not have."
                    : $"A {obj.GetType().Name} value has no NumPy dtype: NumPy would build an object array, which NumSharp does not have.");
            }
            else if (discovered == null)
            {
                // No class found: the requested class gets a second chance (a string class calls str(obj)).
                descr = fixedDType.discover_descr_from_pyobject(obj);
            }
            else
            {
                descr = discovered.discover_descr_from_pyobject(obj);
            }
            if (fixedDType == null)
                return descr;
            return DTypePromotion.CastDescrToDType(descr, fixedDType);
        }

        /// <summary>
        ///     The implicit conversion of a C# array (<c>NDArray x = new[] { 1, 2 }</c>) once its null pass-through ran: text
        ///     values build NumPy's fixed-width string array (the discovery np.array makes), a jagged array of a NumSharp
        ///     element type its nested shape, a rectangular one its own shape; the data is copied.
        /// </summary>
        /// <param name="array">The C# array; not null.</param>
        /// <returns>A new array holding a copy of the elements.</returns>
        /// <exception cref="NotSupportedException">The element type is not one of NumSharp's dtypes.</exception>
        /// <exception cref="ValueError">A ragged jagged array of text (NumPy's inhomogeneous-shape text).</exception>
        internal static NDArray FromManagedArray(Array array)
        {
            // Text values build a fixed-width string array (the same discovery as np.array); everything else is numeric.
            if (Backends.UnmanagedStorage.IsTextInput(array, null))
                return Backends.UnmanagedStorage.FromManagedText(array, null, 0, 'C');

            bool isJaggedArray = array.GetType().GetElementType().IsArray;
            var underlying = array.ResolveElementType();
            if (isJaggedArray)
            {
                var nd = new NDArray(underlying);
                switch (nd.GetTypeCode)
                {
	                // %foreach supported_dtypes,supported_dtypes_lowercase%
	                // case NPTypeCode.#1: return NDArray.FromJaggedArray<#2>(array);
	                // %
	                // default:
		                // throw new NotSupportedException();
                    case NPTypeCode.Boolean: return NDArray.FromJaggedArray<bool>(array);
	                case NPTypeCode.Byte: return NDArray.FromJaggedArray<byte>(array);
	                case NPTypeCode.SByte: return NDArray.FromJaggedArray<sbyte>(array);
	                case NPTypeCode.Int16: return NDArray.FromJaggedArray<short>(array);
	                case NPTypeCode.UInt16: return NDArray.FromJaggedArray<ushort>(array);
	                case NPTypeCode.Int32: return NDArray.FromJaggedArray<int>(array);
	                case NPTypeCode.UInt32: return NDArray.FromJaggedArray<uint>(array);
	                case NPTypeCode.Int64: return NDArray.FromJaggedArray<long>(array);
	                case NPTypeCode.UInt64: return NDArray.FromJaggedArray<ulong>(array);
	                case NPTypeCode.Char: return NDArray.FromJaggedArray<char>(array);
	                case NPTypeCode.Half: return NDArray.FromJaggedArray<Half>(array);
	                case NPTypeCode.Double: return NDArray.FromJaggedArray<double>(array);
	                case NPTypeCode.Single: return NDArray.FromJaggedArray<float>(array);
	                case NPTypeCode.Decimal: return NDArray.FromJaggedArray<decimal>(array);
	                case NPTypeCode.Complex: return NDArray.FromJaggedArray<System.Numerics.Complex>(array);
	                default:
		                throw new NotSupportedException();
                }
            }
            else
            {

                var nd = new NDArray(underlying);
                switch (nd.GetTypeCode)
                {
	                // %foreach supported_dtypes,supported_dtypes_lowercase%
	                // case NPTypeCode.#1: return NDArray.FromMultiDimArray<#2>(array);
	                // %
	                // default:
		                // throw new NotSupportedException();
                    case NPTypeCode.Boolean: return NDArray.FromMultiDimArray<bool>(array);
	                case NPTypeCode.Byte: return NDArray.FromMultiDimArray<byte>(array);
	                case NPTypeCode.SByte: return NDArray.FromMultiDimArray<sbyte>(array);
	                case NPTypeCode.Int16: return NDArray.FromMultiDimArray<short>(array);
	                case NPTypeCode.UInt16: return NDArray.FromMultiDimArray<ushort>(array);
	                case NPTypeCode.Int32: return NDArray.FromMultiDimArray<int>(array);
	                case NPTypeCode.UInt32: return NDArray.FromMultiDimArray<uint>(array);
	                case NPTypeCode.Int64: return NDArray.FromMultiDimArray<long>(array);
	                case NPTypeCode.UInt64: return NDArray.FromMultiDimArray<ulong>(array);
	                case NPTypeCode.Char: return NDArray.FromMultiDimArray<char>(array);
	                case NPTypeCode.Half: return NDArray.FromMultiDimArray<Half>(array);
	                case NPTypeCode.Double: return NDArray.FromMultiDimArray<double>(array);
	                case NPTypeCode.Single: return NDArray.FromMultiDimArray<float>(array);
	                case NPTypeCode.Decimal: return NDArray.FromMultiDimArray<decimal>(array);
	                case NPTypeCode.Complex: return NDArray.FromMultiDimArray<System.Numerics.Complex>(array);
	                default:
		                throw new NotSupportedException();
                }
            }

            throw new NotImplementedException("implicit operator NDArray(Array array)");
        }

        /// <summary>
        ///     Copies an <see cref="IEnumerable{T}"/> into a freshly allocated <typeparamref name="T"/>[].
        ///     Specialised for List&lt;T&gt; and ICollection&lt;T&gt; to skip the enumerator and to
        ///     use <see cref="GC.AllocateUninitializedArray{T}(int, bool)"/> since we overwrite every slot.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="source">The sequence; enumerated once (a lazy one runs here).</param>
        /// <returns>A new array of the sequence's elements, in order.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static T[] ToArrayFast<T>(IEnumerable<T> source)
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
        /// <param name="a">The boxed <c>Memory&lt;T&gt;</c> or <c>ReadOnlyMemory&lt;T&gt;</c>.</param>
        /// <param name="type">Its type (a constructed <c>Memory&lt;&gt;</c> or <c>ReadOnlyMemory&lt;&gt;</c>).</param>
        /// <returns>A new 1-D array of the memory's elements, or null when the element type is not one of NumSharp's
        ///     dtypes (the caller then tries its other conversions).</returns>
        internal static NDArray ConvertMemory(object a, Type type)
        {
            var elementType = type.GetGenericArguments()[0];
            var isReadOnly = type.GetGenericTypeDefinition() == typeof(ReadOnlyMemory<>);

            if (elementType == typeof(bool)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<bool>)a).Span : ((Memory<bool>)a).Span));
            if (elementType == typeof(byte)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<byte>)a).Span : ((Memory<byte>)a).Span));
            if (elementType == typeof(sbyte)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<sbyte>)a).Span : ((Memory<sbyte>)a).Span));
            if (elementType == typeof(short)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<short>)a).Span : ((Memory<short>)a).Span));
            if (elementType == typeof(ushort)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<ushort>)a).Span : ((Memory<ushort>)a).Span));
            if (elementType == typeof(int)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<int>)a).Span : ((Memory<int>)a).Span));
            if (elementType == typeof(uint)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<uint>)a).Span : ((Memory<uint>)a).Span));
            if (elementType == typeof(long)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<long>)a).Span : ((Memory<long>)a).Span));
            if (elementType == typeof(ulong)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<ulong>)a).Span : ((Memory<ulong>)a).Span));
            if (elementType == typeof(char)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<char>)a).Span : ((Memory<char>)a).Span));
            if (elementType == typeof(Half)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<Half>)a).Span : ((Memory<Half>)a).Span));
            if (elementType == typeof(float)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<float>)a).Span : ((Memory<float>)a).Span));
            if (elementType == typeof(double)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<double>)a).Span : ((Memory<double>)a).Span));
            if (elementType == typeof(decimal)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<decimal>)a).Span : ((Memory<decimal>)a).Span));
            if (elementType == typeof(Complex)) return VectorOf(SpanToArrayFast(isReadOnly ? ((ReadOnlyMemory<Complex>)a).Span : ((Memory<Complex>)a).Span));

            return null;
        }

        /// <summary>
        ///     Optimized Span to Array conversion using GC.AllocateUninitializedArray.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="span">The elements.</param>
        /// <returns>A new array holding a copy of them.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static T[] SpanToArrayFast<T>(ReadOnlySpan<T> span)
        {
            var arr = GC.AllocateUninitializedArray<T>(span.Length);
            span.CopyTo(arr);
            return arr;
        }

        /// <summary>
        ///     Converts a non-generic IEnumerator to an NDArray.
        ///     Element type is detected from items with NumPy-like type promotion.
        ///     Empty collections return empty double[] to match NumPy's float64 default.
        /// </summary>
        /// <param name="enumerator">The items, enumerated to their end.</param>
        /// <returns>A new 1-D array of the items' promoted type, or null when that type is not a NumSharp dtype (a nested
        ///     array among the items, for one).</returns>
        /// <exception cref="InvalidCastException">An item that does not convert to the promoted element type.</exception>
        /// <remarks>A null item (Python's None) is DROPPED, where NumPy builds an object array: <c>[7, None]</c> converts to
        ///     <c>[7]</c>. The items promote with NumSharp's scalar rule (<see cref="FindCommonNumericType"/>), so a C#
        ///     <c>int</c> stays int32 where NumPy's Python int is int64.</remarks>
        internal static NDArray ConvertEnumerator(IEnumerator enumerator)
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
                return VectorOf(Array.Empty<double>());

            var elementType = FindCommonNumericType(items);
            return ConvertObjectListToNDArray(items, elementType);
        }

        /// <summary>
        ///     Finds the common numeric type for a list of objects (NumPy-like promotion).
        ///     Uses existing _FindCommonType_Scalar for consistent type promotion.
        /// </summary>
        /// <param name="items">The non-null items (at least one).</param>
        /// <returns>The element type: <see cref="decimal"/> as soon as one item is a decimal, the items' one type when they
        ///     share it, else <c>np._FindCommonType_Scalar</c> of their type codes (NumSharp's scalar promotion, not NumPy's
        ///     array-coercion promotion of the values' discovered dtypes).</returns>
        internal static Type FindCommonNumericType(List<object> items)
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

            var resultCode = np._FindCommonType_Scalar(typeCodes.Slice(0, uniqueCount).ToArray());
            return resultCode.AsType();
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
        internal static NDArray ConvertObjectListToNDArray(List<object> items, Type elementType)
        {
            var span = CollectionsMarshal.AsSpan(items);

            if (elementType == typeof(bool))
            {
                var arr = GC.AllocateUninitializedArray<bool>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is bool v ? v : Converts.ToBoolean(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(byte))
            {
                var arr = GC.AllocateUninitializedArray<byte>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is byte v ? v : Converts.ToByte(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(short))
            {
                var arr = GC.AllocateUninitializedArray<short>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is short v ? v : Converts.ToInt16(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(ushort))
            {
                var arr = GC.AllocateUninitializedArray<ushort>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is ushort v ? v : Converts.ToUInt16(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(int))
            {
                var arr = GC.AllocateUninitializedArray<int>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is int v ? v : Converts.ToInt32(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(uint))
            {
                var arr = GC.AllocateUninitializedArray<uint>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is uint v ? v : Converts.ToUInt32(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(long))
            {
                var arr = GC.AllocateUninitializedArray<long>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is long v ? v : Converts.ToInt64(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(ulong))
            {
                var arr = GC.AllocateUninitializedArray<ulong>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is ulong v ? v : Converts.ToUInt64(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(char))
            {
                var arr = GC.AllocateUninitializedArray<char>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is char v ? v : Converts.ToChar(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(float))
            {
                var arr = GC.AllocateUninitializedArray<float>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is float v ? v : Converts.ToSingle(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(double))
            {
                var arr = GC.AllocateUninitializedArray<double>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is double v ? v : Converts.ToDouble(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(decimal))
            {
                var arr = GC.AllocateUninitializedArray<decimal>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is decimal v ? v : Converts.ToDecimal(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(sbyte))
            {
                var arr = GC.AllocateUninitializedArray<sbyte>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is sbyte v ? v : Converts.ToSByte(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(Half))
            {
                var arr = GC.AllocateUninitializedArray<Half>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is Half v ? v : Converts.ToHalf(span[i]);
                return VectorOf(arr);
            }
            if (elementType == typeof(Complex))
            {
                var arr = GC.AllocateUninitializedArray<Complex>(span.Length);
                for (int i = 0; i < span.Length; i++)
                    arr[i] = span[i] is Complex v ? v : Converts.ToComplex(span[i]);
                return VectorOf(arr);
            }

            return null; // Unsupported element type
        }
    }
}
