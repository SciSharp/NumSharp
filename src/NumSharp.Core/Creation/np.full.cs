using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full(int[] shape, object fill_value)
            => full(new Shape(shape), fill_value);

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full(long[] shape, object fill_value)
            => full(new Shape(shape), fill_value);

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full<T>(int[] shape, object fill_value) where T : unmanaged
            => full(new Shape(shape), fill_value, typeof(T));

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full<T>(long[] shape, object fill_value) where T : unmanaged
            => full(new Shape(shape), fill_value, typeof(T));

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value: a scalar, or an array (an <see cref="NDArray"/> of rank ≥ 1, a C# array) that
        ///     broadcasts into the result, as NumPy's <c>copyto(a, fill_value, casting='unsafe')</c> broadcasts it.</param>
        /// <param name="dtype">
        ///     The desired dtype for the array — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     Default (null) infers from fill_value (an array fill value gives its own dtype).
        /// </param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <exception cref="NotSupportedException"><paramref name="fill_value"/> is null (Python's <c>None</c>) and
        ///     <paramref name="dtype"/> is null: NumPy builds an object array there, a dtype NumSharp does not have.</exception>
        /// <exception cref="TypeError"><paramref name="fill_value"/> is null and <paramref name="dtype"/> is an integer (or
        ///     <c>decimal</c>) dtype, for a non-empty shape: Python's <c>int(None)</c>, raised by NumPy's
        ///     <c>copyto(a, None, casting='unsafe')</c> (an empty result converts nothing and returns).</exception>
        /// <exception cref="ValueError"><paramref name="fill_value"/> is null and <paramref name="dtype"/> is a
        ///     <c>StringDType(coerce=False)</c>; or an array fill value that does not broadcast into the result (<c>could not
        ///     broadcast input array from shape (2,) into shape (3,)</c>), or whose text does not parse into a numeric
        ///     <paramref name="dtype"/>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.full.html
        ///     <para>A null <paramref name="fill_value"/> is Python's <c>None</c>, filled the way NumPy's
        ///     <c>a = empty(shape, dtype); copyto(a, None, casting='unsafe')</c> fills it: NaN (the positive <c>NPY_NAN</c>) for
        ///     a float dtype, <c>nan+nanj</c> for complex, <c>False</c> for bool, <c>'None'</c> (truncated) for <c>S</c> /
        ///     <c>U</c>, a <c>StringDType</c>'s NA object or <c>'None'</c>. Before NumSharp stored None, this returned ZEROS.</para>
        /// </remarks>
        public static NDArray full(Shape shape, object fill_value, DType dtype = null, string device = null)
        {
            ValidateDevice(device);
            // A fill value with dimensions (an NDArray of rank >= 1, a C# array): NumPy's own statements, which broadcast it
            // into the result (FullFromArrayFill). Without dtype= the result takes the fill value's dtype.
            if (IsBroadcastFillValue(fill_value))
                return FullFromArrayFill(shape, fill_value, dtype);
            // StringDType: NumPy's empty + copyto(unsafe) (NDVStringMove.Full).
            if (dtype?.Meta is VStringDTypeMeta)
                return Backends.Strings.NDVStringMove.Full(shape, fill_value, dtype);
            // Fixed-width strings: NumPy's `a = empty(shape, dtype); copyto(a, fill_value, casting='unsafe')` — the dtype is
            // the fill value's own (np.full(2, 'abc') is <U3) or the requested one, an unsized request is ONE character
            // (np.full(2, 'abc', dtype='S') is [b'a', b'a']), and the value is truncated / converted by the copy.
            if (Backends.Strings.NDStringStorage.IsString(dtype) || (dtype == null && (fill_value is string || fill_value is byte[])))
                return Backends.Strings.NDStringArrays.Full(shape, fill_value, dtype);
            // None with no dtype: NumPy's `fill_value = asarray(fill_value)` makes an object array, whose dtype the result
            // takes. NumSharp has none, so the call is refused (it used to throw a NullReferenceException here).
            if (fill_value is null && dtype is null)
                throw NDArray.NoneWithoutDtype("np.full");
            // When dtype is explicitly provided, use it; when null, infer from fill_value.
            var typeCode = dtype != null ? dtype.GetTypeCode() : fill_value.GetType().GetTypeCode();
            return WithLaneSharingDescr(FullCore(shape, fill_value, typeCode), dtype);
        }

        /// <summary>
        ///     Whether a fill value of <c>np.full</c> / <c>np.full_like</c> has dimensions: an <see cref="NDArray"/> of rank
        ///     ≥ 1, or a C# array other than <c>byte[]</c> (a <c>byte[]</c> is Python's <c>bytes</c>, one text value).
        /// </summary>
        /// <param name="fill_value">The caller's fill value (null is Python's <c>None</c>).</param>
        /// <returns>True when NumPy's <c>copyto</c> would broadcast the value rather than store one scalar.</returns>
        /// <remarks>
        ///     A 0-d array is not one: it stores one element through the scalar rule
        ///     (<see cref="UnmanagedStorage.TryCopytoUnsafeScalarElement"/>), which a text 0-d array reaches too (parsed once).
        /// </remarks>
        internal static bool IsBroadcastFillValue(object fill_value)
            => fill_value is NDArray nd ? nd.ndim > 0 : fill_value is Array && fill_value is not byte[];

        /// <summary>
        ///     NumPy's <c>np.full</c> / <c>np.full_like</c> for a fill value with dimensions: <c>fill_value =
        ///     asarray(fill_value)</c> (whose dtype <c>np.full</c> takes when <paramref name="resultDescr"/> is null),
        ///     <c>a = empty(shape, dtype)</c> (<c>empty_like</c>'s layout for <c>full_like</c>), then
        ///     <c>copyto(a, fill_value, casting='unsafe')</c> — the value BROADCASTS into the result and each element is cast
        ///     (a text element parsed into a number, a number formatted into text).
        /// </summary>
        /// <param name="shape">The result's layout: dimensions for <c>np.full</c>, <c>empty_like</c>'s layout for
        ///     <c>np.full_like</c> (a view's layout is re-laid out in C order by the allocation).</param>
        /// <param name="fill_value">The fill value (<see cref="IsBroadcastFillValue"/> holds).</param>
        /// <param name="resultDescr">The result's descriptor, or null for the fill value's own.</param>
        /// <returns>The filled array.</returns>
        /// <exception cref="ValueError">The fill value does not broadcast into the result (NumPy's copyto text, <c>could not
        ///     broadcast input array from shape (2,) into shape (3,)</c>), or a text element does not parse into a numeric
        ///     result (the cast's text).</exception>
        /// <exception cref="OverflowException">A text element parses to an integer outside an integer result.</exception>
        /// <remarks>
        ///     The fill value used to fall to the house scalar conversion, which refused or misread every array of rank
        ///     ≥ 1 (<c>np.full_like(f3, [1, 2, 3])</c> is <c>[1., 2., 3.]</c> in NumPy). Text results keep their own fill
        ///     routes, which broadcast the same way. The allocation is disposed when the copy throws.
        /// </remarks>
        internal static NDArray FullFromArrayFill(Shape shape, object fill_value, DType resultDescr)
        {
            // asarray(fill_value): an NDArray as it is, a C# array converted (np.asanyarray's dtype discovery).
            NDArray converted = fill_value is NDArray ? null : asanyarray(fill_value);
            NDArray fill = converted ?? (NDArray)fill_value;
            try
            {
                var descr = resultDescr ?? fill.dtype;
                // Text results: their routes are NumPy's empty + copyto(unsafe) already, broadcasting included.
                if (descr.Meta is VStringDTypeMeta)
                    return Backends.Strings.NDVStringMove.Full(shape, fill, descr);
                if (Backends.Strings.NDStringStorage.IsString(descr))
                    return Backends.Strings.NDStringArrays.Full(shape, fill, descr);

                // The allocating constructor runs the dimension guard itself (UnmanagedStorage.Allocate).
                var result = new NDArray(descr, shape, false);
                try
                {
                    copyto(result, fill, "unsafe");
                }
                catch
                {
                    // The half-written allocation never reaches the caller.
                    result.Dispose();
                    throw;
                }
                return WithLaneSharingDescr(result, descr);
            }
            finally
            {
                converted?.Dispose();
            }
        }

        /// <summary>
        ///     The storage-lane core behind <see cref="full(Shape,object,DType,string)"/>: a new array of given shape and
        ///     lane, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <param name="typeCode">The desired data-type for the array.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        private static NDArray FullCore(Shape shape, object fill_value, NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            // This path allocates from shape.size directly instead of going through
            // UnmanagedStorage.Allocate, so it needs the dimension guard explicitly: for
            // (2^62, 2^62) the element count ITSELF wraps to 0, and a zero-element request is
            // one the allocator happily satisfies. Checking dimensions rather than the wrapped
            // product is the only way to see it.
            AllocationGuard.CheckDimensions(shape.dimensions, typeCode);

            object fill = FullFillElement(fill_value, typeCode, shape.size);

            // The wrapping constructor installs the shape as given, so the fresh-layout rule of the allocating funnel has
            // to be applied here: a view's Shape (np.full(x.Shape, v) for x = a[::4]) would otherwise address past the
            // `size`-element buffer, and a read-only source's Shape would make the new array read-only.
            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, fill), shape.ForFreshAllocation()));
        }

        /// <summary>
        ///     The element <c>np.full</c> / <c>np.full_like</c> fill with — NumPy's <c>copyto(a, fill_value,
        ///     casting='unsafe')</c> of a scalar (<see cref="UnmanagedStorage.TryCopytoUnsafeScalarElement"/>), boxed as
        ///     <paramref name="typeCode"/>'s exact CLR type.
        /// </summary>
        /// <param name="fill_value">The caller's fill value; <c>null</c> is Python's <c>None</c>.</param>
        /// <param name="typeCode">The new array's dtype.</param>
        /// <param name="size">The new array's element count: <c>None</c> is converted only when there is an element to
        ///     write.</param>
        /// <returns>The boxed element; for a value that is not a scalar (a list, an array with dimensions), the house
        ///     conversion <see cref="Converts.ChangeType(object, NPTypeCode)"/> of it.</returns>
        /// <exception cref="OverflowException">A Python int out of the dtype's integer range (NumPy's OverflowError texts):
        ///     the copy is unsafe, but the Python scalar is packed first.</exception>
        /// <exception cref="TypeError"><c>None</c> into a non-empty integer (or <c>decimal</c>) array.</exception>
        /// <remarks>
        ///     <para>A value of the dtype's own C# type is already the stored element (<c>np.full(3, 0.5)</c>,
        ///     <c>np.full(3, 7, dtype: np.int32)</c>): it returns as is, with no packing.</para>
        ///     <para><c>None</c> reaches NumPy's <c>copyto(a, None, casting='unsafe')</c> as a 0-d OBJECT array, not a
        ///     Python scalar: nothing packs it up front, and the object-to-dtype cast runs once per element written — so
        ///     an empty result never converts it (<c>np.full((0,), None, dtype=int)</c> returns an empty array) while one
        ///     element of an integer dtype raises <c>int(None)</c>'s TypeError (probed 2.4.2). It used to fall to the
        ///     house conversion below, which turned null into ZERO.</para>
        /// </remarks>
        internal static object FullFillElement(object fill_value, NPTypeCode typeCode, long size)
        {
            if (fill_value is null)
                // The empty result never writes its fill element, so the dtype's zero stands for it (a fresh slot is zeroed).
                return size == 0 ? ReductionInitial.Box(new byte[ReductionInitial.SlotBytes], typeCode) : ReductionInitial.BoxNone(typeCode);

            // Only a number qualifies (Type.GetTypeCode unwraps an array type to its element type).
            if (NDArray.IsStoreScalar(fill_value) && fill_value.GetType().GetTypeCode() == typeCode)
                return fill_value;

            Span<byte> element = stackalloc byte[ReductionInitial.SlotBytes];
            if (UnmanagedStorage.TryCopytoUnsafeScalarElement(fill_value, typeCode, element))
                return ReductionInitial.Box(element, typeCode);
            return Converts.ChangeType(fill_value, typeCode);
        }
    }
}
