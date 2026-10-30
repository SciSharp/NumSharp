using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // The Python functions of NumPy's numeric.py that create arrays (Appendix D's NDArray.Numeric.cs): np.full, np.full_like
    // and np.ones, which are `a = empty(...); multiarray.copyto(a, fill_value, casting='unsafe')` in NumPy. By fill value
    // and result dtype: a fill value with dimensions broadcasts through np.copyto, the text classes take their own fill
    // routes, and a scalar is stored into every element of the dtype's storage lane by UnmanagedStorage's FullCore /
    // OnesCore (UnmanagedStorage.Ctors.cs). np.indices' dense and sparse grids live here too (Indices, IndicesSparse).
    // They are the array object's own members (the np-layer dtype plan's D1: functions that create arrays), and the np.*
    // functions are facades over them.
    //
    // The plan's phase P6 made its np.* calls (np.asanyarray, np.copyto) the members their facades dispatch to and moved
    // the file from beside its facades (Creation/) to Backends/, beside the array object's class (plan C3).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>np.full</c> (<c>numeric.py</c>: <c>a = empty(shape, dtype, order); multiarray.copyto(a, fill_value,
        ///     casting='unsafe')</c>), the body of <see cref="np.full(Shape, object, DType, string)"/> once the device was
        ///     checked: a fill value with dimensions broadcasts into the result (<see cref="FullFromArrayFill"/>), a text
        ///     dtype (or, without one, a <c>str</c> / <c>bytes</c> fill value) takes its class's fill route, <c>None</c>
        ///     without a dtype is refused, and any other value is stored into every element of the dtype's storage lane
        ///     (<see cref="UnmanagedStorage.FullCore"/>).
        /// </summary>
        /// <param name="shape">The dimensions. A fresh dense layout is kept; a view's layout is re-laid out in C
        ///     order.</param>
        /// <param name="fill_value">The fill value: a scalar, or an array (an <see cref="NDArray"/> of rank ≥ 1, a C# array)
        ///     that broadcasts into the result; null is Python's <c>None</c>.</param>
        /// <param name="dtype">The requested dtype, or null for the fill value's own: an array fill value's dtype (a 0-d
        ///     array's too), NumPy's discovery of a C# <see cref="BigInteger"/> (a Python int), else the storage lane of the
        ///     value's C# type.</param>
        /// <returns>A new array of <paramref name="shape"/>'s dimensions with <paramref name="fill_value"/> cast into every
        ///     element, labelled with the requested lane-sharing C-type class when there is one
        ///     (<see cref="WithLaneSharingDescr"/>).</returns>
        /// <exception cref="NotSupportedException"><paramref name="fill_value"/> and <paramref name="dtype"/> are both null,
        ///     or the fill value is a <see cref="BigInteger"/> past 64 bits and <paramref name="dtype"/> is null: NumPy builds
        ///     an object array there, a dtype NumSharp does not have.</exception>
        /// <exception cref="ArgumentNullException">The fill value's C# type has no storage lane (<see cref="NPTypeCode.Empty"/>:
        ///     a collection that is not a C# array) and no dtype is given.</exception>
        /// <exception cref="TypeError"><c>None</c> into a non-empty integer (or <c>decimal</c>) array: Python's
        ///     <c>int(None)</c>.</exception>
        /// <exception cref="ValueError"><c>None</c> into a <c>StringDType(coerce=False)</c>; an array fill value that does not
        ///     broadcast into the result or whose text does not parse into a numeric dtype; a negative dimension or a byte
        ///     count past the limit (<see cref="AllocationGuard"/>).</exception>
        /// <exception cref="OverflowException">A Python int outside an integer dtype's range (the copy is unsafe, but the
        ///     Python scalar is packed first).</exception>
        /// <remarks>Without a dtype, a scalar's C# type picks the storage lane (a C# <c>int</c> fills an int32 array), not
        ///     NumPy's discovery of the Python scalar it stands for.</remarks>
        internal static NDArray Full(Shape shape, object fill_value, DType dtype)
        {
            // A fill value with dimensions (an NDArray of rank >= 1, a C# array): NumPy's own statements, which broadcast it
            // into the result (FullFromArrayFill). Without dtype= the result takes the fill value's dtype.
            if (IsBroadcastFillValue(fill_value))
                return FullFromArrayFill(shape, fill_value, dtype);
            // Without dtype=, NumPy's `fill_value = asarray(fill_value); dtype = fill_value.dtype`: a 0-d array keeps its own
            // dtype (np.full(3, np.array(5)) is int64, a 0-d text array fills its text dtype), and a Python int (a C#
            // BigInteger) is discovered by its value - int64, uint64 past int64, and past 64 bits NumPy's object array,
            // which _PyLongDType's discovery refuses (NumSharp has no object dtype). A C# number keeps its C# type, as
            // np.asarray(7) does in NumSharp. Both failed with ArgumentNullException (no storage lane for the type).
            if (dtype is null)
            {
                if (fill_value is NDArray zeroD)
                    dtype = zeroD.dtype;
                else if (fill_value is BigInteger)
                    dtype = DTypeRegistry.PyLong.discover_descr_from_pyobject(fill_value);
            }
            // A class whose items hold references (PyDataType_REFCHK; StringDType is the only one): every element is stored
            // through the class, never as copied bytes, in NumPy's empty + copyto(unsafe) (NDVStringMove.Full).
            if (dtype is not null && dtype.Meta.HasReferences)
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
            return WithLaneSharingDescr(UnmanagedStorage.FullCore(shape, fill_value, typeCode), dtype);
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
            NDArray converted = fill_value is NDArray ? null : AnyArrayFromObject(fill_value, null, 'K');
            NDArray fill = converted ?? (NDArray)fill_value;
            try
            {
                var descr = resultDescr ?? fill.dtype;
                // Text results: their routes are NumPy's empty + copyto(unsafe) already, broadcasting included; a class whose
                // items hold references (PyDataType_REFCHK: StringDType) stores each element through the class.
                if (descr.Meta.HasReferences)
                    return Backends.Strings.NDVStringMove.Full(shape, fill, descr);
                if (Backends.Strings.NDStringStorage.IsString(descr))
                    return Backends.Strings.NDStringArrays.Full(shape, fill, descr);

                // The allocating constructor runs the dimension guard itself (UnmanagedStorage.Allocate).
                var result = new NDArray(descr, shape, false);
                try
                {
                    UnmanagedStorage.Copyto(result, fill, NPY_CASTING.NPY_UNSAFE_CASTING, null);
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
        ///     NumPy's <c>np.full_like</c> (<c>numeric.py</c>: <c>res = empty_like(a, dtype, order, subok, shape);
        ///     multiarray.copyto(res, fill_value, casting='unsafe')</c>), the body of
        ///     <see cref="np.full_like(NDArray, object, DType, char, string)"/> once its class boundaries and the device were
        ///     handled: the result takes <paramref name="a"/>'s dimensions in <c>empty_like</c>'s layout (<c>order='K'</c> keeps
        ///     <paramref name="a"/>'s axis permutation) and <paramref name="a"/>'s dtype unless one is requested, and the fill
        ///     value is cast into it - broadcast when it has dimensions, through the text classes' fill routes for a text
        ///     result, stored into the storage lane otherwise.
        /// </summary>
        /// <param name="a">The prototype: its dimensions, its memory layout and its dtype.</param>
        /// <param name="fill_value">The fill value: a scalar, or an array (an <see cref="NDArray"/> of rank ≥ 1, a C# array)
        ///     that broadcasts into the result; null is Python's <c>None</c>.</param>
        /// <param name="dtype">The result's dtype, or null for <paramref name="a"/>'s. The fill value's own type never
        ///     selects it.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>: the result's memory layout.</param>
        /// <returns>A new array with <paramref name="a"/>'s dimensions and <paramref name="fill_value"/> cast into every
        ///     element, labelled with a lane-sharing C-type class when that is the dtype (as NumPy's <c>empty_like</c> keeps
        ///     the descriptor).</returns>
        /// <exception cref="ArgumentException">An order other than C, F, A or K.</exception>
        /// <exception cref="TypeError"><c>None</c> into a non-empty integer (or <c>decimal</c>) result.</exception>
        /// <exception cref="ValueError">An array fill value that does not broadcast into the result, or whose text does not
        ///     parse into a numeric result; <c>None</c> into a <c>StringDType(coerce=False)</c>; a byte count past the
        ///     limit.</exception>
        /// <exception cref="OverflowException">A Python int outside the result's integer range
        ///     (<c>np.full_like(int8_array, 1000)</c>).</exception>
        internal static NDArray FullLike(NDArray a, object fill_value, DType dtype, char order)
        {
            // NumPy's *_like contract preserves a.dtype unless dtype= is explicitly supplied.
            // fill_value is CAST into that dtype; its CLR type never selects the result dtype.
            var typeCode = dtype != null ? dtype.GetTypeCode() : a.typecode;
            // PyArray_NewLikeArray's layout: order='K' keeps a's axis permutation, not merely C-or-F.
            var shape = Shape.NewLikeLayout(a.Shape, order, (long[])a.Shape.dimensions.Clone());

            // A string result (a's dtype, or a requested one — unsized means one character, NumPy's empty_like) is
            // `res = empty_like(...); copyto(res, fill_value, casting='unsafe')`: the copy truncates / converts the value.
            var resultDescr = dtype ?? a.dtype;
            // A fill value with dimensions broadcasts into empty_like's layout through the same copyto
            // (np.full_like(f3, [1, 2, 3]) is [1., 2., 3.]; a StringDType fill value is parsed per element).
            if (IsBroadcastFillValue(fill_value))
                return FullFromArrayFill(shape, fill_value, resultDescr);
            // A class whose items hold references (PyDataType_REFCHK: StringDType) stores each element through the class.
            if (resultDescr.Meta.HasReferences)
                return Backends.Strings.NDVStringMove.Full(shape, fill_value, resultDescr);
            if (Backends.Strings.NDStringStorage.IsString(resultDescr))
                return Backends.Strings.NDStringArrays.Full(shape, fill_value, resultDescr);

            // Allocates from shape.size directly (not via UnmanagedStorage.Allocate), so the
            // dimension guard has to be explicit here — see np.full.
            AllocationGuard.CheckDimensions(shape.dimensions, typeCode);

            // NumPy's `res = empty_like(...); copyto(res, fill_value, casting='unsafe')`: a Python int is range-checked in
            // the result's dtype (np.full_like(int8_array, 1000) raises), a Python float / complex cast unsafely, and None
            // (a null) is NaN / nan+nanj / False, or int(None)'s TypeError when there is an element to write. The result is
            // labelled with a lane-sharing C-type class when that is the dtype (a's np.longdouble, or a requested np.intc),
            // as NumPy's empty_like keeps the descriptor.
            return WithLaneSharingDescr(
                new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, UnmanagedStorage.FullFillElement(fill_value, typeCode, shape.size)), shape)),
                resultDescr);
        }

        /// <summary>
        ///     NumPy's <c>np.ones</c> (<c>numeric.py</c>: <c>a = empty(shape, dtype, order); multiarray.copyto(a, 1,
        ///     casting='unsafe')</c>), the body of <see cref="np.ones(Shape, DType, string)"/> once the device was checked: a
        ///     text dtype gets the text <c>'1'</c> in every element (<see cref="OnesOfText"/>), any other dtype the Python int
        ///     1 as its class's <c>setitem</c> converts it, written into every element (<see cref="UnmanagedStorage.OnesCore"/>).
        /// </summary>
        /// <param name="shape">The dimensions. A dense layout (C, F or a KEEPORDER permutation at offset 0) is kept, which is
        ///     how <c>ones_like(a, order='K')</c> gets <paramref name="shape"/>'s memory order; a view's layout is re-laid out
        ///     in C order.</param>
        /// <param name="dtype">The dtype, or null for float64.</param>
        /// <returns>A fresh, owned, writeable array of ones, labelled with the requested lane-sharing C-type class when there
        ///     is one.</returns>
        /// <exception cref="ValueError">A negative dimension, or a byte count past the limit
        ///     (<see cref="AllocationGuard"/>).</exception>
        internal static NDArray Ones(Shape shape, DType dtype)
        {
            // Text has no lane fill: a fixed-width string class, and a class whose items hold references (StringDType),
            // take the text routes; every other class converts the Python int 1 through its setitem (OnesCore).
            if (dtype is not null && (dtype.Meta.HasReferences || Backends.Strings.NDStringStorage.IsString(dtype)))
                return OnesOfText(shape, dtype);
            return WithLaneSharingDescr(UnmanagedStorage.OnesCore(shape, dtype ?? DType.Double), dtype);
        }

        /// <summary>
        ///     NumPy's <c>np.ones</c> with an <c>order</c>, the body of <see cref="np.ones(Shape, char, DType)"/>: the order
        ///     resolved (<c>'A'</c> and <c>'K'</c> mean C without a source), the dimensions laid out in it, then the fill of
        ///     <see cref="Ones(Shape, DType)"/>.
        /// </summary>
        /// <param name="shape">The dimensions (only they are read: the layout comes from <paramref name="order"/>).</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <param name="dtype">The dtype, or null for float64.</param>
        /// <returns>A fresh array of ones in the requested layout (the fill is order-independent, so only the flags
        ///     differ).</returns>
        /// <exception cref="ValueError">An order other than C, F, A or K; a negative dimension, or a byte count past the
        ///     limit.</exception>
        internal static NDArray Ones(Shape shape, char order, DType dtype)
        {
            char physical = OrderResolver.Resolve(order);
            var orderedShape = Shape.Adopt(shape.dimensions, physical);
            // The text routes as in Ones(Shape, DType): text has no lane fill.
            if (dtype is not null && (dtype.Meta.HasReferences || Backends.Strings.NDStringStorage.IsString(dtype)))
                return OnesOfText(orderedShape, dtype);
            return WithLaneSharingDescr(UnmanagedStorage.OnesCore(orderedShape, dtype ?? DType.Double), dtype);
        }

        /// <summary>
        ///     <c>np.ones</c> for a string dtype — NumPy's <c>empty</c> + <c>copyto(a, 1, casting='unsafe')</c>: the number one
        ///     written as its <c>str()</c>, <c>'1'</c>, into every element (<see cref="Backends.Strings.NDStringArrays.Full"/>
        ///     with the Python int 1).
        /// </summary>
        /// <param name="shape">The dimensions (a fresh dense layout is kept, a view's re-laid out in C order).</param>
        /// <param name="dtype">The string dtype (sized or not).</param>
        /// <returns>The new string array.</returns>
        private static NDArray OnesOfText(Shape shape, DType dtype)
            // A class whose items hold references (StringDType) stores each element through the class; the int 1 is cast,
            // '1' (coerce does not apply to a cast). A fixed-width class takes the string copy route.
            => dtype.Meta.HasReferences
                ? Backends.Strings.NDVStringMove.Full(shape, 1L, dtype)
                : Backends.Strings.NDStringArrays.Full(shape, 1L, dtype);

        /// <summary>
        ///     Labels a FRESH array that a creation routine allocated by storage lane with the lane-sharing C-type class the
        ///     caller asked for (<c>np.ones(3, np.longdouble)</c> reports <c>'g'</c>, Windows' <c>np.full(2, 7, np.intc)</c>
        ///     <c>'i'</c>, as NumPy's do); any other dtype leaves the array as allocated.
        /// </summary>
        /// <param name="fresh">The array just created, which nothing else references yet.</param>
        /// <param name="dtype">The requested dtype, or null.</param>
        /// <returns><paramref name="fresh"/> itself.</returns>
        /// <remarks>Footgun: never pass an array other code holds a view of — relabelling its storage changes their dtype too.</remarks>
        internal static NDArray WithLaneSharingDescr(NDArray fresh, DType dtype)
        {
            fresh.Storage.AdoptLaneSharingDescr(dtype);
            return fresh;
        }

        /// <summary>
        ///     NumPy's <c>np.indices</c> (<c>numeric.py</c>) with <c>sparse=False</c>, the body of
        ///     <see cref="np.indices(long[], DType)"/> once the dimensions were checked for null: a single dense array of
        ///     shape <c>(len(dimensions), *dimensions)</c> whose d-th sub-array holds each position's d-th coordinate.
        /// </summary>
        /// <param name="dimensions">The grid's sizes. An empty array gives NumPy's <c>(0,)</c> result. Never mutated.</param>
        /// <param name="dtype">The element type, or null for int64 (NumPy's <c>dtype=int</c>, intp). The result carries this
        ///     descriptor itself: a text dtype's width, a C-type class that shares a storage lane.</param>
        /// <returns>The dense coordinate grid, a new array that owns its data.</returns>
        /// <exception cref="ValueError">A dimension is negative (<c>negative dimensions are not allowed</c>) or the grid's
        ///     byte count overflows (<c>array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible
        ///     size.</c>): NumPy's <c>empty((N,) + dimensions)</c>, which runs before any arange.</exception>
        /// <exception cref="TypeError">An arange of the dtype refuses, in axis order and whatever the other sizes (a 0 among
        ///     them included): a class without NumPy's arange fill - a text dtype, <c>arange() not supported for inputs with
        ///     DType &lt;class 'numpy.dtypes.StrDType'&gt;.</c>, whatever its length - or bool with a dimension longer than two
        ///     (<c>arange() is only supported for booleans when the result has at most length 2.</c>).</exception>
        /// <exception cref="NotSupportedException">The dtype's class has no NumSharp storage yet (datetime64, timedelta64), or
        ///     the IL fill kernel is unavailable (dynamic code generation disabled).</exception>
        /// <remarks>
        ///     NumPy's statements: <c>res = empty((N,) + dimensions, dtype=dtype)</c>, then for every axis
        ///     <c>res[i] = arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i+1:])</c>. A result of the int64 storage
        ///     lane (int64 and the C-type classes that share it) is filled by the IL kernel in one pass instead: an int64
        ///     arange never refuses and its values are the kernel's. Every other dtype assigns its own aranges, broadcast
        ///     along the other axes, so the values are NumPy's arange fill (an int8 grid wraps past 127, a float16 one is
        ///     each position rounded once).
        /// </remarks>
        [NDScoped]
        internal static unsafe NDArray Indices(long[] dimensions, DType dtype)
        {
            DType descr = dtype ?? DType.Int64;
            int ndim = dimensions.Length;

            // `res = empty((N,) + dimensions, dtype=dtype)`: the allocation's checks run before any arange - a negative
            // dimension, then a byte count past the address space, NumPy's two ValueErrors in its scan order - and the
            // result keeps the descriptor (an empty tuple gives the (0,) array of a text dtype too).
            var resultShape = new long[ndim + 1];
            resultShape[0] = ndim;
            for (int d = 0; d < ndim; d++)
                resultShape[d + 1] = dimensions[d];
            var res = new NDArray(descr, Shape.Adopt(resultShape), false);
            if (ndim == 0)
                return res;

            try
            {
                if (descr.Meta.TypeCode == NPTypeCode.Int64)
                    FillIndicesInt64(res, dimensions);
                else
                    for (int i = 0; i < ndim; i++)
                    {
                        // `idx = arange(dim, dtype=dtype)`: NumPy's arange refuses a class without a fill function before
                        // it computes a length, and bool past two elements - so an empty grid still raises, in axis order.
                        using var idx = UnmanagedStorage.ArangeObj(0, dimensions[i], 1, descr);
                        var axisShape = new long[ndim];
                        for (int k = 0; k < ndim; k++)
                            axisShape[k] = k == i ? dimensions[i] : 1;
                        // `res[i] = idx.reshape(...)`: the coordinate runs along axis i and broadcasts along the others.
                        using var line = idx.reshape(Shape.Adopt(axisShape));
                        using var slab = res[i];
                        UnmanagedStorage.Copyto(slab, line, NPY_CASTING.NPY_SAME_KIND_CASTING, null);
                    }
            }
            catch
            {
                // An arange refused: the grid never reaches the caller, as NumPy drops its res.
                res.Dispose();
                throw;
            }

            return res;
        }

        /// <summary>
        ///     NumPy's <c>np.indices</c> (<c>numeric.py</c>) with <c>sparse=True</c>, the body of
        ///     <see cref="np.indices_sparse(long[], DType)"/> once the dimensions were checked for null: one array per
        ///     dimension, of shape <c>(1, …, 1, dimensions[d], 1, …, 1)</c>, that broadcasts against the others.
        /// </summary>
        /// <param name="dimensions">The grid's sizes. An empty array gives an empty tuple. Never mutated.</param>
        /// <param name="dtype">The element type of each array, or null for int64.</param>
        /// <returns>One array per dimension, each a VIEW of its arange (its <c>flags.owndata</c> is False, as NumPy's
        ///     reshape result is).</returns>
        /// <exception cref="TypeError">An arange of the dtype refuses, in axis order: a text dtype (no arange fill), whatever
        ///     the length, or bool with a dimension longer than two.</exception>
        /// <exception cref="NotSupportedException">The dtype's class has no NumSharp storage yet.</exception>
        /// <remarks>
        ///     NumPy's statements: <c>res = ()</c>, then for every axis
        ///     <c>res = res + (arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i+1:]),)</c>. There is no
        ///     <c>empty</c> to refuse a negative dimension: its arange is empty and the reshape reads the negative entry as
        ///     the unknown dimension, so that axis is 0 long (the sparse grid of <c>(2, -3)</c> is a <c>(2, 1)</c> and a
        ///     <c>(1, 0)</c> array).
        /// </remarks>
        [NDScoped]
        internal static NDArray[] IndicesSparse(long[] dimensions, DType dtype)
        {
            DType descr = dtype ?? DType.Int64;
            int ndim = dimensions.Length;
            var res = new NDArray[ndim];
            for (int i = 0; i < ndim; i++)
            {
                // `arange(dim, dtype=dtype)`, then the reshape, which keeps a view of it: the aranges own the buffers.
                using var idx = UnmanagedStorage.ArangeObj(0, dimensions[i], 1, descr);
                var axisShape = new long[ndim];
                for (int k = 0; k < ndim; k++)
                    axisShape[k] = k == i ? dimensions[i] : 1;
                // reshape(long[]): a negative entry is the unknown dimension (np.reshape's _fix_unknown_dimension).
                res[i] = idx.reshape(axisShape);
            }

            return res;
        }

        /// <summary>
        ///     Fills a fresh int64-lane <c>(ndim, *dims)</c> coordinate grid with the IL kernel: each slab's runs written as
        ///     block stores, no per-element coordinate arithmetic.
        /// </summary>
        /// <param name="res">The fresh C-contiguous grid of the int64 storage lane, of shape <c>(ndim, *dimensions)</c>.</param>
        /// <param name="dimensions">The grid's sizes: at least one, none negative (the allocation of
        ///     <paramref name="res"/> checked them). Never mutated.</param>
        /// <exception cref="NotSupportedException">The IL fill kernel is unavailable (dynamic code generation disabled).</exception>
        private static unsafe void FillIndicesInt64(NDArray res, long[] dimensions)
        {
            int ndim = dimensions.Length;
            // The element count of one slab. It cannot overflow: the allocation of N such slabs of 8 bytes succeeded.
            long prod = 1;
            for (int d = 0; d < ndim; d++)
                prod *= dimensions[d];
            // An empty grid has nothing to write.
            if (prod == 0)
                return;

            // dimStrides[ndim-1] = 1; dimStrides[d] = dimStrides[d+1] * dims[d+1].
            Span<long> dimStrides = stackalloc long[ndim];
            dimStrides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; d--)
                dimStrides[d] = dimStrides[d + 1] * dimensions[d + 1];

            var kernel = DirectILKernelGenerator.GetIndicesKernel();
            if (kernel == null)
                throw new NotSupportedException("np.indices: IL kernel unavailable");

            long* resPtr = (long*)res.Storage.Address;
            // `dimensions` is already long[], so the kernel's dims pointer pins it directly — no per-call copy into a
            // scratch span is needed (the int[] path widened once, up front).
            fixed (long* dsPtr = dimStrides)
            fixed (long* dPtr = dimensions)
            {
                kernel(resPtr, dsPtr, dPtr, ndim, prod);
            }
        }
    }
}
