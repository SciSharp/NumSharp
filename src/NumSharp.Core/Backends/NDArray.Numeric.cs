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
    // OnesCore (UnmanagedStorage.Ctors.cs). np.indices' dense and sparse grids live here too (indices, indices_sparse), and
    // numeric.py's other array-object functions: zeros_like / ones_like (empty_like's checks and layout, then the fill),
    // isfortran (a flags read), roll (empty_like plus two slice copies) and isscalar (a type test over C# values). They are
    // the array object's own members (the np-layer dtype plan's D1: functions that create arrays), and the np.* functions
    // are facades over them.
    //
    // The plan's phase P6 made its np.* calls (np.asanyarray, np.copyto) the members their facades dispatch to and moved
    // the file from beside its facades (Creation/) to Backends/, beside the array object's class (plan C3); P8 moved the
    // functions without dtype logic here with their calls converted the same way (np.zeros / np.ones / np.empty_like).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>np.full</c> (<c>numeric.py</c>: <c>a = empty(shape, dtype, order); multiarray.copyto(a, fill_value,
        ///     casting='unsafe')</c>), the body of <see cref="np.full(Shape, object, DType, char, string)"/> once the device was
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
        /// <exception cref="OverflowError">A Python int outside an integer dtype's range (the copy is unsafe, but the
        ///     Python scalar is packed first).</exception>
        /// <remarks>Without a dtype, a scalar's C# type picks the storage lane (a C# <c>int</c> fills an int32 array), not
        ///     NumPy's discovery of the Python scalar it stands for.</remarks>
        internal static NDArray full(Shape shape, object fill_value, DType dtype)
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
        /// <exception cref="OverflowError">A text element parses to an integer outside an integer result.</exception>
        /// <remarks>
        ///     The fill value used to fall to the house scalar conversion, which refused or misread every array of rank
        ///     ≥ 1 (<c>np.full_like(f3, [1, 2, 3])</c> is <c>[1., 2., 3.]</c> in NumPy). Text results keep their own fill
        ///     routes, which broadcast the same way. The allocation is disposed when the copy throws.
        /// </remarks>
        internal static NDArray FullFromArrayFill(Shape shape, object fill_value, DType resultDescr)
        {
            // asarray(fill_value): an NDArray as it is, a C# array converted (np.asanyarray's dtype discovery).
            NDArray converted = fill_value is NDArray ? null : asanyarray(fill_value, null, 'K');
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
        /// <param name="shapeOverride">NumPy's <c>shape=</c> override, or an empty (default) <see cref="Shape"/> for
        ///     <paramref name="a"/>'s dimensions; <c>'K'</c> lays a shape of another rank out in C order.</param>
        /// <returns>A new array with <paramref name="a"/>'s dimensions (or the override's) and <paramref name="fill_value"/> cast into every
        ///     element, labelled with a lane-sharing C-type class when that is the dtype (as NumPy's <c>empty_like</c> keeps
        ///     the descriptor).</returns>
        /// <exception cref="ArgumentException">An order other than C, F, A or K.</exception>
        /// <exception cref="TypeError"><c>None</c> into a non-empty integer (or <c>decimal</c>) result.</exception>
        /// <exception cref="ValueError">An array fill value that does not broadcast into the result, or whose text does not
        ///     parse into a numeric result; <c>None</c> into a <c>StringDType(coerce=False)</c>; a byte count past the
        ///     limit.</exception>
        /// <exception cref="OverflowError">A Python int outside the result's integer range
        ///     (<c>np.full_like(int8_array, 1000)</c>).</exception>
        /// <remarks>NumPy's <c>subok=True</c>: <c>[NewSubtype("a")]</c> creates the result as the operand's class (a matrix
        ///     view; a chararray), and a chararray operand's finalize rule runs on the result dtype here, where
        ///     <c>empty_like</c> allocates - after the order check, before the fill.</remarks>
        [NewSubtype("a")]
        internal static NDArray full_like(NDArray a, object fill_value, DType dtype, char order, Shape shapeOverride = default)
        {
            // NumPy's *_like contract preserves a.dtype unless dtype= is explicitly supplied.
            // fill_value is CAST into that dtype; its CLR type never selects the result dtype.
            var typeCode = dtype != null ? dtype.GetTypeCode() : a.typecode;
            // PyArray_NewLikeArrayWithShape's layout: order='K' keeps a's axis permutation (C when shape= changes the rank),
            // not merely C-or-F; an empty (default) override is "no shape=".
            var dims = shapeOverride.IsEmpty ? (long[])a.Shape.dimensions.Clone() : (long[])shapeOverride;
            var shape = Shape.NewLikeLayout(a.Shape, order, dims);
            // empty_like allocates the result as a chararray operand's class (subok=True) at this point: its finalize rule
            // refuses a dtype that is not string data BEFORE the fill could raise its own error.
            if (NDSubclass.StandsForCharArray(a))
                NDCharArray.Finalize(dtype ?? a.dtype);

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
        ///     casting='unsafe')</c>), the body of <see cref="np.ones(Shape, DType, char, string)"/> once the device was checked: a
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
        internal static NDArray ones(Shape shape, DType dtype)
        {
            // Text has no lane fill: a fixed-width string class, and a class whose items hold references (StringDType),
            // take the text routes; every other class converts the Python int 1 through its setitem (OnesCore).
            if (dtype is not null && (dtype.Meta.HasReferences || Backends.Strings.NDStringStorage.IsString(dtype)))
                return OnesOfText(shape, dtype);
            return WithLaneSharingDescr(UnmanagedStorage.OnesCore(shape, dtype ?? DType.Double), dtype);
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

        // ---------------------------- np.empty_like / np.zeros_like / np.ones_like ----------------------------

        /// <summary>
        ///     NumPy's <c>np.empty_like</c> (<c>PyArray_NewLikeArrayWithShape</c>), the body of
        ///     <see cref="np.empty_like(NDArray, DType, Shape, char, string)"/>: the argument checks in NumPy's order (the
        ///     order's letter, then the device), the dimensions (the prototype's, or <paramref name="shape"/>'s), the layout
        ///     (<c>'K'</c> keeps the prototype's axis permutation), then an uninitialized allocation.
        /// </summary>
        /// <param name="prototype">The prototype (not null): its dimensions unless <paramref name="shape"/> is given, its
        ///     dtype unless <paramref name="dtype"/> is, and for <c>'K'</c> its axis permutation.</param>
        /// <param name="dtype">The result dtype, or null for the prototype's.</param>
        /// <param name="shape">The result's dimensions, or an empty <see cref="Shape"/> for the prototype's.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c> in either case.</param>
        /// <param name="device">Null or <c>"cpu"</c>.</param>
        /// <returns>A new, owned array of arbitrary content.</returns>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K (reported before the device);
        ///     <paramref name="device"/> is neither null nor <c>"cpu"</c>; a byte count past the limit.</exception>
        /// <remarks>NumPy's <c>subok=True</c>: <c>[NewSubtype("prototype")]</c> creates the result as the operand's class
        ///     after the allocation - a matrix view, or a chararray whose finalize rule refuses a dtype that is not string
        ///     data (empty_like fills nothing, so that error comes last, as in NumPy).</remarks>
        [NewSubtype("prototype")]
        internal static NDArray empty_like(NDArray prototype, DType dtype, Shape shape, char order, string device)
        {
            // NumPy's empty_like converts its arguments in order (npy_parse_arguments): the order's letter before the device,
            // so an unknown order is reported first (np.zeros_like(a, order='Z', device='cuda') raises the order's ValueError,
            // probed 2.4.2). The layout itself is resolved after the device, as before.
            OrderResolver.Convert(order);
            DeviceArguments.ValidateDevice(device);
            var dims = shape.IsEmpty ? (long[])prototype.Shape.dimensions.Clone() : (long[])shape;
            // PyArray_NewLikeArrayWithShape's layout: order='K' keeps the prototype's axis permutation (C when shape=
            // changes the rank), not merely C-or-F.
            var resolvedShape = Shape.NewLikeLayout(prototype.Shape, order, dims);
            return new NDArray(dtype ?? prototype.dtype, resolvedShape, false);
        }

        /// <summary>
        ///     NumPy's <c>np.zeros_like</c> (<c>numeric.py</c>: <c>res = empty_like(a, dtype, order, subok, shape, device);
        ///     multiarray.copyto(res, 0, casting='unsafe')</c>), the body of
        ///     <see cref="np.zeros_like(NDArray, DType, char, string)"/> after its class rules: empty_like's argument checks in
        ///     its order (the order's letter, then the device), the result laid out like <paramref name="a"/>, zero-filled.
        /// </summary>
        /// <param name="a">The prototype (not null): its dimensions, its dtype unless <paramref name="dtype"/> is given, and,
        ///     for <c>'K'</c>, its axis permutation.</param>
        /// <param name="dtype">The result dtype, or null for <paramref name="a"/>'s.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c> in either case (<c>'K'</c> keeps
        ///     <paramref name="a"/>'s axis order: a transposed operand gives a transposed result, a broadcast one an F
        ///     result).</param>
        /// <param name="shape">NumPy's <c>shape=</c> override, or an empty (default) <see cref="Shape"/> for
        ///     <paramref name="a"/>'s dimensions; <c>'K'</c> lays a shape of another rank out in C order.</param>
        /// <param name="device">Null or <c>"cpu"</c>.</param>
        /// <returns>A new, owned, writeable array of zeros (a text dtype's empty strings).</returns>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K (<c>order must be one of 'C', 'F',
        ///     'A', or 'K' (got 'Z')</c>), checked before the device; <paramref name="device"/> is neither null nor
        ///     <c>"cpu"</c>; a byte count past the limit.</exception>
        /// <remarks>NumPy's <c>subok=True</c>: <c>[NewSubtype("a")]</c> creates the result as the operand's class (a matrix
        ///     view; a chararray), and a chararray operand's finalize rule runs on the result dtype here, where
        ///     <c>empty_like</c> allocates - after the order and device checks, before the fill.</remarks>
        [NewSubtype("a")]
        internal static NDArray zeros_like(NDArray a, DType dtype, char order, Shape shape, string device)
        {
            // NumPy's empty_like converts its arguments in order (npy_parse_arguments): the order's letter before the device,
            // so an unknown order is reported first (np.zeros_like(a, order='Z', device='cuda') raises the order's ValueError,
            // probed 2.4.2). The layout itself is resolved after the device, as before.
            OrderResolver.Convert(order);
            DeviceArguments.ValidateDevice(device);
            // PyArray_NewLikeArrayWithShape's layout: order='K' keeps a's axis permutation (a transposed a gives a
            // transposed result, a broadcast one an F result; C when shape= changes the rank), not merely C-or-F.
            var dims = shape.IsEmpty ? (long[])a.Shape.dimensions.Clone() : (long[])shape;
            var resolvedShape = Shape.NewLikeLayout(a.Shape, order, dims);
            // empty_like allocates the result as a chararray operand's class (subok=True) at this point: its finalize rule
            // refuses a dtype that is not string data before the fill.
            if (NDSubclass.StandsForCharArray(a))
                NDCharArray.Finalize(dtype ?? a.dtype);
            // np.zeros(shape, dtype)'s statement: a zero-filled allocation, which keeps the dense layout just resolved.
            return new NDArray(dtype ?? a.dtype, resolvedShape, true);
        }

        /// <summary>
        ///     NumPy's <c>np.ones_like</c> (<c>numeric.py</c>: <c>res = empty_like(a, dtype, order, subok, shape, device);
        ///     multiarray.copyto(res, 1, casting='unsafe')</c>), the body of
        ///     <see cref="np.ones_like(NDArray, DType, char, string)"/> after its class rules: empty_like's argument checks in
        ///     its order, the result laid out like <paramref name="a"/>, filled with the Python int 1 as the dtype converts it
        ///     (the text <c>'1'</c> for a string dtype).
        /// </summary>
        /// <param name="a">The prototype (not null): its dimensions, its dtype unless <paramref name="dtype"/> is given, and,
        ///     for <c>'K'</c>, its axis permutation.</param>
        /// <param name="dtype">The result dtype, or null for <paramref name="a"/>'s.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c> in either case.</param>
        /// <param name="shape">NumPy's <c>shape=</c> override, or an empty (default) <see cref="Shape"/> for
        ///     <paramref name="a"/>'s dimensions; <c>'K'</c> lays a shape of another rank out in C order.</param>
        /// <param name="device">Null or <c>"cpu"</c>.</param>
        /// <returns>A new, owned, writeable array of ones, labelled with a requested lane-sharing C-type class.</returns>
        /// <exception cref="ValueError"><paramref name="order"/> is not one of C, F, A, K, checked before the device;
        ///     <paramref name="device"/> is neither null nor <c>"cpu"</c>; a byte count past the limit.</exception>
        /// <remarks>NumPy's <c>subok=True</c>: <c>[NewSubtype("a")]</c> creates the result as the operand's class (a matrix
        ///     view; a chararray), and a chararray operand's finalize rule runs on the result dtype here, where
        ///     <c>empty_like</c> allocates - after the order and device checks, before the fill.</remarks>
        [NewSubtype("a")]
        internal static NDArray ones_like(NDArray a, DType dtype, char order, Shape shape, string device)
        {
            // A string array (or dtype) is NumPy's empty_like + copyto(res, 1, casting='unsafe'): the text '1' in every
            // element, through ones' string route.
            // NumPy's empty_like converts its arguments in order (npy_parse_arguments): the order's letter before the device,
            // so an unknown order is reported first (np.zeros_like(a, order='Z', device='cuda') raises the order's ValueError,
            // probed 2.4.2). The layout itself is resolved after the device, as before.
            OrderResolver.Convert(order);
            DeviceArguments.ValidateDevice(device);
            // PyArray_NewLikeArrayWithShape's layout: order='K' keeps a's axis permutation (C when shape= changes the
            // rank), not merely C-or-F.
            var dims = shape.IsEmpty ? (long[])a.Shape.dimensions.Clone() : (long[])shape;
            var resolvedShape = Shape.NewLikeLayout(a.Shape, order, dims);
            // empty_like allocates the result as a chararray operand's class (subok=True) at this point: its finalize rule
            // refuses a dtype that is not string data before the fill.
            if (NDSubclass.StandsForCharArray(a))
                NDCharArray.Finalize(dtype ?? a.dtype);
            // np.ones(shape, dtype)'s statement: the array object's ones (np.ones' device check is the null device's).
            return ones(resolvedShape, dtype ?? a.dtype);
        }

        // ---------------------------- np.isfortran ----------------------------

        /// <summary>
        ///     NumPy's <c>np.isfortran</c> (<c>numeric.py</c>: <c>return a.flags.fnc</c>): whether <paramref name="a"/> is
        ///     F-contiguous and NOT C-contiguous, a read of the cached flags.
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <returns>True for column-major memory order only: a 1-d, 0-d or empty array is both C- and F-contiguous, so it
        ///     answers False (use <c>flags.f_contiguous</c> for F-contiguity alone).</returns>
        internal static bool isfortran(NDArray a) => a.Shape.IsFContiguous && !a.Shape.IsContiguous;

        // ---------------------------- np.roll ----------------------------

        /// <summary>
        ///     NumPy's <c>np.roll</c> (<c>numeric.py</c>) after the facade's class rules, statement for statement: with no axis
        ///     the raveled array is rolled along axis 0 and reshaped back; otherwise
        ///     <code>
        ///     axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        ///     broadcasted = broadcast(shift, axis)              # ndim > 1: ValueError
        ///     shifts = {ax: 0 for ax in range(a.ndim)}
        ///     for sh, ax in broadcasted: shifts[ax] += int(sh)
        ///     rolls[ax] = ((:-offset, offset:), (-offset:, :offset)) for each offset = shifts[ax] % (a.shape[ax] or 1) != 0
        ///     result = empty_like(a); for indices in product(*rolls): result[res_index] = a[arr_index]
        ///     </code>
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <param name="shift">The places to shift: a scalar (0-d) or a 1-D array, broadcast against the axes; every element is
        ///     read as Python's <c>int()</c> reads NumPy's scalar of it (an integer exactly, a float truncated toward zero, a
        ///     bool as 0 / 1, a complex by its real part, text parsed as a Python int literal), and the shifts of one axis are
        ///     SUMMED in arbitrary precision before Python's modulo by the axis length, so no value overflows.</param>
        /// <param name="axis">The axis or the axes (repeats allowed: their shifts add up), or null to roll the flattened
        ///     array.</param>
        /// <returns>A new array of <paramref name="a"/>'s shape and dtype: <paramref name="a"/>'s memory order (empty_like's
        ///     <c>'K'</c>) along axes, also when every offset is 0 (a <c>copy('K')</c>); C order with no axis (the reshape of
        ///     the raveled roll).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="shift"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>'s rank (the axis as given).</exception>
        /// <exception cref="IncorrectShapeException">The shift does not broadcast against the axes (np.broadcast's <c>shape
        ///     mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (3,) and arg 1
        ///     with shape (2,).</c>).</exception>
        /// <exception cref="ValueError">The broadcast has two dimensions or more (<c>'shift' and 'axis' should be scalars or
        ///     1D sequences</c>); a NaN shift (<c>cannot convert float NaN to integer</c>); a text shift that is no int
        ///     literal; a 0-d <paramref name="a"/> with the empty tuple (Python's <c>not enough values to unpack (expected 2,
        ///     got 0)</c>: <c>zip(*())</c> of the one empty product).</exception>
        /// <exception cref="OverflowError">An infinite shift (<c>cannot convert float infinity to integer</c>).</exception>
        /// <exception cref="TypeError">A shift element <c>int()</c> refuses (a StringDType NA None).</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the ravel and the inner roll of the flat case, and every slice view of the
        ///     copies, are released; the result is yielded. It sees plain arrays only: the woven public function hands it the
        ///     plain view of a subclass operand and answers for the class itself (NDSubclass). NumPy's ComplexWarning for a
        ///     complex shift is not raised (NumSharp has no warnings).</remarks>
        [NDScoped /* the ravel and the inner roll of the flat case, and the slice views of the copies, are released; the result is yielded */]
        [NewSubtype("a")]
        internal static NDArray roll(NDArray a, NDArray shift, NDAxis? axis)
        {
            if (shift is null)
                throw new ArgumentNullException(nameof(shift));
            // `if axis is None: return roll(a.ravel(), shift, 0).reshape(a.shape)`.
            if (axis is null)
                return roll(a.ravel(), shift, 0).reshape(new Shape(a.Shape.dimensions, copy: false));

            // `axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)`: NumPy's AxisError, the axis as given; an
            // integer is a tuple of one.
            int nd = a.ndim;
            int[] axes = axis.Value.Normalize(nd, allowDuplicate: true);

            // `broadcasted = broadcast(shift, axis)`: the shift's shape against the axes' (k,), np.broadcast's mismatch text;
            // then the dimension check.
            Shape spread = Shape.MultiIterShape(new[] { shift.Shape, new Shape(new long[] { axes.Length }, copy: false) });
            if (spread.NDim > 1)
                throw new ValueError("'shift' and 'axis' should be scalars or 1D sequences");

            // `for sh, ax in broadcasted: shifts[ax] += int(sh)` - in C order over the 1-D broadcast; a shift of one element
            // (0-d, or length 1) repeats, an axis tuple of one names the same axis every time. The sums are Python ints.
            var sums = new BigInteger[nd];
            long count = spread.size;
            bool shiftRepeats = shift.size == 1;
            bool unicode = shift.typecode == NPTypeCode.String && !shift.Storage.IsVString && shift.dtype.kind == 'U';
            for (long i = 0; i < count; i++)
            {
                int ax = axes.Length == 1 ? axes[0] : axes[i];
                sums[ax] += PythonIntOfShift(shift.GetAtIndex(shiftRepeats ? 0 : i), unicode);
            }

            // Python's `offset %= a.shape[ax] or 1`: never negative, and 0 for an empty axis ("nothing matters").
            var offsets = new long[nd];
            bool any = false;
            for (int d = 0; d < nd; d++)
            {
                long n = a.Shape.dimensions[d] == 0 ? 1 : a.Shape.dimensions[d];
                BigInteger r = BigInteger.Remainder(sums[d], n);
                offsets[d] = (long)(r.Sign < 0 ? r + n : r);
                any |= offsets[d] != 0;
            }

            // `zip(*indices)` of the ONE empty product a 0-d input gives: Python cannot unpack it into two names.
            if (nd == 0)
                throw new ValueError("not enough values to unpack (expected 2, got 0)");

            // NumPy still runs empty_like + the slice copies when every offset is 0, so the result keeps a's memory order
            // (order='K': a transposed / broadcast input stays transposed / F), which copy('K') reproduces.
            if (!any)
                return a.copy('K');

            // np.empty_like(a)'s statements: an uninitialised array of a's dtype in a's K layout (every element is written by
            // the copies below: the 2^m slice pairs of the m rolled axes cover the result exactly once).
            var result = new NDArray(a.dtype, Shape.NewLikeLayout(a.Shape, 'K', (long[])a.Shape.dimensions.Clone()), false);

            // `for indices in itertools.product(*rolls)`: an odometer over the rolled axes, each either its body
            // (result[offset:] = a[:-offset]) or its tail (result[:offset] = a[-offset:]); an axis without an offset is ':'
            // on both sides. The product's order (the first axis slowest) is NumPy's.
            var rolled = new int[nd];
            int m = 0;
            for (int d = 0; d < nd; d++)
                if (offsets[d] != 0)
                    rolled[m++] = d;
            var src = new Slice[nd];
            var dst = new Slice[nd];
            for (long pick = 0; pick < 1L << m; pick++)
            {
                for (int d = 0; d < nd; d++)
                    src[d] = dst[d] = Slice.All;
                for (int j = 0; j < m; j++)
                {
                    int d = rolled[j];
                    long offset = offsets[d];
                    bool tail = ((pick >> (m - 1 - j)) & 1) != 0;
                    src[d] = tail ? new Slice(-offset, null) : new Slice(null, -offset);
                    dst[d] = tail ? new Slice(null, offset) : new Slice(offset, null);
                }
                result[dst] = a[src];
            }

            return result;
        }

        /// <summary>
        ///     Python's <c>int(sh)</c> of one element of the broadcast shift, as <c>np.broadcast</c>'s iteration hands it to
        ///     np.roll: NumPy's scalar of the element (an <c>np.str_</c> for a <c>U</c> array, a Python str for StringDType).
        /// </summary>
        /// <param name="element">The element, boxed by <see cref="NDArray.GetAtIndex(long)"/>.</param>
        /// <param name="unicode">The shift is a <c>U</c> array: an invalid literal is quoted as <c>np.str_('…')</c>.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="ValueError">A NaN, or text that is no int literal.</exception>
        /// <exception cref="OverflowError">An infinity.</exception>
        /// <exception cref="TypeError">A value <c>int()</c> refuses (None).</exception>
        private static BigInteger PythonIntOfShift(object element, bool unicode)
        {
            switch (element)
            {
                case Complex c:
                    // NumPy's complex scalar __int__: a ComplexWarning, then int() of the real part.
                    return PythonInt.From(c.Real);
                case string s when unicode:
                    return PythonInt.Parse(s, 10, numpyStr: true);
                default:
                    return PythonInt.From(element);
            }
        }

        /// <summary>
        ///     NumPy's <c>np.indices</c> (<c>numeric.py</c>) with <c>sparse=False</c>, the body of
        ///     <see cref="np.indices(Shape, DType, bool)"/> once the dimensions were read: a single dense array of
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
        internal static unsafe NDArray indices(long[] dimensions, DType dtype)
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
            var res = new NDArray(descr, new Shape(resultShape, copy: false), false);
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
                        using var line = idx.reshape(new Shape(axisShape, copy: false));
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
        ///     <see cref="np.indices(Shape, DType, bool)"/> (<c>sparse: true</c>) once the dimensions were checked for null: one array per
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
        internal static NDArray[] indices_sparse(long[] dimensions, DType dtype)
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

        // ---------------------------- np.isscalar ----------------------------

        /// <summary>
        ///     NumPy's <c>np.isscalar</c> (<c>numeric.py</c>: <c>isinstance(element, generic) or type(element) in ScalarType
        ///     or isinstance(element, numbers.Number)</c>) over C# values: a NumPy scalar (a 0-d NDArray that
        ///     <see cref="NDArray.IsNumPyScalar"/> calls one), a <see cref="System.Numerics.Complex"/>, a string, a bool, a
        ///     primitive, a decimal, and the C# spellings of the other NumPy and Python scalars — a <see cref="Half"/>
        ///     (<c>np.float16</c>), a <see cref="BigInteger"/> (a Python int of any size), a <see cref="Char8"/>
        ///     (<c>np.bytes_</c> / <c>bytes</c> of length 1) and a <see cref="DateTime64"/> (<c>np.datetime64</c>).
        /// </summary>
        /// <param name="obj">Any object, null included.</param>
        /// <returns>True for the scalars above; false for null, an explicit 0-d array (<c>np.array(5)</c>,
        ///     <c>np.asarray(5)</c>), any NDArray with dimensions, a C# array, a <see cref="Type"/> and any other
        ///     object.</returns>
        /// <remarks>
        ///     <para>
        ///     A <see cref="Type"/> is a class object, never a scalar: <c>isscalar(typeof(int))</c> is False, as NumPy's
        ///     <c>isscalar(int)</c> and <c>isscalar(np.float64)</c> are (a class is neither a <c>generic</c> nor a
        ///     <c>numbers.Number</c>).
        ///     </para>
        ///     <para>
        ///     NumSharp's NumPy scalar is a 0-d array, so the answer for an NDArray is <see cref="NDArray.IsNumPyScalar"/>:
        ///     <c>np.isscalar(a.sum())</c> and <c>np.isscalar(NDArray.Scalar(1.5))</c> are True and
        ///     <c>np.isscalar(np.array(5))</c> is False, as in NumPy. The accepted difference: a 0-d array built another way
        ///     (<c>np.zeros(())</c>, <c>x.reshape(())</c>) answers True, where NumPy's ndarray answers False.
        ///     </para>
        /// </remarks>
        internal static bool isscalar(object obj)
        {
            if (obj is null)
                return false;

            // The common answers first, each a method-table compare: type.IsPrimitive checks for Boolean, Byte, SByte, Int16,
            // UInt16, Int32, UInt32, Int64, UInt64, IntPtr, UIntPtr, Char, Double and Single; then the other scalar types,
            // none of which can have a subclass, so exact type tests decide them. NumPy's float16 scalar (a generic), a
            // Python int of any size (a numbers.Number), np.bytes_ / bytes and np.datetime64 (generics) are among them: no
            // CLR primitive, they answered False before the np-layer plan's NL11. Only an NDArray needs the hierarchy walk
            // of a class test (its subclasses), so it comes last.
            Type type = obj.GetType();
            if (type.IsPrimitive)
                return true;
            if (type == typeof(string) || type == typeof(Complex) || type == typeof(decimal) || type == typeof(Half)
                || type == typeof(BigInteger) || type == typeof(Char8) || type == typeof(DateTime64))
                return true;
            // A 0-d NDArray is NumPy's scalar unless np.array(x) / np.asarray(x) built it as an explicit 0-d ARRAY, which
            // NumPy's isinstance(element, generic) rejects (np.isscalar(np.array(5)) is False). IsNumPyScalar is that rule.
            if (obj is NDArray nd)
                return nd.IsNumPyScalar;

            // Everything else: a C# array (a Python list), any other object, and a Type - a class object, which NumPy's
            // isscalar(int) answers False for. A Type used to be read as the type it names, so typeof(int) answered True
            // (NL11).
            return false;
        }
    }
}
