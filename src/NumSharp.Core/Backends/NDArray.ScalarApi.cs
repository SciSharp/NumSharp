using System;
using NumSharp.Backends;

namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // NumPy's scalar C-API (numpy/_core/src/multiarray/scalarapi.c) over NumSharp's scalar model.
    //
    // NumSharp has no scalar classes: a NumPy scalar (np.float64(1.5), a.sum(), np.dot(v, v)) is a 0-d NDArray, and a 0-d
    // NDArray IS a scalar unless the caller built it as an array with np.array(x) / np.asarray(x) (IsExplicitZeroDimArray).
    // There is no scalar flag and no extra state: IsNumPyScalar computes the answer, and a scalar is immutable through
    // NumPy's own WRITEABLE flag, cleared where NumPy hands a scalar out.
    //
    // Every scalar passes one of NumPy's own seams, so a later phase that gives the scalars classes of their own
    // (NDFloat64 : NDFloating : ... : NDGeneric : NDArray) changes only what the seams build:
    //   - PyArray_Return: a fresh 0-d result becomes the scalar (the reductions, the products, NDArray.Scalar(...));
    //   - IsNumPyScalar: every rule that tells a scalar from a 0-d array reads only this (np.isscalar, the scalar-class
    //     isinstance, numpy.polynomial's operand protocol, numpy.matrix's scaling rule).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class NDArray
    {
        /// <summary>
        ///     True when this array stands for a NumPy SCALAR (<c>numpy.generic</c>): a 0-d array that was not built as an
        ///     explicit 0-d ARRAY by <c>np.array(x)</c> / <c>np.asarray(x)</c> (<see cref="IsExplicitZeroDimArray"/>). It is
        ///     computed, not stored: a 0-d array is a scalar by default.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Read this, never <c>ndim == 0</c> or the explicit mark directly, wherever NumPy's answer depends on scalar
        ///     against 0-d ndarray (<see cref="np.isscalar"/>, <see cref="NDScalarType.__instancecheck__"/>, numpy.matrix's
        ///     <c>m * x</c>, numpy.polynomial's operand protocol): it is the one place the later scalar-subclass phase turns
        ///     into a class test (<c>this is NDGeneric</c>).
        ///     </para>
        ///     <para>
        ///     <b>Accepted difference from NumPy:</b> a 0-d array built any other way than the two explicit spellings
        ///     (<c>np.zeros(())</c>, <c>x.reshape(())</c>, <c>np.squeeze(x)</c>, a copy of an explicit 0-d array) answers
        ///     true, where NumPy's object is an ndarray. Only <c>np.array</c> / <c>np.asarray</c> of a scalar argument mark
        ///     their result (see <see cref="IsExplicitZeroDimArray"/> for why the kernels' own 0-d allocations cannot).
        ///     </para>
        ///     <para>
        ///     A scalar is not necessarily read-only: NumPy's return points (<see cref="PyArray_Return"/>) clear WRITEABLE,
        ///     while a 0-d array allocated another way (<c>np.zeros(())</c>) keeps it, so <c>fill</c> still works there.
        ///     </para>
        /// </remarks>
        internal bool IsNumPyScalar => Shape.IsScalar && !IsExplicitZeroDimArray;

        /// <summary>
        ///     NumPy's <c>PyArray_Return</c> (C-API slot 76): the step that turns a FRESH result into what NumPy hands the
        ///     caller - the array itself when it has dimensions, NumPy's scalar when it is 0-d. NumSharp's scalar is the 0-d
        ///     array itself, made what NumPy's scalar is: read-only (WRITEABLE cleared) and never an explicit 0-d array.
        /// </summary>
        /// <returns>
        ///     This array. The scalar-subclass phase may return a NEW object here (the scalar of the array's class), so a
        ///     caller should return or store the RESULT of this call, not the array it called it on.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     <b>Where NumPy calls it, NumSharp calls this:</b> every exit of a reduction that hands back a 0-d result
        ///     (<c>axis=None</c>, an axis that reduces the last dimension away, <c>keepdims</c> over a 0-d input; probed 2.4.2:
        ///     <c>np.sum(a).flags.writeable is False</c>, and even <c>np.sum(np.array(5), keepdims=True)</c> comes back a
        ///     read-only scalar), the product gufuncs' 0-d answer without <c>out=</c> (<c>GufuncGuard.Deliver</c>: NumPy's
        ///     <c>replace_with_wrapped_result_and_return</c> → <c>npy_apply_wrap</c> with <c>return_scalar</c>),
        ///     <c>np.dot</c> / <c>linalg.det</c> / the index routines' 0-d answers, and the scalar constructors
        ///     (<see cref="Scalar{T}(T)"/> and its overloads: NumPy's scalar <c>tp_new</c> ends in <c>PyArray_ToScalar</c>,
        ///     the same conversion).
        ///     </para>
        ///     <para>
        ///     <b>Footgun - the array must be FRESH.</b> NumPy's function copies the element into a new scalar object and
        ///     releases the array; this one changes the array in place, so it must be an array nobody else holds: never an
        ///     <c>out=</c> operand (NumPy returns the out array itself, writeable; probed:
        ///     <c>np.sum(a, out=np.array(0)).flags.writeable is True</c>), never a caller's operand returned unchanged, never a
        ///     view of another array's memory (a scalar owns its value: a view would change when its base changes). A caller
        ///     with such an array copies it first (<c>r.copy().PyArray_Return()</c>).
        ///     </para>
        ///     <para>
        ///     A no-op for an array with dimensions, so every exit of a function can route through here: the
        ///     <c>keepdims</c> path reshapes to <c>(1,)*ndim</c> first and passes untouched.
        ///     </para>
        ///     <para>
        ///     The typed wrappers (<c>NDArray&lt;bool&gt;</c> of <c>np.all</c> / <c>np.any</c>, the <c>NDArray&lt;long&gt;</c>
        ///     index routines, <see cref="NDCharArray"/>'s scalar stand-ins) call it as a statement and keep their own
        ///     reference, which is exact while this returns the same object; the scalar-subclass phase revisits them.
        ///     </para>
        /// </remarks>
        internal NDArray PyArray_Return()
        {
            if (Shape.IsScalar)
            {
                // A scalar is immutable: NumPy's scalar object has no writable buffer (flags.writeable is False).
                if (Shape.IsWriteable)
                    Storage.SetShapeUnsafe(Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
                // A fresh result never carries the explicit-array mark; clearing it keeps the contract "what comes out of
                // here is a scalar" (IsNumPyScalar) even for a caller that broke the fresh-array rule above.
                IsExplicitZeroDimArray = false;
            }
            return this;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_Scalar</c> (C-API slot 60) of one element: NumPy's scalar holding a COPY of the element a
        ///     0-d view storage points at - read-only, owning its value, never a view that changes with the array it came
        ///     from.
        /// </summary>
        /// <param name="item">A 0-d storage whose one element is the item (a basic-index view of an array; it holds no
        ///     counted reference, so the caller only keeps the ARRAY it views alive until this returns).</param>
        /// <returns>A new read-only 0-d array (<see cref="IsNumPyScalar"/>): for a numeric dtype the element's bytes under
        ///     the descriptor of the element's CLASS (<see cref="ScalarDescr"/>); for bool NumPy's canonical True / False
        ///     (<c>PyArrayScalar_RETURN_BOOL_FROM_LONG</c>: any nonzero byte is True, a byte-reinterpreted view included); for
        ///     fixed-width text the content without its trailing NULs, as wide as the content (NumPy trims the scalar's
        ///     itemsize: <c>np.array(['a'], 'U5')[0]</c> is a <c>&lt;U1</c> scalar; NumSharp's 0-d text is at least one
        ///     character wide); for <c>StringDType</c> the element under the same descriptor.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="item"/> is null.</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy calls it from <c>PyArray_Return</c> / <c>PyArray_ToScalar</c> and from element access
        ///     (<c>array_item</c>: <c>a[i]</c>, <c>a[i, j]</c>, <c>z[()]</c>). Its callers here are element accesses that hold
        ///     a 0-d view: <see cref="ItemScalar"/>'s text route and <see cref="NDMatrix"/>'s <c>m[i, j]</c>; the numeric item
        ///     paths (<see cref="ItemScalar"/>, <see cref="NDArray{TDType}"/>'s typed twin) copy from the element's offset
        ///     through the same <see cref="CopyItem"/> without building the view. The later scalar-subclass phase builds the
        ///     scalar of the element's class here and in those paths.
        ///     </para>
        ///     <para>
        ///     <b>Footgun - lifetime.</b> The element's raw bytes are read from the array's buffer: the caller must keep the
        ///     array the view belongs to rooted until this returns (<see cref="NDAnchoredAttribute"/> on the calling
        ///     instance method), or a temporary's buffer may be pooled under the copy.
        ///     </para>
        /// </remarks>
        internal static unsafe NDArray PyArray_Scalar(UnmanagedStorage item)
        {
            if (item is null)
                throw new ArgumentNullException(nameof(item));

            // Text: the element read back (S / U without its trailing NULs, NumPy's itemsize trim) and its width discovered
            // again; StringDType's element (NumPy's NPY_USE_GETITEM: the Python str or the NA object) keeps its descriptor.
            if (item.TypeCode == NPTypeCode.String)
            {
                DType descr = item.Descr;
                object element = item.GetAtIndex(0);
                var text = descr.Meta is VStringDTypeMeta
                    ? Backends.UnmanagedStorage.FromAnyScalar(element, descr)
                    : Backends.UnmanagedStorage.FromAnyScalar(element, null);
                return text.PyArray_Return();
            }

            var scalar = new NDArray(ScalarDescr(item.Descr), Shape.Scalar, false);
            CopyItem(item, item.Shape.offset, (byte*)scalar.Address);
            return scalar.PyArray_Return();
        }

        /// <summary>
        ///     The byte half of <see cref="PyArray_Scalar"/> for a numeric (non-text) element: copies one element of a
        ///     storage into a scalar's own buffer, canonicalizing a bool.
        /// </summary>
        /// <param name="storage">A numeric storage holding the element (an array's own storage, or a 0-d view of it; the
        ///     caller keeps the array rooted until this returns).</param>
        /// <param name="offset">The element's offset in ELEMENTS from the storage's buffer start (a 0-d view's
        ///     <c>Shape.offset</c>, or <see cref="ItemOffset"/> of an index).</param>
        /// <param name="destination">The new scalar's buffer: at least the element's itemsize bytes, of the SAME type code
        ///     (the caller allocated it under <see cref="ScalarDescr"/> of <paramref name="storage"/>'s descriptor).</param>
        /// <remarks>
        ///     NumSharp's storage is always native byte order, so the bytes copy as they are (NumPy's <c>PyArray_Scalar</c>
        ///     byte-swaps a non-native element here). Shared by the untyped item path and <see cref="NDArray{TDType}"/>'s
        ///     typed one, so the two cannot drift: a typed array's item is a typed object, built by its own class around the
        ///     same copy.
        /// </remarks>
        internal static unsafe void CopyItem(UnmanagedStorage storage, long offset, byte* destination)
        {
            int itemsize = storage.Descr.itemsize;
            byte* src = storage.Address + offset * itemsize;
            if (storage.TypeCode == NPTypeCode.Boolean)
                // PyArrayScalar_RETURN_BOOL_FROM_LONG: np.True_ / np.False_, whatever nonzero byte the element holds.
                *destination = (byte)(*src != 0 ? 1 : 0);
            else
                Buffer.MemoryCopy(src, destination, itemsize, itemsize);
        }

        /// <summary>
        ///     The element offset a full integer index names, with NumPy's bounds check: the item path's replacement for
        ///     a storage view it would only read one element through.
        /// </summary>
        /// <param name="slices">One integer <see cref="Slice"/> per dimension (none for a 0-d array): the caller tested
        ///     <see cref="IsItemIndex"/>, so the count matches and every entry is an index.</param>
        /// <returns>The element's offset in ELEMENTS from this array's buffer start (<c>Shape.offset</c> plus each index,
        ///     a negative one counted from its axis' end, times its axis' stride; a broadcast axis' stride 0 included).</returns>
        /// <exception cref="IndexError">An index outside <c>[-size, size)</c> of its axis: NumPy's
        ///     <c>index 5 is out of bounds for axis 0 with size 3</c>, the index as given and the axis in this array -
        ///     <see cref="UnmanagedStorage.GetView(Slice[])"/>'s text for the same index, checked in the same axis order.</exception>
        /// <remarks>
        ///     The storage view this replaces cost about 190 ns per element read (a view storage and its sliced shape, for
        ///     one element); the arithmetic is a few nanoseconds. Measured on a float64 (3, 4) array, Release, pinned to two
        ///     CPUs, best of 7 x 1M: <see cref="ItemScalar"/> plus its Dispose 628 ns through the view, 476 ns from the
        ///     offset. What is left is the 0-d array's own allocation (about 420 ns with its Dispose, every NumSharp array's
        ///     floor); the old 0-d VIEW of the element cost 262 ns.
        /// </remarks>
        private protected long ItemOffset(Slice[] slices)
        {
            long[] dims = Shape.dimensions;
            long[] strides = Shape.strides;
            long offset = Shape.offset;
            for (int axis = 0; axis < slices.Length; axis++)
            {
                long index = slices[axis].Start ?? 0;
                long dim = dims[axis];
                if (index < -dim || index >= dim)
                    throw new IndexError(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                        $"index {index} is out of bounds for axis {axis} with size {dim}"));
                if (index < 0)
                    index += dim;
                offset += index * strides[axis];
            }
            return offset;
        }

        /// <summary>
        ///     The descriptor a NumPy scalar of a numeric element carries: the canonical instance of the element descriptor's
        ///     CLASS - the scalar's type - never the array's own descriptor instance.
        /// </summary>
        /// <param name="descr">The array's (numeric) element descriptor.</param>
        /// <returns>The class singleton: the same C-type class (an <c>'i'</c> array's item is an <c>np.intc</c>, char
        ///     <c>'i'</c>, beside the <c>'l'</c> class of the same width on Windows), without the instance's metadata.</returns>
        /// <remarks>
        ///     NumPy's scalar of a builtin type stores only its value; its <c>dtype</c> is
        ///     <c>PyArray_DescrFromScalar</c> → <c>PyArray_DescrFromTypeObject(type(scalar))</c>, the builtin descriptor of the
        ///     scalar's class (probed 2.4.2: an array of <c>np.dtype('f8', metadata={'k': 1})</c> keeps the metadata while
        ///     its item's <c>dtype.metadata</c> is <c>None</c>; <c>np.array([1], 'i')[0].dtype.char</c> is <c>'i'</c>). Footgun:
        ///     <see cref="DType.From(NPTypeCode)"/> is NOT this - it picks the type code's default class and turns an
        ///     <c>np.intc</c> item into <c>'l'</c>.
        /// </remarks>
        internal static DType ScalarDescr(DType descr) => descr.Meta.singleton;

        /// <summary>
        ///     NumPy's element access (<c>array_item</c>): the item a full integer index selects (an integer for every
        ///     dimension; <c>()</c> of a 0-d array) as NumPy's scalar, through <see cref="PyArray_Scalar"/>.
        /// </summary>
        /// <param name="slices">The index: one integer <see cref="Slice"/> per dimension (none for a 0-d array).</param>
        /// <returns>A new read-only 0-d array holding a copy of the element.</returns>
        /// <exception cref="IndexError">An integer is out of bounds for its axis (NumPy's text, <see cref="ItemOffset"/>).</exception>
        /// <remarks>
        ///     <para>
        ///     A numeric element is copied straight from its offset (<see cref="ItemOffset"/>): no view storage is built
        ///     for the one element read. Text keeps the view route: <see cref="PyArray_Scalar"/> reads the element back
        ///     through it, trimmed, and discovers the scalar's width again.
        ///     </para>
        ///     <para>
        ///     Anchored: the element's bytes are read after this array's last use in the body, so the weaver keeps
        ///     <c>this</c> rooted until the copy is made (a temporary indexed in place, <c>np.arange(5)[2]</c>, cannot be
        ///     collected under it).
        ///     </para>
        /// </remarks>
        [NDAnchored]
        internal unsafe NDArray ItemScalar(Slice[] slices)
        {
            if (Storage.TypeCode == NPTypeCode.String)
                return PyArray_Scalar(Storage.GetView(slices));
            // The offset (and NumPy's bounds error) before the allocation: a bad index allocates nothing.
            long offset = ItemOffset(slices);
            var scalar = new NDArray(ScalarDescr(Storage.Descr), Shape.Scalar, false);
            CopyItem(Storage, offset, (byte*)scalar.Address);
            return scalar.PyArray_Return();
        }

        /// <summary>
        ///     NumPy's <c>gentype_subscript</c>: ANY index of a NumPy SCALAR indexes the 0-d ARRAY of the scalar's value
        ///     (<c>PyArray_FromScalar</c>), so a view (<c>s[...]</c>, <c>s[None]</c>) is a writeable ndarray over a copy,
        ///     never a view of the immutable scalar, and a key that 0-d array rejects raises NumPy's scalar text.
        /// </summary>
        /// <param name="key">The index tuple (any key: basic or advanced; not <c>null</c>).</param>
        /// <returns>What the 0-d array of the value gives for <paramref name="key"/> (probed 2.4.2:
        ///     <c>np.float64(5)[...]</c> is <c>array(5.)</c>, writeable, <c>owndata</c> False; <c>[None]</c> and
        ///     <c>[np.array(True)]</c> are <c>array([5.])</c>; <c>[()]</c> is the scalar <c>np.float64(5.0)</c>).</returns>
        /// <exception cref="IndexError">The 0-d array rejects the key (<c>s[0]</c>, <c>s[:]</c>, <c>s[[0]]</c>,
        ///     <c>s[1.5]</c>, a slice-notation text that does not parse): NumPy's <c>invalid index to scalar variable.</c>,
        ///     whatever the 0-d array's own error was - <c>gen_arrtype_subscript</c> replaces it with
        ///     <c>PyErr_SetString(PyExc_IndexError, ...)</c>.</exception>
        /// <exception cref="ArgumentNullException">An entry of <paramref name="key"/> is <c>null</c> (a C# misuse with no
        ///     Python equivalent: NumPy's <c>None</c> is <see cref="Slice.NewAxis"/>), kept as the indexer raises it.</exception>
        /// <remarks>
        ///     <para>
        ///     Scoped: the 0-d array of the value (<see cref="asarray(NDArray, DType, char, bool?)"/> copies a read-only
        ///     scalar into a fresh explicit 0-d array) is a temporary this call disposes; the returned view keeps that
        ///     array's buffer alive with its own counted reference. A rare path (indexing a scalar), so the scope's cost
        ///     stays off the hot indexer. Through an <see cref="NDArray{TDType}"/> the result is typed: the temporary is a
        ///     plain array, and its typed twin is built by the typed getter.
        ///     </para>
        ///     <para>
        ///     The rejected key's own error rides along as <see cref="Exception.InnerException"/>: NumPy discards it (the
        ///     IndexError's <c>__context__</c> is None), but a debugger still shows why the key was refused. The type and
        ///     the text, NumPy's observable contract, are NumPy's.
        ///     </para>
        /// </remarks>
        [NDScoped]
        internal NDArray ScalarSubscript(object[] key)
        {
            NDArray value = asarray(this, null, 'K', null);
            try
            {
                return value[key];
            }
            // NumPy replaces ANY failure of the 0-d array's subscript; only what has no Python counterpart passes: a null
            // entry (C# misuse) and running out of memory (no index error at all).
            catch (Exception e) when (e is not ArgumentNullException && e is not OutOfMemoryException)
            {
                throw new IndexError("invalid index to scalar variable.", e);
            }
        }

        /// <summary>
        ///     Whether a basic, non-item index of this array takes NumPy's scalar route (<see cref="ScalarSubscript"/>: a view
        ///     of a writeable COPY of the value) instead of viewing this array's own buffer.
        /// </summary>
        /// <remarks>
        ///     True only for a scalar NumPy would have handed out as a scalar object: a 0-d array that is no explicit array
        ///     (<see cref="IsNumPyScalar"/>), read-only (cleared at <see cref="PyArray_Return"/>: the scalar results, items and
        ///     constructors), and not a C# literal's conversion (<see cref="WasScalarArgument"/>: NumPy's Python scalar, an
        ///     operand, never an indexed object). A 0-d array NumSharp allocated another way (<c>np.zeros(())</c>, writeable)
        ///     keeps being indexed as the array it is, so a write through its <c>[...]</c> view still reaches it.
        /// </remarks>
        internal bool IsScalarSubscript => IsNumPyScalar && !Shape.IsWriteable && !WasScalarArgument;

        /// <summary>
        ///     NumPy's <c>out=...</c> for a FRESH ufunc result: the result stays an ARRAY - writeable, and an explicit 0-d
        ///     array when it is 0-d - where the ufunc returns NumPy's scalar without it
        ///     (<c>np.subtract(np.array(2.0), 1.0, out=...)</c> is <c>array(1.)</c>, not <c>np.float64(1.0)</c>).
        /// </summary>
        /// <returns>This array, writeable.</returns>
        /// <remarks>
        ///     <para>
        ///     The library's NumPy ports call it exactly where NumPy passes <c>out=...</c> and then writes into the result
        ///     (<c>_methods._var</c>: <c>x = um.subtract(arr, arrmean, out=...)</c>, then <c>um.square(x, out=x)</c>), so a
        ///     0-d input does not meet a read-only scalar.
        ///     </para>
        ///     <para>
        ///     <b>Footgun - the array must be the caller's own fresh ufunc result.</b> It undoes <see cref="PyArray_Return"/>
        ///     in place: called on a scalar somebody else holds (<c>a.sum()</c> handed in by a caller, an
        ///     <see cref="Scalar{T}(T)"/>), it would make that scalar writeable.
        ///     </para>
        /// </remarks>
        internal NDArray AsArrayResult()
        {
            if (Shape.IsScalar)
            {
                if (!Shape.IsWriteable)
                    Storage.SetShapeUnsafe(Shape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE));
                MarkExplicitZeroDimArray();
            }
            return this;
        }
    }
}
