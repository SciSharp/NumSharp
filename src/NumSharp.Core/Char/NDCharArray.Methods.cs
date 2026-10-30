using System;
using NumSharp.Backends;

// =============================================================================
// NDCharArray.Methods.cs — the ndarray methods a chararray inherits, returned as chararrays
// =============================================================================
//
// numpy.char.chararray overrides none of the view / copy methods: ndarray creates their results with the SUBCLASS and
// calls __array_finalize__, so each one is a chararray — or raises the finalize ValueError when its dtype is not string
// data (astype(int), getfield(np.uint32)). C# binds these members by static type, so each is re-declared here with
// `new` and returns NDCharArray. They take the same arguments as the NDArray members, minus out=: a call that passes an
// out binds the NDArray member and returns what that returns.
//
// OWNERSHIP follows NumPy: a method that allocates (copy, flatten, astype, take, repeat, compress, byteswap) returns a
// chararray that owns its data; a view method (T, transpose, reshape, ravel of a contiguous array, diagonal, view,
// getfield) a chararray that does not; a method NumPy answers with `self` (real, astype(copy=False) of the same dtype,
// squeeze without a length-1 axis, to_device) returns this very object.
//
// THE ONE ORDER QUIRK. view(dtype) checks the finalize rule against the dtype the array HAS (PyArray_View builds the
// subclass object first and sets the new dtype afterwards), so c.view(np.uint32) is a uint32 chararray; astype and
// getfield build the subclass object with the NEW dtype, so they raise.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        #region Conversion and copies

        /// <summary>A copy of the chararray cast to <paramref name="dtype"/>, as a chararray (NumPy's <c>c.astype(dtype)</c>).</summary>
        /// <param name="dtype">The target dtype: a string dtype (an unsized <c>"U"</c> / <c>"S"</c> keeps the element width), or
        ///     <c>int8</c> (the one numeric dtype the finalize rule accepts).</param>
        /// <param name="copy"><c>false</c> returns this chararray ITSELF when it already has the dtype.</param>
        /// <returns>A new chararray that owns its data, or this chararray.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">The dtype is not string data (<c>Can only create a chararray from string data.</c>),
        ///     raised BEFORE any element is converted, as NumPy allocates the subclass result before its cast loop.</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U</c> → <c>S</c> cast of a non-ASCII character.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> → <c>U</c> cast of a byte above 0x7F.</exception>
        [NDScoped]
        public new NDCharArray astype(DType dtype, bool copy = true) => astype(dtype, copy, 'K');

        /// <summary>A copy of the chararray cast to <paramref name="dtype"/> with a memory order and a casting rule, as a
        ///     chararray.</summary>
        /// <param name="dtype">The target dtype (see <see cref="astype(DType, bool)"/>).</param>
        /// <param name="copy"><c>false</c> returns this chararray itself when nothing has to change.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <param name="casting">NumPy's casting rule name (<c>"unsafe"</c>, <c>"same_kind"</c>, <c>"safe"</c>,
        ///     <c>"equiv"</c>, <c>"no"</c>).</param>
        /// <returns>A new chararray that owns its data, or this chararray.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is <c>null</c>.</exception>
        /// <exception cref="InvalidCastException">The cast is not allowed under <paramref name="casting"/> (checked first,
        ///     as NumPy does).</exception>
        /// <exception cref="ValueError">The dtype is not string data (the finalize rule, after the casting rule and before
        ///     any element is converted), or an unknown order / casting name.</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U</c> → <c>S</c> cast of a non-ASCII character.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> → <c>U</c> cast of a byte above 0x7F.</exception>
        [NDScoped]
        public new NDCharArray astype(DType dtype, bool copy, char order, string casting = "unsafe")
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));

            // A string array or a string target: NumPy's array_astype over descriptors, with the subclass's finalize rule
            // between its casting rule and the allocation of the result (the hook).
            if (typecode == NPTypeCode.String || dtype.Meta is StringDTypeMeta || dtype.Meta is VStringDTypeMeta)
                return From(Backends.Strings.NDStringCasting.AsType(this, dtype, copy, order, casting, Finalize), this);

            // A chararray over int8 / char data (or one a view(dtype) left numeric) cast to a number: ndarray's astype.
            // NumPy parses the order, then the casting name, before anything else (an unknown name raises even when nothing
            // has to change); it returns the array itself for copy=False of the same dtype whose layout already satisfies the
            // order, before it allocates anything (so even a uint32 chararray answers itself there); otherwise its casting
            // rule, then the allocation (the finalize rule), then the cast loop. The order test is the one the string path
            // uses.
            OrderResolver.Resolve(order, Shape);
            var rule = DTypeCasting.ParseCastingSameValue(casting, out _);
            if (!copy && dtype.Equals(this.dtype) && Backends.Strings.NDStringCasting.OrderSatisfied(order, Shape))
                return this;
            ValidateAstypeCasting(dtype.GetTypeCode(), rule);
            Finalize(dtype);
            return From(base.astype(dtype, copy, order, casting), this);
        }

        /// <summary>A copy of the chararray, as a chararray (NumPy's <c>c.copy(order)</c>).</summary>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <returns>A new chararray that owns its data.</returns>
        /// <exception cref="ArgumentException">An unknown order.</exception>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        /// <remarks>The result is allocated as a chararray and filled in place — <c>ndarray.copy</c>'s layout
        ///     (<c>NDIter.CopyAsShape</c>) and fill (<c>NDIter.Copy</c>) without the plain intermediate array; NumPy's order:
        ///     the order is checked before the copy exists, the finalize rule when it is created.</remarks>
        [NDAnchored] // the block move reads this array's storage after its last use: rooted until the copy returns
        public new NDCharArray copy(char order = 'C')
        {
            var shape = Backends.Iteration.NDIter.CopyAsShape(Shape, order);
            Finalize(dtype);
            // The result buffer, filled completely below (no zero fill, not even a U array's NPY_NEEDS_INIT one): a
            // fixed-width string buffer straight from AllocateString (this array's descriptor is sized and native), else
            // (int8 / char data) the general allocation of the descriptor.
            UnmanagedStorage storage;
            if (Storage.IsFixedString)
                storage = UnmanagedStorage.AllocateString(shape, dtype, fillZeros: false);
            else
            {
                storage = new UnmanagedStorage(dtype);
                storage.Allocate(shape, dtype, fillZeros: false, overwriteAll: true);
            }

            var result = new NDCharArray(storage) { TensorEngine = TensorEngine };
            try
            {
                // The usual copy — S / U data the result lays out the same way (C into C, F into F) — is one block move,
                // without the general copy's dispatch and checks (the result is fresh, writeable and of this shape); any
                // other layout (a strided view, KEEPORDER over a permutation, int8 data) takes the general copy.
                if (!Storage.IsFixedString || !Backends.Strings.NDStringCopy.TryCopyDenseBlock(result.Storage, Storage))
                    Backends.Iteration.NDIter.Copy(result, this);
            }
            catch
            {
                // A copy that cannot finish (nothing here converts, but the fill is the general one) must not hand out
                // the uninitialized buffer.
                result.Dispose();
                throw;
            }

            return result;
        }

        /// <summary>
        ///     A new chararray object over the same memory (NumPy's <c>c.view()</c>), optionally reading the bytes as another
        ///     dtype. NumPy checks the finalize rule against the dtype this chararray HAS and only then changes it, so
        ///     <c>c.view(np.uint32)</c> is a uint32 chararray (whose own views and copies raise).
        /// </summary>
        /// <param name="dtype">The dtype to read the bytes as, or <c>null</c> to keep it. A dtype of another size changes the
        ///     length of the last axis.</param>
        /// <returns>A view; writes reach this chararray.</returns>
        /// <exception cref="ValueError">This chararray holds data that is not string data (the rule reads the current dtype),
        ///     or one of NumPy's itemsize errors.</exception>
        /// <exception cref="TypeError">A view to or from a <c>StringDType</c> descriptor.</exception>
        [NDScoped]
        public new NDCharArray view(DType dtype = null)
        {
            Finalize(this.dtype);
            return TakeOver(base.view(dtype), this);
        }

        /// <summary>
        ///     NumPy's <c>c.view(type)</c>: a new view of this chararray as the array class <paramref name="type"/>
        ///     (<c>typeof(NDArray)</c> for the plain array, <c>typeof(NDCharArray)</c>, <c>typeof(NDMatrix)</c>), or — when
        ///     <paramref name="type"/> is not an array class — read as the dtype that CLR type names.
        /// </summary>
        /// <param name="type">The array class, or a CLR element type (<c>typeof(uint)</c>).</param>
        /// <returns>A view of the class asked for; <see cref="NDArray"/> is its static type.</returns>
        /// <exception cref="ValueError">The finalize rule of the class asked for refuses this chararray's dtype, or a CLR type
        ///     that is neither an array class nor a dtype.</exception>
        /// <exception cref="NotSupportedException">An array class with no NumPy counterpart (<c>NDArray&lt;T&gt;</c>).</exception>
        /// <remarks>Declared here because the chararray's own <see cref="view(DType)"/> would otherwise take a
        ///     <see cref="Type"/> argument through its implicit conversion to <see cref="DType"/>.</remarks>
        public new NDArray view(Type type) => base.view(type);

        /// <summary>NumPy's <c>c.view(dtype, type)</c>: a view of this chararray as the array class <paramref name="type"/>,
        ///     its bytes read as <paramref name="dtype"/>.</summary>
        /// <param name="dtype">The dtype to read the bytes as, or <c>null</c> to keep it.</param>
        /// <param name="type">The array class (<c>typeof(NDArray)</c>, <c>typeof(NDCharArray)</c>, <c>typeof(NDMatrix)</c>),
        ///     or <c>null</c> for this chararray's own class.</param>
        /// <returns>A view of the class asked for.</returns>
        /// <exception cref="ValueError">The finalize rule of the class asked for refuses the dtype this chararray HAS, or
        ///     <paramref name="type"/> is not an array class (<c>Type must be a sub-type of ndarray type</c>).</exception>
        /// <exception cref="NotSupportedException">An array class with no NumPy counterpart.</exception>
        public new NDArray view(DType dtype, Type type) => type is null ? view(dtype) : base.view(dtype, type);

        /// <summary>The chararray with its bytes swapped, as a chararray (NumPy's <c>c.byteswap(inplace)</c>): every 4-byte
        ///     code unit of a <c>U</c> chararray reversed (the result usually holds units above <c>U+10FFFF</c>), an <c>S</c>
        ///     chararray unchanged.</summary>
        /// <param name="inplace"><c>true</c> swaps this chararray's own bytes and returns this chararray.</param>
        /// <returns>A new chararray that owns its data, or this chararray when <paramref name="inplace"/> is <c>true</c>.</returns>
        /// <exception cref="ValueError"><paramref name="inplace"/> on a read-only chararray, or a copy of a chararray that holds
        ///     data that is not string data.</exception>
        [NDScoped]
        public new NDCharArray byteswap(bool inplace = false) => From(base.byteswap(inplace), this);

        /// <summary>A field of each element read as <paramref name="dtype"/> at a byte offset, as a chararray view (NumPy's
        ///     <c>c.getfield(dtype, offset)</c>): <c>getfield("U1", 4)</c> is the second character of each <c>U</c> element.</summary>
        /// <param name="dtype">The dtype of the field; it must be string data (or int8), since NumPy builds the subclass view
        ///     with it.</param>
        /// <param name="offset">The byte offset inside each element.</param>
        /// <returns>A view of the field.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">The field does not fit inside an element (NumPy's texts, checked first), or the field's
        ///     dtype is not string data.</exception>
        [NDScoped]
        public new NDCharArray getfield(DType dtype, int offset = 0) => From(base.getfield(dtype, offset), this);

        /// <summary>Array-API device transfer: only <c>"cpu"</c> exists, and the chararray itself is returned.</summary>
        /// <param name="device">Must be <c>"cpu"</c>.</param>
        /// <param name="stream">Must be <c>null</c>.</param>
        /// <returns>This chararray.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="device"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">Another device, or a stream.</exception>
        public new NDCharArray to_device(string device, object stream = null) => (NDCharArray)base.to_device(device, stream);

        #endregion

        #region Shape and axes

        /// <summary>NumPy's <c>c.T</c>: the transpose, as a chararray.</summary>
        /// <value>A view with the axes reversed (writes reach this chararray).</value>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        /// <remarks>Built straight on the storage (the engine's transposed shape over a new alias): the same view
        ///     <c>base.transpose()</c> returns, without its intermediate array (a third of the cost of a 1K <c>c.T</c>).</remarks>
        public new NDCharArray T
        {
            get
            {
                Finalize(dtype);
                return new NDCharArray(Storage.Alias(Backends.DefaultEngine.ReversedAxesShape(Shape))) { TensorEngine = TensorEngine };
            }
        }

        /// <summary>NumPy's <c>c.mT</c>: the two last axes exchanged, as a chararray.</summary>
        /// <value>A view.</value>
        /// <exception cref="ArgumentException">Fewer than two axes (<c>matrix transpose with ndim &lt; 2 is undefined</c>).</exception>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        public new NDCharArray mT
        {
            [NDScoped] get => From(base.mT, this);
        }

        /// <summary>The chararray with its axes permuted (NumPy's <c>c.transpose(*axes)</c>); without an argument the axes
        ///     reversed.</summary>
        /// <param name="axes">NumPy's <c>*axes</c>: the permutation as separate integers (<c>c.transpose(1, 0)</c>) or one
        ///     array (NumPy's tuple), or none (or null) to reverse the axes.</param>
        /// <returns>A view.</returns>
        /// <exception cref="ArgumentException">The permutation does not match the axes.</exception>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        /// <remarks>One member, NumPy's signature (<c>ndarray.transpose</c> takes no keyword). It hides
        ///     <see cref="NDArray.transpose(int[])"/>, whose parameter is the same array.</remarks>
        [NDScoped]
        public new NDCharArray transpose(params int[] axes) => From(base.transpose(axes is { Length: > 0 } ? axes : null), this);

        /// <summary>The chararray with two axes exchanged (NumPy's <c>c.swapaxes(a, b)</c>).</summary>
        /// <param name="axis1">The first axis.</param>
        /// <param name="axis2">The second axis.</param>
        /// <returns>A view.</returns>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        [NDScoped]
        public new NDCharArray swapaxes(int axis1, int axis2) => From(base.swapaxes(axis1, axis2), this);

        /// <summary>The elements under another shape, as a chararray (NumPy's <c>c.reshape(shape)</c>).</summary>
        /// <param name="newShape">The requested dimensions; one may be -1 (inferred).</param>
        /// <returns>A view when the layout allows it, otherwise a copy.</returns>
        /// <exception cref="IncorrectShapeException">The requested size differs from the chararray's size.</exception>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        [NDScoped]
        public new NDCharArray reshape(Shape newShape) => From(base.reshape(newShape), this);

        /// <inheritdoc cref="reshape(Shape)"/>
        /// <param name="newShape">The requested dimensions.</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c> or <c>'A'</c>: the index order the elements are read and placed in.</param>
        [NDScoped]
        public new NDCharArray reshape(Shape newShape, char order) => From(base.reshape(newShape, order), this);

        /// <inheritdoc cref="reshape(Shape)"/>
        /// <param name="newShape">The requested dimensions.</param>
        [NDScoped]
        public new NDCharArray reshape(ref Shape newShape) => From(base.reshape(ref newShape), this);

        /// <inheritdoc cref="reshape(Shape)"/>
        /// <param name="shape">The requested dimensions; one may be -1.</param>
        [NDScoped]
        public new NDCharArray reshape(int[] shape) => From(base.reshape(shape), this);

        /// <inheritdoc cref="reshape(Shape)"/>
        /// <param name="shape">The requested dimensions; one may be -1.</param>
        [NDScoped]
        public new NDCharArray reshape(params long[] shape) => From(base.reshape(shape), this);

        /// <inheritdoc cref="reshape(Shape)"/>
        /// <param name="shape">The requested dimensions; one may be -1.</param>
        [NDScoped]
        public new NDCharArray reshape(params ReadOnlySpan<long> shape) => From(base.reshape(shape), this);

        /// <summary>The elements as one axis, as a chararray (NumPy's <c>c.ravel()</c>): a VIEW when the layout allows it,
        ///     a copy otherwise.</summary>
        /// <returns>A 1-d chararray.</returns>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        [NDScoped]
        public new NDCharArray ravel() => From(base.ravel(), this);

        /// <inheritdoc cref="ravel()"/>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <exception cref="ArgumentException">An unknown order.</exception>
        [NDScoped]
        public new NDCharArray ravel(char order) => From(base.ravel(order), this);

        /// <summary>A COPY of the elements as one axis, as a chararray (NumPy's <c>c.flatten(order)</c>).</summary>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <returns>A new 1-d chararray that owns its data.</returns>
        /// <exception cref="ArgumentException">An unknown order.</exception>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        [NDScoped]
        public new NDCharArray flatten(char order = 'C') => From(base.flatten(order), this);

        /// <summary>
        ///     The chararray without its length-1 axes (NumPy's <c>c.squeeze(axis)</c>).
        /// </summary>
        /// <param name="axis">The axis to remove, or <c>null</c> for every axis of length 1.</param>
        /// <returns>This chararray ITSELF when <paramref name="axis"/> is <c>null</c> and no axis has length 1 (as NumPy);
        ///     otherwise a chararray view.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError"><paramref name="axis"/> names an axis longer than 1, or this chararray holds data that
        ///     is not string data and an axis is removed (NumPy wraps that plain view as an ndarray; the C# type here is a
        ///     chararray).</exception>
        [NDScoped]
        public new NDCharArray squeeze(int? axis = null)
        {
            if (axis is null)
            {
                // PyArray_Squeeze returns self when no axis has length 1.
                bool unit = false;
                foreach (long extent in Shape.dimensions)
                    unit |= extent == 1;
                if (!unit)
                    return this;
            }

            return From(base.squeeze(axis), this);
        }

        #endregion

        #region Selection

        /// <summary>The diagonal, as a chararray (NumPy's <c>c.diagonal(offset, axis1, axis2)</c>).</summary>
        /// <param name="offset">0 for the main diagonal, positive above it, negative below.</param>
        /// <param name="axis1">The first axis of the plane.</param>
        /// <param name="axis2">The second axis of the plane.</param>
        /// <returns>A READ-ONLY view of the diagonal.</returns>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ArgumentException">The two axes are the same.</exception>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        [NDScoped]
        public new NDCharArray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1) => From(base.diagonal(offset, axis1, axis2), this);

        /// <summary>
        ///     Elements selected by index along an axis, as a chararray (NumPy's <c>c.take(indices, axis, out, mode)</c>).
        ///     Without an axis the chararray is read flat.
        /// </summary>
        /// <param name="indices">The indices: an integer array, or one C# integer (<c>c.take(1)</c>, NumPy's 0-d index, which
        ///     selects one element).</param>
        /// <param name="axis">The axis, or <c>null</c> to index the flattened chararray.</param>
        /// <param name="out">NumPy's <c>out=</c>: a chararray of the result's shape that receives the elements and is
        ///     returned, or null. A plain <see cref="NDArray"/> out binds <see cref="NDArray.take(NDArray, int?, NDArray, string)"/>
        ///     instead and returns that array, as NumPy returns its <c>out</c> whatever its class.</param>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c> or <c>"clip"</c> for an index out of bounds.</param>
        /// <returns>A new chararray that owns its data (or <paramref name="out"/>); for a 0-d result the element as a
        ///     read-only 0-d stand-in, NOT stripped (NumPy's <c>PyArray_Return</c> gives the element's NumPy scalar, which is
        ///     not <c>__getitem__</c>'s stripped value).</returns>
        /// <exception cref="IndexError">An index is out of bounds under <c>"raise"</c>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ArgumentException">An index array of a non-integer dtype, or an unknown mode.</exception>
        /// <exception cref="ValueError">This chararray holds data that is not string data; NumPy's out= errors (another shape,
        ///     read-only).</exception>
        /// <remarks>One overload, NumPy's signature: <paramref name="indices"/> takes NumPy's int or int array_like through
        ///     NumSharp's implicit conversions, so the one-index overload this member used to have adds nothing.</remarks>
        [NDScoped]
        public NDCharArray take(NDArray indices, int? axis = null, NDCharArray @out = null, string mode = "raise")
        {
            if (@out is null)
                return TakeResult(base.take(indices, axis, null, mode));
            // ndarray.take(out=) writes the elements into out and returns it — its scalar for a 0-d result (PyArray_Return).
            var written = base.take(indices, axis, @out, mode);
            return written.ndim == 0 ? TakeResult(written) : @out;
        }

        /// <summary>The chararray form of a <c>take</c> result: NumPy's <c>PyArray_Return</c> turns a 0-d result into the
        ///     element's scalar.</summary>
        /// <param name="result">The plain result (reclaimed by the caller's scope).</param>
        /// <returns>The unstripped scalar stand-in for a 0-d result, otherwise the chararray over the result.</returns>
        /// <exception cref="ValueError">This chararray holds data that is not string data.</exception>
        private NDCharArray TakeResult(NDArray result)
        {
            // NumPy allocates the result as the subclass before PyArray_Return looks at its rank: the rule applies to both.
            Finalize(result.dtype);
            return result.ndim == 0 ? ScalarOf(result.Storage, strip: false) : From(result, this);
        }

        /// <summary>The elements where <paramref name="condition"/> is true, as a chararray (NumPy's
        ///     <c>c.compress(condition, axis, out)</c>); without an axis the flattened elements.</summary>
        /// <param name="condition">A 1-d bool array.</param>
        /// <param name="axis">The axis to select along, or <c>null</c>.</param>
        /// <param name="out">NumPy's <c>out=</c>: a chararray of the result's shape that receives the elements and is
        ///     returned, or null. A plain <see cref="NDArray"/> out binds <see cref="NDArray.compress"/> instead and returns
        ///     that array, as NumPy returns its <c>out</c> whatever its class.</param>
        /// <returns>A new chararray that owns its data, or <paramref name="out"/>.</returns>
        /// <exception cref="ValueError"><paramref name="condition"/> is not 1-d, or this chararray holds data that is not
        ///     string data; NumPy's out= errors (compress is a take of the condition's nonzero indices: <c>output array does
        ///     not match result of ndarray.take</c>).</exception>
        /// <exception cref="IndexError"><paramref name="condition"/> is longer than the axis.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <remarks>One member, NumPy's signature.</remarks>
        [NDScoped]
        public NDCharArray compress(NDArray condition, int? axis = null, NDCharArray @out = null)
        {
            if (@out is null)
                return From(base.compress(condition, axis, null), this);
            // ndarray.compress(out=) fills out and returns it (compress is never 0-d).
            base.compress(condition, axis, @out);
            return @out;
        }

        /// <summary>Each element repeated, as a chararray (NumPy's <c>c.repeat(repeats, axis)</c>); without an axis the
        ///     flattened elements.</summary>
        /// <param name="repeats">The repeat counts: one C# integer for every element (<c>c.repeat(2)</c>, NumPy's
        ///     <c>np.asarray(2)</c>), or an integer array of one count per element of the axis.</param>
        /// <param name="axis">The axis to repeat along, or <c>null</c>.</param>
        /// <returns>A new chararray that owns its data.</returns>
        /// <exception cref="ValueError">A negative count, counts that do not match the axis, or this chararray holds data
        ///     that is not string data.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <remarks>One member, NumPy's signature: every <see cref="NDArray.repeat(NDArray, int?)"/> call form converts to
        ///     this <see cref="NDArray"/> parameter, so none of NDArray's repeat overloads binds through a chararray.</remarks>
        [NDScoped]
        public new NDCharArray repeat(NDArray repeats, int? axis = null) => From(base.repeat(repeats, axis), this);

        #endregion

        #region Complex parts

        /// <summary>
        ///     The real part, as a chararray: for string data the chararray ITSELF (NumPy's <c>c.real is c</c>). Assigning
        ///     stores into it.
        /// </summary>
        /// <value>This chararray.</value>
        /// <exception cref="ValueError">(set) The chararray is read-only, or the value does not broadcast.</exception>
        public new NDCharArray real
        {
            [NDScoped] get => From(base.real, this);
            set => base.real = value;
        }

        /// <summary>
        ///     The imaginary part, as a chararray: for string data a READ-ONLY chararray of the same dtype and shape holding
        ///     empty strings (NumPy's zero-filled subclass array). Assigning raises: there is no imaginary part to set.
        /// </summary>
        /// <value>A new read-only chararray that owns its (empty) data.</value>
        /// <exception cref="ValueError">(get) This chararray holds data that is not string data.</exception>
        /// <exception cref="TypeError">(set) Always for string data (<c>array does not have imaginary part to set</c>).</exception>
        public new NDCharArray imag
        {
            [NDScoped] get => From(base.imag, this);
            set => base.imag = value;
        }

        #endregion
    }
}
