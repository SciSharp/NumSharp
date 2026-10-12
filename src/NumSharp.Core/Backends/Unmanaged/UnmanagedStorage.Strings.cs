using System;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        /// <summary>
        ///     True when this storage holds fixed-width strings (<c>S</c> or <c>U</c>): its lane is
        ///     <see cref="NPTypeCode.String"/>, its <see cref="InternalArray"/> is a <see cref="StringArraySlice"/>, and its
        ///     <see cref="Descr"/> is the sized string descriptor.
        /// </summary>
        public bool IsString => _typecode == NPTypeCode.String;

        /// <summary>
        ///     The <see cref="StringArraySlice"/> of a string storage (null for every other lane) — the bytes plus the element
        ///     width they are read with.
        /// </summary>
        internal StringArraySlice StringSlice => InternalArray as StringArraySlice;

        /// <summary>
        ///     Allocates a string storage of <paramref name="shape"/>'s dimensions — the string half of NumPy's
        ///     <c>PyArray_NewFromDescr</c>: the descriptor is made allocatable first (an unsized <c>S0</c>/<c>U0</c> request
        ///     becomes one character, a non-native order becomes native; <see cref="NDStringStorage.AllocationDescr"/>), the
        ///     dimensions are checked against the BYTE count (<see cref="AllocationGuard"/>), and only a fresh layout of the
        ///     shape is kept (<see cref="Shape.ForFreshAllocation"/>).
        /// </summary>
        /// <param name="shape">The dimensions (and, when it is a fresh C/F/permuted layout, the layout).</param>
        /// <param name="descr">A string descriptor (sized or not).</param>
        /// <param name="fillZeros">True to NUL-fill (<c>np.zeros</c>; every element reads empty); false leaves pool noise for
        /// a caller that writes every byte (<c>np.empty</c> does not promise zeros, but NumPy's <c>U</c> needs init, so the
        /// public creation routes zero-fill anyway).</param>
        /// <returns>A new owning string storage.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="shape"/> is uninitialized.</exception>
        /// <exception cref="ValueError">A negative dimension or a byte count that overflows (<see cref="AllocationGuard"/>).</exception>
        internal static UnmanagedStorage AllocateString(Shape shape, DType descr, bool fillZeros)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));
            var allocDescr = NDStringStorage.AllocationDescr(descr);
            AllocationGuard.CheckDimensions(shape.dimensions, allocDescr.itemsize);
            var fresh = shape.ForFreshAllocation();
            var storage = new UnmanagedStorage();
            storage._Allocate(fresh, StringArraySlice.Allocate(allocDescr, fresh.size, fillZeros));
            return storage;
        }

        /// <summary>
        ///     The string half of <see cref="CopyTo(void*)"/>: every element's raw bytes (NUL-padded ASCII for <c>S</c>,
        ///     UCS-4 code units for <c>U</c>) written to <paramref name="address"/> in C order — <c>tobytes()</c>'s layout.
        ///     A C-contiguous view is one block move from its first element; any other layout is first copied into a fresh
        ///     C-contiguous buffer by the string copy route (whole elements, no conversion: the class and width stay).
        /// </summary>
        /// <param name="address">The destination; it must hold <c>Count * DTypeSize</c> bytes.</param>
        /// <exception cref="ArgumentNullException"><paramref name="address"/> is null.</exception>
        internal unsafe void CopyStringBytesTo(void* address)
        {
            if (address == null)
                throw new ArgumentNullException(nameof(address));
            long bytes = Count * (long)DTypeSize;
            if (bytes == 0)
                return;
            if (Shape.IsContiguous)
            {
                // A contiguous view starts at its offset (a slice re-seats Address and keeps offset 0, a strided parent
                // keeps the base address with a non-zero offset: Address + offset * itemsize covers both).
                Buffer.MemoryCopy(Address + Shape.offset * (long)DTypeSize, address, bytes, bytes);
                return;
            }
            // Any other layout: a C-ordered copy of the same descriptor (the copy route's whole-element kernels), then
            // one block move. The wrapper takes its own reference on this storage (released with it) and the temporary
            // goes back to the pool here.
            using var view = new NDArray(this);
            using var compact = view.copy('C');
            Buffer.MemoryCopy(compact.Storage.Address, address, bytes, bytes);
        }

        /// <summary>
        ///     Assignment into a string storage (or of a string value), <c>this[indices] = value</c> — NumPy's
        ///     <c>array_assign_subscript</c> for the string dtypes: a coordinate that names ONE element accepts only a 0-d
        ///     value (<c>setting an array element with a sequence</c>, without the numeric message's trailing period —
        ///     <c>UNICODE_setitem</c>'s text); a sub-array target takes any value that broadcasts to it, extra LEADING size-1
        ///     axes of the value squeezed away first; the bytes move through the string copy route (encode / convert,
        ///     truncate, NUL-pad).
        /// </summary>
        /// <param name="value">The value array: text, or numbers into a string storage (each written as its
        ///     <c>str()</c>), or text into a numeric storage (each parsed by the storage dtype's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c>, the copy route's conversion).</param>
        /// <param name="indices">Normalised (non-negative, in-range) leading coordinates.</param>
        /// <exception cref="ValueError">A sequence into one element, a value that does not broadcast to the target, or a text
        ///     that does not parse into a numeric storage.</exception>
        /// <exception cref="OverflowException">An integer text outside a numeric storage's dtype.</exception>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        internal void SetDataString(NDArray value, int[] indices)
        {
            var valueShape = value.Shape;
            if (indices.Length == _shape.NDim && !valueShape.IsScalar)
                throw new ValueError("setting an array element with a sequence");

            var target = GetData(indices);
            var targetShape = target.Shape;
            var source = value.Storage;

            // NumPy squeezes extra LEADING size-1 axes of the value toward the target's rank before broadcasting
            // (a[...] = [['m', 'n', 'o']] into a (3,) array).
            int drop = valueShape.NDim - targetShape.NDim;
            if (drop > 0)
            {
                bool leadingOnes = true;
                for (int i = 0; i < drop; i++)
                    if (valueShape.dimensions[i] != 1)
                    {
                        leadingOnes = false;
                        break;
                    }
                if (leadingOnes)
                {
                    var dims = valueShape.dimensions.AsSpan(drop).ToArray();
                    var strides = valueShape.strides.AsSpan(drop).ToArray();
                    source = source.Alias(new Shape(dims, strides, valueShape.offset, valueShape.bufferSize));
                }
            }

            if (!Broadcastable(source.Shape.dimensions, targetShape.dimensions))
                throw new ValueError($"could not broadcast input array from shape {Tuple(valueShape.dimensions)} into shape {Tuple(targetShape.dimensions)}");
            if (targetShape.size == 0)
            {
                // Nothing is written. NumPy assigns an ARRAY value with PyArray_AssignArray, which converts nothing for an
                // empty target (np.zeros(0, 'S1')[...] = np.array(['é']) succeeds). A SCALAR value — a C# literal, NumPy's
                // Python str — is first packed into this array's dtype (PyArray_Pack), so a text that cannot convert raises
                // even when the target selects nothing (a[...] = 'é' into an empty S array): that conversion is run and
                // discarded here. Into a numeric array a Python str literal (NDArray.WasPythonStr) was already parsed by
                // SetData's pack before it got here, and a 0-d text ARRAY converts nothing; formatting a number as text
                // cannot fail, so a numeric value leaves nothing to report.
                if (valueShape.IsScalar && value.Storage.IsString && IsString && value.dtype.Meta != Descr.Meta)
                    value.astype(Descr, true, 'K', "unsafe").Dispose();
                return;
            }
            Iteration.NDIter.Copy(target, source);
        }

        /// <summary>
        ///     <c>this[indices] = value</c> for a C# text scalar into a string storage: the value becomes a 0-d array of this
        ///     storage's CLASS (its width discovered, so a value longer than the element is truncated by the copy, exactly as
        ///     NumPy's setitem truncates), then broadcasts over the target.
        /// </summary>
        /// <param name="value">A <see cref="string"/>, <see cref="char"/>, <c>byte[]</c> or null (None).</param>
        /// <param name="indices">The leading coordinates (normalised by the array path).</param>
        /// <returns>True when <paramref name="value"/> was a text scalar and has been assigned.</returns>
        internal bool TrySetTextScalar(object value, int[] indices)
        {
            if (!(value is string || value is char || value is byte[] || value is null))
                return false;
            // The class of THIS array decides how the value is encoded (str into S is ASCII-encoded, bytes into U decoded).
            var cls = (StringDTypeMeta)Descr.Meta;
            using var scalar = NDStringArrays.FromScalar(value, cls.Singleton);
            SetData(scalar, indices);
            return true;
        }

        /// <summary>NumPy's broadcast compatibility of a source shape to a target shape (right-aligned, 1 stretches).</summary>
        /// <param name="source">The source dimensions.</param>
        /// <param name="target">The target dimensions.</param>
        /// <returns>True when the source broadcasts to the target.</returns>
        private static bool Broadcastable(long[] source, long[] target)
        {
            if (source.Length > target.Length)
                return false;
            int shift = target.Length - source.Length;
            for (int i = 0; i < source.Length; i++)
                if (source[i] != 1 && source[i] != target[shift + i])
                    return false;
            return true;
        }

        /// <summary>A shape as NumPy prints it in assignment errors: <c>(3,)</c>, <c>(2,3)</c>.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The text.</returns>
        private static string Tuple(long[] dims) => dims.Length == 1 ? $"({dims[0]},)" : "(" + string.Join(",", dims) + ")";

        /// <summary>
        ///     The string descriptor a width-less request means: this storage's own descriptor when it already is a string
        ///     storage and no type was named (re-allocation keeps the width), else the <c>U</c> class singleton, which the
        ///     allocator turns into NumPy's default <c>&lt;U1</c> (<c>np.zeros(3, str)</c> is <c>&lt;U1</c>).
        /// </summary>
        /// <param name="requested">The requested CLR type, or null for "keep this storage's type".</param>
        /// <returns>A string descriptor (possibly unsized).</returns>
        private DType StringRequest(Type requested)
            => requested == null && _descr?.Meta is StringDTypeMeta ? _descr : DTypeRegistry.Str.Singleton;

        /// <summary>
        ///     A fresh buffer of <paramref name="count"/> elements of THIS storage's element type — the descriptor's width for
        ///     a string storage (a <see cref="StringArraySlice"/> of the same descriptor), the lane's for a numeric one. The
        ///     generic clone paths size their result with it, because <see cref="ArraySlice.Allocate(NPTypeCode, long, bool)"/>
        ///     cannot know a string lane's width.
        /// </summary>
        /// <param name="count">The number of elements.</param>
        /// <param name="fillZeros">True to zero-fill (NUL-fill for strings).</param>
        /// <returns>The new slice.</returns>
        internal IArraySlice AllocateLike(long count, bool fillZeros)
            => _typecode == NPTypeCode.String
                ? StringArraySlice.Allocate(StringSlice.Descr, count, fillZeros)
                : ArraySlice.Allocate(_typecode, count, fillZeros);

        /// <summary>
        ///     Re-tags a FRESH, C-contiguous unit array (<c>uint8</c> for <c>S</c>, <c>uint32</c> for <c>U</c>, its last axis
        ///     the characters of one element) as the string array it holds — no byte moves. The string results of the unit
        ///     routes (fancy and boolean indexing of a string array run on its unit view) come back this way: the numeric
        ///     gather wrote whole elements contiguously, so element <c>i</c> already occupies bytes
        ///     <c>[i·itemsize, (i+1)·itemsize)</c>.
        /// </summary>
        /// <param name="units">A unit storage the caller owns (C-contiguous; last axis = <c>descr</c>'s characters).</param>
        /// <param name="descr">The sized, native string descriptor of the result.</param>
        /// <returns>An owning string storage over the same buffer, of the unit shape without its last axis. Its reference
        ///     counting forwards to <paramref name="units"/>' block, so the caller may release its own reference at once.</returns>
        /// <exception cref="InvalidOperationException"><paramref name="units"/> is not C-contiguous, or its last axis is not
        ///     one element of <paramref name="descr"/>.</exception>
        internal static unsafe UnmanagedStorage StringsOverUnits(UnmanagedStorage units, DType descr)
        {
            var shape = units.Shape;
            var cls = (StringDTypeMeta)descr.Meta;
            int nd = shape.NDim - 1;
            long chars = descr.itemsize / cls.UnitSize;
            if (nd < 0 || shape.dimensions[nd] != chars || !shape.IsContiguous)
                throw new InvalidOperationException("A string result must come from a C-contiguous unit array whose last axis is one element.");

            var dims = new long[nd];
            Array.Copy(shape.dimensions, dims, nd);
            long elements = shape.size / Math.Max(chars, 1);
            byte* first = units.Address + shape.offset * (long)cls.UnitSize;
            var bytes = ArraySlice.WrapShared<byte>(first, elements * descr.itemsize, units.InternalArray);

            var storage = new UnmanagedStorage();
            storage._Allocate(nd == 0 ? Shape.NewScalar() : new Shape(dims), new StringArraySlice(bytes, descr));
            storage.Engine = units.Engine;
            return storage;
        }

        /// <summary>
        ///     <c>ndarray.view(dtype)</c> when the view or this storage is a fixed-width string — NumPy's
        ///     <c>array_descr_set</c> at the DESCRIPTOR level: the same bytes, read as <paramref name="descr"/>. An equal
        ///     itemsize keeps every axis (any layout, a broadcast view stays read-only); a changed itemsize rescales the
        ///     contiguous last axis (<see cref="ReinterpretShape"/>: <c>U3 → U1</c> triples it, <c>S3 → S9</c> divides it by
        ///     three, <c>uint32 → U2</c> halves it). Nothing is converted: a <c>U</c> element read as <c>uint32</c> is its
        ///     code points, a <c>uint8</c> array read as <c>S2</c> pairs its bytes.
        /// </summary>
        /// <param name="descr">The view's descriptor: any storage dtype, or a SIZED native string dtype.</param>
        /// <returns>A storage sharing this storage's memory (its references counted on this storage's block, so the
        ///     buffer outlives a disposed owner while the view lives).</returns>
        /// <exception cref="ValueError">NumPy's itemsize errors (<see cref="ReinterpretShape"/>), including an UNSIZED string
        ///     target (<c>'U'</c>, <c>'S'</c>: <c>When changing to a smaller dtype, its size must be a divisor of the size of
        ///     original dtype</c>).</exception>
        /// <exception cref="InvalidOperationException">A view whose byte offset or outer byte stride is not a whole number of
        ///     new elements (NumSharp keeps them in elements).</exception>
        /// <exception cref="NotSupportedException">A non-native byte order (NumPy keeps the bytes and swaps on every read;
        ///     NumSharp stores every dtype natively), or a descriptor without storage (<c>datetime64</c> before Stage C).</exception>
        internal unsafe UnmanagedStorage AliasAsDescr(DType descr)
        {
            if (descr is null)
                throw new ArgumentNullException(nameof(descr));
            var meta = descr.Meta;
            bool toString = meta is StringDTypeMeta;
            if (!descr.isnative)
                throw new NotSupportedException(
                    $"Viewing as {descr.ToString(true)} reads the bytes in a non-native byte order, which NumPy keeps byte-swapped " +
                    "but NumSharp does not store (every NumSharp dtype is native).");
            var code = toString ? NPTypeCode.String : descr.GetTypeCode(); // a descriptor-only class raises "no storage yet"
            if (toString ? descr.Equals(Descr) : code == _typecode)
                return Alias();

            int oldSize = DTypeSize;
            int newSize = toString ? (int)descr.itemsize : code.SizeOf();
            Shape shape;
            long count;
            if (newSize == oldSize)
            {
                // A pure retag: every axis, stride and offset stays (same element size, so the units do not change).
                shape = _shape;
                count = InternalArray.BytesLength / newSize;
            }
            else
            {
                shape = ReinterpretShape(oldSize, newSize, out count);
            }

            // A view inherits non-writeability (a broadcast or 'r' memmap stays read-only) and re-reports ALIGNED.
            if (!_shape.IsWriteable && shape.IsWriteable)
                shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            if ((shape._flags & (int)ArrayFlags.ALIGNED) == 0)
                shape = shape.WithFlags(flagsToSet: ArrayFlags.ALIGNED);

            // The alias counts its references on THIS storage's block (WrapShared): disposing the owner while the view
            // lives does not free the buffer under it.
            IArraySlice slice = toString
                ? new StringArraySlice(ArraySlice.WrapShared<byte>(InternalArray.Address, count * newSize, InternalArray), descr)
                : ArraySlice.WrapShared(code, InternalArray.Address, count, InternalArray);

            var r = new UnmanagedStorage();
            r._shape = shape;
            r._typecode = code;
            r._dtype = code.AsType();
            r.SetInternalArray(slice);
            r.Count = shape.size;
            r._baseStorage = _baseStorage ?? this;
            r.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
            r.Engine = Engine;
            return r;
        }

        /// <summary>
        ///     The UNIT VIEW of a string storage: the same memory as an ordinary numeric storage — <c>uint8</c> for <c>S</c>,
        ///     <c>uint32</c> (one UCS-4 code point) for <c>U</c> — with ONE extra trailing axis of the element's characters.
        ///     A string array of shape <c>D</c> and width <c>c</c> characters becomes a unit array of shape <c>D + (c,)</c>
        ///     whose outer strides and offset are scaled by <c>c</c> and whose new axis has stride 1, so broadcast axes stay
        ///     broadcast (stride 0) and every view layout (strided, reversed, transposed, offset) carries over exactly.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     This is how every string MOVEMENT runs on the existing numeric machinery — copy, clone, string↔string casts,
        ///     <c>copyto</c> — with no string-specific loop: the result is a bare storage (no <see cref="NDArray"/>, so no
        ///     counted reference), its <c>_baseStorage</c> is this storage's owner, and a non-writeable string array gives a
        ///     non-writeable unit view. For <c>U</c> the <c>uint32</c> slice is a <see cref="ArraySlice.WrapShared{T}"/>
        ///     alias whose reference counting forwards to the string slice, so the bytes stay alive as long as the view.
        ///     </para>
        /// </remarks>
        /// <returns>The unit view.</returns>
        /// <exception cref="InvalidOperationException">This is not a string storage.</exception>
        internal unsafe UnmanagedStorage StringUnitView()
        {
            var slice = StringSlice ?? throw new InvalidOperationException($"A unit view needs a string storage, not {_typecode}.");
            int chars = slice.Chars;

            IArraySlice units = slice.IsUnicode
                ? ArraySlice.WrapShared<uint>(slice.Bytes.VoidAddress, slice.Bytes.Count / 4, slice)
                : slice.Bytes;

            var shape = _shape;
            int nd = shape.dimensions.Length;
            var dims = new long[nd + 1];
            var strides = new long[nd + 1];
            for (int i = 0; i < nd; i++)
            {
                dims[i] = shape.dimensions[i];
                strides[i] = shape.strides[i] * chars; // element stride → unit stride (0 stays 0: broadcast survives)
            }
            dims[nd] = chars;
            strides[nd] = 1; // the characters of one element are contiguous

            var unitShape = new Shape(dims, strides, shape.offset * chars, shape.bufferSize * chars);
            if (!shape.IsWriteable && unitShape.IsWriteable)
                unitShape = unitShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            var view = new UnmanagedStorage();
            view._Allocate(unitShape, units);
            view._baseStorage = _baseStorage ?? this;
            view.WriteProtected = WriteProtected;
            view.OnReshaped(); // a view never owns its buffer
            view.Engine = Engine;
            return view;
        }
    }
}
