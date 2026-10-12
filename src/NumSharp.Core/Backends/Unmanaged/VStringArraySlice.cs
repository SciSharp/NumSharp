using System;
using System.Collections;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    /// <summary>
    ///     The ELEMENT view of a NEP 55 <c>StringDType</c> buffer — the <see cref="IArraySlice"/> a <c>T</c>
    ///     <see cref="UnmanagedStorage"/> holds as its <see cref="UnmanagedStorage.InternalArray"/>. The elements are NumPy's
    ///     16-byte packed cells (<see cref="Cells"/>, an ordinary <see cref="ArraySlice{T}"/> of <see cref="byte"/>, so pooling,
    ///     reference counting and views are the numeric lanes'); the text of elements longer than 15 bytes lives in the
    ///     buffer's <see cref="Allocator"/> (an arena and heap strings), which the cells' memory block owns.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Views share, copies duplicate.</b> <see cref="Slice(long, long)"/> (contiguous slicing, views, sub-arrays) shares
    ///     the cells AND the allocator: a write through a view is a write into the same buffer. <see cref="Clone"/> allocates a
    ///     new buffer with a new allocator and DUPLICATES every string into it (NumPy's <c>NpyString_dup</c>), so the copy's
    ///     arena is compact and its cells never point into the source's memory.
    ///     </para>
    ///     <para>
    ///     <b>No raw element access.</b> A cell is meaningful only with its own allocator: an arena offset read with another
    ///     buffer's arena reads someone else's text, and a heap pointer copied into another buffer would be freed twice by the
    ///     two clear loops. So the typed accessors refuse, <see cref="Clone{T}"/> refuses, and the raw byte accessors
    ///     (<see cref="AsSpan{T}"/>, <see cref="CopyTo{T}(Span{T})"/>) expose the cells only as BYTES to read
    ///     (NumPy's <c>tobytes()</c> of a <c>T</c> array is exactly these cells); writing such bytes into another
    ///     <c>StringDType</c> buffer is never valid.
    ///     </para>
    ///     <para>
    ///     <b>Element values.</b> <see cref="GetIndex(long)"/> is NumPy's <c>stringdtype_getitem</c>: a <see cref="string"/>, or
    ///     the descriptor's <c>na_object</c> for an NA cell (<c>''</c> when the descriptor has none).
    ///     <see cref="SetIndex(long, object)"/> is <c>stringdtype_setitem</c>: the NA object packs an NA cell; any other value
    ///     is stored as its Python <c>str()</c> when the descriptor coerces (a <see cref="string"/> or <see cref="char"/> as
    ///     itself, <c>bytes</c> decoded as strict UTF-8), else only text is accepted.
    ///     </para>
    /// </remarks>
    public sealed unsafe class VStringArraySlice : IArraySlice
    {
        /// <summary>The error text of a non-text value under <c>coerce=False</c> (<c>as_pystring</c>).</summary>
        internal const string CoercionDisabledText = "StringDType only allows string data when string coercion is disabled.";

        /// <summary>
        ///     Wraps cells that belong to <paramref name="allocator"/> as <c>cells.Count / 16</c> elements of
        ///     <paramref name="descr"/>.
        /// </summary>
        /// <param name="cells">The cells (a whole number of 16-byte cells).</param>
        /// <param name="allocator">The allocator of the buffer the cells belong to.</param>
        /// <param name="descr">A <c>StringDType</c> descriptor.</param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="descr"/> is not a <c>StringDType</c> descriptor, or the cells
        ///     are not a whole number of 16-byte cells.</exception>
        internal VStringArraySlice(ArraySlice<byte> cells, VStringAllocator allocator, DType descr)
        {
            Allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
            if (descr is null)
                throw new ArgumentNullException(nameof(descr));
            if (descr.Meta is not VStringDTypeMeta)
                throw new ArgumentException($"{descr.ToString(true)} is not a StringDType descriptor.", nameof(descr));
            if (cells.Count % VStringAllocator.CellSize != 0)
                throw new ArgumentException($"A buffer of {cells.Count} bytes is not a whole number of 16-byte cells.", nameof(cells));
            Cells = cells;
            Descr = descr;
            Count = cells.Count / VStringAllocator.CellSize;
        }

        /// <summary>
        ///     Allocates a buffer of <paramref name="count"/> elements of <paramref name="descr"/>: ZERO-FILLED cells (every
        ///     element is <c>''</c>, NumPy's <c>NPY_NEEDS_INIT</c>) and a new allocator, both owned by one memory block whose
        ///     release runs the clear loop (frees the heap strings and the arena) before the cells go back to the pool.
        /// </summary>
        /// <param name="descr">A <c>StringDType</c> descriptor.</param>
        /// <param name="count">The number of elements (the caller has already run <see cref="AllocationGuard"/> on 16-byte
        ///     elements, so <c>count * 16</c> cannot overflow).</param>
        /// <returns>The new slice (the only reference to its block).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="descr"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="descr"/> is not a <c>StringDType</c> descriptor.</exception>
        /// <exception cref="OutOfMemoryException">The cells cannot be allocated.</exception>
        public static VStringArraySlice Allocate(DType descr, long count)
        {
            if (descr is null)
                throw new ArgumentNullException(nameof(descr));
            if (descr.Meta is not VStringDTypeMeta)
                throw new ArgumentException($"{descr.ToString(true)} is not a StringDType descriptor.", nameof(descr));
            long bytes = count * VStringAllocator.CellSize;
            var allocator = new VStringAllocator();
            IntPtr ptr = SizeBucketedBufferPool.TakeZeroed(bytes);
            // The block OWNS the allocator: its release (the last view's Release, or the block's finalizer) runs NumPy's
            // clear loop over every cell of the WHOLE buffer, frees the arena, and only then returns the cells to the pool —
            // so no heap string outlives the buffer and no cell outlives its allocator.
            var block = new UnmanagedMemoryBlock<byte>((byte*)ptr, bytes, () =>
            {
                allocator.ClearAndFree((byte*)ptr, count);
                SizeBucketedBufferPool.Return(ptr, bytes);
            });
            return new VStringArraySlice(new ArraySlice<byte>(block), allocator, descr);
        }

        /// <summary>The packed cells: <see cref="Count"/> × 16 bytes.</summary>
        public ArraySlice<byte> Cells { get; }

        /// <summary>The allocator of the buffer these cells belong to (shared by every view of it).</summary>
        internal VStringAllocator Allocator { get; }

        /// <summary>The element descriptor (a <c>StringDType</c> instance: its <c>na_object</c> and <c>coerce</c>).</summary>
        public DType Descr { get; }

        /// <summary>True when the descriptor has a missing-data object (its NA cells read as that object).</summary>
        public bool HasNAObject => Descr.VString.HasNAObject;

        // ---------------------------------------------------------------------------------------------
        // IMemoryBlock — element semantics
        // ---------------------------------------------------------------------------------------------

        /// <summary>The number of string ELEMENTS (cells, not bytes).</summary>
        public long Count { get; }

        /// <summary>Bytes per element: one 16-byte cell.</summary>
        public int ItemLength => VStringAllocator.CellSize;

        /// <summary>Total cell bytes (<see cref="Count"/> × 16; the text in the allocator is not counted, as NumPy's <c>nbytes</c>).</summary>
        public long BytesLength => Count * VStringAllocator.CellSize;

        /// <summary>Always <see cref="NPTypeCode.String"/>: the lane shared with <c>S</c> / <c>U</c> (decision D4; the descriptor's
        ///     class tells the layouts apart).</summary>
        public NPTypeCode TypeCode => NPTypeCode.String;

        /// <summary>The first cell.</summary>
        public byte* Address => Cells.Address;

        /// <inheritdoc/>
        void* IMemoryBlock.Address => Cells.VoidAddress;

        /// <summary>The cells' memory block (pool, reference count, and the allocator's lifetime).</summary>
        public IMemoryBlock MemoryBlock => Cells.MemoryBlock;

        // ---------------------------------------------------------------------------------------------
        // Element access (stringdtype_getitem / stringdtype_setitem)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Element <paramref name="index"/> as its C# value (<c>stringdtype_getitem</c>): the text as a <see cref="string"/>,
        ///     or the descriptor's <c>na_object</c> for an NA cell (<c>''</c> when the descriptor has none).
        /// </summary>
        /// <param name="index">The element index.</param>
        /// <returns>A <see cref="string"/>, or the NA object (possibly null, NumPy's <c>None</c>).</returns>
        /// <exception cref="OutOfMemoryException"><c>Failed to load string in StringDType getitem</c> (a corrupt cell), or a
        ///     text too long for a .NET string.</exception>
        public object GetIndex(long index)
        {
            Allocator.Acquire();
            try
            {
                int isNull = Allocator.Load(Address + index * VStringAllocator.CellSize, out byte* buf, out long size);
                if (isNull < 0)
                    throw new OutOfMemoryException("Failed to load string in StringDType getitem");
                if (isNull == 1)
                    return HasNAObject ? Descr.VString.NAObject : string.Empty;
                return Decode(buf, size);
            }
            finally
            {
                Allocator.Release();
            }
        }

        /// <summary>
        ///     Stores <paramref name="value"/> into element <paramref name="index"/> (<c>stringdtype_setitem</c>): the
        ///     descriptor's NA object (by NumPy's <c>na_eq_cmp</c>) packs an NA cell; any other value is converted by
        ///     <see cref="AsPyString"/>, encoded as strict UTF-8 and packed.
        /// </summary>
        /// <param name="index">The element index.</param>
        /// <param name="value">Any C# value standing in for a Python object (see <see cref="NDPyObject"/>).</param>
        /// <exception cref="ValueError">A value that is not text under <c>coerce=False</c>.</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8.</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>Failed to pack string during StringDType setitem</c>.</exception>
        [System.Runtime.CompilerServices.SkipLocalsInit]
        public void SetIndex(long index, object value)
        {
            // NumPy compares with the NA object and converts the value before it packs; both can raise, so they run before
            // the cell is touched (an error leaves the element as it was). The UTF-8 form goes into stack space (a text whose
            // form could be longer gets a heap array): no allocation per stored text.
            bool isNA = IsNAValue(value);
            Span<byte> scratch = stackalloc byte[StackEncodeBytes];
            ReadOnlySpan<byte> utf8 = isNA ? default : EncodeForStore(AsPyString(value, Descr.VString.Coerce), scratch);
            Allocator.Acquire();
            try
            {
                PackValue(Address + index * VStringAllocator.CellSize, isNA, utf8);
            }
            finally
            {
                Allocator.Release();
            }
        }

        /// <summary>
        ///     Stores <paramref name="values"/> into elements 0 … <c>values.Count - 1</c>, each by setitem (see
        ///     <see cref="SetIndex(long, object)"/>), with the allocator locked ONCE for the whole run — <c>np.array(values,
        ///     dtype=StringDType)</c> filling a fresh buffer.
        /// </summary>
        /// <param name="values">The values, in element order (no more than <see cref="Count"/>).</param>
        /// <exception cref="ValueError">A value that is not text under <c>coerce=False</c>.</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8.</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>Failed to pack string during StringDType setitem</c>.</exception>
        /// <remarks>An error leaves the elements stored before it written (the caller discards the half-filled buffer, as
        ///     NumPy discards its half-built array).</remarks>
        [System.Runtime.CompilerServices.SkipLocalsInit]
        internal void SetRange(System.Collections.Generic.IReadOnlyList<object> values)
        {
            Span<byte> scratch = stackalloc byte[StackEncodeBytes];
            bool coerce = Descr.VString.Coerce;
            bool hasNA = Descr.VString.HasNAObject; // no NA object: no value can be NA, skip the comparison per element
            Allocator.Acquire();
            try
            {
                for (int i = 0; i < values.Count; i++)
                {
                    object value = values[i];
                    // The common element, a short ASCII string, written straight into its cell (what Pack gives for it).
                    if (value is string text && !(hasNA && IsNAValue(value))
                        && TryWriteShortAscii(Address + (long)i * VStringAllocator.CellSize, text))
                        continue;
                    bool isNA = hasNA && IsNAValue(value);
                    ReadOnlySpan<byte> utf8 = isNA ? default : EncodeForStore(AsPyString(value, coerce), scratch);
                    PackValue(Address + (long)i * VStringAllocator.CellSize, isNA, utf8);
                }
            }
            finally
            {
                Allocator.Release();
            }
        }

        /// <summary>The stack space an element store encodes its text into (a text whose UTF-8 form could be longer — up to
        ///     3 bytes per UTF-16 unit — uses a heap array instead).</summary>
        private const int StackEncodeBytes = 768;

        /// <summary>
        ///     Writes a non-empty ASCII text of at most 15 characters straight into a cell that holds no arena or heap text —
        ///     the short-string cell <c>NpyString_pack</c> produces there (<c>free</c> releases nothing, <c>newsize</c> sets the
        ///     flags and length, the bytes after the text are zero) without the UTF-8 transcoder and the allocator calls.
        /// </summary>
        /// <param name="cell">The cell (the allocator lock is held).</param>
        /// <param name="text">The text.</param>
        /// <returns>False, writing nothing, for an empty, longer or non-ASCII text or a cell that holds arena / heap text.</returns>
        private static bool TryWriteShortAscii(byte* cell, string text)
        {
            int length = text.Length;
            byte flags = cell[15];
            if (length == 0 || length > VStringAllocator.ShortMaxSize
                || (flags & VStringAllocator.Long) != 0 || (flags != 0 && (flags & VStringAllocator.OutsideArena) == 0)
                || !System.Text.Ascii.IsValid(text))
                return false;
            *(ulong*)cell = 0;
            *(ulong*)(cell + 8) = 0;
            for (int k = 0; k < length; k++)
                cell[k] = (byte)text[k];
            cell[15] = (byte)(VStringAllocator.Initialized | VStringAllocator.OutsideArena | length);
            return true;
        }

        /// <summary>The strict UTF-8 form of a stored text, in <paramref name="scratch"/> when it surely fits there.</summary>
        /// <param name="text">The text.</param>
        /// <param name="scratch">Stack space of <see cref="StackEncodeBytes"/> bytes.</param>
        /// <returns>The UTF-8 bytes (a slice of <paramref name="scratch"/>, or a new array).</returns>
        /// <exception cref="UnicodeEncodeError">A lone surrogate.</exception>
        private static ReadOnlySpan<byte> EncodeForStore(string text, Span<byte> scratch)
            => text.Length <= scratch.Length / 3 ? scratch.Slice(0, NDUtf8.Encode(text, scratch)) : NDUtf8.Encode(text);

        /// <summary>Element access by index (see <see cref="GetIndex(long)"/> / <see cref="SetIndex(long, object)"/>).</summary>
        /// <param name="index">The element index.</param>
        public object this[long index]
        {
            get => GetIndex(index);
            set => SetIndex(index, value);
        }

        /// <summary>Typed element access is meaningless for a <c>StringDType</c> cell; refused.</summary>
        /// <typeparam name="T">Any unmanaged type.</typeparam>
        /// <param name="index">Ignored.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always.</exception>
        public T GetIndex<T>(long index) where T : unmanaged
            => throw new NotSupportedException($"A {Descr} element is text; read it with GetIndex(long) (a string or the NA object), not as {typeof(T).Name}.");

        /// <summary>Typed element writes are meaningless for a <c>StringDType</c> cell; refused.</summary>
        /// <typeparam name="T">Any unmanaged type.</typeparam>
        /// <param name="index">Ignored.</param>
        /// <param name="value">Ignored.</param>
        /// <exception cref="NotSupportedException">Always.</exception>
        public void SetIndex<T>(long index, T value) where T : unmanaged
            => throw new NotSupportedException($"A {Descr} element is text; write it with SetIndex(long, object), not as {typeof(T).Name}.");

        /// <summary>
        ///     Writes <paramref name="value"/> into every element (NumPy's <c>fill</c>: the value is converted ONCE, then packed
        ///     into each cell — a long text gets its own allocation per element, as NumPy's duplicating copy gives it).
        /// </summary>
        /// <param name="value">The value (see <see cref="SetIndex(long, object)"/>); it is converted even when there are no
        ///     elements, as NumPy packs it into a scalar first.</param>
        /// <exception cref="ValueError">See <see cref="SetIndex(long, object)"/>.</exception>
        /// <exception cref="UnicodeDecodeError">See <see cref="SetIndex(long, object)"/>.</exception>
        /// <exception cref="UnicodeEncodeError">See <see cref="SetIndex(long, object)"/>.</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="SetIndex(long, object)"/>.</exception>
        public void Fill(object value)
        {
            bool isNA = IsNAValue(value);
            byte[] utf8 = isNA ? null : NDUtf8.Encode(AsPyString(value, Descr.VString.Coerce));
            if (Count == 0)
                return;
            Allocator.Acquire();
            try
            {
                for (long i = 0; i < Count; i++)
                    PackValue(Address + i * VStringAllocator.CellSize, isNA, utf8);
            }
            finally
            {
                Allocator.Release();
            }
        }

        /// <summary>Whether a value is this descriptor's NA object (NumPy's <c>na_eq_cmp(value, na_object)</c>; never true
        ///     without one).</summary>
        /// <param name="value">The value.</param>
        /// <returns>True when the value packs an NA cell.</returns>
        internal bool IsNAValue(object value)
        {
            var p = Descr.VString;
            return p.HasNAObject && NDPyObject.Equal(value, p.NAObject);
        }

        /// <summary>
        ///     NumPy's <c>as_pystring</c>: the Python <c>str</c> a value is stored as. A <see cref="string"/> (and a
        ///     <see cref="char"/>, the one-character <c>str</c> of every NumSharp string store) is itself; under
        ///     <c>coerce=False</c> anything else raises; otherwise <c>bytes</c> are decoded as strict UTF-8 and any other value
        ///     is its <c>str()</c> (<see cref="NDPyObject.Str"/>).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="coerce">The descriptor's <c>coerce</c>.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ValueError">A value that is not text under <c>coerce=False</c>.</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8.</exception>
        internal static string AsPyString(object value, bool coerce)
        {
            switch (value)
            {
                case string s:
                    return s;
                case char c:
                    return c.ToString();
            }
            if (!coerce)
                throw new ValueError(CoercionDisabledText);
            if (value is byte[] bytes)
                return NDUtf8.DecodeStrict(bytes); // assume UTF-8 encoding (PyUnicode_FromStringAndSize)
            return NDPyObject.Str(value);
        }

        /// <summary>Packs a converted value into a cell (the allocator lock must be held).</summary>
        /// <param name="cell">The cell.</param>
        /// <param name="isNA">True to pack an NA cell.</param>
        /// <param name="utf8">The UTF-8 text (ignored for an NA cell).</param>
        /// <exception cref="OutOfMemoryException">NumPy's setitem <c>MemoryError</c> texts.</exception>
        private void PackValue(byte* cell, bool isNA, ReadOnlySpan<byte> utf8)
        {
            if (isNA)
            {
                if (Allocator.PackNull(cell) < 0)
                    throw new OutOfMemoryException("Failed to pack null string during StringDType setitem");
                return;
            }
            fixed (byte* b = utf8)
            {
                if (Allocator.Pack(cell, b, utf8.Length) < 0)
                    throw new OutOfMemoryException("Failed to pack string during StringDType setitem");
            }
        }

        /// <summary>Decodes a cell's UTF-8 text (always valid: every write encoded it strictly).</summary>
        /// <param name="buf">The first byte.</param>
        /// <param name="size">The byte count.</param>
        /// <returns>The text.</returns>
        /// <exception cref="OutOfMemoryException">A text longer than a .NET string can be.</exception>
        private static string Decode(byte* buf, long size)
        {
            if (size > int.MaxValue)
                throw new OutOfMemoryException($"A StringDType element of {size} bytes is longer than a .NET string can be.");
            return NDUtf8.DecodeTrusted(new ReadOnlySpan<byte>(buf, (int)size));
        }

        // ---------------------------------------------------------------------------------------------
        // Slicing / cloning — element units
        // ---------------------------------------------------------------------------------------------

        /// <summary>The elements from <paramref name="start"/> to the end, sharing the cells and the allocator.</summary>
        /// <param name="start">The first element.</param>
        /// <returns>A slice over the same buffer.</returns>
        public IArraySlice Slice(long start) => Slice(start, Count - start);

        /// <summary><paramref name="count"/> elements from <paramref name="start"/>, sharing the cells and the allocator.</summary>
        /// <param name="start">The first element.</param>
        /// <param name="count">The number of elements.</param>
        /// <returns>A slice over the same buffer (a write through it is a write into this buffer).</returns>
        public IArraySlice Slice(long start, long count)
            => new VStringArraySlice(Cells.Slice(start * VStringAllocator.CellSize, count * VStringAllocator.CellSize), Allocator, Descr);

        /// <summary>
        ///     A deep copy: a new buffer with a new allocator and every string DUPLICATED into it (NumPy's <c>NpyString_dup</c>
        ///     per element, in order, so the copy's arena is compact); NA cells stay NA.
        /// </summary>
        /// <returns>The copy.</returns>
        /// <exception cref="OutOfMemoryException">NumPy's <c>Failed to allocate string in string to string cast</c>.</exception>
        public IArraySlice Clone()
        {
            var copy = Allocate(Descr, Count);
            try
            {
                VStringAllocator.AcquirePair(Allocator, copy.Allocator);
                try
                {
                    byte* src = Address, dst = copy.Address;
                    // The fresh copy takes every medium / long string at its arena's cursor: reserve the exact total once
                    // (VStringAllocator.TryReserve) instead of growing by 25 % steps.
                    if (Allocator.HoldsLongStrings)
                    {
                        long need = 0;
                        for (long i = 0; i < Count; i++)
                            need += VStringAllocator.ArenaBytesForDuplicate(src + i * VStringAllocator.CellSize);
                        copy.Allocator.TryReserve(need);
                    }
                    for (long i = 0; i < Count; i++)
                        VStringAllocator.FreeAndCopy(Allocator, copy.Allocator, src + i * VStringAllocator.CellSize,
                            dst + i * VStringAllocator.CellSize, "string to string cast");
                }
                finally
                {
                    VStringAllocator.ReleasePair(Allocator, copy.Allocator);
                }
            }
            catch
            {
                // The half-built copy owns heap strings already and nothing references it yet: free it now (its clear loop
                // frees them). A plain Release would be a stray decrement on a block no one has counted.
                copy.DangerousFree();
                throw;
            }
            return copy;
        }

        /// <inheritdoc/>
        object ICloneable.Clone() => Clone();

        /// <summary>A typed clone would copy cells away from their allocator; refused.</summary>
        /// <typeparam name="T">Any unmanaged type.</typeparam>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always: clone the slice itself (<see cref="Clone()"/>).</exception>
        public ArraySlice<T> Clone<T>() where T : unmanaged
            => throw new NotSupportedException($"A {Descr} buffer's cells refer to its own allocator; copy it with Clone(), not as raw {typeof(T).Name} values.");

        // ---------------------------------------------------------------------------------------------
        // Raw cell access (read-only bytes)
        // ---------------------------------------------------------------------------------------------

        /// <summary>The cells as raw bytes (NumPy's <c>tobytes()</c> of a <c>T</c> array); only <see cref="byte"/>.</summary>
        /// <typeparam name="T"><see cref="byte"/>.</typeparam>
        /// <returns>A span over the cells, no copy. Writing it into another buffer's cells is never valid.</returns>
        /// <exception cref="NotSupportedException">Any other <typeparamref name="T"/>.</exception>
        public UnmanagedSpan<T> AsSpan<T>() where T : unmanaged
        {
            CheckByte<T>();
            return new UnmanagedSpan<T>((T*)Cells.Address, Cells.Count);
        }

        /// <summary>Copies the raw cells into <paramref name="destination"/> (the destination's element type only sizes it).</summary>
        /// <typeparam name="T">Any element type; the destination must hold at least <see cref="BytesLength"/> bytes.</typeparam>
        /// <param name="destination">The target span.</param>
        /// <exception cref="ArgumentException"><paramref name="destination"/> is too short.</exception>
        public void CopyTo<T>(Span<T> destination)
        {
            long sourceBytes = Cells.Count;
            long destinationBytes = (long)destination.Length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
            if ((ulong)sourceBytes > (ulong)destinationBytes)
                throw new ArgumentException("Destination was too short.");
            Buffer.MemoryCopy(Cells.Address, System.Runtime.CompilerServices.Unsafe.AsPointer(ref destination.GetPinnableReference()), destinationBytes, sourceBytes);
        }

        /// <summary>Copies the raw cells into <paramref name="destination"/> (see <see cref="CopyTo{T}(Span{T})"/>).</summary>
        /// <typeparam name="T">Any unmanaged type; the destination must hold at least <see cref="BytesLength"/> bytes.</typeparam>
        /// <param name="destination">The target span.</param>
        /// <exception cref="ArgumentException"><paramref name="destination"/> is too short.</exception>
        public void CopyTo<T>(UnmanagedSpan<T> destination) where T : unmanaged
        {
            long sourceBytes = Cells.Count;
            long destinationBytes = destination.Length * sizeof(T);
            if ((ulong)sourceBytes > (ulong)destinationBytes)
                throw new ArgumentException("Destination was too short.");
            Buffer.MemoryCopy(Cells.Address, System.Runtime.CompilerServices.Unsafe.AsPointer(ref destination.GetPinnableReference()), destinationBytes, sourceBytes);
        }

        /// <summary>A reference to the first cell byte.</summary>
        /// <typeparam name="T"><see cref="byte"/>.</typeparam>
        /// <returns>The reference.</returns>
        /// <exception cref="NotSupportedException">Any other <typeparamref name="T"/>.</exception>
        public ref T GetPinnableReference<T>() where T : unmanaged
        {
            CheckByte<T>();
            return ref *(T*)Cells.Address;
        }

        /// <summary>Rejects every raw unit type but <see cref="byte"/>.</summary>
        /// <typeparam name="T">The requested unit type.</typeparam>
        /// <exception cref="NotSupportedException">Any other <typeparamref name="T"/>.</exception>
        private void CheckByte<T>() where T : unmanaged
        {
            if (typeof(T) != typeof(byte))
                throw new NotSupportedException($"A {Descr} buffer exposes its cells only as raw bytes, not as {typeof(T).Name}.");
        }

        /// <summary>
        ///     The elements as a managed array: <c>string[]</c> when the descriptor has no NA object, else <c>object[]</c>
        ///     (an NA cell is the NA object).
        /// </summary>
        /// <returns>The values in buffer order.</returns>
        /// <exception cref="OutOfMemoryException">See <see cref="GetIndex(long)"/>.</exception>
        public Array ToArray()
        {
            if (!HasNAObject)
            {
                var strings = new string[Count];
                for (long i = 0; i < Count; i++)
                    strings[i] = (string)GetIndex(i);
                return strings;
            }
            var values = new object[Count];
            for (long i = 0; i < Count; i++)
                values[i] = GetIndex(i);
            return values;
        }

        // ---------------------------------------------------------------------------------------------
        // Lifetime — forwarded to the cells' block (which owns the allocator)
        // ---------------------------------------------------------------------------------------------

        /// <inheritdoc/>
        public void DangerousFree() => Cells.DangerousFree();

        /// <inheritdoc/>
        public bool TryAddRef() => Cells.TryAddRef();

        /// <inheritdoc/>
        public void Release() => Cells.Release();

        /// <inheritdoc/>
        public void Abandon() => Cells.Abandon();

        /// <inheritdoc/>
        public bool IsReleased => Cells.IsReleased;

        /// <inheritdoc/>
        public bool IsUniquelyReferenced => Cells.IsUniquelyReferenced;

        /// <summary>Enumerates the elements as C# values (see <see cref="GetIndex(long)"/>).</summary>
        /// <returns>The enumerator.</returns>
        public IEnumerator GetEnumerator()
        {
            for (long i = 0; i < Count; i++)
                yield return GetIndex(i);
        }

        /// <inheritdoc/>
        public override string ToString() => $"VStringArraySlice({Descr}, Count={Count})";
    }
}
