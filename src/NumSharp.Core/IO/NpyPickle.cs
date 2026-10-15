using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.IO
{
    /// <summary>
    ///     The Python pickle NumPy 2.4.2 writes for a NEP 55 <c>StringDType</c> array in a <c>.npy</c> file (behind the
    ///     header <c>{'descr': '|O', ...}</c>), byte for byte, and a reader for exactly that pickle — the owner's decision D3
    ///     (<c>docs/plans/strings-phase1-completion.md</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why a pickle.</b> A <c>StringDType</c> cell holds text inside its buffer's allocator, so it has no flat byte
    ///     image; NumPy's <c>dtype_to_descr</c> therefore "lies that the dtype is object" and <c>write_array</c> runs
    ///     <c>pickle.dump(array, fp, protocol=4)</c>. The array reduces (<c>array_reduce</c>, methods.c) to
    ///     <c>numpy._core.multiarray._reconstruct(numpy.ndarray, (0,), b'b')</c> plus the state
    ///     <c>(1, shape, dtype, is_fortran, [every element as a Python str, an NA element as the na_object])</c>, and the dtype
    ///     to <c>numpy._core._internal._convert_to_stringdtype_kwargs(coerce[, na_object])</c>.
    ///     </para>
    ///     <para>
    ///     <b>Byte identity.</b> <see cref="Pickler"/> replays CPython 3.12's C pickler (<c>_pickle.c</c>) for that object
    ///     graph: protocol 4 framing (a frame is committed at the first <c>save()</c> that finds it at 64 KiB or more, a
    ///     frame under 4 bytes loses its header, a text of 64 KiB or more is written outside any frame), <c>MEMOIZE</c> for
    ///     every memoized object, and <c>BINGET</c> / <c>LONG_BINGET</c> where the SAME Python object comes back. The only
    ///     elements CPython hands out twice are its cached strings (<c>''</c> and the one-character strings below U+0100,
    ///     which <c>PyUnicode_FromStringAndSize</c> returns as singletons) and the descriptor's own na_object, so those are
    ///     the only memo hits modelled; every other text is a fresh <c>str</c> written in full.
    ///     </para>
    ///     <para>
    ///     <b>The reader is not an unpickler.</b> <see cref="ReadStringDTypeArray"/> runs the binary opcodes of protocols
    ///     2–5 over a stack of inert values (a global is a pair of names, a call is a record), then accepts only NumPy's
    ///     <c>StringDType</c> array graph and builds the array through <c>stringdtype_setitem</c>'s rules
    ///     (<see cref="VStringArraySlice.SetIndex(long, object)"/>), as NumPy's <c>_setlist_pkl</c> does. Nothing a pickle
    ///     names is ever executed, so loading such a file cannot run code; any other pickle (an object array, a subclass, a
    ///     foreign missing-data object) is refused by name.
    ///     </para>
    ///     <para>
    ///     <b>No string per element.</b> The reader keeps the whole pickle in one unmanaged buffer and every text as an
    ///     offset into it; a text element's UTF-8 bytes (validated as strict UTF-8 when the opcode was read) are packed into
    ///     its cell as they are. Loading a million elements therefore allocates no million strings, decodes and re-encodes
    ///     nothing, and leaves the garbage collector nothing to promote.
    ///     </para>
    /// </remarks>
    internal static unsafe class NpyPickle
    {
        // ---------------------------------------------------------------------------------------------------------------
        // Opcodes (CPython's pickletools), the subset NumPy's StringDType pickle and the binary protocols use.
        // ---------------------------------------------------------------------------------------------------------------

        private const byte MARK = 0x28, STOP = 0x2E, POP = 0x30, POP_MARK = 0x31, DUP = 0x32, BINFLOAT = 0x47,
            BINBYTES = 0x42, SHORT_BINBYTES = 0x43, BININT = 0x4A, BININT1 = 0x4B, BININT2 = 0x4D, NONE = 0x4E,
            REDUCE = 0x52, BINUNICODE = 0x58, EMPTY_LIST = 0x5D, APPEND = 0x61, BUILD = 0x62, GLOBAL = 0x63,
            APPENDS = 0x65, BINGET = 0x68, LONG_BINGET = 0x6A, LIST = 0x6C, BINPUT = 0x71, LONG_BINPUT = 0x72,
            TUPLE = 0x74, EMPTY_TUPLE = 0x29, PROTO = 0x80, TUPLE1 = 0x85, TUPLE2 = 0x86, TUPLE3 = 0x87, NEWTRUE = 0x88,
            NEWFALSE = 0x89, LONG1 = 0x8A, LONG4 = 0x8B, SHORT_BINUNICODE = 0x8C, BINUNICODE8 = 0x8D, BINBYTES8 = 0x8E,
            STACK_GLOBAL = 0x93, MEMOIZE = 0x94, FRAME = 0x95;

        /// <summary>The module NumPy 2.x names <c>_reconstruct</c> under (NumPy 1.x's spelling is accepted on read).</summary>
        private const string ReconstructModule = "numpy._core.multiarray";

        /// <summary>The module of <c>_convert_to_stringdtype_kwargs</c>, the reducer of a <c>StringDType</c> descriptor.</summary>
        private const string InternalModule = "numpy._core._internal";

        /// <summary>The refusal for every pickle that is not NumPy's pickle of a <c>StringDType</c> array.</summary>
        private const string OnlyStringDType =
            "NumSharp reads only the pickle NumPy writes for a StringDType array (numpy.save of a StringDType array); ";

        // ---------------------------------------------------------------------------------------------------------------
        // Writing
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Writes NumPy's protocol-4 pickle of <paramref name="array"/> — the bytes <c>pickle.dump(array, fp,
        ///     protocol=4)</c> produces in NumPy 2.4.2 — to <paramref name="stream"/> at its current position.
        /// </summary>
        /// <param name="stream">An open, writable stream (the <c>.npy</c> header is already written).</param>
        /// <param name="array">A <c>StringDType</c> array of any layout; its elements are pickled in C (logical) order, as
        ///     NumPy's <c>_getlist_pkl</c> walks them, and <c>is_fortran</c> is set for an F-contiguous-and-not-C array.</param>
        /// <exception cref="ArgumentException"><paramref name="array"/> is not a <c>StringDType</c> array.</exception>
        /// <exception cref="NotSupportedException">The descriptor's missing-data object has no pickle form NumSharp writes
        ///     (only <c>None</c>, a float, a string, a bool and a Python int do).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt cell (NumPy's <c>Failed to load string in StringDType getitem</c>).</exception>
        internal static void WriteStringDTypeArray(Stream stream, NDArray array)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            if (array is null) throw new ArgumentNullException(nameof(array));
            if (!array.Storage.IsVString)
                throw new ArgumentException($"{array.dtype} is not a StringDType array.", nameof(array));

            DType descr = array.dtype;
            VStringParameters p = descr.VString;
            if (p.HasNAObject)
                CheckPicklableNA(p.na_object);

            // is_fortran is PyArray_ISFORTRAN of the array itself (not of the C copy the walk reads).
            bool isFortran = !array.Shape.IsContiguous && array.Shape.IsFContiguous;
            // The elements go out in C logical order (PyArray_IterNew); a C-contiguous source is walked in place, any other
            // layout through a C-contiguous copy of the same values (the copy duplicates the strings into its own allocator).
            NDArray src = array.Shape.IsContiguous ? array : array.copy('C');
            try
            {
                var pickler = new Pickler(stream);
                pickler.Dump(src, descr, p, isFortran);
            }
            finally
            {
                if (!ReferenceEquals(src, array))
                    src.Dispose();
            }
        }

        /// <summary>
        ///     Refuses a missing-data object that has no Python pickle form this writer emits. NumPy pickles any object; the
        ///     ones a C# descriptor can hold and that pickle as plain atoms or strings are <c>None</c> (<see langword="null"/>),
        ///     a float (<see cref="double"/>, <see cref="float"/>), a <c>str</c>, a <c>bool</c> and a Python <c>int</c>.
        /// </summary>
        /// <param name="na">The descriptor's missing-data object.</param>
        /// <exception cref="NotSupportedException">Any other object (a <see cref="Half"/> is NumPy's <c>np.float16</c> scalar,
        ///     whose pickle names <c>numpy._core.multiarray.scalar</c>; a <see cref="decimal"/> has no NumPy analog).</exception>
        private static void CheckPicklableNA(object na)
        {
            switch (na)
            {
                case null:
                case double:
                case float:
                case string:
                case bool:
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                    return;
                default:
                    throw new NotSupportedException(
                        $"A StringDType missing-data object of type {na.GetType().Name} has no pickle form NumSharp writes; " +
                        "np.save writes None, a float, a string, a bool or an integer as the na_object.");
            }
        }

        /// <summary>
        ///     CPython 3.12's C pickler (<c>_pickle.c</c>) for protocol 4, reduced to the calls NumPy's <c>StringDType</c>
        ///     array graph makes: the output buffer with its open frame, the framing rules, the memo and the opcodes.
        /// </summary>
        /// <remarks>
        ///     Every public <c>Save*</c> method starts with <see cref="Boundary"/>, because CPython's <c>save()</c> starts with
        ///     <c>_Pickler_OpcodeBoundary</c> for every object it saves (atoms included): that is where a frame of 64 KiB or
        ///     more is committed, so the frame sizes depend on exactly which saves happen in which order.
        /// </remarks>
        private sealed class Pickler
        {
            /// <summary><c>FRAME_HEADER_SIZE</c>: the <c>FRAME</c> opcode and its 8-byte length.</summary>
            private const int FrameHeaderSize = 9;

            /// <summary><c>FRAME_SIZE_TARGET</c>: a frame is committed once it holds this many bytes, and a text or bytes
            ///     payload at least this long is written outside any frame.</summary>
            private const int FrameSizeTarget = 64 * 1024;

            /// <summary><c>FRAME_SIZE_MIN</c>: a committed frame shorter than this loses its header.</summary>
            private const int FrameSizeMin = 4;

            /// <summary><c>BATCHSIZE</c>: list items are appended in <c>MARK … APPENDS</c> batches of this many.</summary>
            private const int BatchSize = 1000;

            /// <summary>The cache key of a text no other save can meet again (a fresh <c>str</c>).</summary>
            private const int NoKey = int.MinValue;

            /// <summary>The cache key of the descriptor's string missing-data object (when it is not a cached string).</summary>
            private const int NAKey = int.MaxValue;

            /// <summary>The cache key of CPython's empty string.</summary>
            private const int EmptyKey = -1;

            private readonly Stream _stream;

            /// <summary>The output not yet written to the stream: committed frames and the open frame.</summary>
            private byte[] _buffer = new byte[8192];

            private int _length;

            /// <summary>Where the open frame's header starts in <see cref="_buffer"/>, or -1 (no open frame).</summary>
            private int _frameStart = -1;

            /// <summary>CPython's <c>self->framing</c>: off for <c>PROTO</c> and while a large payload bypasses the frames.</summary>
            private bool _framing;

            /// <summary>The memo's size: the index the next <c>MEMOIZE</c> gives its object.</summary>
            private int _memoSize;

            /// <summary>The memo index of each object a later save can meet again, by cache key (CPython's cached strings
            ///     by code point, <see cref="EmptyKey"/> for <c>''</c>, <see cref="NAKey"/> for the string na_object).</summary>
            private readonly Dictionary<int, int> _memo = new Dictionary<int, int>();

            /// <summary>Starts a pickler that writes to <paramref name="stream"/>.</summary>
            /// <param name="stream">The output stream.</param>
            internal Pickler(Stream stream) => _stream = stream;

            /// <summary>
            ///     <c>dump()</c> of the array: <c>PROTO 4</c>, the array's reduce (<c>_reconstruct</c>, its arguments, the
            ///     state with the descriptor's reduce and the element list), <c>STOP</c>, and the last frame committed.
            /// </summary>
            /// <param name="src">The elements, C-contiguous.</param>
            /// <param name="descr">The array's descriptor.</param>
            /// <param name="p">The descriptor's parameters.</param>
            /// <param name="isFortran">The state's <c>is_fortran</c> flag.</param>
            internal void Dump(NDArray src, DType descr, VStringParameters p, bool isFortran)
            {
                // PROTO is written before framing starts (dump() turns framing on after it), so it stays outside every frame.
                WriteByte(PROTO);
                WriteByte(4);
                _framing = true;

                // save(array) -> save_reduce: save(callable), save(args), REDUCE, memoize the array.
                Boundary();
                SaveGlobal(ReconstructModule, "_reconstruct");
                Boundary();
                SaveGlobal("numpy", "ndarray");                      // args[0]: the type (tuple elements: one save each)
                Boundary();
                SaveLong(0);                                          // args[1]: (0,)
                WriteByte(TUPLE1);
                MemoPut();
                Boundary();
                WriteByte(SHORT_BINBYTES);                            // args[2]: b'b', the dummy typecode
                WriteByte(1);
                WriteByte((byte)'b');
                MemoPut();
                WriteByte(TUPLE3);
                MemoPut();
                WriteByte(REDUCE);
                MemoPut();                                            // the array itself

                // save(state): a 5-tuple, so MARK ... TUPLE; then BUILD.
                Boundary();
                WriteByte(MARK);
                SaveLong(1);                                          // the pickle version
                SaveShape(src.Shape.dimensions);
                SaveDescr(p);
                SaveBool(isFortran);
                SaveElements(src, p);
                WriteByte(TUPLE);
                MemoPut();
                WriteByte(BUILD);

                WriteByte(STOP);
                CommitFrame();
                Flush();
            }

            /// <summary>
            ///     <c>save()</c> of the shape tuple: <c>()</c> is <c>EMPTY_TUPLE</c> (never memoized), one to three dimensions
            ///     take <c>TUPLE1</c>–<c>TUPLE3</c>, more take <c>MARK … TUPLE</c>; a non-empty tuple is memoized.
            /// </summary>
            /// <param name="dims">The dimensions.</param>
            private void SaveShape(long[] dims)
            {
                Boundary();
                if (dims.Length == 0)
                {
                    WriteByte(EMPTY_TUPLE);
                    return;
                }
                if (dims.Length > 3)
                    WriteByte(MARK);
                foreach (long d in dims)
                    SaveLong(d);
                WriteByte(dims.Length switch { 1 => TUPLE1, 2 => TUPLE2, 3 => TUPLE3, _ => TUPLE });
                MemoPut();
            }

            /// <summary>
            ///     <c>save()</c> of the descriptor: its reduce <c>(_convert_to_stringdtype_kwargs, (coerce[, na_object]))</c>,
            ///     <c>coerce</c> as the C int NumPy's <c>Py_BuildValue("(iO)")</c> makes, then the descriptor memoized.
            /// </summary>
            /// <param name="p">The descriptor's parameters.</param>
            private void SaveDescr(VStringParameters p)
            {
                Boundary();
                Boundary();
                SaveGlobal(InternalModule, "_convert_to_stringdtype_kwargs");
                Boundary();
                SaveLong(p.coerce ? 1 : 0);
                if (p.HasNAObject)
                {
                    SaveNA(p.na_object);
                    WriteByte(TUPLE2);
                }
                else
                {
                    WriteByte(TUPLE1);
                }
                MemoPut();
                WriteByte(REDUCE);
                MemoPut();
            }

            /// <summary>
            ///     <c>save()</c> of the element list: <c>EMPTY_LIST</c> memoized, then one item and <c>APPEND</c> for a
            ///     one-element list, or <c>MARK</c> + up to 1000 items + <c>APPENDS</c> per batch (CPython's
            ///     <c>batch_list_exact</c>: a last batch of one still uses <c>MARK … APPENDS</c>).
            /// </summary>
            /// <param name="src">The C-contiguous elements.</param>
            /// <param name="p">The descriptor's parameters (for an NA element's object).</param>
            private void SaveElements(NDArray src, VStringParameters p)
            {
                Boundary();
                WriteByte(EMPTY_LIST);
                MemoPut();
                long n = src.size;
                if (n == 0)
                    return;
                var slice = (VStringArraySlice)src.Storage.InternalArray;
                byte* cells = (byte*)src.Storage.Address + src.Shape.offset * (long)VStringAllocator.CellSize;
                if (n == 1)
                {
                    SaveElement(slice, cells, p);
                    WriteByte(APPEND);
                    return;
                }
                long done = 0;
                while (done < n)
                {
                    WriteByte(MARK);
                    long batchEnd = Math.Min(n, done + BatchSize);
                    for (; done < batchEnd; done++)
                        SaveElement(slice, cells + done * VStringAllocator.CellSize, p);
                    WriteByte(APPENDS);
                }
            }

            /// <summary>
            ///     <c>save()</c> of one element's Python object: the na_object for an NA cell, else the text — a cached string
            ///     (<c>''</c>, a one-character string below U+0100) or the string na_object meets its memo entry again
            ///     (<c>BINGET</c>), any other text is a fresh <c>str</c> written in full and memoized.
            /// </summary>
            /// <param name="slice">The cells' slice (for its allocator).</param>
            /// <param name="cell">The cell.</param>
            /// <param name="p">The descriptor's parameters.</param>
            /// <exception cref="OutOfMemoryException">A corrupt cell (NumPy's getitem text).</exception>
            private void SaveElement(VStringArraySlice slice, byte* cell, VStringParameters p)
            {
                Boundary();
                byte[] large = null;
                VStringAllocator allocator = slice.Allocator;
                // The text is read under the allocator's lock (an arena or heap pointer is valid only while it is held);
                // writes into the in-memory buffer never touch the stream, so they happen under the lock too. A payload of
                // 64 KiB or more is copied out and written after the lock is released (it goes straight to the stream).
                allocator.Acquire();
                try
                {
                    int isNull = allocator.Load(cell, out byte* buf, out long size);
                    if (isNull < 0)
                        throw new OutOfMemoryException("Failed to load string in StringDType getitem");
                    if (isNull == 1)
                    {
                        // A descriptor without an NA object reads an NA cell as '' (stringdtype_getitem).
                        if (p.HasNAObject)
                            SaveNAWithoutBoundary(p.na_object);
                        else
                            SaveTextWithoutBoundary(null, 0, EmptyKey);
                        return;
                    }
                    int key = CacheKey(buf, size);
                    if (size >= FrameSizeTarget && !(key != NoKey && _memo.ContainsKey(key)))
                    {
                        if (size > int.MaxValue)
                            throw new NotSupportedException("A StringDType element of 2 GiB or more cannot be pickled by NumSharp.");
                        large = new ReadOnlySpan<byte>(buf, (int)size).ToArray();
                    }
                    else
                    {
                        SaveTextWithoutBoundary(buf, size, key);
                    }
                }
                finally
                {
                    allocator.Release();
                }
                if (large != null)
                    SaveTextWithoutBoundary(large, NoKey);
            }

            /// <summary>
            ///     The cache key of an element's text: CPython returns its cached object for <c>''</c> and for a one-character
            ///     string below U+0100 (one ASCII byte, or the two UTF-8 bytes <c>C2 xx</c> / <c>C3 xx</c>), a fresh one otherwise.
            /// </summary>
            /// <param name="buf">The UTF-8 bytes.</param>
            /// <param name="size">Their number.</param>
            /// <returns><see cref="EmptyKey"/>, the code point, or <see cref="NoKey"/>.</returns>
            private static int CacheKey(byte* buf, long size)
            {
                if (size == 0)
                    return EmptyKey;
                if (size == 1 && buf[0] < 0x80)
                    return buf[0];
                if (size == 2 && (buf[0] == 0xC2 || buf[0] == 0xC3))
                    return ((buf[0] & 0x1F) << 6) | (buf[1] & 0x3F);
                return NoKey;
            }

            /// <summary>The cache key of a C# string as the Python object it stands for (see <see cref="CacheKey"/>); the
            ///     string missing-data object keeps <see cref="NAKey"/> unless it is a cached string.</summary>
            /// <param name="s">The string.</param>
            /// <returns>Its key.</returns>
            private static int NAStringKey(string s)
            {
                if (s.Length == 0)
                    return EmptyKey;
                if (s.Length == 1 && s[0] < 0x100)
                    return s[0];
                return NAKey;
            }

            /// <summary><c>save()</c> of the missing-data object (in the descriptor's reduce).</summary>
            /// <param name="na">The object (checked by <see cref="CheckPicklableNA"/>).</param>
            private void SaveNA(object na)
            {
                Boundary();
                SaveNAWithoutBoundary(na);
            }

            /// <summary>The missing-data object's opcodes: <c>NONE</c>, <c>BINFLOAT</c>, a bool, an int, or the string (its
            ///     first save memoizes it, every later save — an NA element — is a <c>BINGET</c>).</summary>
            /// <param name="na">The object.</param>
            private void SaveNAWithoutBoundary(object na)
            {
                switch (na)
                {
                    case null:
                        WriteByte(NONE);
                        return;
                    case double d:
                        WriteFloat(d);
                        return;
                    case float f:
                        WriteFloat(f);
                        return;
                    case bool b:
                        WriteByte(b ? NEWTRUE : NEWFALSE);
                        return;
                    case string s:
                        byte[] utf8 = NDUtf8.Encode(s);
                        SaveTextWithoutBoundary(utf8, NAStringKey(s));
                        return;
                    case BigInteger big:
                        WriteLong(big);
                        return;
                    case ulong u:
                        WriteLong(new BigInteger(u));
                        return;
                    default:
                        WriteLong(new BigInteger(Convert.ToInt64(na, System.Globalization.CultureInfo.InvariantCulture)));
                        return;
                }
            }

            /// <summary>A text from managed bytes (the string na_object, or a large element copied out of its cell).</summary>
            /// <param name="utf8">The UTF-8 bytes.</param>
            /// <param name="key">Its cache key.</param>
            private void SaveTextWithoutBoundary(byte[] utf8, int key)
            {
                fixed (byte* p = utf8)
                    SaveTextWithoutBoundary(p, utf8.Length, key);
            }

            /// <summary>
            ///     <c>save()</c> of a <c>str</c> after the boundary check: a memo hit is <c>BINGET</c> / <c>LONG_BINGET</c>;
            ///     otherwise <c>write_unicode_binary</c> (<c>SHORT_BINUNICODE</c> up to 255 bytes, <c>BINUNICODE</c> up to
            ///     <c>2^32 - 1</c>, else <c>BINUNICODE8</c>) and <c>MEMOIZE</c>, the index recorded for a key a later save can meet.
            /// </summary>
            /// <param name="utf8">The UTF-8 bytes (may be null when <paramref name="size"/> is 0).</param>
            /// <param name="size">Their number.</param>
            /// <param name="key">The cache key (<see cref="NoKey"/> for a fresh object).</param>
            private void SaveTextWithoutBoundary(byte* utf8, long size, int key)
            {
                if (key != NoKey && _memo.TryGetValue(key, out int index))
                {
                    MemoGet(index);
                    return;
                }
                Span<byte> header = stackalloc byte[9];
                int headerLength;
                if (size <= 0xFF)
                {
                    header[0] = SHORT_BINUNICODE;
                    header[1] = (byte)size;
                    headerLength = 2;
                }
                else if ((ulong)size <= 0xFFFFFFFFUL)
                {
                    header[0] = BINUNICODE;
                    BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(1), (uint)size);
                    headerLength = 5;
                }
                else
                {
                    header[0] = BINUNICODE8;
                    BinaryPrimitives.WriteUInt64LittleEndian(header.Slice(1), (ulong)size);
                    headerLength = 9;
                }
                WriteBytesPayload(header.Slice(0, headerLength), new ReadOnlySpan<byte>(utf8, checked((int)size)));
                int memo = MemoPut();
                if (key != NoKey)
                    _memo[key] = memo;
            }

            /// <summary>
            ///     <c>save_global()</c> for protocol 4: the module name and the qualified name as memoized strings (fresh
            ///     objects: no memo hits), <c>STACK_GLOBAL</c>, and the global memoized.
            /// </summary>
            /// <param name="module">The module name (ASCII).</param>
            /// <param name="name">The qualified name (ASCII).</param>
            private void SaveGlobal(string module, string name)
            {
                Boundary();
                byte[] m = Encoding.ASCII.GetBytes(module);
                SaveTextWithoutBoundary(m, NoKey);
                Boundary();
                byte[] n = Encoding.ASCII.GetBytes(name);
                SaveTextWithoutBoundary(n, NoKey);
                WriteByte(STACK_GLOBAL);
                MemoPut();
            }

            /// <summary><c>save()</c> of a Python int (an atom: never memoized).</summary>
            /// <param name="value">The value.</param>
            private void SaveLong(long value)
            {
                Boundary();
                WriteLong(value);
            }

            /// <summary><c>save()</c> of a Python bool: <c>NEWTRUE</c> / <c>NEWFALSE</c> (an atom).</summary>
            /// <param name="value">The value.</param>
            private void SaveBool(bool value)
            {
                Boundary();
                WriteByte(value ? NEWTRUE : NEWFALSE);
            }

            /// <summary>
            ///     <c>save_long()</c>'s opcodes: <c>BININT1</c> / <c>BININT2</c> / <c>BININT</c> for a value in the signed
            ///     32-bit range (chosen by its little-endian bytes: <c>BININT</c> when byte 2 or 3 is set, which includes every
            ///     negative value), <c>LONG1</c> / <c>LONG4</c> with the minimal two's-complement bytes otherwise.
            /// </summary>
            /// <param name="value">The value.</param>
            private void WriteLong(BigInteger value)
            {
                if (value >= int.MinValue && value <= int.MaxValue)
                {
                    int v = (int)value;
                    Span<byte> b = stackalloc byte[5];
                    BinaryPrimitives.WriteInt32LittleEndian(b.Slice(1), v);
                    if (b[4] != 0 || b[3] != 0)
                    {
                        b[0] = BININT;
                        Write(b);
                    }
                    else if (b[2] != 0)
                    {
                        b[0] = BININT2;
                        Write(b.Slice(0, 3));
                    }
                    else
                    {
                        b[0] = BININT1;
                        Write(b.Slice(0, 2));
                    }
                    return;
                }
                // CPython takes (bits / 8) + 1 bytes and drops a redundant 0xFF sign byte of a negative value: the minimal
                // two's-complement form, which BigInteger.ToByteArray also gives.
                byte[] bytes = value.ToByteArray(isUnsigned: false, isBigEndian: false);
                if (bytes.Length < 256)
                {
                    WriteByte(LONG1);
                    WriteByte((byte)bytes.Length);
                }
                else
                {
                    Span<byte> h = stackalloc byte[5];
                    h[0] = LONG4;
                    BinaryPrimitives.WriteInt32LittleEndian(h.Slice(1), bytes.Length);
                    Write(h);
                }
                Write(bytes);
            }

            /// <summary><c>save_float()</c>: <c>BINFLOAT</c> and the double's 8 bytes big-endian (its bits as given, a NaN's
            ///     sign and payload included).</summary>
            /// <param name="value">The value.</param>
            private void WriteFloat(double value)
            {
                Span<byte> b = stackalloc byte[9];
                b[0] = BINFLOAT;
                BinaryPrimitives.WriteInt64BigEndian(b.Slice(1), BitConverter.DoubleToInt64Bits(value));
                Write(b);
            }

            /// <summary><c>memo_put()</c> for protocol 4: <c>MEMOIZE</c>, the object taking the next index.</summary>
            /// <returns>The index.</returns>
            private int MemoPut()
            {
                WriteByte(MEMOIZE);
                return _memoSize++;
            }

            /// <summary><c>memo_get()</c>: <c>BINGET</c> for an index below 256, else <c>LONG_BINGET</c>.</summary>
            /// <param name="index">The memo index.</param>
            private void MemoGet(int index)
            {
                if (index < 256)
                {
                    WriteByte(BINGET);
                    WriteByte((byte)index);
                    return;
                }
                Span<byte> b = stackalloc byte[5];
                b[0] = LONG_BINGET;
                BinaryPrimitives.WriteInt32LittleEndian(b.Slice(1), index);
                Write(b);
            }

            /// <summary>
            ///     <c>_Pickler_write_bytes()</c>: a payload of 64 KiB or more commits the open frame and is written, with its
            ///     header, outside any frame (framing is switched off for it and back on after, so the next write opens a new
            ///     frame); a smaller one is an ordinary write.
            /// </summary>
            /// <param name="header">The opcode and length.</param>
            /// <param name="data">The payload.</param>
            private void WriteBytesPayload(ReadOnlySpan<byte> header, ReadOnlySpan<byte> data)
            {
                if (data.Length < FrameSizeTarget)
                {
                    Write(header);
                    Write(data);
                    return;
                }
                CommitFrame();
                bool framing = _framing;
                _framing = false;
                Write(header);
                Flush();
                _stream.Write(data);
                _framing = framing;
            }

            /// <summary>
            ///     <c>_Pickler_OpcodeBoundary()</c>: commits the open frame once it holds <see cref="FrameSizeTarget"/> bytes
            ///     or more, and writes the committed output to the stream (CPython flushes it to the file at this point too).
            /// </summary>
            private void Boundary()
            {
                if (!_framing || _frameStart == -1)
                    return;
                if (_length - _frameStart - FrameHeaderSize >= FrameSizeTarget)
                {
                    CommitFrame();
                    Flush();
                }
            }

            /// <summary>
            ///     <c>_Pickler_CommitFrame()</c>: fills the open frame's header with <c>FRAME</c> and its length, or removes the
            ///     header when the frame holds fewer than <see cref="FrameSizeMin"/> bytes; no frame is open afterwards.
            /// </summary>
            private void CommitFrame()
            {
                if (!_framing || _frameStart == -1)
                    return;
                int frameLength = _length - _frameStart - FrameHeaderSize;
                if (frameLength >= FrameSizeMin)
                {
                    _buffer[_frameStart] = FRAME;
                    BinaryPrimitives.WriteUInt64LittleEndian(_buffer.AsSpan(_frameStart + 1), (ulong)frameLength);
                }
                else
                {
                    Buffer.BlockCopy(_buffer, _frameStart + FrameHeaderSize, _buffer, _frameStart, frameLength);
                    _length -= FrameHeaderSize;
                }
                _frameStart = -1;
            }

            /// <summary>Writes the buffered output to the stream; only called with no frame open (a header waits to be filled).</summary>
            private void Flush()
            {
                if (_length == 0)
                    return;
                _stream.Write(_buffer, 0, _length);
                _length = 0;
            }

            /// <summary>One opcode byte (see <see cref="Write"/>).</summary>
            /// <param name="b">The byte.</param>
            private void WriteByte(byte b)
            {
                Span<byte> one = stackalloc byte[1];
                one[0] = b;
                Write(one);
            }

            /// <summary>
            ///     <c>_Pickler_Write()</c>: opens a frame first (9 placeholder bytes) when framing is on and no frame is open,
            ///     then appends. It never writes to the stream, so it is safe under an allocator lock.
            /// </summary>
            /// <param name="data">The bytes.</param>
            private void Write(ReadOnlySpan<byte> data)
            {
                bool newFrame = _framing && _frameStart == -1;
                int need = data.Length + (newFrame ? FrameHeaderSize : 0);
                if (_length + need > _buffer.Length)
                    Array.Resize(ref _buffer, Math.Max(_buffer.Length * 2, _length + need));
                if (newFrame)
                {
                    _frameStart = _length;
                    _length += FrameHeaderSize;
                }
                data.CopyTo(_buffer.AsSpan(_length));
                _length += data.Length;
            }
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Reading
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>The pickle's <c>None</c> (a C# null on the stack would be ambiguous with "no value").</summary>
        private sealed class PyNoneValue
        {
            internal static readonly PyNoneValue Instance = new PyNoneValue();
        }

        /// <summary>A pickled global: the names only, never resolved or executed.</summary>
        private sealed class PyGlobalRef
        {
            internal PyGlobalRef(string module, string name)
            {
                Module = module;
                Name = name;
            }

            internal string Module { get; }
            internal string Name { get; }

            public override string ToString() => $"{Module}.{Name}";
        }

        /// <summary>A pickled tuple.</summary>
        private sealed class PyTupleValue
        {
            internal PyTupleValue(object[] items) => Items = items;
            internal object[] Items { get; }
        }

        /// <summary>
        ///     One value on the reader's stack, in its memo or in a pickled list: a text that stays in the reader's pickle
        ///     buffer (its UTF-8 bytes at <see cref="Offset"/>, <see cref="Length"/> long — no C# string is made for it), or any
        ///     other value as an object (a boxed number, <see cref="PyNoneValue"/>, a tuple, a call, a list, bytes).
        /// </summary>
        /// <remarks>
        ///     A pickle of a million-element <c>StringDType</c> array pushes, memoizes and appends a million texts. As offsets,
        ///     those three steps are struct copies: no string per element, no decode / encode round trip (the bytes go into
        ///     the cells as they are), and nothing for the garbage collector to promote while the memo keeps every element
        ///     reachable. Measured on one million elements: 341 ms with a string per element, NumPy 230 ms. The
        ///     <see langword="default"/> value (a null <see cref="Obj"/>) is an empty memo slot.
        /// </remarks>
        private readonly struct PyValue
        {
            /// <summary>The tag object of a text value, so that <see langword="default"/> stays distinct (an empty memo slot).</summary>
            private static readonly object TextTag = new object();

            /// <summary>The value, or <see cref="TextTag"/> for a text, or null for an empty memo slot.</summary>
            internal readonly object Obj;

            /// <summary>A text's first byte in the pickle buffer.</summary>
            internal readonly long Offset;

            /// <summary>A text's UTF-8 byte count.</summary>
            internal readonly int Length;

            private PyValue(object obj, long offset, int length)
            {
                Obj = obj;
                Offset = offset;
                Length = length;
            }

            /// <summary>A non-text value.</summary>
            /// <param name="obj">The value (never null: <c>None</c> is <see cref="PyNoneValue.Instance"/>).</param>
            /// <returns>The reader value.</returns>
            internal static PyValue Of(object obj) => new PyValue(obj, 0, 0);

            /// <summary>A text in the pickle buffer.</summary>
            /// <param name="offset">Its first byte in the buffer.</param>
            /// <param name="length">Its UTF-8 byte count.</param>
            /// <returns>The reader value.</returns>
            internal static PyValue Text(long offset, int length) => new PyValue(TextTag, offset, length);

            /// <summary>True for a text (its bytes are in the pickle buffer).</summary>
            internal bool IsText => ReferenceEquals(Obj, TextTag);

            /// <summary>True for a memo slot no <c>MEMOIZE</c> / <c>BINPUT</c> has filled.</summary>
            internal bool IsEmpty => Obj == null;
        }

        /// <summary>A pickled list: its items as reader values, so the texts of the element list stay in the pickle buffer.</summary>
        private sealed class PyList
        {
            /// <summary>The items (the first <see cref="Count"/> slots).</summary>
            internal PyValue[] Items = new PyValue[16];

            /// <summary>The item count.</summary>
            internal int Count;

            /// <summary>Appends one item (<c>APPEND</c>).</summary>
            /// <param name="value">The item.</param>
            internal void Add(PyValue value)
            {
                if (Count == Items.Length)
                    Grow(Count + 1);
                Items[Count++] = value;
            }

            /// <summary>Appends <paramref name="count"/> items from a stack slice (<c>APPENDS</c>, <c>LIST</c>).</summary>
            /// <param name="source">The stack.</param>
            /// <param name="start">The first item's index in <paramref name="source"/>.</param>
            /// <param name="count">The item count.</param>
            internal void AddRange(PyValue[] source, int start, int count)
            {
                long needed = (long)Count + count;
                if (needed > Items.Length)
                    Grow(needed);
                Array.Copy(source, start, Items, Count, count);
                Count += count;
            }

            /// <summary>Doubles the capacity until it holds <paramref name="needed"/> items.</summary>
            /// <param name="needed">The required capacity.</param>
            /// <exception cref="FormatException">More items than a .NET array can hold.</exception>
            private void Grow(long needed)
            {
                if (needed > Array.MaxLength)
                    throw new FormatException($"A pickled list of {needed} items is longer than NumSharp can read.");
                int capacity = (int)Math.Min(Array.MaxLength, Math.Max(needed, (long)Items.Length * 2));
                Array.Resize(ref Items, capacity);
            }
        }

        /// <summary>A pickled call (<c>REDUCE</c>), with the state a later <c>BUILD</c> gives it; never executed.</summary>
        private sealed class PyCall
        {
            internal PyCall(object callable, object args)
            {
                Callable = callable;
                Args = args;
            }

            internal object Callable { get; }
            internal object Args { get; }
            internal object State { get; set; }
        }

        /// <summary>
        ///     Reads NumPy's pickle of a <c>StringDType</c> array from <paramref name="stream"/> (positioned at the pickle,
        ///     protocol 2–5) and builds the array: the shape and <c>is_fortran</c> layout of the pickle (NumPy ignores the
        ///     <c>.npy</c> header's for an object array), the descriptor from <c>_convert_to_stringdtype_kwargs(coerce[,
        ///     na_object])</c>, and the elements stored by setitem in C logical order (<c>_setlist_pkl</c>): an element equal to
        ///     the na_object becomes an NA value, a text is stored as itself.
        /// </summary>
        /// <param name="stream">The stream; read exactly up to the pickle's <c>STOP</c> (frames are read whole, as their
        ///     lengths say), so arrays saved one after another into one stream load one after another.</param>
        /// <returns>The new <c>StringDType</c> array (owned by the caller).</returns>
        /// <exception cref="FormatException">The pickle is truncated or malformed.</exception>
        /// <exception cref="NotSupportedException">A valid pickle that is not NumPy's pickle of a <c>StringDType</c> array
        ///     (an object array, an ndarray subclass, a missing-data object of another type, a protocol 0 / 1 pickle).</exception>
        internal static NDArray ReadStringDTypeArray(Stream stream)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            // The reader owns unmanaged memory: the pickle bytes the element texts still point into. It is freed however the
            // load ends, and only after BuildArray has packed every text into the array's cells.
            using var reader = new PickleReader(stream);
            object root = reader.Load();
            return BuildArray(root, reader);
        }

        /// <summary>
        ///     Checks the object graph against NumPy's <c>StringDType</c> array pickle and builds the array.
        /// </summary>
        /// <param name="root">The value <c>STOP</c> returned.</param>
        /// <param name="reader">The reader that produced <paramref name="root"/>: its buffer holds the element texts.</param>
        /// <returns>The array.</returns>
        /// <exception cref="NotSupportedException">The graph is not that pickle.</exception>
        /// <exception cref="FormatException">The state is inconsistent (an element count that is not the shape's).</exception>
        /// <exception cref="ValueError">An element that is not text under <c>coerce=False</c> (setitem's refusal).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>Failed to pack string during StringDType setitem</c>.</exception>
        private static NDArray BuildArray(object root, PickleReader reader)
        {
            if (root is not PyCall call || call.Callable is not PyGlobalRef rec || rec.Name != "_reconstruct"
                || (rec.Module != ReconstructModule && rec.Module != "numpy.core.multiarray"))
                throw new NotSupportedException(OnlyStringDType + $"this pickle holds {Describe(root)}.");
            if (call.Args is not PyTupleValue args || args.Items.Length != 3 || args.Items[0] is not PyGlobalRef cls
                || cls.Module != "numpy" || cls.Name != "ndarray")
                throw new NotSupportedException(OnlyStringDType + $"this pickle reconstructs {Describe(call.Args)}, not a numpy.ndarray.");
            if (call.State is not PyTupleValue state || (state.Items.Length != 5 && state.Items.Length != 4))
                throw new FormatException("The ndarray pickle has no (version, shape, dtype, is_fortran, data) state.");

            // array_setstate takes a 5-tuple (version 1) or the older 4-tuple without the version.
            int at = state.Items.Length == 5 ? 1 : 0;
            if (at == 1 && !(state.Items[0] is long version && version == 1))
                throw new FormatException($"Unsupported ndarray pickle state version {Describe(state.Items[0])}.");
            long[] shape = ShapeOf(state.Items[at]);
            DType descr = DescrOf(state.Items[at + 1]);
            bool isFortran = state.Items[at + 2] switch
            {
                bool b => b,
                long l => l != 0,
                _ => throw new FormatException($"The ndarray pickle's is_fortran flag is {Describe(state.Items[at + 2])}."),
            };
            if (state.Items[at + 3] is not PyList items)
                throw new FormatException("object pickle not returning list");

            long count = 1;
            foreach (long d in shape)
                count = checked(count * d);
            if (items.Count != count)
                throw new FormatException(
                    $"The ndarray pickle lists {items.Count} elements for an array of shape ({string.Join(", ", shape)}).");

            // NumPy builds the array in the pickled order, then setitem()s the list in C logical order (PyArray_IterNew): a C
            // array takes the list as it is, an F array element by element at its F position, so the strings reach the arena
            // in the same order as in NumPy and the cells come out identical.
            bool fortran = isFortran && shape.Length > 1;
            NDArray result = fortran
                ? np.empty(Shape.Adopt((long[])shape.Clone()), 'F', descr)
                : np.empty(Shape.Adopt((long[])shape.Clone()), descr);
            try
            {
                StoreElements((VStringArraySlice)result.Storage.InternalArray, items, reader, shape, fortran);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     NumPy's <c>_setlist_pkl</c>: stores the pickled elements by setitem in C logical order — element <c>k</c> of
        ///     the list into the <c>k</c>-th element of a C walk, which sits at its F memory position in an F array.
        /// </summary>
        /// <param name="slice">The new array's (zero-filled) cells.</param>
        /// <param name="items">The pickled list, one item per element.</param>
        /// <param name="reader">The reader whose buffer holds the texts.</param>
        /// <param name="shape">The array's dimensions.</param>
        /// <param name="fortran">True for an F array of two or more dimensions.</param>
        /// <exception cref="ValueError">A non-text element under <c>coerce=False</c>.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's setitem <c>MemoryError</c> texts.</exception>
        /// <remarks>
        ///     <para>
        ///     A text element is packed straight from the pickle's UTF-8 bytes. The reader validated them when it read the
        ///     opcode (strict UTF-8: CPython's <c>surrogatepass</c> decode followed by setitem's strict encode keeps exactly
        ///     those texts), so the bytes ARE the strict encoding setitem would produce, and no C# string is made. Against a
        ///     string na_object a text is compared by its bytes: UTF-8 is one-to-one, so byte equality is <c>na_eq_cmp</c>'s
        ///     string equality, and no other na_object can equal a text. Any other element (<c>None</c>, a float, a bool, an
        ///     int) takes <see cref="VStringArraySlice.SetIndex(long, object)"/>, setitem itself.
        ///     </para>
        ///     <para>
        ///     The allocator stays locked for the whole run (the lock is reentrant, so the setitem fallback can take it again)
        ///     and the arena is reserved once for every long text: <see cref="VStringAllocator.TryReserve"/> changes only the
        ///     arena's capacity, never an offset, so the cells are the ones NumPy's element-by-element setitem produces.
        ///     </para>
        /// </remarks>
        private static void StoreElements(VStringArraySlice slice, PyList items, PickleReader reader, long[] shape, bool fortran)
        {
            VStringParameters p = slice.Descr.VString;
            byte[] naUtf8 = p.HasNAObject && p.na_object is string naText ? NDUtf8.Encode(naText) : null;
            PyValue[] values = items.Items;
            int n = items.Count;
            long arenaBytes = 0;
            for (int i = 0; i < n; i++)
                if (values[i].IsText)
                    arenaBytes += VStringAllocator.ArenaBytesForText(values[i].Length);

            long[] coords = fortran ? new long[shape.Length] : null;
            VStringAllocator allocator = slice.Allocator;
            allocator.Acquire();
            try
            {
                allocator.TryReserve(arenaBytes);
                for (int i = 0; i < n; i++)
                {
                    long index = i;
                    if (fortran)
                    {
                        // The F memory index of the C-order coordinates (dimension 0 varies fastest in memory), then the
                        // C-order odometer step to the next element.
                        index = 0;
                        for (int d = shape.Length - 1; d >= 0; d--)
                            index = index * shape[d] + coords[d];
                        for (int d = shape.Length - 1; d >= 0; d--)
                        {
                            if (++coords[d] < shape[d])
                                break;
                            coords[d] = 0;
                        }
                    }

                    PyValue item = values[i];
                    if (!item.IsText)
                    {
                        slice.SetIndex(index, item.Obj is PyNoneValue ? null : item.Obj);
                        continue;
                    }
                    byte* cell = slice.Address + index * VStringAllocator.CellSize;
                    byte* text = reader.TextAt(item);
                    int rc = naUtf8 != null && new ReadOnlySpan<byte>(text, item.Length).SequenceEqual(naUtf8)
                        ? allocator.PackNull(cell)
                        : allocator.Pack(cell, text, item.Length);
                    if (rc < 0)
                        throw new OutOfMemoryException("Failed to pack string during StringDType setitem");
                }
            }
            finally
            {
                allocator.Release();
            }
        }

        /// <summary>The pickled shape: a tuple of non-negative Python ints.</summary>
        /// <param name="value">The state's shape entry.</param>
        /// <returns>The dimensions.</returns>
        /// <exception cref="FormatException">Not a tuple of non-negative ints.</exception>
        private static long[] ShapeOf(object value)
        {
            if (value is not PyTupleValue t)
                throw new FormatException($"The ndarray pickle's shape is {Describe(value)}, not a tuple.");
            var dims = new long[t.Items.Length];
            for (int i = 0; i < dims.Length; i++)
            {
                if (t.Items[i] is not long d || d < 0)
                    throw new FormatException($"The ndarray pickle's shape holds {Describe(t.Items[i])}.");
                dims[i] = d;
            }
            return dims;
        }

        /// <summary>
        ///     The pickled descriptor: <c>_convert_to_stringdtype_kwargs(coerce)</c> or <c>(coerce, na_object)</c>, the
        ///     na_object being <c>None</c>, a float, a string, a bool or a Python int.
        /// </summary>
        /// <param name="value">The state's dtype entry.</param>
        /// <returns>A new <c>StringDType</c> descriptor.</returns>
        /// <exception cref="NotSupportedException">Another descriptor (an object array's <c>numpy.dtype('O')</c>) or another
        ///     missing-data object.</exception>
        private static DType DescrOf(object value)
        {
            if (value is not PyCall dcall || dcall.Callable is not PyGlobalRef g
                || g.Module != InternalModule || g.Name != "_convert_to_stringdtype_kwargs")
                throw new NotSupportedException(OnlyStringDType + $"this array's dtype is pickled as {Describe(value)}.");
            if (dcall.Args is not PyTupleValue a || a.Items.Length < 1 || a.Items.Length > 2)
                throw new FormatException("The StringDType pickle does not hold (coerce[, na_object]).");
            bool coerce = a.Items[0] switch
            {
                bool b => b,
                long l => l != 0,
                BigInteger big => !big.IsZero,
                _ => throw new FormatException($"The StringDType pickle's coerce is {Describe(a.Items[0])}."),
            };
            if (a.Items.Length == 1)
                return np.dtypes.StringDType(coerce);
            object na = a.Items[1] switch
            {
                PyNoneValue => null,
                double or string or bool or long or BigInteger => a.Items[1],
                _ => throw new NotSupportedException(
                    OnlyStringDType + $"its na_object is {Describe(a.Items[1])} (NumSharp reads None, a float, a string, a bool or an int)."),
            };
            return np.dtypes.StringDType(na, coerce);
        }

        /// <summary>A short description of a pickled value for an error text.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The description.</returns>
        private static string Describe(object value) => value switch
        {
            null => "nothing",
            PyNoneValue => "None",
            PyGlobalRef g => $"the global {g}",
            PyCall c => $"a call to {Describe(c.Callable)}",
            PyTupleValue t => $"a tuple of {t.Items.Length}",
            PyList l => $"a list of {l.Count}",
            string s => $"the str '{(s.Length > 40 ? s.Substring(0, 40) + "…" : s)}'",
            byte[] b => $"bytes of length {b.Length}",
            _ => $"a {value.GetType().Name} ({value})",
        };

        /// <summary>
        ///     The binary pickle opcodes of protocols 2–5 over a stack of inert values (CPython's <c>Unpickler</c>, without
        ///     <c>find_class</c>: a global stays a pair of names and a <c>REDUCE</c> a record).
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     <b>One buffer for the whole pickle.</b> Every byte read from the stream is appended to one unmanaged buffer: a
        ///     frame as a whole (CPython reads a frame whole too), and outside the frames each payload as it is read. A text
        ///     stays there as an offset (<see cref="PyValue"/>) until <see cref="BuildArray"/> packs it into its cell, so the
        ///     buffer lives until the array is built and is returned by <see cref="Dispose"/>. Outside a frame nothing is read
        ///     ahead: the reader consumes exactly the bytes up to <c>STOP</c>, so arrays saved one after another into one
        ///     stream load one after another. The buffer comes from <see cref="SizeBucketedBufferPool"/> (the library's
        ///     chokepoint for native memory: the pool accounts it, and the guard-page mode protects its end) and grows by
        ///     taking a buffer twice as large and copying.
        ///     </para>
        ///     <para>
        ///     <b>Hostile lengths.</b> A frame or payload length comes from the file. The buffer grows only by what the stream
        ///     can still deliver (a seekable stream's remaining length) and otherwise only as the bytes arrive, doubling, as
        ///     <see cref="NpyFormat.ReadBytes"/> does — a short file cannot make the reader allocate what it claims.
        ///     </para>
        /// </remarks>
        private sealed class PickleReader : IDisposable
        {
            /// <summary>The first capacity of the pickle buffer, and of the growth step of a stream with no known length.</summary>
            private const long MinBuffer = 64 * 1024;

            /// <summary>A memo index this far past the filled slots goes to <see cref="_sparseMemo"/>, not the array (a
            ///     <c>LONG_BINPUT</c> may name any index; the array must not grow to it).</summary>
            private const long SparseMemoGap = 1 << 16;

            private static readonly object True = true, False = false;

            private readonly Stream _stream;

            /// <summary>Every byte read so far (frames whole, payloads outside frames as read); a pooled buffer of
            ///     <see cref="_capacity"/> bytes, returned by <see cref="Dispose"/>.</summary>
            private byte* _data;

            /// <summary>The byte size <see cref="_data"/> was taken with (the pool needs it back at the return).</summary>
            private long _capacity;

            /// <summary>The bytes held in <see cref="_data"/>.</summary>
            private long _length;

            /// <summary>The next unread byte. <c>_pos &lt; _length</c> only inside a frame: outside one, a read appends the
            ///     bytes it takes from the stream and leaves <c>_pos == _length</c>.</summary>
            private long _pos;

            private PyValue[] _stack = new PyValue[64];
            private int _sp;

            /// <summary>The MARK positions (CPython's metastack).</summary>
            private readonly Stack<int> _marks = new Stack<int>();

            /// <summary>The memo (CPython's memo array); an empty slot is <see langword="default"/>.</summary>
            private PyValue[] _memo = new PyValue[64];

            /// <summary>The memo slots far past the filled ones (see <see cref="SparseMemoGap"/>); null until one is needed.</summary>
            private Dictionary<long, PyValue> _sparseMemo;

            /// <summary>CPython's <c>memo_len</c>: the number of filled slots, which is the index <c>MEMOIZE</c> uses.</summary>
            private long _memoLen;

            /// <summary>Starts a reader at the stream's current position.</summary>
            /// <param name="stream">The stream.</param>
            internal PickleReader(Stream stream) => _stream = stream;

            /// <summary>Returns the pickle buffer to the pool (every text offset becomes invalid).</summary>
            public void Dispose()
            {
                if (_data != null)
                {
                    SizeBucketedBufferPool.Return((IntPtr)_data, _capacity);
                    _data = null;
                    _capacity = _length = _pos = 0;
                }
            }

            /// <summary>The first byte of a text value in the pickle buffer (valid until <see cref="Dispose"/>; the buffer no
            ///     longer moves once <see cref="Load"/> has returned).</summary>
            /// <param name="text">A value for which <see cref="PyValue.IsText"/> is true.</param>
            /// <returns>The pointer.</returns>
            internal byte* TextAt(PyValue text) => _data + text.Offset;

            /// <summary>Runs the opcodes up to <c>STOP</c>.</summary>
            /// <returns>The value on top of the stack at <c>STOP</c>, as an object (a text as a <see cref="string"/>).</returns>
            /// <exception cref="FormatException">A truncated or malformed pickle.</exception>
            /// <exception cref="UnicodeDecodeError">A text that is not strict UTF-8.</exception>
            /// <exception cref="NotSupportedException">An opcode outside the binary protocols' value set (a protocol 0 / 1
            ///     text opcode, a dict, a set, a persistent id, an extension code, an out-of-band buffer).</exception>
            internal object Load()
            {
                while (true)
                {
                    byte op = ReadByte();
                    switch (op)
                    {
                        case PROTO:
                            int proto = ReadByte();
                            if (proto < 2 || proto > 5)
                                throw new NotSupportedException($"NumSharp reads pickle protocols 2 to 5, not protocol {proto}.");
                            break;
                        case FRAME:
                        {
                            ulong length = BinaryPrimitives.ReadUInt64LittleEndian(ReadExact(8));
                            if (_pos < _length)
                                throw new FormatException("beginning of a new frame before end of current frame");
                            if (length > long.MaxValue)
                                throw new FormatException($"A pickle frame of {length} bytes is too large.");
                            Append((long)length, "pickle frame");
                            break;
                        }
                        case STOP:
                            if (_marks.Count != 0 || _sp != 1)
                                throw new FormatException("The pickle's STOP did not leave exactly one value.");
                            return Materialize(_stack[0]);
                        case MARK:
                            _marks.Push(_sp);
                            break;
                        case POP:
                            if (_marks.Count > 0 && _marks.Peek() == _sp)
                                _marks.Pop();
                            else
                                Pop();
                            break;
                        case POP_MARK:
                            _sp = PopMark();
                            break;
                        case DUP:
                            Push(Peek());
                            break;
                        case NONE:
                            Push(PyValue.Of(PyNoneValue.Instance));
                            break;
                        case NEWTRUE:
                            Push(PyValue.Of(True));
                            break;
                        case NEWFALSE:
                            Push(PyValue.Of(False));
                            break;
                        case BININT:
                            Push(PyValue.Of((long)BinaryPrimitives.ReadInt32LittleEndian(ReadExact(4))));
                            break;
                        case BININT1:
                            Push(PyValue.Of((long)ReadByte()));
                            break;
                        case BININT2:
                            Push(PyValue.Of((long)BinaryPrimitives.ReadUInt16LittleEndian(ReadExact(2))));
                            break;
                        case LONG1:
                            Push(PyValue.Of(ReadLong(ReadByte())));
                            break;
                        case LONG4:
                        {
                            int n = BinaryPrimitives.ReadInt32LittleEndian(ReadExact(4));
                            if (n < 0)
                                throw new FormatException("LONG pickle has negative byte count");
                            Push(PyValue.Of(ReadLong(n)));
                            break;
                        }
                        case BINFLOAT:
                            Push(PyValue.Of(BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64BigEndian(ReadExact(8)))));
                            break;
                        case SHORT_BINUNICODE:
                            PushText(ReadByte());
                            break;
                        case BINUNICODE:
                            PushText(BinaryPrimitives.ReadUInt32LittleEndian(ReadExact(4)));
                            break;
                        case BINUNICODE8:
                            PushText(BinaryPrimitives.ReadUInt64LittleEndian(ReadExact(8)));
                            break;
                        case SHORT_BINBYTES:
                            Push(PyValue.Of(ReadPayloadArray(ReadByte())));
                            break;
                        case BINBYTES:
                            Push(PyValue.Of(ReadPayloadArray(BinaryPrimitives.ReadUInt32LittleEndian(ReadExact(4)))));
                            break;
                        case BINBYTES8:
                            Push(PyValue.Of(ReadPayloadArray(BinaryPrimitives.ReadUInt64LittleEndian(ReadExact(8)))));
                            break;
                        case EMPTY_TUPLE:
                            Push(PyValue.Of(new PyTupleValue(Array.Empty<object>())));
                            break;
                        case TUPLE1:
                        case TUPLE2:
                        case TUPLE3:
                        {
                            int n = op - TUPLE1 + 1;
                            var items = new object[n];
                            for (int i = n - 1; i >= 0; i--)
                                items[i] = Materialize(Pop());
                            Push(PyValue.Of(new PyTupleValue(items)));
                            break;
                        }
                        case TUPLE:
                        {
                            int start = PopMark();
                            var items = new object[_sp - start];
                            for (int i = 0; i < items.Length; i++)
                                items[i] = Materialize(_stack[start + i]);
                            _sp = start;
                            Push(PyValue.Of(new PyTupleValue(items)));
                            break;
                        }
                        case EMPTY_LIST:
                            Push(PyValue.Of(new PyList()));
                            break;
                        case LIST:
                        {
                            int start = PopMark();
                            var list = new PyList();
                            list.AddRange(_stack, start, _sp - start);
                            _sp = start;
                            Push(PyValue.Of(list));
                            break;
                        }
                        case APPEND:
                        {
                            PyValue item = Pop();
                            if (Peek().Obj is not PyList list)
                                throw new NotSupportedException(OnlyStringDType + "it appends to an object that is not a list.");
                            list.Add(item);
                            break;
                        }
                        case APPENDS:
                        {
                            // CPython's do_append: the list is the value just below the mark, and it must lie above the
                            // enclosing mark; nothing to append is not an error (nor a check of the target).
                            int start = PopMark();
                            if (start <= Fence)
                                throw new FormatException("unpickling stack underflow");
                            if (start == _sp)
                                break;
                            if (_stack[start - 1].Obj is not PyList list)
                                throw new NotSupportedException(OnlyStringDType + "it appends to an object that is not a list.");
                            list.AddRange(_stack, start, _sp - start);
                            _sp = start;
                            break;
                        }
                        case GLOBAL:
                        {
                            string module = ReadLine();
                            string name = ReadLine();
                            Push(PyValue.Of(new PyGlobalRef(module, name)));
                            break;
                        }
                        case STACK_GLOBAL:
                        {
                            PyValue name = Pop(), module = Pop();
                            if (!module.IsText || !name.IsText)
                                throw new FormatException("STACK_GLOBAL requires str");
                            Push(PyValue.Of(new PyGlobalRef((string)Materialize(module), (string)Materialize(name))));
                            break;
                        }
                        case REDUCE:
                        {
                            PyValue args = Pop(), callable = Pop();
                            Push(PyValue.Of(new PyCall(Materialize(callable), Materialize(args))));
                            break;
                        }
                        case BUILD:
                        {
                            PyValue state = Pop();
                            if (Peek().Obj is not PyCall target || target.State != null)
                                throw new NotSupportedException(OnlyStringDType + "it builds an object NumPy's pickle does not build.");
                            target.State = Materialize(state);
                            break;
                        }
                        case MEMOIZE:
                            MemoPut(_memoLen, Peek());
                            break;
                        case BINPUT:
                            MemoPut(ReadByte(), Peek());
                            break;
                        case LONG_BINPUT:
                            MemoPut(BinaryPrimitives.ReadUInt32LittleEndian(ReadExact(4)), Peek());
                            break;
                        case BINGET:
                            Push(MemoGet(ReadByte()));
                            break;
                        case LONG_BINGET:
                            Push(MemoGet(BinaryPrimitives.ReadUInt32LittleEndian(ReadExact(4))));
                            break;
                        default:
                            throw new NotSupportedException(
                                OnlyStringDType + $"it holds the pickle opcode 0x{op:X2}, which that pickle never uses.");
                    }
                }
            }

            /// <summary>A reader value as an object: a text decoded to a <see cref="string"/> (it was validated when read),
            ///     anything else as it is.</summary>
            /// <param name="value">The value.</param>
            /// <returns>The object.</returns>
            private object Materialize(PyValue value)
                => value.IsText ? NDUtf8.DecodeTrusted(new ReadOnlySpan<byte>(_data + value.Offset, value.Length)) : value.Obj;

            /// <summary>
            ///     CPython's <c>_Unpickler_MemoPut</c>: fills memo slot <paramref name="index"/>; filling an empty slot counts
            ///     towards <see cref="_memoLen"/>, the index the next <c>MEMOIZE</c> takes.
            /// </summary>
            /// <param name="index">The slot.</param>
            /// <param name="value">The value.</param>
            private void MemoPut(long index, PyValue value)
            {
                if (index >= _memo.Length)
                {
                    if (index - _memoLen > SparseMemoGap || index >= Array.MaxLength)
                    {
                        // An index far past the filled slots (a hand-made LONG_BINPUT): a dictionary entry, so a short file
                        // cannot make the memo array grow to a huge index.
                        _sparseMemo ??= new Dictionary<long, PyValue>();
                        if (!_sparseMemo.ContainsKey(index))
                            _memoLen++;
                        _sparseMemo[index] = value;
                        return;
                    }
                    Array.Resize(ref _memo, (int)Math.Min(Array.MaxLength, Math.Max(index + 1, (long)_memo.Length * 2)));
                }
                // A slot the array has grown over may already be filled in the dictionary: it moves, and still counts once
                // (CPython's memo is one array, so memo_len counts every index once).
                if (_memo[index].IsEmpty && !(_sparseMemo != null && _sparseMemo.Remove(index)))
                    _memoLen++;
                _memo[index] = value;
            }

            /// <summary>The memo entry <paramref name="index"/>.</summary>
            /// <param name="index">The memo index.</param>
            /// <returns>The value.</returns>
            /// <exception cref="FormatException">No such entry.</exception>
            private PyValue MemoGet(long index)
            {
                if (index < _memo.Length && !_memo[index].IsEmpty)
                    return _memo[index];
                if (_sparseMemo != null && _sparseMemo.TryGetValue(index, out PyValue v))
                    return v;
                throw new FormatException($"Memo value not found at index {index}");
            }

            /// <summary>A little-endian two's-complement int of <paramref name="n"/> bytes (<c>LONG1</c> / <c>LONG4</c>).</summary>
            /// <param name="n">The byte count (0 is the int 0).</param>
            /// <returns>A <see cref="long"/> when it fits, else a <see cref="BigInteger"/>.</returns>
            private object ReadLong(long n)
            {
                if (n == 0)
                    return 0L;
                // Take first: it may move the buffer, and `_data + Take(...)` would read the old pointer before the call.
                long at = Take(n, "pickle data");
                var value = new BigInteger(new ReadOnlySpan<byte>(_data + at, (int)n), isUnsigned: false, isBigEndian: false);
                return value >= long.MinValue && value <= long.MaxValue ? (long)value : value;
            }

            /// <summary>
            ///     Reads a UTF-8 text payload of <paramref name="size"/> bytes and pushes it as an offset into the buffer.
            /// </summary>
            /// <param name="size">The byte count.</param>
            /// <exception cref="FormatException">A payload too large for NumSharp, or the pickle ends first.</exception>
            /// <exception cref="UnicodeDecodeError">The bytes are not strict UTF-8.</exception>
            /// <remarks>
            ///     CPython decodes the payload when it reads the opcode (with <c>surrogatepass</c>), so a malformed text fails
            ///     here, before any later opcode. NumSharp checks strict UTF-8 at once: a surrogate passes CPython's decode but
            ///     fails NumPy's setitem (its strict encode), so the strict check keeps exactly the texts NumPy can store.
            /// </remarks>
            private void PushText(ulong size)
            {
                if (size > int.MaxValue)
                    throw new FormatException($"A pickled payload of {size} bytes is too large for NumSharp.");
                int n = (int)size;
                long at = Take(n, "pickle data");
                NDUtf8.ThrowIfInvalid(new ReadOnlySpan<byte>(_data + at, n));
                Push(PyValue.Text(at, n));
            }

            /// <summary>A payload of <paramref name="size"/> bytes as a new array (a pickled <c>bytes</c>).</summary>
            /// <param name="size">The byte count.</param>
            /// <returns>The bytes.</returns>
            private byte[] ReadPayloadArray(ulong size)
            {
                if (size > int.MaxValue)
                    throw new FormatException($"A pickled payload of {size} bytes is too large for NumSharp.");
                int n = (int)size;
                long at = Take(n, "pickle data"); // first: Take may move the buffer
                return new ReadOnlySpan<byte>(_data + at, n).ToArray();
            }

            /// <summary>Exactly <paramref name="n"/> bytes (an opcode argument), consumed at once: the span points into the
            ///     buffer, which the next read may move.</summary>
            /// <param name="n">The count.</param>
            /// <returns>The bytes.</returns>
            private ReadOnlySpan<byte> ReadExact(int n)
            {
                // Take first: it may move the buffer, and `_data + Take(...)` would read the old pointer before the call.
                long at = Take(n, "pickle data");
                return new ReadOnlySpan<byte>(_data + at, n);
            }

            /// <summary>
            ///     Takes the next <paramref name="n"/> bytes: from the current frame when it holds them, else (no frame open)
            ///     from the stream, appended to the buffer.
            /// </summary>
            /// <param name="n">The count.</param>
            /// <param name="what">What is read, for the truncation message.</param>
            /// <returns>The offset of the first byte in the buffer.</returns>
            /// <exception cref="FormatException">The read runs past the frame's end, or the pickle ends first.</exception>
            private long Take(long n, string what)
            {
                if (_pos + n > _length)
                {
                    if (_pos < _length)
                        throw new FormatException("pickle exhausted before end of frame");
                    Append(n, what);
                }
                long at = _pos;
                _pos += n;
                return at;
            }

            /// <summary>
            ///     Appends exactly <paramref name="n"/> bytes of the stream to the buffer (a frame, or a payload outside one).
            /// </summary>
            /// <param name="n">The count, as the pickle claims it.</param>
            /// <param name="what">What is read, for the truncation message.</param>
            /// <exception cref="FormatException"><c>EOF: reading {what}, expected {n} bytes got {m}</c>: the stream ends first.</exception>
            /// <exception cref="OutOfMemoryException">The buffer cannot grow.</exception>
            private void Append(long n, string what)
            {
                if (n == 0)
                    return;
                // What the stream can plausibly deliver: a seekable stream's remaining bytes; else the claim is honored only
                // as the bytes arrive, the step doubling with what has arrived.
                long plausible = n;
                if (_stream.CanSeek)
                    plausible = Math.Min(n, Math.Max(0L, _stream.Length - _stream.Position));
                long step = Math.Max(1, _stream.CanSeek ? plausible : Math.Min(n, MinBuffer));
                long got = 0;
                while (got < n)
                {
                    long want = Math.Min(n - got, Math.Max(step, got));
                    Reserve(_length + got + want);
                    long read = ReadInto(_data + _length + got, want);
                    got += read;
                    if (read < want)
                        break; // the stream ended
                }
                if (got != n)
                    throw new FormatException($"EOF: reading {what}, expected {n} bytes got {got}");
                _length += n;
            }

            /// <summary>Reads up to <paramref name="count"/> bytes into <paramref name="dst"/>, looping because
            ///     <see cref="Stream.Read(Span{byte})"/> may return fewer bytes than asked for.</summary>
            /// <param name="dst">The destination in the buffer.</param>
            /// <param name="count">The count.</param>
            /// <returns>The bytes read (fewer only at the end of the stream).</returns>
            private long ReadInto(byte* dst, long count)
            {
                long total = 0;
                while (total < count)
                {
                    int chunk = (int)Math.Min(count - total, int.MaxValue);
                    int read = _stream.Read(new Span<byte>(dst + total, chunk));
                    if (read == 0)
                        break;
                    total += read;
                }
                return total;
            }

            /// <summary>Grows the buffer to hold <paramref name="needed"/> bytes (doubling: a new pooled buffer, the bytes
            ///     held copied over, the old one returned; the offsets stay valid, the pointers do not).</summary>
            /// <param name="needed">The required capacity.</param>
            /// <exception cref="OutOfMemoryException">The allocation failed (the old buffer stays in place and is returned
            ///     by <see cref="Dispose"/>).</exception>
            private void Reserve(long needed)
            {
                if (needed <= _capacity)
                    return;
                long capacity = Math.Max(needed, Math.Max(MinBuffer, _capacity * 2));
                // Take before giving anything back: a failed take leaves the reader's buffer as it was.
                byte* grown = (byte*)SizeBucketedBufferPool.Take(capacity);
                if (_data != null)
                {
                    Buffer.MemoryCopy(_data, grown, capacity, _length);
                    SizeBucketedBufferPool.Return((IntPtr)_data, _capacity);
                }
                _data = grown;
                _capacity = capacity;
            }

            /// <summary>One byte: from the current frame, else straight from the stream (an opcode outside the frames is
            ///     not kept: nothing refers to it later).</summary>
            /// <returns>The byte.</returns>
            /// <exception cref="FormatException">The pickle ends.</exception>
            private byte ReadByte()
            {
                if (_pos < _length)
                    return _data[_pos++];
                int b = _stream.ReadByte();
                if (b < 0)
                    throw new FormatException("pickle data was truncated");
                return (byte)b;
            }

            /// <summary>A newline-terminated ASCII line (<c>GLOBAL</c>'s two names).</summary>
            /// <returns>The line without its newline.</returns>
            private string ReadLine()
            {
                var sb = new StringBuilder();
                while (true)
                {
                    byte b = ReadByte();
                    if (b == (byte)'\n')
                        return sb.ToString();
                    sb.Append((char)b);
                }
            }

            /// <summary>The stack position of the innermost mark (CPython's <c>fence</c>): values below it are out of reach.</summary>
            private int Fence => _marks.Count > 0 ? _marks.Peek() : 0;

            private void Push(PyValue value)
            {
                if (_sp == _stack.Length)
                    Array.Resize(ref _stack, _stack.Length * 2);
                _stack[_sp++] = value;
            }

            private PyValue Peek()
            {
                if (_sp <= Fence)
                    throw new FormatException("unpickling stack underflow");
                return _stack[_sp - 1];
            }

            private PyValue Pop()
            {
                PyValue v = Peek();
                _sp--;
                return v;
            }

            /// <summary>Removes the innermost mark.</summary>
            /// <returns>Its stack position: the values pushed since are <c>_stack[position .. _sp)</c>.</returns>
            /// <exception cref="FormatException">No mark.</exception>
            private int PopMark()
            {
                if (_marks.Count == 0)
                    throw new FormatException("could not find MARK");
                return _marks.Pop();
            }
        }
    }
}
