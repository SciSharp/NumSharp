using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        // Where read_rows puts the fields it converts. NumPy writes each field straight into its growing result array through
        // the dtype's set_from_ucs4 function (textreading/field_types.c: npy_to_double, npy_to_int64, …, npy_to_string,
        // npy_to_unicode, and npy_to_generic — PyArray_Pack of the field's str — for every other dtype, StringDType
        // included). A sink does the same per dtype family: it stages the converted elements in element (row-major) order
        // and builds the 2-D result once the row count is known, which is what NumPy's PyArray_Resize to row_count ends with.

        /// <summary>
        ///     The destination of one <c>np.loadtxt</c> call's converted fields: stages the elements in row-major order and
        ///     builds the 2-D result once every row is read.
        /// </summary>
        /// <remarks>
        ///     A sink owns pooled staging memory; <see cref="Dispose"/> returns it, so the caller disposes the sink whether the
        ///     read succeeds or raises. The result <see cref="Finish"/> returns does not depend on the sink afterwards.
        /// </remarks>
        private abstract class LoadTxtSink : IDisposable
        {
            /// <summary>The dtype's <c>str()</c> as NumPy's conversion error names it (<c>%S</c> of the descriptor):
            ///     <c>float64</c>, <c>&lt;U5</c>, <c>|S2</c>, <c>StringDType(na_object=None)</c>, or <c>object</c> for an unsized
            ///     string dtype (NumPy reads it as an object array first).</summary>
            internal abstract string DescrText { get; }

            /// <summary>
            ///     Converts one field and appends it as the next element (NumPy's <c>set_from_ucs4</c>, or
            ///     <c>npy_to_generic_with_converter</c> when the column has a converter).
            /// </summary>
            /// <param name="field">The field's characters (the tokenizer's buffer; valid only during the call).</param>
            /// <param name="converter">The column's converter, or null.</param>
            /// <returns>True when stored; false when the dtype's parser refused the text without an error of its own, which
            ///     the caller reports as NumPy's <c>could not convert string … at row r, column c.</c> without a cause.</returns>
            /// <exception cref="Exception">Any error of the converter or of the value's store, which the caller chains as the
            ///     cause of that same message (NumPy's <c>npy_PyErr_ChainExceptionsCause</c>).</exception>
            internal abstract bool Store(ReadOnlySpan<char> field, Func<string, object> converter);

            /// <summary>
            ///     Converts and appends the first <paramref name="count"/> fields of a row that has no converter and no
            ///     <c>usecols</c> selection: <see cref="Store"/> for each field in order, with the row's loop inside the sink so
            ///     a sink can run its own conversion per field without a virtual call each.
            /// </summary>
            /// <param name="row">The tokenizer holding the row.</param>
            /// <param name="count">The number of fields to store (the row's field count).</param>
            /// <param name="cause">The error that stopped the row, or null (also when a parser refused a field without
            ///     one).</param>
            /// <returns><paramref name="count"/> when every field was stored, else the index of the field that was not, which
            ///     the caller reports as NumPy's <c>could not convert string … at row r, column c.</c> with
            ///     <paramref name="cause"/> chained.</returns>
            internal virtual int StoreRow(LoadTxtTokenizer row, int count, out Exception cause)
            {
                // The row's characters once; each field is a slice of them.
                ReadOnlySpan<char> chars = row.RowChars;
                int[] starts = row.FieldStarts, ends = row.FieldEnds;
                int i = 0;
                try
                {
                    for (; i < count; i++)
                    {
                        if (!Store(chars.Slice(starts[i], ends[i] - starts[i]), null))
                        {
                            cause = null;
                            return i;
                        }
                    }
                }
                catch (Exception e)
                {
                    // The caller chains it as the cause, as it chains an error of Store.
                    cause = e;
                    return i;
                }
                cause = null;
                return count;
            }

            /// <summary>Builds the result: a fresh C-ordered <c>(rows, cols)</c> array that owns its data (NumPy's
            ///     <c>read_rows</c> result, whatever <c>ndmin</c> does to it afterwards).</summary>
            /// <param name="rows">The number of data rows read.</param>
            /// <param name="cols">The number of columns (1 when no row was read and none could be discovered).</param>
            /// <returns>The array.</returns>
            /// <exception cref="OutOfMemoryException">The result cannot be allocated, or a string cannot be packed.</exception>
            internal abstract NDArray Finish(long rows, int cols);

            /// <summary>
            ///     Sizes the staging for a source that bounds its rows (<see cref="LoadTxtStream.RemainingLinesHint"/>): a sink
            ///     of fixed-size elements allocates its result at once and writes the elements straight into it, so the
            ///     doubling buffer's copies and the final copy into the result are skipped. Called once, at the first data
            ///     row, before any element is stored.
            /// </summary>
            /// <param name="rows">The most rows the read can produce (at least 1).</param>
            /// <param name="cols">The number of result columns.</param>
            internal virtual void Reserve(long rows, int cols) { }

            /// <summary>The largest result <see cref="Reserve"/> allocates up front: past it the rows a source bounds may be
            ///     mostly comment lines, and the doubling buffer's extra copy is a small share of such a read anyway.</summary>
            protected const long PresizeMaxBytes = 1L << 26;

            /// <summary>
            ///     Allocates a presized result (see <see cref="Reserve"/>): its buffer becomes the staging buffer.
            /// </summary>
            /// <param name="descr">The result's descriptor.</param>
            /// <param name="rows">The rows to allocate.</param>
            /// <param name="cols">The columns.</param>
            /// <param name="itemSize">The element size in bytes.</param>
            /// <param name="buffer">The result's buffer.</param>
            /// <param name="capacity">Its size in bytes.</param>
            /// <returns>The result, or null when it would exceed <see cref="PresizeMaxBytes"/>.</returns>
            /// <exception cref="OutOfMemoryException">The result cannot be allocated.</exception>
            protected static unsafe NDArray AllocatePresized(DType descr, long rows, int cols, int itemSize, out byte* buffer, out long capacity)
            {
                buffer = null;
                capacity = 0;
                if (rows <= 0 || cols <= 0 || itemSize <= 0 || rows > PresizeMaxBytes / cols / itemSize)
                    return null;
                var result = new NDArray(descr, new Shape(rows, cols), false);
                buffer = (byte*)result.Storage.Address;
                capacity = rows * cols * itemSize;
                return result;
            }

            /// <summary>
            ///     Leaves a presized result when the source produced more rows than it bounded (a converter grew the list):
            ///     the staged elements move to a pooled buffer that grows as usual, and the presized result is released.
            /// </summary>
            /// <param name="presized">The presized result (cleared).</param>
            /// <param name="buffer">The staging buffer (replaced by the pooled one).</param>
            /// <param name="capacity">Its size (replaced).</param>
            /// <param name="used">The bytes staged.</param>
            /// <param name="needed">The free bytes the caller is about to write.</param>
            /// <exception cref="OutOfMemoryException">The pool cannot provide the buffer.</exception>
            protected static unsafe void LeavePresized(ref NDArray presized, ref byte* buffer, ref long capacity, long used, long needed)
            {
                byte* staged = buffer;
                buffer = null;
                capacity = 0;
                Grow(ref buffer, ref capacity, 0, used + needed);
                if (used > 0)
                    Buffer.MemoryCopy(staged, buffer, capacity, used);
                presized.Dispose();
                presized = null;
            }

            /// <summary>
            ///     The result of a presized read: the presized array itself when every row it was sized for arrived, else an
            ///     exact array of the rows read (the bound counted lines that held no row: blanks, comments, a quoted field's
            ///     continuation), the presized one released.
            /// </summary>
            /// <param name="descr">The result's descriptor.</param>
            /// <param name="presized">The presized result (cleared).</param>
            /// <param name="buffer">Its buffer, the staging buffer (cleared).</param>
            /// <param name="capacity">Its size (cleared).</param>
            /// <param name="presizedRows">The rows it was allocated with.</param>
            /// <param name="rows">The rows read.</param>
            /// <param name="cols">The columns.</param>
            /// <param name="itemSize">The element size in bytes.</param>
            /// <returns>The result.</returns>
            /// <exception cref="OutOfMemoryException">The exact array cannot be allocated.</exception>
            protected static unsafe NDArray TakePresized(DType descr, ref NDArray presized, ref byte* buffer, ref long capacity,
                long presizedRows, long rows, int cols, int itemSize)
            {
                NDArray result = presized;
                byte* staged = buffer;
                presized = null;
                buffer = null;
                capacity = 0;
                if (rows == presizedRows)
                    return result;
                using (result)
                {
                    var exact = new NDArray(descr, new Shape(rows, cols), false);
                    long bytes = rows * cols * itemSize;
                    if (bytes > 0)
                        Buffer.MemoryCopy(staged, (byte*)exact.Storage.Address, bytes, bytes);
                    return exact;
                }
            }

            /// <summary>Returns the staging memory to the pool (idempotent).</summary>
            public virtual void Dispose() { }

            /// <summary>
            ///     The sink of <paramref name="descr"/>'s family: <c>StringDType</c>, a fixed-width string class (<c>U</c> /
            ///     <c>S</c>, sized or not), or a numeric dtype.
            /// </summary>
            /// <param name="descr">The result's descriptor.</param>
            /// <param name="byteConverters">NumPy's <c>encoding='bytes'</c> (<c>python_byte_converters</c>): a converter's
            ///     input is the field encoded as Latin-1.</param>
            /// <returns>The sink.</returns>
            internal static LoadTxtSink Create(DType descr, bool byteConverters) => descr.Meta switch
            {
                VStringDTypeMeta => new LoadTxtVStringSink(descr, byteConverters),
                StringDTypeMeta => new LoadTxtFixedStringSink(descr, byteConverters),
                _ => new LoadTxtNumericSink(descr, byteConverters),
            };

            /// <summary>
            ///     The text a converter receives (<c>call_converter_function</c>): the field's str, or — under
            ///     <c>encoding='bytes'</c> — its Latin-1 bytes, which a C# converter receives as the same characters (each
            ///     at most <c>U+00FF</c>), after the same encode check.
            /// </summary>
            /// <param name="field">The field.</param>
            /// <param name="byteConverters">True under <c>encoding='bytes'</c>.</param>
            /// <returns>The converter's argument.</returns>
            /// <exception cref="UnicodeEncodeError">Under <c>encoding='bytes'</c>, a character past <c>U+00FF</c>
            ///     (<c>'latin-1' codec can't encode character '中' in position 0: ordinal not in range(256)</c>).</exception>
            protected static string ConverterInput(ReadOnlySpan<char> field, bool byteConverters)
            {
                if (byteConverters)
                    NDStringCodec.ThrowIfNotEncodable(field, '\u00FF', "latin-1", "ordinal not in range(256)");
                return field.ToString();
            }

            /// <summary>
            ///     A pooled staging buffer that grows by doubling: NumPy grows its result by ~25 % blocks of at least 8 KiB
            ///     (<c>MIN_BLOCK_SIZE</c>); only the capacity sequence differs, never an element.
            /// </summary>
            /// <param name="buffer">The current buffer (null before the first growth).</param>
            /// <param name="capacity">Its size in bytes.</param>
            /// <param name="used">The bytes in use (kept by the copy).</param>
            /// <param name="needed">The free bytes the caller is about to write.</param>
            /// <exception cref="OutOfMemoryException">The pool cannot provide the larger buffer.</exception>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            protected static unsafe void EnsureFree(ref byte* buffer, ref long capacity, long used, long needed)
            {
                if (capacity - used < needed)
                    Grow(ref buffer, ref capacity, used, needed);
            }

            /// <summary>The growth step of <see cref="EnsureFree"/>, kept out of line.</summary>
            /// <param name="buffer">The current buffer (null before the first growth).</param>
            /// <param name="capacity">Its size in bytes.</param>
            /// <param name="used">The bytes in use (kept by the copy).</param>
            /// <param name="needed">The free bytes the caller is about to write.</param>
            /// <exception cref="OutOfMemoryException">The pool cannot provide the larger buffer.</exception>
            /// <remarks>Never inlined: the pool's allocation is a P/Invoke, and a P/Invoke inlined into a caller makes that
            ///     caller set up an inlined call frame in its prolog on every call, taken or not — measured as a fifth of a
            ///     short row's cost in the row stores.</remarks>
            [MethodImpl(MethodImplOptions.NoInlining)]
            protected static unsafe void Grow(ref byte* buffer, ref long capacity, long used, long needed)
            {
                long grown = Math.Max(capacity * 2, 1L << 13);
                while (grown - used < needed)
                    grown *= 2;
                var fresh = (byte*)SizeBucketedBufferPool.Take(grown);
                if (used > 0)
                    Buffer.MemoryCopy(buffer, fresh, grown, used);
                // The old buffer goes back with the size it was taken with (the pool's rule: the size selects the bucket
                // and tells whether the buffer carries an alignment header).
                if (buffer != null)
                    SizeBucketedBufferPool.Return((IntPtr)buffer, capacity);
                buffer = fresh;
                capacity = grown;
            }

            /// <summary>Returns a staging buffer to the pool and clears the reference (safe to call twice).</summary>
            /// <param name="buffer">The buffer, or null.</param>
            /// <param name="capacity">Its size in bytes.</param>
            protected static unsafe void ReturnBuffer(ref byte* buffer, ref long capacity)
            {
                if (buffer == null)
                    return;
                SizeBucketedBufferPool.Return((IntPtr)buffer, capacity);
                buffer = null;
                capacity = 0;
            }
        }

        /// <summary>
        ///     A numeric dtype's fields (NumPy's <c>npy_to_bool</c> / <c>npy_to_int*</c> / <c>npy_to_uint*</c> /
        ///     <c>npy_to_float</c> / <c>npy_to_double</c> / <c>npy_to_cfloat</c> / <c>npy_to_cdouble</c>, the strict parsers of
        ///     <c>np.loadtxt.Parsers.cs</c>), each element written in the dtype's own bytes; a converter's value is stored by
        ///     <c>PyArray_Pack</c> (<see cref="ReductionInitial.PackStore"/>).
        /// </summary>
        private sealed unsafe class LoadTxtNumericSink : LoadTxtSink
        {
            private readonly DType _descr;
            private readonly NPTypeCode _tc;
            private readonly int _itemSize;
            private readonly bool _byteConverters;
            private byte* _buffer;
            private long _capacity, _used;
            // A presized result (Reserve): _buffer is its buffer, not the pool's.
            private NDArray _presized;
            private long _presizedRows;

            /// <summary>Creates the sink.</summary>
            /// <param name="descr">The numeric descriptor.</param>
            /// <param name="byteConverters">NumPy's <c>encoding='bytes'</c>.</param>
            internal LoadTxtNumericSink(DType descr, bool byteConverters)
            {
                _descr = descr;
                _tc = descr.GetTypeCode();
                _itemSize = _tc.SizeOf();
                _byteConverters = byteConverters;
            }

            /// <inheritdoc/>
            internal override void Reserve(long rows, int cols)
            {
                if (_buffer != null)
                    return;
                _presized = AllocatePresized(_descr, rows, cols, _itemSize, out _buffer, out _capacity);
                _presizedRows = rows;
            }

            /// <summary>Room for <paramref name="needed"/> more bytes (a presized result left only when it is full).</summary>
            /// <param name="needed">The bytes about to be written.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private void EnsureRoom(long needed)
            {
                if (_capacity - _used < needed)
                    GrowRoom(needed);
            }

            /// <summary>The growth step of <see cref="EnsureRoom"/>, out of line (see <see cref="LoadTxtSink.Grow"/>).</summary>
            /// <param name="needed">The bytes about to be written.</param>
            [MethodImpl(MethodImplOptions.NoInlining)]
            private void GrowRoom(long needed)
            {
                if (_presized is not null)
                    LeavePresized(ref _presized, ref _buffer, ref _capacity, _used, needed);
                else
                    Grow(ref _buffer, ref _capacity, _used, needed);
            }

            /// <inheritdoc/>
            internal override string DescrText => _tc.AsNumpyDtypeName();

            /// <inheritdoc/>
            /// <remarks>
            ///     <para>
            ///     Without a converter the field is parsed by the dtype's strict parser (<c>LoadtxtWriteField</c>), which
            ///     refuses text without an error — NumPy's parsers return -1 with no exception set, so its message has no cause.
            ///     </para>
            ///     <para>
            ///     A converter's value follows NumPy's setitem rules for its KIND (<see cref="ReductionInitial.PackStore"/>,
            ///     probed against NumPy 2.4.2): a C# float into an integer dtype truncates toward zero; a C# integer out of the
            ///     dtype's range raises NumPy's <c>Python integer 300 out of bounds for uint8</c> where a <see cref="Half"/> (a
            ///     NumPy scalar) wraps; a <see cref="string"/> parses with Python's <c>int()</c> / <c>float()</c> /
            ///     <c>complex()</c>; a bool dtype takes Python truthiness; <c>None</c> (null) is False for bool, NumPy's positive
            ///     quiet NaN for a float dtype and <c>int(None)</c>'s TypeError for an integer dtype. Each error propagates, and
            ///     the reader chains it as the cause of its <c>could not convert</c> message.
            ///     </para>
            /// </remarks>
            [SkipLocalsInit]
            internal override bool Store(ReadOnlySpan<char> field, Func<string, object> converter)
            {
                EnsureRoom(_itemSize);
                var element = new Span<byte>(_buffer + _used, _itemSize);
                if (converter == null)
                {
                    if (!LoadtxtWriteField(_tc, field, element))
                        return false;
                }
                else
                {
                    object value = converter(ConverterInput(field, _byteConverters));
                    // PyArray_Pack into a slot of the widest element, then the element's own bytes into the row.
                    Span<byte> slot = stackalloc byte[ReductionInitial.SlotBytes];
                    ReductionInitial.PackStore(value, _tc, slot);
                    slot.Slice(0, _itemSize).CopyTo(element);
                }
                _used += _itemSize;
                return true;
            }

            /// <inheritdoc/>
            /// <remarks>The row's elements are reserved at once and each field goes straight to the dtype's strict parser, the
            ///     same parse <see cref="Store"/> runs without a converter.</remarks>
            internal override int StoreRow(LoadTxtTokenizer row, int count, out Exception cause)
            {
                EnsureRoom((long)_itemSize * count);
                // The row's characters once, and the element position in a local: the writes go through raw pointers, after
                // which the JIT must assume this object's fields changed.
                ReadOnlySpan<char> chars = row.RowChars;
                int[] starts = row.FieldStarts, ends = row.FieldEnds;
                NPTypeCode tc = _tc;
                int itemSize = _itemSize;
                byte* element = _buffer + _used;
                int i = 0;
                try
                {
                    for (; i < count; i++, element += itemSize)
                    {
                        if (!LoadtxtWriteField(tc, chars.Slice(starts[i], ends[i] - starts[i]), new Span<byte>(element, itemSize)))
                        {
                            _used = element - _buffer;
                            cause = null;
                            return i;
                        }
                    }
                }
                catch (Exception e)
                {
                    // A float16 field is Python's float() of the text, whose ValueError NumPy chains as the cause.
                    _used = element - _buffer;
                    cause = e;
                    return i;
                }
                _used = element - _buffer;
                cause = null;
                return count;
            }

            /// <inheritdoc/>
            internal override NDArray Finish(long rows, int cols)
            {
                if (_presized is not null)
                    return TakePresized(_descr, ref _presized, ref _buffer, ref _capacity, _presizedRows, rows, cols, _itemSize);
                var result = new NDArray(_descr, new Shape(rows, cols), false);
                long bytes = rows * cols * _itemSize;
                if (bytes > 0)
                    Buffer.MemoryCopy(_buffer, result.Storage.Address, bytes, bytes);
                return result;
            }

            /// <inheritdoc/>
            public override void Dispose()
            {
                // A presized result's buffer is the result's own, never the pool's.
                if (_presized is not null)
                {
                    _presized.Dispose();
                    _presized = null;
                    _buffer = null;
                    _capacity = 0;
                }
                ReturnBuffer(ref _buffer, ref _capacity);
            }
        }

        /// <summary>
        ///     A fixed-width string dtype's fields. A sized class is NumPy's <c>npy_to_unicode</c> (the field's first
        ///     <c>elsize / 4</c> code points, NUL-padded) or <c>npy_to_string</c> (the first <c>elsize</c> characters, each at
        ///     most <c>U+00FF</c>, stored as their Latin-1 bytes); an unsized class (<c>'U'</c>, <c>'S'</c>) is NumPy's object
        ///     read followed by <c>astype</c>, so the widest element decides the width and the conversion error names
        ///     <c>object</c>.
        /// </summary>
        private sealed unsafe class LoadTxtFixedStringSink : LoadTxtSink
        {
            private readonly DType _descr;
            private readonly bool _unicode;
            private readonly int _chars;
            private readonly int _itemSize;
            private readonly bool _byteConverters;
            // A sized class: the elements' bytes, written as the result will hold them. An unsized class: the values the
            // object array would hold (NumPy's _read reads S0 / U0 into an object array, then casts it) — as objects when a
            // field goes through Store (a converter, usecols), or, for rows StoreRow takes, as their text staged in _buffer
            // (U: UTF-16 units, S: Latin-1 bytes) with each element's end offset in _ends and the widest element's width
            // (code points for U, bytes for S): the cast's result without a string object per field.
            private byte* _buffer;
            private long _capacity, _used;
            private readonly List<object> _values;
            private byte* _ends;
            private long _endsCapacity, _staged, _stagedWidth;
            // A sized class's presized result (Reserve): _buffer is its buffer, not the pool's.
            private NDArray _presized;
            private long _presizedRows;

            /// <summary>Creates the sink.</summary>
            /// <param name="descr">A <c>U</c> or <c>S</c> descriptor (sized or not).</param>
            /// <param name="byteConverters">NumPy's <c>encoding='bytes'</c>.</param>
            internal LoadTxtFixedStringSink(DType descr, bool byteConverters)
            {
                _descr = descr;
                _unicode = ((StringDTypeMeta)descr.Meta).IsUnicode;
                _itemSize = descr.itemsize;
                _chars = _unicode ? _itemSize / 4 : _itemSize;
                _byteConverters = byteConverters;
                if (_itemSize == 0)
                    _values = new List<object>(256);
            }

            /// <inheritdoc/>
            internal override string DescrText => _itemSize == 0 ? "object" : _descr.ToString();

            /// <inheritdoc/>
            /// <remarks>A sized class only: an unsized class's width is known after the last row.</remarks>
            internal override void Reserve(long rows, int cols)
            {
                if (_itemSize == 0 || _buffer != null)
                    return;
                _presized = AllocatePresized(_descr, rows, cols, _itemSize, out _buffer, out _capacity);
                _presizedRows = rows;
            }

            /// <summary>Room for <paramref name="needed"/> more bytes of a sized class's elements (a presized result left
            ///     only when it is full).</summary>
            /// <param name="needed">The bytes about to be written.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private void EnsureRoom(long needed)
            {
                if (_capacity - _used < needed)
                    GrowRoom(needed);
            }

            /// <summary>The growth step of <see cref="EnsureRoom"/>, out of line (see <see cref="LoadTxtSink.Grow"/>).</summary>
            /// <param name="needed">The bytes about to be written.</param>
            [MethodImpl(MethodImplOptions.NoInlining)]
            private void GrowRoom(long needed)
            {
                if (_presized is not null)
                    LeavePresized(ref _presized, ref _buffer, ref _capacity, _used, needed);
                else
                    Grow(ref _buffer, ref _capacity, _used, needed);
            }

            /// <inheritdoc/>
            /// <remarks>
            ///     <para>
            ///     A sized <c>S</c> field fails without an error when one of its first <c>elsize</c> characters is past
            ///     <c>U+00FF</c> (<c>to_string</c> checks only what it stores: <c>'a中'</c> fits <c>S1</c>, not <c>S2</c>). An
            ///     unsized <c>S</c> field is encoded WHOLE as Latin-1 (NumPy's <c>c_byte_converters</c>), so a character past
            ///     <c>U+00FF</c> raises the <see cref="UnicodeEncodeError"/> NumPy chains as the cause.
            ///     </para>
            ///     <para>
            ///     A converter's value is stored by the class's setitem (<c>UNICODE_setitem</c> / <c>STRING_setitem</c>:
            ///     <see cref="NDStringStorage.WriteElement"/> — <c>str()</c> of a number, <c>'None'</c> for null, ASCII for a
            ///     <c>U</c> element's bytes and an <c>S</c> element's text, a sequence refused), its errors chained as the
            ///     cause. Under an unsized class NumPy keeps the converter's object and casts the array afterwards, so its
            ///     errors surface from <see cref="Finish"/> without the row and column.
            ///     </para>
            /// </remarks>
            internal override bool Store(ReadOnlySpan<char> field, Func<string, object> converter)
            {
                if (_itemSize == 0)
                {
                    // Elements staged by StoreRow come first in element order (a call uses one path for all its rows, so
                    // this only guards the order).
                    if (_staged > 0)
                        MoveStagedToValues();
                    if (converter != null)
                        _values.Add(converter(ConverterInput(field, _byteConverters)));
                    else if (_unicode)
                        _values.Add(field.ToString());
                    else
                    {
                        NDStringCodec.ThrowIfNotEncodable(field, '\u00FF', "latin-1", "ordinal not in range(256)");
                        _values.Add(Latin1Bytes(field));
                    }
                    return true;
                }

                EnsureRoom(_itemSize);
                byte* element = _buffer + _used;
                if (converter != null)
                {
                    object value = converter(ConverterInput(field, _byteConverters));
                    NDStringStorage.WriteElement(element, _unicode, _chars, value);
                }
                else if (_unicode)
                    NDStringCodec.WriteUcs4(field, (uint*)element, _chars);
                else if (!WriteLatin1Element(field, element, _chars))
                    return false;
                _used += _itemSize;
                return true;
            }

            /// <inheritdoc/>
            /// <remarks>A sized class reserves the row's elements at once and writes each field as <see cref="Store"/> does
            ///     without a converter (neither writer raises). An unsized class stages each field's text (see
            ///     <see cref="StageText"/>), which <see cref="Finish"/> casts as NumPy casts its object array.</remarks>
            internal override int StoreRow(LoadTxtTokenizer row, int count, out Exception cause)
            {
                if (_itemSize == 0)
                {
                    // Objects stored by Store come first in element order (never the case within one call).
                    if (_values.Count > 0)
                        return base.StoreRow(row, count, out cause);
                    EnsureFree(ref _ends, ref _endsCapacity, _staged * sizeof(long), (long)count * sizeof(long));
                    ReadOnlySpan<char> rowText = row.RowChars;
                    int[] rowStarts = row.FieldStarts, rowEnds = row.FieldEnds;
                    int i = 0;
                    try
                    {
                        for (; i < count; i++)
                            StageText(rowText.Slice(rowStarts[i], rowEnds[i] - rowStarts[i]));
                    }
                    catch (Exception e)
                    {
                        // An unsized S field past Latin-1: the UnicodeEncodeError NumPy chains as the cause.
                        cause = e;
                        return i;
                    }
                    cause = null;
                    return count;
                }
                cause = null;
                EnsureRoom((long)_itemSize * count);
                // The element writes go through raw pointers, after which the JIT must assume this object's fields changed:
                // the loop works on locals and stores the position once.
                byte* element = _buffer + _used;
                int itemSize = _itemSize, chars = _chars;
                ReadOnlySpan<char> text = row.RowChars;
                int[] starts = row.FieldStarts, ends = row.FieldEnds;
                if (_unicode)
                {
                    for (int i = 0; i < count; i++, element += itemSize)
                        NDStringCodec.WriteUcs4(text.Slice(starts[i], ends[i] - starts[i]), (uint*)element, chars);
                }
                else
                {
                    for (int i = 0; i < count; i++, element += itemSize)
                    {
                        if (!WriteLatin1Element(text.Slice(starts[i], ends[i] - starts[i]), element, chars))
                        {
                            _used = element - _buffer;
                            return i;
                        }
                    }
                }
                _used = element - _buffer;
                return count;
            }

            /// <summary>
            ///     <c>npy_to_string</c>: the field's first <paramref name="chars"/> characters as their Latin-1 bytes, the rest
            ///     of the element zero-filled. Only the stored characters must be at most <c>U+00FF</c>; the rest of the field
            ///     is never looked at.
            /// </summary>
            /// <param name="field">The field.</param>
            /// <param name="element">The element's first byte.</param>
            /// <param name="chars">The element's width (<c>itemsize</c>).</param>
            /// <returns>False when a stored character is past <c>U+00FF</c> (the element may then hold some of the bytes; the
            ///     caller does not keep it).</returns>
            private static bool WriteLatin1Element(ReadOnlySpan<char> field, byte* element, int chars)
            {
                int n = Math.Min(field.Length, chars);
                if (n <= 16)
                {
                    // A short field: the test and the narrowing in one pass (a vectorized range search costs more to start
                    // than these few characters take).
                    for (int i = 0; i < n; i++)
                    {
                        char c = field[i];
                        if (c > '\u00FF')
                            return false;
                        element[i] = (byte)c;
                    }
                }
                else
                {
                    var stored = field.Slice(0, n);
                    if (stored.IndexOfAnyExceptInRange('\0', '\u00FF') >= 0)
                        return false;
                    for (int i = 0; i < n; i++)
                        element[i] = (byte)stored[i];
                }
                if (chars - n <= 16)
                {
                    for (int i = n; i < chars; i++)
                        element[i] = 0;
                }
                else
                {
                    new Span<byte>(element + n, chars - n).Clear();
                }
                return true;
            }

            /// <summary>The Latin-1 bytes of a text whose characters are all at most <c>U+00FF</c>.</summary>
            /// <param name="text">The text (checked by the caller).</param>
            /// <returns>One byte per character.</returns>
            private static byte[] Latin1Bytes(ReadOnlySpan<char> text)
            {
                var bytes = new byte[text.Length];
                for (int i = 0; i < text.Length; i++)
                    bytes[i] = (byte)text[i];
                return bytes;
            }

            /// <summary>
            ///     Appends one field of an unsized class to the staged text: what <see cref="Store"/> keeps as an object (the
            ///     field's str for U, its Latin-1 bytes for S, the whole field encoded first), kept as the text itself.
            /// </summary>
            /// <param name="field">The field.</param>
            /// <exception cref="UnicodeEncodeError">An S field with a character past <c>U+00FF</c>
            ///     (<c>c_byte_converters</c> encodes the whole field as Latin-1).</exception>
            /// <remarks>The element count's room in <see cref="_ends"/> is reserved by the caller for the row.</remarks>
            private void StageText(ReadOnlySpan<char> field)
            {
                int n = field.Length;
                long width = n;
                if (_unicode)
                {
                    EnsureFree(ref _buffer, ref _capacity, _used, 2L * n);
                    char* dst = (char*)(_buffer + _used);
                    bool surrogate = false;
                    for (int j = 0; j < n; j++)
                    {
                        char c = field[j];
                        surrogate |= (uint)(c - '\uD800') <= '\uDFFF' - '\uD800';
                        dst[j] = c;
                    }
                    // The cast's width counts code points: a surrogate pair is one (len() of the str).
                    if (surrogate)
                        width = NDStringCodec.CodePointCount(field);
                    _used += 2L * n;
                }
                else
                {
                    EnsureFree(ref _buffer, ref _capacity, _used, n);
                    byte* dst = _buffer + _used;
                    for (int j = 0; j < n; j++)
                    {
                        char c = field[j];
                        if (c > '\u00FF')
                        {
                            // Raises CPython's UnicodeEncodeError for the first character past Latin-1.
                            NDStringCodec.ThrowIfNotEncodable(field, '\u00FF', "latin-1", "ordinal not in range(256)");
                        }
                        dst[j] = (byte)c;
                    }
                    _used += n;
                }
                ((long*)_ends)[_staged++] = _used;
                if (width > _stagedWidth)
                    _stagedWidth = width;
            }

            /// <summary>Moves the staged elements into <see cref="_values"/> as the objects <see cref="Store"/> keeps
            ///     (strings for U, Latin-1 byte arrays for S), so the two can follow each other in element order.</summary>
            private void MoveStagedToValues()
            {
                long start = 0;
                for (long k = 0; k < _staged; k++)
                {
                    long stop = ((long*)_ends)[k];
                    if (_unicode)
                        _values.Add(new string((char*)(_buffer + start), 0, (int)((stop - start) / 2)));
                    else
                        _values.Add(new ReadOnlySpan<byte>(_buffer + start, (int)(stop - start)).ToArray());
                    start = stop;
                }
                _staged = 0;
                _stagedWidth = 0;
                _used = 0;
            }

            /// <inheritdoc/>
            /// <exception cref="TypeError">An unsized class whose widest value exceeds the class's maximum width.</exception>
            /// <exception cref="ValueError">An unsized class holding a converter's sequence (NumPy's cast of its object
            ///     array: <c>setting an array element with a sequence</c>).</exception>
            internal override NDArray Finish(long rows, int cols)
            {
                if (_itemSize == 0)
                {
                    // NumPy's object read, then astype(U0 / S0): the widest element decides the width, and an empty result
                    // is one character wide.
                    if (rows == 0)
                        return new NDArray(NDStringStorage.AllocationDescr(_descr), new Shape(0, cols), false);
                    if (_staged > 0)
                        return FinishStaged(rows, cols);
                    return NDStringArrays.FromValues(new[] { rows, (long)cols }, _values, _descr);
                }

                if (_presized is not null)
                    return TakePresized(_descr, ref _presized, ref _buffer, ref _capacity, _presizedRows, rows, cols, _itemSize);
                var result = new NDArray(_descr, new Shape(rows, cols), false);
                long bytes = rows * cols * _itemSize;
                if (bytes > 0)
                    Buffer.MemoryCopy(_buffer, result.Storage.Address, bytes, bytes);
                return result;
            }

            /// <summary>
            ///     The unsized class's result from the staged text: NumPy's <c>astype</c> of its object array of the fields
            ///     (<see cref="NDStringArrays.FromValues"/> over the same strings / byte arrays) — the widest element decides
            ///     the width, each element is written by the class's setitem.
            /// </summary>
            /// <param name="rows">The number of rows (at least one).</param>
            /// <param name="cols">The number of columns.</param>
            /// <returns>The array.</returns>
            /// <exception cref="TypeError">The widest element exceeds the class's maximum width (<c>string too large to store
            ///     inside array.</c>).</exception>
            private NDArray FinishStaged(long rows, int cols)
            {
                DType descr = NDStringArrays.UnsizedTextDescr((StringDTypeMeta)_descr.Meta, _stagedWidth);
                var storage = UnmanagedStorage.AllocateString(new Shape(rows, cols), descr, fillZeros: false);
                var slice = storage.StringSlice;
                byte* p = slice.Address;
                int itemSize = slice.ItemSize;
                int chars = slice.Chars;
                long start = 0;
                for (long k = 0; k < _staged; k++, p += itemSize)
                {
                    long stop = ((long*)_ends)[k];
                    if (_unicode)
                    {
                        // UNICODE_setitem of the str: its code points, NUL-padded (the width holds the widest one).
                        NDStringCodec.WriteUcs4(new ReadOnlySpan<char>(_buffer + start, (int)((stop - start) / 2)), (uint*)p, chars);
                    }
                    else
                    {
                        // STRING_setitem of the bytes: copied, NUL-padded.
                        int n = (int)(stop - start);
                        Buffer.MemoryCopy(_buffer + start, p, itemSize, n);
                        if (n < itemSize)
                            new Span<byte>(p + n, itemSize - n).Clear();
                    }
                    start = stop;
                }
                return new NDArray(storage);
            }

            /// <inheritdoc/>
            public override void Dispose()
            {
                // A presized result's buffer is the result's own, never the pool's.
                if (_presized is not null)
                {
                    _presized.Dispose();
                    _presized = null;
                    _buffer = null;
                    _capacity = 0;
                }
                ReturnBuffer(ref _buffer, ref _capacity);
                ReturnBuffer(ref _ends, ref _endsCapacity);
            }
        }

        /// <summary>
        ///     A <c>StringDType</c>'s fields (NumPy's <c>npy_to_generic</c>: <c>PyArray_Pack</c> of the field's str, i.e.
        ///     <c>stringdtype_setitem</c>): a field equal to a string <c>na_object</c> is an NA element, any other is its
        ///     UTF-8 text; a converter's value is NA when it equals the <c>na_object</c> (<c>na_eq_cmp</c>), else its
        ///     <c>as_pystring</c> text (a non-<c>str</c> refused under <c>coerce=False</c>), and a 0-d array is CAST into the
        ///     descriptor (<c>cast_raw_scalar_item</c>).
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     <b>Two phases.</b> Every element's UTF-8 is staged in element order (one pooled byte buffer, one 8-byte record
        ///     per element). <see cref="Finish"/> then allocates the result, reserves the arena ONCE for the exact bytes, and
        ///     packs the cells in element order under one lock. NumPy packs each field as it reads it, also in element
        ///     (row-major) order — so the arena offsets, and with them every cell, are NumPy's; only the arena's capacity
        ///     sequence differs (<see cref="VStringAllocator.TryReserve"/>).
        ///     </para>
        ///     <para>
        ///     A short text (1 to 15 bytes) is written straight into its zeroed cell — the cell <c>NpyString_pack</c> gives
        ///     there (<see cref="VStringAllocator.newemptysize"/>: the bytes, then
        ///     <c>NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA | size</c>); an empty text leaves the zero cell,
        ///     NumPy's empty string.
        ///     </para>
        /// </remarks>
        private sealed unsafe class LoadTxtVStringSink : LoadTxtSink
        {
            private readonly DType _descr;
            private readonly VStringParameters _p;
            private readonly bool _byteConverters;
            private readonly string _naText;
            // The UTF-8 of every text element, in element order.
            private byte* _text;
            private long _textCapacity, _textUsed;
            // Per element: (the end offset of its UTF-8 in _text << 1) | 1 for an NA element.
            private byte* _records;
            private long _recordsCapacity, _count;
            // The arena bytes the elements append (VStringAllocator.ArenaBytesForText of each text).
            private long _arenaBytes;

            /// <summary>Creates the sink.</summary>
            /// <param name="descr">The <c>StringDType</c> descriptor.</param>
            /// <param name="byteConverters">NumPy's <c>encoding='bytes'</c>.</param>
            internal LoadTxtVStringSink(DType descr, bool byteConverters)
            {
                _descr = descr;
                _p = descr.VString;
                _byteConverters = byteConverters;
                // na_eq_cmp(field, na_object): a field (a str) can only equal a str NA object (the boundary map compares
                // text ordinally and never equal to a non-text object).
                _naText = _p.has_string_na ? (string)_p.na_object : null;
            }

            /// <inheritdoc/>
            internal override string DescrText => _descr.ToString();

            /// <inheritdoc/>
            /// <exception cref="UnicodeEncodeError">A text with a lone surrogate (CPython's text, the position in the
            ///     field).</exception>
            /// <exception cref="ValueError">A converter's non-text value under <c>coerce=False</c>
            ///     (<c>StringDType only allows string data when string coercion is disabled.</c>).</exception>
            /// <exception cref="UnicodeDecodeError">A converter's <c>bytes</c> that are not UTF-8.</exception>
            internal override bool Store(ReadOnlySpan<char> field, Func<string, object> converter)
            {
                if (converter == null)
                {
                    if (_naText != null && field.SequenceEqual(_naText.AsSpan()))
                        AddNA();
                    else
                        AddText(field);
                    return true;
                }

                object value = converter(ConverterInput(field, _byteConverters));
                if (value is NDArray nd && nd.ndim == 0)
                {
                    // PyArray_Pack's 0-d ndarray branch (an ndarray is no known scalar type, so the DType discovery gives
                    // None): cast_raw_scalar_item — the element CAST into this descriptor, as astype casts it.
                    AddCastScalar(nd);
                    return true;
                }
                if (_p.HasNAObject && NDPyObject.Equal(value, _p.na_object))
                    AddNA();
                else
                    AddText(VStringArraySlice.AsPyString(value, _p.coerce));
                return true;
            }

            /// <summary>Records an NA element.</summary>
            private void AddNA() => AddRecord((_textUsed << 1) | 1);

            /// <summary>Stages a text element's strict UTF-8.</summary>
            /// <param name="text">The text.</param>
            /// <exception cref="UnicodeEncodeError">A lone surrogate.</exception>
            private void AddText(ReadOnlySpan<char> text)
            {
                // At most 3 UTF-8 bytes per UTF-16 unit; a text whose bound does not fit a span is counted exactly first.
                long bound = (long)text.Length * 3;
                if (bound > int.MaxValue)
                    bound = NDUtf8.GetByteCount(text);
                EnsureFree(ref _text, ref _textCapacity, _textUsed, bound);
                int written = NDUtf8.Encode(text, new Span<byte>(_text + _textUsed, (int)bound));
                _textUsed += written;
                _arenaBytes += VStringAllocator.ArenaBytesForText(written);
                AddRecord(_textUsed << 1);
            }

            /// <summary>Stages UTF-8 bytes that are already valid (a cast's cell).</summary>
            /// <param name="utf8">The first byte.</param>
            /// <param name="size">The byte count.</param>
            private void AddUtf8(byte* utf8, long size)
            {
                EnsureFree(ref _text, ref _textCapacity, _textUsed, size);
                if (size > 0)
                    Buffer.MemoryCopy(utf8, _text + _textUsed, size, size);
                _textUsed += size;
                _arenaBytes += VStringAllocator.ArenaBytesForText(size);
                AddRecord(_textUsed << 1);
            }

            /// <summary>
            ///     Stages a 0-d array's element cast into this descriptor (<c>cast_raw_scalar_item</c>): a number becomes its
            ///     text whatever <c>coerce</c> says, a <c>StringDType</c> element keeps its NA.
            /// </summary>
            /// <param name="scalar">The 0-d array.</param>
            /// <exception cref="OutOfMemoryException">The cast's cell cannot be read.</exception>
            private void AddCastScalar(NDArray scalar)
            {
                using var cast = scalar.astype(_descr);
                var slice = cast.Storage.VStringSlice;
                var allocator = slice.Allocator;
                allocator.acquire_allocator();
                try
                {
                    byte* cell = slice.Address + cast.Shape.offset * VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
                    int isNull = allocator.load(cell, out byte* buf, out long size);
                    if (isNull < 0)
                        throw new OutOfMemoryException("Failed to load string in StringDType getitem");
                    if (isNull == 1)
                        AddNA();
                    else
                        AddUtf8(buf, size);
                }
                finally
                {
                    allocator.release_allocator();
                }
            }

            /// <summary>Appends one element record.</summary>
            /// <param name="record">The record.</param>
            private void AddRecord(long record)
            {
                EnsureFree(ref _records, ref _recordsCapacity, _count * sizeof(long), sizeof(long));
                ((long*)_records)[_count++] = record;
            }

            /// <inheritdoc/>
            internal override NDArray Finish(long rows, int cols)
            {
                var result = new NDArray(_descr, new Shape(rows, cols), true);
                try
                {
                    long count = rows * cols;
                    var slice = result.Storage.VStringSlice;
                    var allocator = slice.Allocator;
                    byte* cells = slice.Address;
                    var records = (long*)_records;
                    allocator.acquire_allocator();
                    try
                    {
                        // The exact arena the cells take, in one growth (the offsets are the cursor's, as NumPy's).
                        allocator.TryReserve(_arenaBytes);
                        long start = 0;
                        for (long i = 0; i < count; i++)
                        {
                            long record = records[i];
                            long end = record >> 1;
                            byte* cell = cells + i * VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
                            if ((record & 1) != 0)
                            {
                                if (allocator.pack_null(cell) < 0)
                                    throw new OutOfMemoryException("Failed to pack null string during StringDType setitem");
                            }
                            else
                            {
                                long size = end - start;
                                if (size > VStringAllocator.NPY_SHORT_STRING_MAX_SIZE)
                                {
                                    if (allocator.pack(cell, _text + start, size) < 0)
                                        throw new OutOfMemoryException("Failed to pack string during StringDType setitem");
                                }
                                else if (size > 0)
                                {
                                    // The short cell NpyString_pack writes into a fresh (zero) cell.
                                    Buffer.MemoryCopy(_text + start, cell, VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING, size);
                                    cell[15] = (byte)(VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA | size);
                                }
                            }
                            start = end;
                        }
                    }
                    finally
                    {
                        allocator.release_allocator();
                    }
                    return result;
                }
                catch
                {
                    result.Dispose();
                    throw;
                }
            }

            /// <inheritdoc/>
            public override void Dispose()
            {
                ReturnBuffer(ref _text, ref _textCapacity);
                ReturnBuffer(ref _records, ref _recordsCapacity);
            }
        }
    }
}
