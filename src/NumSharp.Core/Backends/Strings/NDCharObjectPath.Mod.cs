using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Printing;

// =============================================================================
// NDCharObjectPath.Mod.cs — np.strings.mod's driver and CPython's % formatting per element (strings phase 2, P2.6
// milestone 3)
// =============================================================================
//
// NUMPY'S STRUCTURE (numpy/_core/strings.py mod / _to_bytes_or_str_array, multiarraymodule.c _vec_string_with_args)
//   mod(a, values) = _to_bytes_or_str_array(_vec_string(a, np.object_, '__mod__', (values,)), a)
//   _vec_string checks that a is text ("string operation on non-string array"), picks bytes.__mod__ for S and
//   str.__mod__ for U and StringDType, converts values with np.asarray, broadcasts the two with a multi-iterator (NumPy's
//   "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (2,) and arg 1
//   with shape (3,)." text), and calls, per element in C order, the method with both operands' scalars (PyArray_ToScalar,
//   a first, values second: a U one-unit text past U+10FFFF raises CPython's SystemError there). The method's descriptor
//   refuses a StringDType NA that is not a str ("descriptor '__mod__' requires a 'str' object but received a 'NoneType'").
//   The first element (C order) that raises ends the call. The object results then become the answer:
//     * no element: result.astype(a.dtype) — an empty array of a's own dtype (a StringDType descriptor kept whole);
//     * else ret = np.asarray(result.tolist()): U (str results) or S (bytes results) as wide as the longest result, at
//       least 1 — trailing NULs count for the width though a U / S element does not keep them —, then
//         S / U:        ret.astype(type(a.dtype)(n)), the same dtype;
//         StringDType:  ret.astype(StringDType()) — a FRESH default descriptor (a's NA object is dropped), each element
//                       the U element read back (its trailing NULs gone), a surrogate or a unit past U+10FFFF raising
//                       TypeError "Invalid unicode code point found" — after every element was formatted.
//
// CPYTHON'S % (Objects/unicodeobject.c PyUnicode_Format for str, Objects/bytesobject.c _PyBytes_FormatEx for bytes)
//   The argument is one object: a tuple gives its items one per spec (an item left over is "not all arguments converted
//   during string formatting" / "... bytes formatting"); any other object is the single argument (a second spec finds
//   "not enough arguments for format string"), and a MAPPING — an object with a subscript slot that is not a tuple and not
//   a str (str formatting) or not a tuple / bytes / str (bytes formatting) — also feeds %(key) specs and silences the
//   left-over check. NumPy's numeric scalars, np.bool and np.float16 are mappings (they subscript), and so are a dict, a
//   list and an ndarray; a bytes value is one for str formatting only. So '%s' % 5 and 'abc' % np.int64(5) (no spec, no
//   error) differ from 'abc' % 'x' (TypeError). A key lookup's own error is the answer: a NumPy scalar's IndexError
//   "invalid index to scalar variable.", a bytes value's TypeError "byte indices must be integers or slices, not str", a
//   list's "list indices must be integers or slices, not str" (or "... not bytes"), an ndarray's IndexError, a dict's
//   KeyError (its text the key's repr: "'a'", "b'a'"). The parse and its error order are ModFormat's (NDStringMod.cs).
//   The conversions, per spec:
//     str formatting: s / r / a are str() / repr() / ascii() cut to the precision and padded to the width in code
//       points; d i u x X o, e E f F g G and c follow PrintfFormatter (CPython's conversions for NumPy scalars and Python
//       objects: np.float64 repr is 'np.float64(1.5)', %d of a float truncates, %x of a float or np.bool refuses, %c takes
//       an integer code point or a one-character str).
//     bytes formatting: s / b are the value's bytes (a bytes value; the buffer of a NumPy scalar — the raw element bytes,
//       np.str_ its UCS-4 code units —; a Python str or object without a buffer is TypeError "%b requires a bytes-like
//       object, or an object that implements __bytes__, not 'str'"); r / a are ascii(repr()); c a single byte or an
//       integer in range(256) (OverflowError "%c arg not in range(256)", TypeError "%c requires an integer in range(256) or
//       a single byte"); the numbers as str formatting writes them, except that ANY failure of float() is TypeError "float
//       argument required, not str"; an unsupported conversion byte past 0x7F is OverflowError "character argument not in
//       range(0x110000)" (CPython formats the signed char with %c), one below it shows the byte raw.
//
// HOW IT IS COMPUTED HERE
//   One C-order NDIter walk over (a, values) with the object path's readers (VecArg / Operand: NumPy's scalars per element).
//   Each distinct format element is compiled once (ModFormat, cached by its units; the common call broadcasts ONE format
//   over the values, read and compiled once per chunk). An element is formatted over the format's code points (U /
//   StringDType) or bytes (S) and moved into the result at once: a StringDType result packs each element into its cell (the
//   U round trip's rules; an invalid code point raises after the walk), a U / S result writes each element into its row
//   (ModWalkState: the width chosen from the first elements, rows re-laid out for a longer one, narrowed once at the end).
//   A format of ONE positional argument takes the fast paths of NDCharObjectPath.ModFast.cs (a chunk loop that writes a
//   number's or a text's field straight into its row; FloatDigits' integer-arithmetic digits for floats); every other
//   format runs CPython's argument machinery below. C# helpers per element: the phase-2 plan's technique for the
//   object-path family.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDCharObjectPath
    {
        /// <summary>CPython's TypeError when a single non-mapping argument (or a tuple item) is left over (str formatting).</summary>
        private const string NotAllConvertedStrText = "not all arguments converted during string formatting";

        /// <summary>CPython's TypeError when an argument is left over (bytes formatting).</summary>
        private const string NotAllConvertedBytesText = "not all arguments converted during bytes formatting";

        /// <summary>CPython's TypeError when a spec finds no argument left.</summary>
        private const string NotEnoughArgumentsText = "not enough arguments for format string";

        /// <summary>CPython's TypeError for a <c>%(key)</c> spec whose argument is not a mapping.</summary>
        private const string FormatRequiresMappingText = "format requires a mapping";

        /// <summary>CPython's TypeError for a <c>*</c> width or precision whose argument is not a Python int.</summary>
        private const string StarWantsIntText = "* wants int";

        /// <summary>NumPy's IndexError when a NumPy scalar is subscripted (a <c>%(key)</c> lookup on np.int64 / np.bool ...).</summary>
        private const string ScalarIndexText = "invalid index to scalar variable.";

        /// <summary>NumPy's IndexError when an ndarray is subscripted by a str or bytes key.</summary>
        private const string ArrayIndexText =
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices";

        /// <summary>NumPy's TypeError when a StringDType result holds a code point UTF-8 cannot encode (the U → StringDType cast).</summary>
        private const string InvalidCodePointText = "Invalid unicode code point found";

        /// <summary>
        ///     NumPy's <c>np.strings.mod(a, values)</c> (see the file header): every element of <paramref name="a"/> formatted
        ///     with CPython's <c>%</c> against the broadcast element of <paramref name="values"/>.
        /// </summary>
        /// <param name="a">The format strings (<c>np.asarray</c>'d by the caller).</param>
        /// <param name="values">The values, converted as NumPy's <c>np.asarray</c> converts them (null is Python's None).</param>
        /// <returns>A fresh C-ordered <c>U</c> / <c>S</c> array as wide as the longest result (at least 1), a default
        ///     <c>StringDType</c> array for a <c>StringDType</c> operand, or an empty array of <paramref name="a"/>'s dtype over an
        ///     empty broadcast.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is not text; a conversion the value's type refuses; the argument
        ///     count; a <c>%(key)</c> without a mapping; a <c>StringDType</c> NA that is not a str; a <c>StringDType</c> result
        ///     with an invalid code point.</exception>
        /// <exception cref="ValueError">A malformed format (<c>incomplete format</c>, <c>unsupported format character</c> ...);
        ///     a NaN under an integer conversion; a ragged <paramref name="values"/>.</exception>
        /// <exception cref="OverflowException">An infinity under an integer conversion; a <c>%c</c> argument out of range;
        ///     an int too large for a float conversion; a bytes format's unsupported conversion byte past 0x7F.</exception>
        /// <exception cref="IndexError">A <c>%(key)</c> lookup on a NumPy scalar or an ndarray.</exception>
        /// <exception cref="KeyError">A <c>%(key)</c> missing from a dict.</exception>
        /// <exception cref="SystemError">A <c>U</c> one-unit element past U+10FFFF (CPython's scalar).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's multi-iterator text).</exception>
        /// <exception cref="NotSupportedException">A value with no NumSharp form (an object array of lists).</exception>
        internal static NDArray Mod(NDArray a, object values)
        {
            var loop = NDStringMethods.ResolveOne(a);
            // _vec_string picks bytes.__mod__ for S, str.__mod__ for U and StringDType, and refuses anything else — before
            // values is converted (a ragged values list with a numeric a is still the text error).
            if (loop == NDStringMethods.TextLoop.None)
                throw new TypeError(NonStringArrayText);
            using var vals = VecArg.Convert(values, "values");
            if (vals.Lists is not null)
                throw new NotSupportedException("An object array of lists as np.strings.mod's values has no NumSharp form.");
            var outer = np.broadcast_shapes(a.Shape, vals.BroadcastShape);
            // _to_bytes_or_str_array: an empty result is result.astype(a.dtype) — a's dtype whole, nothing formatted.
            if (outer.size == 0)
                return new NDArray(a.dtype, outer.Clean(), true);

            using var self = VecArg.Convert(a, "a");
            var readers = BuildReaders(self, vals);
            // A StringDType result is a FRESH default descriptor (a's NA object dropped): its cells are packed as the walk
            // formats them, so no arena of the whole result and no second pass exist (see ModWalkState.PackElement).
            NDArray packed = loop == NDStringMethods.TextLoop.VString
                ? new NDArray(np.dtypes.StringDType(), outer.Clean(), true)
                : null;
            ModWalkState w = null;
            try
            {
                w = new ModWalkState(readers, loop, outer.size, packed, outer);
                Walk(new List<NDArray> { a, vals.Array }, readers, (dataptrs, strides, n, aux) => ModRun(dataptrs, strides, n, w));
                if (packed is null)
                    return w.FinishRows();
                // The U → StringDType cast runs after every element was formatted: a formatting error of any element wins
                // over a code point UTF-8 cannot encode.
                if (w.InvalidCodePoint)
                    throw new TypeError(InvalidCodePointText);
                return packed;
            }
            catch
            {
                packed?.Dispose();
                throw;
            }
            finally
            {
                w?.Dispose();
                foreach (var r in readers)
                    r?.Dispose();
            }
        }

        // =============================================================================================================
        // The walk's state: readers, the compiled-format cache, the result arena
        // =============================================================================================================

        /// <summary>
        ///     The state of one <c>np.strings.mod</c> call shared by its NDIter chunks: the operand readers, the compiled-format
        ///     cache, the arena an element is formatted into (code points for str formatting, bytes for bytes formatting), the
        ///     result it then moves into (a <c>U</c> / <c>S</c> row, or a <c>StringDType</c> cell), and the scratch of values
        ///     derived from Python objects.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     The arena holds one element at a time: as soon as an element is formatted it is stored into the result
        ///     (<see cref="StoreRow"/> / <see cref="PackElement"/>) and the arena is emptied, so a large call never builds a
        ///     second copy of its whole result and never reads it back (at 100K elements that copy and its pass cost 10–30 ns
        ///     per element). A <c>U</c> / <c>S</c> result's width is NumPy's longest element, unknown until the end: the first
        ///     <see cref="RowSample"/> elements are kept in the arena to choose a width, a longer element later re-lays the
        ///     rows out (<see cref="GrowRows"/>), and <see cref="FinishRows"/> narrows the rows once if the width overshot.
        ///     </para>
        ///     <para>
        ///     Owns the arena, the result until it is handed out, and the derived scratch (pool buffers, returned by
        ///     <see cref="Dispose"/>); the readers belong to <see cref="Mod"/>, which releases them. The walk is single-threaded
        ///     (NDIter's C-order chunks), so nothing is synchronized.
        ///     </para>
        /// </remarks>
        private sealed class ModWalkState : IDisposable
        {
            /// <summary>The operand readers: a's, then values'.</summary>
            [NDBorrowed] // Mod creates and disposes the readers around the walk
            internal readonly Operand[] Readers;

            /// <summary>True for bytes formatting (an <c>S</c> operand: the arena holds bytes), false for str formatting
            ///     (<c>U</c> / <c>StringDType</c>: the arena holds code points).</summary>
            internal readonly bool Bytes;

            /// <summary>Whether a's elements live in the array's own memory (<c>S</c> / <c>U</c>): then an element at the same
            ///     address with the same length IS the last compiled format, no comparison needed.</summary>
            private readonly bool _stableFormat;

            /// <summary>The pool buffer behind <see cref="_arena"/>.</summary>
            private NDArray _arenaArray;

            /// <summary>The arena: the element being formatted (units: 4 bytes for str, 1 for bytes) — or, while the row width is
            ///     still being sampled, the first elements back to back.</summary>
            private byte* _arena;

            /// <summary>The arena's capacity in units.</summary>
            private long _arenaCap;

            /// <summary>The number of units written into the arena.</summary>
            internal long Count;

            /// <summary>The next element's C-order position.</summary>
            internal long Position;

            /// <summary>The longest element so far, in units (a <c>U</c> / <c>S</c> result is that wide, at least 1).</summary>
            internal long MaxLength;

            /// <summary>The last compiled format, or null before the first element.</summary>
            private ModFormat _format;

            /// <summary>The units the last format was compiled from (a copy, for formats read from scratch memory).</summary>
            private byte[] _formatUnits = Array.Empty<byte>();

            /// <summary>The address the last format was read from (meaningful for <see cref="_stableFormat"/>).</summary>
            private byte* _formatAddress;

            /// <summary>The last format's length in units.</summary>
            private long _formatLength = -1;

            /// <summary>
            ///     The formats compiled in this call besides the last one (a per-element format array cycling through a few
            ///     formats would otherwise recompile at every element), filled round-robin; an entry owns a copy of its
            ///     units.
            /// </summary>
            private readonly (ModFormat Format, byte[] Units, long Length)[] _cache = new (ModFormat, byte[], long)[FormatCacheSize];

            /// <summary>The cache slot the next compiled format replaces.</summary>
            private int _cacheNext;

            /// <summary>The number of compiled formats a call keeps (a format array of more distinct formats than this recompiles
            ///     the ones it evicted; each entry costs one units copy).</summary>
            private const int FormatCacheSize = 16;

            /// <summary>The pool buffer behind <see cref="_derivedPoints"/>, or null before the first derived str.</summary>
            private NDArray _derivedPointsArray;

            /// <summary>The code points of the current derived str (a tuple item, a dict value).</summary>
            private uint* _derivedPoints;

            /// <summary>The capacity of <see cref="_derivedPoints"/>.</summary>
            private long _derivedPointsCap;

            /// <summary>The pool buffer behind <see cref="_derivedBytes"/>, or null before the first derived bytes value.</summary>
            private NDArray _derivedBytesArray;

            /// <summary>The bytes of the current derived bytes value or np.float16.</summary>
            private byte* _derivedBytes;

            /// <summary>The capacity of <see cref="_derivedBytes"/>.</summary>
            private long _derivedBytesCap;

            /// <summary>Creates the state of one call.</summary>
            /// <param name="readers">The operand readers (a's, values').</param>
            /// <param name="loop">a's loop: bytes or str formatting.</param>
            /// <param name="size">The number of elements (the broadcast size, at least 1).</param>
            /// <param name="packed">The StringDType result to pack each element into as it is formatted (see
            ///     <see cref="PackElement"/>), or null for a U / S result built row by row (<see cref="StoreRow"/>).</param>
            /// <param name="outer">The broadcast shape (the result's shape).</param>
            internal ModWalkState(Operand[] readers, NDStringMethods.TextLoop loop, long size, NDArray packed, Shape outer)
            {
                Readers = readers;
                Bytes = loop == NDStringMethods.TextLoop.Bytes;
                _stableFormat = loop != NDStringMethods.TextLoop.VString;
                _size = size;
                if (packed is not null)
                {
                    // The arena holds one element at a time (packed, then reset), so it starts small and no lengths are
                    // kept. The fresh result's allocator is locked for the walk: no other thread can reach a new array, so
                    // locking it out of the operands' lock order cannot deadlock.
                    Packing = true;
                    _packAllocator = packed.Storage.VStringSlice.Allocator;
                    _packCells = (byte*)packed.Storage.Address;
                    _packAllocator.acquire_allocator();
                    _packLocked = true;
                    GrowArena(256, exact: true);
                    return;
                }
                _resultShape = outer.Clean();
                // Room for the sampled elements at a typical length; the arena doubles past it.
                GrowArena(RowSample * 32, exact: true);
            }

            /// <summary>True when each element is packed into the StringDType result as soon as it is formatted (a
            ///     StringDType operand), false when each element goes into a <c>U</c> / <c>S</c> row (<see cref="StoreRow"/>).</summary>
            internal readonly bool Packing;

            /// <summary>Whether an element held a code point UTF-8 cannot encode (a surrogate, a unit past U+10FFFF): the
            ///     U → StringDType cast's TypeError, raised by <see cref="Mod"/> only after every element was formatted.</summary>
            internal bool InvalidCodePoint;

            /// <summary>The packed result's allocator (locked from construction to <see cref="Dispose"/>).</summary>
            private readonly VStringAllocator _packAllocator;

            /// <summary>The packed result's first cell.</summary>
            private readonly byte* _packCells;

            /// <summary>Whether <see cref="_packAllocator"/> is still locked by this walk.</summary>
            private bool _packLocked;

            /// <summary>The managed buffer an element's UTF-8 is encoded into before it is packed (grown as needed).</summary>
            private byte[] _packUtf8 = new byte[256];

            /// <summary>
            ///     Packs the element just formatted (the last <paramref name="length"/> code points of the arena) into the
            ///     StringDType result's cell at the walk's position, then empties the arena: NumPy's
            ///     <c>np.asarray(result.tolist()).astype(StringDType())</c> per element — the U element's reading (its trailing
            ///     NULs dropped), encoded as UTF-8. An element holding a surrogate or a unit past U+10FFFF marks
            ///     <see cref="InvalidCodePoint"/> and is not packed (the call raises once the walk is over), and no later
            ///     element is packed either.
            /// </summary>
            /// <param name="length">The element's length in code points.</param>
            /// <exception cref="OutOfMemoryException">The result's arena cannot grow.</exception>
            internal void PackElement(long length)
            {
                uint* src = (uint*)_arena + (Count - length);
                Count = 0;
                if (InvalidCodePoint)
                    return;
                long len = length;
                while (len > 0 && src[len - 1] == 0)
                    len--;
                if (len == 0)
                    return;   // the zeroed cell is the empty string already
                if (_packUtf8.Length < len * 4)
                    _packUtf8 = new byte[Math.Max(len * 4, _packUtf8.Length * 2L)];
                int bytes = 0;
                fixed (byte* buf = _packUtf8)
                {
                    for (long k = 0; k < len; k++)
                    {
                        uint cp = src[k];
                        if (cp < 0x80)
                        {
                            buf[bytes++] = (byte)cp;   // ASCII: every number's text, most formats
                            continue;
                        }
                        if (cp > 0x10FFFF || cp - 0xD800u < 0x800u)
                        {
                            InvalidCodePoint = true;
                            return;
                        }
                        bytes += new System.Text.Rune(cp).EncodeToUtf8(new Span<byte>(buf + bytes, 4));
                    }
                    if (_packAllocator.pack(_packCells + Position * VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING, buf, bytes) < 0)
                        throw new OutOfMemoryException("Failed to allocate string in unicode to string cast");
                }
            }

            /// <summary>The number of elements the walk formats (the broadcast size).</summary>
            private readonly long _size;

            /// <summary>The number of first elements kept in the arena to choose the <c>U</c> / <c>S</c> result's width.</summary>
            internal const int RowSample = 64;

            /// <summary>The <c>U</c> / <c>S</c> result's shape (the broadcast shape, C order).</summary>
            private readonly Shape _resultShape;

            /// <summary>The <c>U</c> / <c>S</c> result being filled, or null while the width is sampled (and after
            ///     <see cref="FinishRows"/> handed it out).</summary>
            private NDArray _rows;

            /// <summary>The first row of <see cref="_rows"/>.</summary>
            private byte* _rowBase;

            /// <summary>The width of <see cref="_rows"/> in units (at least <see cref="MaxLength"/>).</summary>
            private int _rowWidth;

            /// <summary>How many times the rows were re-laid out for a longer element (the first few grow to the exact length,
            ///     later ones by half again, so a call re-lays out O(log width) times).</summary>
            private int _relayouts;

            /// <summary>The lengths of the sampled elements (the arena holds them back to back until the rows start).</summary>
            private readonly long[] _sampleLengths = new long[RowSample];

            /// <summary>
            ///     Stores the element just formatted (the last <paramref name="length"/> units of the arena) as the row at the
            ///     walk's position, then empties the arena. While the width is still being sampled the element stays in the
            ///     arena; the sample's last element starts the rows (<see cref="StartRows"/>).
            /// </summary>
            /// <param name="length">The element's length in units.</param>
            /// <exception cref="OutOfMemoryException">The result cannot be allocated.</exception>
            internal void StoreRow(long length)
            {
                if (length > MaxLength)
                    MaxLength = length;
                if (_rows is null)
                {
                    _sampleLengths[Position] = length;
                    if (Position + 1 == RowSample || Position + 1 == _size)
                        StartRows();
                    return;
                }
                if (length > _rowWidth)
                    GrowRows(length);
                // The vector row copy reads whole vectors from the element's start: room past it in the arena.
                Reserve(VectorRowReach);
                CopyRow(_arena + (Count - length) * UnitSize, length, Position);
                Count = 0;
            }

            /// <summary>Whether the <c>U</c> / <c>S</c> rows exist (the width was chosen): the chunk path then formats an
            ///     element straight into its row (<see cref="CommitDirectRow"/>).</summary>
            internal bool RowsActive => _rows is not null;

            /// <summary>The first row of the <c>U</c> / <c>S</c> result (valid until the rows are re-laid out).</summary>
            internal byte* RowBase => _rowBase;

            /// <summary>The rows' width in units.</summary>
            internal int RowWidth => _rowWidth;

            /// <summary>The number of elements of the walk (the result's size).</summary>
            internal long Size => _size;

            /// <summary>
            ///     Finishes an element the chunk path formatted STRAIGHT into the row at the walk's position (no arena, no
            ///     copy): zeros to the row's width — or, for an element longer than the width (its units ran into the following
            ///     rows' space, which no element has written yet and which the caller checked lies inside the result), a
            ///     re-layout of the rows written so far and the element moved into its wider row.
            /// </summary>
            /// <param name="length">The element's length in units.</param>
            /// <exception cref="OutOfMemoryException">The wider result cannot be allocated.</exception>
            internal void CommitDirectRow(long length)
            {
                int unit = UnitSize;
                byte* rowStart = _rowBase + Position * _rowWidth * unit;
                if (length > MaxLength)
                    MaxLength = length;
                if (length <= _rowWidth)
                {
                    long pad = (_rowWidth - length) * unit;
                    if (pad > 0)
                        new Span<byte>(rowStart + length * unit, (int)pad).Clear();
                    return;
                }
                // The re-layout frees the old rows: park the element in the arena first (it holds nothing in rows mode).
                Reserve(length + VectorRowReach);
                Buffer.MemoryCopy(rowStart, _arena, length * unit, length * unit);
                GrowRows(length);
                CopyRow(_arena, length, Position);
            }

            /// <summary>The size of an arena / result unit in bytes: 1 for bytes formatting, 4 for str formatting.</summary>
            private int UnitSize => Bytes ? 1 : 4;

            /// <summary>The <c>U</c> / <c>S</c> descriptor of a given width.</summary>
            /// <param name="width">The width in characters.</param>
            /// <returns>The descriptor.</returns>
            private DType RowDescr(int width) => Bytes ? DTypeRegistry.Bytes.Descr(width) : DTypeRegistry.Str.Descr(width);

            /// <summary>
            ///     Allocates the result at the sample's longest length (at least 1) and stores the sampled elements, which the
            ///     arena holds back to back, as its first rows; the arena then holds one element at a time.
            /// </summary>
            /// <exception cref="OutOfMemoryException">The result cannot be allocated.</exception>
            private void StartRows()
            {
                _rowWidth = checked((int)Math.Max(1, MaxLength));
                _rows = new NDArray(RowDescr(_rowWidth), _resultShape, false);
                _rowBase = (byte*)_rows.Storage.Address;
                // The vector row copies load whole vectors from an element's start: room past the last sampled element.
                Reserve(VectorRowReach);
                byte* src = _arena;
                for (long i = 0; i <= Position; i++)
                {
                    CopyRow(src, _sampleLengths[i], i);
                    src += _sampleLengths[i] * UnitSize;
                }
                Count = 0;
            }

            /// <summary>
            ///     Re-lays the rows written so far out at a larger width (zero-padded), for an element longer than the current
            ///     width: the exact length for the first few re-layouts, at least half again afterwards (so increasing lengths
            ///     cost O(log width) passes; <see cref="FinishRows"/> narrows an overshoot once).
            /// </summary>
            /// <param name="length">The new element's length.</param>
            /// <exception cref="OutOfMemoryException">The result cannot be allocated.</exception>
            private void GrowRows(long length)
            {
                long wanted = _relayouts < 4 ? length : Math.Max(length, _rowWidth + _rowWidth / 2L);
                int width = checked((int)wanted);
                var grown = new NDArray(RowDescr(width), _resultShape, false);
                CopyRows(_rowBase, _rowWidth, (byte*)grown.Storage.Address, width, Position);
                _rows.Dispose();
                _rows = grown;
                _rowBase = (byte*)grown.Storage.Address;
                _rowWidth = width;
                _relayouts++;
            }

            /// <summary>
            ///     Hands the finished <c>U</c> / <c>S</c> result out: as wide as the longest element (at least 1) — narrowed by
            ///     one copy when the rows' width overshot it.
            /// </summary>
            /// <returns>The result (owned by the caller from now on).</returns>
            /// <exception cref="OutOfMemoryException">The narrowed result cannot be allocated.</exception>
            internal NDArray FinishRows()
            {
                int width = checked((int)Math.Max(1, MaxLength));
                if (width != _rowWidth)
                {
                    var narrow = new NDArray(RowDescr(width), _resultShape, false);
                    CopyRows(_rowBase, _rowWidth, (byte*)narrow.Storage.Address, width, _size);
                    _rows.Dispose();
                    _rows = narrow;
                }
                var result = _rows;
                _rows = null;
                _rowBase = null;
                return result;
            }

            /// <summary>How far past an element's start the vector row copy reads, in arena units: four 8-unit vectors of a
            ///     str row (32 units) or two 32-byte vectors of a bytes row (64 units), at most 64.</summary>
            private const int VectorRowReach = 64;

            /// <summary>
            ///     Writes one element into row <paramref name="row"/>: its units, then zeros to the row's width. A row of up to
            ///     32 code points (str) or 64 bytes (bytes) is written by whole masked 32-byte stores, which may also cover the
            ///     start of the following rows — rewritten when their elements are stored, as rows are stored in C order — but
            ///     never past the result's end (the last rows take the exact copy).
            /// </summary>
            /// <param name="src">The element's units (the arena has <see cref="VectorRowReach"/> units of room past them).</param>
            /// <param name="length">Their number (at most the row width).</param>
            /// <param name="row">The row (C-order position).</param>
            private void CopyRow(byte* src, long length, long row)
            {
                int width = _rowWidth;
                if (Bytes)
                {
                    byte* dst = _rowBase + row * width;
                    int vectors = (width + 31) / 32;
                    // The stores reach vectors * 32 bytes from the row's start: they must end inside the result.
                    if (Vector256.IsHardwareAccelerated && vectors <= 2 && row * width + vectors * 32L <= _size * width)
                    {
                        // The lane numbers 0..31 (Vector256<T>.Indices is .NET 9+; this file builds for net8.0 too).
                        var lanes = Vector256.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
                        var count = Vector256.Create((byte)length);
                        (Vector256.Load(src) & Vector256.LessThan(lanes, count)).Store(dst);
                        if (vectors == 2)
                            (Vector256.Load(src + 32) & Vector256.LessThan(lanes + Vector256.Create((byte)32), count)).Store(dst + 32);
                        return;
                    }
                    Buffer.MemoryCopy(src, dst, length, length);
                    new Span<byte>(dst + length, (int)(width - length)).Clear();
                }
                else
                {
                    uint* dst = (uint*)_rowBase + row * width;
                    int vectors = (width + 7) / 8;
                    if (Vector256.IsHardwareAccelerated && vectors <= 4 && row * width + vectors * 8L <= _size * width)
                    {
                        var lanes = Vector256.Create(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u);
                        var count = Vector256.Create((uint)length);
                        uint* s = (uint*)src;
                        for (int k = 0; k < vectors; k++)
                        {
                            (Vector256.Load(s + 8 * k) & Vector256.LessThan(lanes, count)).Store(dst + 8 * k);
                            lanes += Vector256.Create(8u);
                        }
                        return;
                    }
                    Buffer.MemoryCopy(src, dst, length * 4, length * 4);
                    new Span<uint>(dst + length, (int)(width - length)).Clear();
                }
            }

            /// <summary>Copies the first <paramref name="rows"/> rows of one width into rows of another (zero-padded when
            ///     wider; a narrower width never cuts an element, which <see cref="MaxLength"/> bounds).</summary>
            /// <param name="src">The source rows.</param>
            /// <param name="srcWidth">Their width in units.</param>
            /// <param name="dst">The destination rows.</param>
            /// <param name="dstWidth">Their width in units.</param>
            /// <param name="rows">The number of rows.</param>
            private void CopyRows(byte* src, int srcWidth, byte* dst, int dstWidth, long rows)
            {
                int unit = UnitSize;
                long keep = Math.Min(srcWidth, dstWidth) * (long)unit;
                long pad = (dstWidth - Math.Min(srcWidth, dstWidth)) * (long)unit;
                for (long i = 0; i < rows; i++)
                {
                    byte* s = src + i * srcWidth * unit;
                    byte* d = dst + i * dstWidth * unit;
                    Buffer.MemoryCopy(s, d, keep, keep);
                    if (pad > 0)
                        new Span<byte>(d + keep, (int)pad).Clear();
                }
            }

            /// <summary>The arena's first unit (valid until the next growth).</summary>
            internal byte* Arena => _arena;

            /// <summary>
            ///     The compiled format of a format element: the last one when the element has the same units, else one of the
            ///     call's cached formats with the same units, else a fresh compile (remembered as the last one and cached).
            /// </summary>
            /// <param name="units">The element's units (code points for str, bytes for bytes).</param>
            /// <param name="length">Their number.</param>
            /// <returns>The compiled format.</returns>
            /// <exception cref="OverflowException">A format longer than <see cref="int.MaxValue"/> units.</exception>
            internal ModFormat FormatFor(byte* units, long length)
            {
                long bytes = Bytes ? length : length * 4;
                var span = new ReadOnlySpan<byte>(units, checked((int)bytes));
                if (_format is not null && length == _formatLength)
                {
                    // S / U elements stay where they are for the whole walk: the same address is the same text. A StringDType
                    // text is decoded into the reader's scratch, which every element reuses: compare the units.
                    if (_stableFormat && units == _formatAddress)
                        return _format;
                    if (span.SequenceEqual(_formatUnits.AsSpan(0, (int)bytes)))
                    {
                        _formatAddress = units;
                        return _format;
                    }
                }
                for (int k = 0; k < _cache.Length; k++)
                {
                    ref var entry = ref _cache[k];
                    if (entry.Format is null)
                        break;
                    if (entry.Length == length && span.SequenceEqual(entry.Units))
                    {
                        _format = entry.Format;
                        _formatUnits = entry.Units;
                        _formatLength = length;
                        _formatAddress = units;
                        return _format;
                    }
                }
                _format = Bytes ? ModFormat.Compile(units, length) : ModFormat.Compile((uint*)units, length);
                // The copy is owned by its cache entry (never written again), so the last-format check may share it.
                _formatUnits = span.ToArray();
                _formatAddress = units;
                _formatLength = length;
                _cache[_cacheNext] = (_format, _formatUnits, length);
                _cacheNext = (_cacheNext + 1) % _cache.Length;
                return _format;
            }

            /// <summary>Makes room for <paramref name="units"/> more units in the arena.</summary>
            /// <param name="units">The number of units about to be written.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal void Reserve(long units)
            {
                if (Count + units > _arenaCap)
                    GrowArena(Count + units);
            }

            /// <summary>Grows the arena to hold at least <paramref name="units"/> units, keeping what was written (doubling, so
            ///     the copies cost O(total), unless <paramref name="exact"/> asks for that capacity itself).</summary>
            /// <param name="units">The capacity needed.</param>
            /// <param name="exact">Allocate <paramref name="units"/> exactly (the first allocation, a projection), not at least
            ///     double the current capacity.</param>
            /// <exception cref="OutOfMemoryException">The arena cannot be allocated.</exception>
            private void GrowArena(long units, bool exact = false)
            {
                long cap = exact ? Math.Max(units, 64) : Math.Max(units, Math.Max(64, _arenaCap * 2));
                long unitSize = Bytes ? 1 : 4;
                var grown = new NDArray(NPTypeCode.Byte, Shape.Vector(cap * unitSize), false);
                byte* p = (byte*)grown.Storage.Address;
                if (Count > 0)
                    Buffer.MemoryCopy(_arena, p, Count * unitSize, Count * unitSize);
                _arenaArray?.Dispose();
                _arenaArray = grown;
                _arena = p;
                _arenaCap = cap;
            }

            /// <summary>Appends one code point (str formatting).</summary>
            /// <param name="cp">The code point.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal void AppendPoint(uint cp)
            {
                Reserve(1);
                ((uint*)_arena)[Count++] = cp;
            }

            /// <summary>Appends code points (str formatting).</summary>
            /// <param name="src">The code points.</param>
            /// <param name="count">Their number.</param>
            internal void AppendPoints(uint* src, long count)
            {
                if (count <= 0)
                    return;
                Reserve(count);
                Buffer.MemoryCopy(src, (uint*)_arena + Count, count * 4, count * 4);
                Count += count;
            }

            /// <summary>Appends <paramref name="count"/> spaces (a width's padding; code points or bytes by the mode).</summary>
            /// <param name="count">The number of spaces (nothing when not positive).</param>
            internal void AppendSpaces(long count)
            {
                if (count <= 0)
                    return;
                Reserve(count);
                if (Bytes)
                    new Span<byte>(_arena + Count, checked((int)count)).Fill((byte)' ');
                else
                    new Span<uint>((uint*)_arena + Count, checked((int)count)).Fill(' ');
                Count += count;
            }

            /// <summary>
            ///     Appends a .NET string as code points (str formatting): a surrogate pair is one code point, a lone surrogate
            ///     unit stays a code point of its own (CPython's str holds lone surrogates).
            /// </summary>
            /// <param name="s">The text.</param>
            internal void AppendText(string s)
            {
                Reserve(s.Length);
                uint* dst = (uint*)_arena + Count;
                long k = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    char c = s[i];
                    if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                    {
                        dst[k++] = (uint)char.ConvertToUtf32(c, s[i + 1]);
                        i++;
                    }
                    else
                    {
                        dst[k++] = c;
                    }
                }
                Count += k;
            }

            /// <summary>
            ///     Appends a text conversion's field (str formatting): <paramref name="s"/> cut to <paramref name="precision"/>
            ///     code points, padded with spaces to <paramref name="width"/> code points (on the right when
            ///     <paramref name="left"/>).
            /// </summary>
            /// <param name="s">The <c>str()</c> / <c>repr()</c> text.</param>
            /// <param name="width">The minimum width in code points.</param>
            /// <param name="precision">The maximum number of code points kept (-1 for all).</param>
            /// <param name="left">Left-justify.</param>
            internal void AppendTextField(string s, int width, int precision, bool left)
            {
                long start = Count;
                Reserve(s.Length);
                uint* dst = (uint*)_arena + Count;
                long k = 0;
                for (int i = 0; i < s.Length && (precision < 0 || k < precision); i++)
                {
                    char c = s[i];
                    if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                    {
                        dst[k++] = (uint)char.ConvertToUtf32(c, s[i + 1]);
                        i++;
                    }
                    else
                    {
                        dst[k++] = c;
                    }
                }
                // The text is committed BEFORE any padding is reserved: a growth copies only what Count covers.
                Count = start + k;
                long pad = width - k;
                if (pad <= 0)
                    return;
                if (left)
                {
                    AppendSpaces(pad);
                    return;
                }
                // Right-justify: the text was written first to count its code points; it moves past the padding (an
                // overlapping move, which Span.CopyTo performs as if through a temporary).
                Reserve(pad);
                uint* p = (uint*)_arena + start;
                new Span<uint>(p, checked((int)k)).CopyTo(new Span<uint>(p + pad, checked((int)k)));
                new Span<uint>(p, checked((int)pad)).Fill(' ');
                Count = start + pad + k;
            }

            /// <summary>Appends bytes (bytes formatting).</summary>
            /// <param name="src">The bytes.</param>
            /// <param name="count">Their number.</param>
            internal void AppendBytes(byte* src, long count)
            {
                if (count <= 0)
                    return;
                Reserve(count);
                Buffer.MemoryCopy(src, _arena + Count, count, count);
                Count += count;
            }

            /// <summary>Appends one byte (bytes formatting).</summary>
            /// <param name="b">The byte.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal void AppendByte(byte b)
            {
                Reserve(1);
                _arena[Count++] = b;
            }

            /// <summary>Appends one ASCII unit in the arena's mode (a code point for str formatting, a byte for bytes
            ///     formatting).</summary>
            /// <param name="c">The ASCII character.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal void AppendUnit(byte c)
            {
                Reserve(1);
                if (Bytes)
                    _arena[Count++] = c;
                else
                    ((uint*)_arena)[Count++] = c;
            }

            /// <summary>Appends <paramref name="count"/> copies of an ASCII character in the arena's mode (a field's spaces or
            ///     zeros).</summary>
            /// <param name="c">The character.</param>
            /// <param name="count">The number of copies (nothing when not positive).</param>
            internal void AppendFill(byte c, long count)
            {
                if (count <= 0)
                    return;
                Reserve(count);
                if (Bytes)
                    new Span<byte>(_arena + Count, checked((int)count)).Fill(c);
                else
                    new Span<uint>((uint*)_arena + Count, checked((int)count)).Fill(c);
                Count += count;
            }

            /// <summary>Appends ASCII bytes in the arena's mode: widened to code points for str formatting, copied for bytes
            ///     formatting (the fast path's digit bodies).</summary>
            /// <param name="src">The ASCII bytes.</param>
            /// <param name="count">Their number.</param>
            internal void AppendAscii(byte* src, int count)
            {
                if (count <= 0)
                    return;
                Reserve(count);
                if (Bytes)
                {
                    Buffer.MemoryCopy(src, _arena + Count, count, count);
                }
                else
                {
                    uint* dst = (uint*)_arena + Count;
                    for (int i = 0; i < count; i++)
                        dst[i] = src[i];
                }
                Count += count;
            }

            /// <summary>Appends an ASCII .NET string as bytes (bytes formatting: every conversion that writes text writes ASCII).</summary>
            /// <param name="s">The text (ASCII).</param>
            /// <param name="limit">The maximum number of characters written (-1 for all).</param>
            internal void AppendAscii(string s, int limit = -1)
            {
                int n = limit >= 0 && limit < s.Length ? limit : s.Length;
                Reserve(n);
                byte* dst = _arena + Count;
                for (int i = 0; i < n; i++)
                    dst[i] = (byte)s[i];
                Count += n;
            }

            /// <summary>
            ///     The code points of a derived str (a tuple item, a dict value that is a C# string), in the call's derived
            ///     scratch: a surrogate pair joined, a lone surrogate kept. Valid until the next derived str.
            /// </summary>
            /// <param name="s">The text.</param>
            /// <param name="count">Receives the number of code points.</param>
            /// <returns>The code points.</returns>
            internal uint* DerivedPoints(string s, out long count)
            {
                if (s.Length > _derivedPointsCap)
                {
                    long cap = Math.Max(s.Length, Math.Max(64, _derivedPointsCap * 2));
                    var grown = new NDArray(NPTypeCode.UInt32, Shape.Vector(cap), false);
                    _derivedPointsArray?.Dispose();
                    _derivedPointsArray = grown;
                    _derivedPoints = (uint*)grown.Storage.Address;
                    _derivedPointsCap = cap;
                }
                long n = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    char c = s[i];
                    if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                    {
                        _derivedPoints[n++] = (uint)char.ConvertToUtf32(c, s[i + 1]);
                        i++;
                    }
                    else
                    {
                        _derivedPoints[n++] = c;
                    }
                }
                count = n;
                return _derivedPoints;
            }

            /// <summary>Room for <paramref name="n"/> bytes of a derived value (a bytes object, an np.float16), in the call's
            ///     derived scratch. Valid until the next derived bytes value.</summary>
            /// <param name="n">The number of bytes.</param>
            /// <returns>The scratch.</returns>
            internal byte* DerivedBytes(long n)
            {
                if (n > _derivedBytesCap)
                {
                    long cap = Math.Max(n, Math.Max(64, _derivedBytesCap * 2));
                    var grown = new NDArray(NPTypeCode.Byte, Shape.Vector(cap), false);
                    _derivedBytesArray?.Dispose();
                    _derivedBytesArray = grown;
                    _derivedBytes = (byte*)grown.Storage.Address;
                    _derivedBytesCap = cap;
                }
                return _derivedBytes;
            }

            /// <summary>Returns the arena, the lengths and the derived scratch to the buffer pool.</summary>
            public void Dispose()
            {
                if (_packLocked)
                {
                    _packLocked = false;
                    _packAllocator.release_allocator();
                }
                _arenaArray?.Dispose();
                _rows?.Dispose();   // a result never handed out (the walk raised)
                _derivedPointsArray?.Dispose();
                _derivedBytesArray?.Dispose();
                _arenaArray = _rows = _derivedPointsArray = _derivedBytesArray = null;
                _arena = null;
                _derivedPoints = null;
                _derivedBytes = null;
            }
        }

        // =============================================================================================================
        // The walk and the per-element formatting
        // =============================================================================================================

        /// <summary>One NDIter chunk of the walk: each element's two scalars, its formatting, its length.</summary>
        /// <param name="dataptrs">The operand pointers (a, values).</param>
        /// <param name="strides">Their byte strides.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="w">The call's state.</param>
        /// <exception cref="Exception">The first element's error (see <see cref="Mod"/>).</exception>
        private static void ModRun(void** dataptrs, long* strides, long n, ModWalkState w)
        {
            byte* pa = (byte*)dataptrs[0];
            long sa = strides[0];
            byte* pv = (byte*)dataptrs[1];
            long sv = strides[1];
            var formatReader = w.Readers[0];
            var valueReader = w.Readers[1];
            Elem format = default;
            ModFormat program = null;
            if (sa == 0 && n > 0)
            {
                // A format that does not move within the chunk (one format broadcast over the values: NumPy's common call)
                // is read and compiled once — its object, and so its compiled steps, are the same for every element. A
                // StringDType format's decoded text lives in the format reader's scratch, which only a read of that reader
                // replaces. PyArray_ToScalar of a still comes first (a U one-unit text past U+10FFFF raises here).
                formatReader.Read(pa, out format);
                program = CompileFormat(in format, w);
                // One positional argument and a value kind the chunk path writes: the whole chunk in one loop. A str format
                // must pass the descriptor check first (a StringDType NA format is refused element by element below).
                if (!ModFastPathsDisabled && program is not null && program.FastSpec >= 0 && (w.Bytes || IsStr(format))
                    && TryModFastChunk(program, in format, pv, sv, n, w))
                    return;
            }
            for (long i = 0; i < n; i++, pa += sa, pv += sv)
            {
                // PyArray_ToScalar of both operands, a first (a U one-unit text past U+10FFFF raises here); a moving format
                // is read and compiled per element (compiled formats are cached by their units).
                if (sa != 0)
                {
                    formatReader.Read(pa, out format);
                    program = CompileFormat(in format, w);
                }
                valueReader.Read(pv, out var value);
                long start = w.Count;
                if (w.Bytes)
                    ModBytesElement(in format, program, in value, w);
                else
                    ModStrElement(in format, program, in value, w);
                RecordLength(w, w.Count - start);
            }
        }

        /// <summary>The compiled steps of a format element, or null for a non-text format (a StringDType NA object has no
        ///     units; its element refuses it before compiling would matter).</summary>
        /// <param name="format">The format element.</param>
        /// <param name="w">The call's state (the compiled-format cache).</param>
        /// <returns>The compiled format, or null.</returns>
        private static ModFormat CompileFormat(in Elem format, ModWalkState w)
            => format.Kind is ElemKind.NpStr or ElemKind.Str or ElemKind.NpBytes or ElemKind.Bytes
                ? w.FormatFor(format.Units, format.Length)
                : null;

        /// <summary>
        ///     PyUnicode_Format's argument state: the current argument (the value, or a key lookup's result), a tuple's items
        ///     and CPython's <c>arglen</c> / <c>argidx</c> counters (-1 / -2 for a single argument), and whether the value is a
        ///     mapping.
        /// </summary>
        private struct ModArgs
        {
            /// <summary>The single argument the next spec takes (the value, or the last key lookup's result).</summary>
            internal Elem Current;

            /// <summary>The value (the mapping a key is looked up in).</summary>
            internal Elem Mapping;

            /// <summary>A tuple value's items, or null.</summary>
            internal ITuple Items;

            /// <summary>CPython's <c>arglen</c>: a tuple's length, or -1 for a single argument.</summary>
            internal long ArgLen;

            /// <summary>CPython's <c>argidx</c>: the next tuple item, or -2 before a single argument is taken.</summary>
            internal long ArgIdx;

            /// <summary>CPython's <c>dict != NULL</c>: the value is a mapping (it feeds keys and silences the left-over
            ///     check).</summary>
            internal bool IsMapping;

            /// <summary>The state of one formatting call over <paramref name="value"/>.</summary>
            /// <param name="value">The argument.</param>
            /// <param name="bytesFormat">True for bytes formatting (its mapping rule excludes bytes values).</param>
            internal ModArgs(in Elem value, bool bytesFormat)
            {
                Current = value;
                Mapping = value;
                Items = value.Kind == ElemKind.Object ? value.Obj as ITuple : null;
                if (Items is not null)
                {
                    ArgLen = Items.Length;
                    ArgIdx = 0;
                    IsMapping = false;   // a tuple is never the mapping
                }
                else
                {
                    ArgLen = -1;
                    ArgIdx = -2;
                    IsMapping = bytesFormat ? IsBytesMapping(in value) : IsStrMapping(in value);
                }
            }
        }

        /// <summary>CPython's <c>getnextarg</c>: the next tuple item (converted to an element), or the single argument once.</summary>
        /// <param name="args">The argument state.</param>
        /// <param name="w">The call's state (the derived scratch a tuple item lives in).</param>
        /// <param name="e">Receives the argument.</param>
        /// <exception cref="TypeError"><c>not enough arguments for format string</c>.</exception>
        private static void NextArg(ref ModArgs args, ModWalkState w, out Elem e)
        {
            if (args.ArgIdx < args.ArgLen)
            {
                args.ArgIdx++;
                if (args.ArgLen < 0)
                {
                    e = args.Current;
                    return;
                }
                ElemOfObject(args.Items[(int)(args.ArgIdx - 1)], w, out e);
                return;
            }
            throw new TypeError(NotEnoughArgumentsText);
        }

        /// <summary>
        ///     One <c>str.__mod__</c> call (U / StringDType): the descriptor's check, then PyUnicode_Format over the compiled
        ///     format and the value, written into the arena.
        /// </summary>
        /// <param name="format">a's element (a str, or a StringDType NA object).</param>
        /// <param name="program">The format's compiled steps (null for a non-str format, which is refused first).</param>
        /// <param name="value">values' element.</param>
        /// <param name="w">The call's state.</param>
        /// <exception cref="Exception">CPython's error for the element (see <see cref="Mod"/>).</exception>
        private static void ModStrElement(in Elem format, ModFormat program, in Elem value, ModWalkState w)
        {
            // str.__mod__'s slot wrapper refuses a non-str self before anything is formatted.
            if (!IsStr(format))
                throw new TypeError($"descriptor '__mod__' requires a 'str' object but received a '{TypeName(format)}'");
            uint* src = (uint*)format.Units;
            try
            {
                // A format of one positional argument: its literal runs around one conversion, without the argument state.
                if (!ModFastPathsDisabled && program.FastSpec >= 0 && TryModFastElement(program, (byte*)src, in value, w))
                    return;
                var args = new ModArgs(in value, bytesFormat: false);
                foreach (ref readonly var op in program.Ops.AsSpan())
                {
                    switch (op.Kind)
                    {
                        case ModFormat.OpKind.Text:
                            w.AppendPoints(src + op.Start, op.Length);
                            break;
                        case ModFormat.OpKind.Percent:
                            w.AppendPoint('%');
                            break;
                        default:
                            RunStrSpec(in op, src, ref args, w);
                            break;
                    }
                }
                // PyUnicode_Format's last check: an argument no spec took (silenced by a mapping).
                if (args.ArgIdx < args.ArgLen && !args.IsMapping)
                    throw new TypeError(NotAllConvertedStrText);
            }
            catch (PrintfArgumentException e)
            {
                // PrintfFormatter's TypeErrors carry the savetxt marker type; np.strings.mod surfaces CPython's plain
                // TypeError with the same text, as NumPy does.
                throw new TypeError(e.Message);
            }
        }

        /// <summary>
        ///     One spec of str formatting: the key lookup, the parse error at CPython's point, the <c>*</c> arguments, the
        ///     value, the conversion character's check, then the conversion.
        /// </summary>
        /// <param name="op">The spec.</param>
        /// <param name="src">The format's code points (the key's source).</param>
        /// <param name="args">The argument state.</param>
        /// <param name="w">The call's state.</param>
        /// <exception cref="Exception">CPython's error for the spec.</exception>
        private static void RunStrSpec(in ModFormat.Op op, uint* src, ref ModArgs args, ModWalkState w)
        {
            if (op.HasKey)
            {
                // unicode_format_arg_parse: the mapping is required before the key is read; then an unbalanced key; then
                // the lookup, whose result becomes the single argument of the rest of the spec.
                if (!args.IsMapping)
                    throw new TypeError(FormatRequiresMappingText);
                if (op.Failure == ModFormat.ParseFailure.IncompleteKey)
                    throw new ValueError("incomplete format key");
                LookupStr(in args.Mapping, src + op.Start, op.Length, w, out args.Current);
                args.ArgLen = -1;
                args.ArgIdx = -2;
            }
            if (op.Failure == ModFormat.ParseFailure.WidthTooBig)
                throw new ValueError("width too big");
            bool left = op.Left;
            long width = op.Width;
            if (op.WidthStar)
            {
                NextArg(ref args, w, out var star);
                width = StarWidth(in star);
                // A negative '*' width left-justifies; CPython negates it as a Py_ssize_t, so the most negative value stays
                // negative and pads nothing.
                if (width < 0)
                {
                    left = true;
                    width = unchecked(-width);
                }
            }
            if (op.Failure == ModFormat.ParseFailure.PrecisionTooBig)
                throw new ValueError("precision too big");
            int precision = op.Precision;
            if (op.PrecisionStar)
            {
                NextArg(ref args, w, out var star);
                precision = StarPrecision(in star);
                if (precision < 0)
                    precision = 0;   // a negative '*' precision is 0, not "none"
            }
            if (op.Failure == ModFormat.ParseFailure.Incomplete)
                throw new ValueError("incomplete format");

            // The value is taken BEFORE the conversion character is checked: '%s %y' % 5 reports the missing argument.
            NextArg(ref args, w, out var v);
            int conv = op.Conversion;
            if (!IsStrConversion(conv))
                throw UnsupportedStrConversion(conv, op.ConversionIndex);
            // A width past int.MaxValue cannot be met by a .NET string: it pads to int.MaxValue, whose allocation raises
            // OutOfMemoryException after the conversion's own checks (where CPython's MemoryError comes too).
            int w32 = width < 0 ? 0 : width > int.MaxValue ? int.MaxValue : (int)width;
            ConvertStr((char)conv, left, in op, w32, precision, in v, w);
        }

        /// <summary>
        ///     One conversion of str formatting into the arena: a str value's own code points for <c>s</c> / <c>c</c> (and its
        ///     repr for <c>r</c> / <c>a</c>); every other value through <see cref="PrintfFormatter"/>'s CPython conversions.
        /// </summary>
        /// <param name="conv">The conversion character (one <see cref="IsStrConversion"/> accepts).</param>
        /// <param name="left">Left-justify (the <c>-</c> flag or a negative <c>*</c> width).</param>
        /// <param name="op">The spec (its numeric flags).</param>
        /// <param name="width">The minimum width (0 for none).</param>
        /// <param name="precision">The precision (-1 for none).</param>
        /// <param name="v">The value.</param>
        /// <param name="w">The call's state.</param>
        /// <exception cref="TypeError">A conversion the value's type refuses.</exception>
        /// <exception cref="ValueError">A NaN under an integer conversion; an int with more than 4300 digits.</exception>
        /// <exception cref="OverflowException">An infinity under an integer conversion; a <c>%c</c> code point out of range;
        ///     an int too large for a float; <c>precision too large</c>.</exception>
        private static void ConvertStr(char conv, bool left, in ModFormat.Op op, int width, int precision, in Elem v, ModWalkState w)
        {
            switch (v.Kind)
            {
                case ElemKind.NpStr:
                case ElemKind.Str:
                    TextIntoStr(conv, left, width, precision, in v, w);
                    return;
                case ElemKind.NpBytes:
                    // np.bytes_ subclasses bytes: str() and repr() are its repr (repr names np.bytes_), numbers refuse it.
                    w.AppendText(PrintfFormatter.ApplyNumPyScalar(conv, left, op.Plus, op.Space, op.Zero, op.Alt, width,
                        precision, BytesArray(in v), NPTypeCode.String));
                    return;
                case ElemKind.Bytes:
                    w.AppendText(PrintfFormatter.ApplyPyObject(conv, left, op.Plus, op.Space, op.Zero, op.Alt, width,
                        precision, BytesArray(in v)));
                    return;
                case ElemKind.Number:
                    w.AppendText(PrintfFormatter.ApplyNumPyScalar(conv, left, op.Plus, op.Space, op.Zero, op.Alt, width,
                        precision, BoxNumber(in v), v.Number));
                    return;
                default:
                    w.AppendText(PrintfFormatter.ApplyPyObject(conv, left, op.Plus, op.Space, op.Zero, op.Alt, width,
                        precision, v.Obj));
                    return;
            }
        }

        /// <summary>
        ///     A str value (<c>np.str_</c> / Python <c>str</c>) under str formatting, over its code points (a <c>U</c> unit past
        ///     U+10FFFF included, which no .NET string holds): <c>s</c> the text, <c>r</c> / <c>a</c> its repr / ascii, <c>c</c>
        ///     its one character; the numeric conversions refuse it with CPython's texts.
        /// </summary>
        /// <param name="conv">The conversion character.</param>
        /// <param name="left">Left-justify.</param>
        /// <param name="width">The minimum width in code points.</param>
        /// <param name="precision">The maximum number of code points kept by <c>s</c> / <c>r</c> / <c>a</c> (-1 for all).</param>
        /// <param name="v">The value.</param>
        /// <param name="w">The call's state.</param>
        /// <exception cref="TypeError"><c>%c requires int or char</c>; <c>%d format: a real number is required, not
        ///     numpy.str_</c>; <c>%x format: an integer is required, not str</c>; <c>must be real number, not str</c>.</exception>
        private static void TextIntoStr(char conv, bool left, int width, int precision, in Elem v, ModWalkState w)
        {
            uint* units = (uint*)v.Units;
            long n = v.Length;
            switch (conv)
            {
                case 's':
                {
                    long keep = precision >= 0 && precision < n ? precision : n;
                    long pad = width - keep;
                    if (pad > 0 && !left)
                        w.AppendSpaces(pad);
                    w.AppendPoints(units, keep);
                    if (pad > 0 && left)
                        w.AppendSpaces(pad);
                    return;
                }
                case 'r':
                case 'a':
                {
                    string repr = PyRepr.StrCodePoints(new ReadOnlySpan<uint>(units, checked((int)n)));
                    if (v.Kind == ElemKind.NpStr)
                        repr = "np.str_(" + repr + ")";
                    if (conv == 'a')
                        repr = PyRepr.Ascii(repr);
                    w.AppendTextField(repr, width, precision, left);
                    return;
                }
                case 'c':
                {
                    // formatchar: a str of exactly one character is its own %c; the precision does not cut it.
                    if (n != 1)
                        throw new TypeError("%c requires int or char");
                    if (width > 1 && !left)
                        w.AppendSpaces(width - 1);
                    w.AppendPoint(units[0]);
                    if (width > 1 && left)
                        w.AppendSpaces(width - 1);
                    return;
                }
                case 'd':
                case 'i':
                case 'u':
                    throw new TypeError($"%{conv} format: a real number is required, not {TypeName(v)}");
                case 'x':
                case 'X':
                case 'o':
                    throw new TypeError($"%{conv} format: an integer is required, not {TypeName(v)}");
                default:
                    throw new TypeError($"must be real number, not {TypeName(v)}");
            }
        }

        /// <summary>
        ///     One <c>bytes.__mod__</c> call (S): _PyBytes_FormatEx over the compiled format and the value, written into the
        ///     arena.
        /// </summary>
        /// <param name="format">a's element (an np.bytes_).</param>
        /// <param name="program">The format's compiled steps.</param>
        /// <param name="value">values' element.</param>
        /// <param name="w">The call's state.</param>
        /// <exception cref="Exception">CPython's error for the element (see <see cref="Mod"/>).</exception>
        private static void ModBytesElement(in Elem format, ModFormat program, in Elem value, ModWalkState w)
        {
            byte* src = format.Units;
            try
            {
                // A format of one positional argument: its literal runs around one conversion, without the argument state.
                if (!ModFastPathsDisabled && program.FastSpec >= 0 && TryModFastElement(program, src, in value, w))
                    return;
                var args = new ModArgs(in value, bytesFormat: true);
                foreach (ref readonly var op in program.Ops.AsSpan())
                {
                    switch (op.Kind)
                    {
                        case ModFormat.OpKind.Text:
                            w.AppendBytes(src + op.Start, op.Length);
                            break;
                        case ModFormat.OpKind.Percent:
                            w.AppendByte((byte)'%');
                            break;
                        default:
                            RunBytesSpec(in op, src, ref args, w);
                            break;
                    }
                }
                if (args.ArgIdx < args.ArgLen && !args.IsMapping)
                    throw new TypeError(NotAllConvertedBytesText);
            }
            catch (PrintfArgumentException e)
            {
                throw new TypeError(e.Message);
            }
        }

        /// <summary>
        ///     One spec of bytes formatting: the same steps as <see cref="RunStrSpec"/> with bytes keys, the bytes
        ///     conversions and the bytes formatter's unsupported-character rule.
        /// </summary>
        /// <param name="op">The spec.</param>
        /// <param name="src">The format's bytes (the key's source).</param>
        /// <param name="args">The argument state.</param>
        /// <param name="w">The call's state.</param>
        /// <exception cref="Exception">CPython's error for the spec.</exception>
        private static void RunBytesSpec(in ModFormat.Op op, byte* src, ref ModArgs args, ModWalkState w)
        {
            if (op.HasKey)
            {
                if (!args.IsMapping)
                    throw new TypeError(FormatRequiresMappingText);
                if (op.Failure == ModFormat.ParseFailure.IncompleteKey)
                    throw new ValueError("incomplete format key");
                LookupBytes(in args.Mapping, src + op.Start, op.Length, w, out args.Current);
                args.ArgLen = -1;
                args.ArgIdx = -2;
            }
            if (op.Failure == ModFormat.ParseFailure.WidthTooBig)
                throw new ValueError("width too big");
            bool left = op.Left;
            long width = op.Width;
            if (op.WidthStar)
            {
                NextArg(ref args, w, out var star);
                width = StarWidth(in star);
                if (width < 0)
                {
                    left = true;
                    width = unchecked(-width);
                }
            }
            // The bytes formatter's own text for the precision's overflow (str formatting says "precision too big").
            if (op.Failure == ModFormat.ParseFailure.PrecisionTooBig)
                throw new ValueError("prec too big");
            int precision = op.Precision;
            if (op.PrecisionStar)
            {
                NextArg(ref args, w, out var star);
                precision = StarPrecision(in star);
                if (precision < 0)
                    precision = 0;
            }
            if (op.Failure == ModFormat.ParseFailure.Incomplete)
                throw new ValueError("incomplete format");

            NextArg(ref args, w, out var v);
            int conv = op.Conversion;
            if (!IsBytesConversion(conv))
                throw UnsupportedBytesConversion(conv, op.ConversionIndex);
            // _PyBytes_FormatEx turns 'i' into 'd' before converting, so its errors name %d ("%d format: a real number is
            // required, not str" for a '%i' spec); str formatting keeps the 'i'.
            if (conv == 'i')
                conv = 'd';
            int w32 = width < 0 ? 0 : width > int.MaxValue ? int.MaxValue : (int)width;
            ConvertBytes((char)conv, left, in op, w32, precision, in v, w);
        }

        /// <summary>
        ///     One conversion of bytes formatting into the arena (see the file header): <c>s</c> / <c>b</c> the value's bytes,
        ///     <c>r</c> / <c>a</c> its ascii repr, <c>c</c> one byte, the numbers as str formatting writes them (ASCII).
        /// </summary>
        /// <param name="conv">The conversion character (one <see cref="IsBytesConversion"/> accepts).</param>
        /// <param name="left">Left-justify.</param>
        /// <param name="op">The spec (its numeric flags).</param>
        /// <param name="width">The minimum width.</param>
        /// <param name="precision">The precision (-1 for none).</param>
        /// <param name="v">The value.</param>
        /// <param name="w">The call's state.</param>
        /// <exception cref="TypeError">A conversion the value refuses (<c>%b requires a bytes-like object ...</c>, <c>%c requires
        ///     an integer in range(256) or a single byte</c>, <c>float argument required, not str</c>, the integer
        ///     conversions' texts).</exception>
        /// <exception cref="ValueError">A NaN under an integer conversion.</exception>
        /// <exception cref="OverflowException"><c>%c arg not in range(256)</c>; an infinity under an integer conversion;
        ///     <c>precision too large</c>.</exception>
        /// <exception cref="NotSupportedException">An ndarray object's bytes with no C-order copy in NumSharp.</exception>
        private static void ConvertBytes(char conv, bool left, in ModFormat.Op op, int width, int precision, in Elem v, ModWalkState w)
        {
            switch (conv)
            {
                case 'r':
                case 'a':
                {
                    // %r is an alias of %a for bytes: ascii(repr(v)), cut to the precision, padded with spaces.
                    string repr = PyRepr.Ascii(ReprText(in v));
                    int n = precision >= 0 && precision < repr.Length ? precision : repr.Length;
                    if (width > n && !left)
                        w.AppendSpaces(width - n);
                    w.AppendAscii(repr, n);
                    if (width > n && left)
                        w.AppendSpaces(width - n);
                    return;
                }
                case 's':
                case 'b':
                {
                    // format_obj: a bytes value, else the buffer protocol; the precision cuts the bytes.
                    byte[] ndBytes = null;
                    byte* data;
                    long n;
                    if (v.Kind == ElemKind.Object && v.Obj is NDArray nd)
                    {
                        // An ndarray object exposes its data: the C-order bytes (PyBUF_FULL_RO copied contiguous).
                        ndBytes = nd.tobytes('C');
                        n = ndBytes.Length;
                        data = null;
                    }
                    else if (!TryBuffer(in v, out data, out n))
                    {
                        throw new TypeError("%b requires a bytes-like object, or an object that implements __bytes__, not '"
                                            + TypeName(v) + "'");
                    }
                    if (precision >= 0 && precision < n)
                        n = precision;
                    if (width > n && !left)
                        w.AppendSpaces(width - n);
                    if (ndBytes is not null)
                    {
                        fixed (byte* p = ndBytes)
                            w.AppendBytes(p, n);
                    }
                    else
                    {
                        w.AppendBytes(data, n);
                    }
                    if (width > n && left)
                        w.AppendSpaces(width - n);
                    return;
                }
                case 'c':
                {
                    byte b = ByteConverter(in v);
                    if (width > 1 && !left)
                        w.AppendSpaces(width - 1);
                    w.AppendByte(b);
                    if (width > 1 && left)
                        w.AppendSpaces(width - 1);
                    return;
                }
                case 'e':
                case 'E':
                case 'f':
                case 'F':
                case 'g':
                case 'G':
                    // formatfloat (bytes): PyFloat_AsDouble, and ANY failure becomes this one TypeError (an int too large for
                    // a float included, which str formatting reports as OverflowError).
                    if (!HasFloatValue(in v))
                        throw new TypeError("float argument required, not " + TypeName(v));
                    w.AppendAscii(NumberText(conv, left, in op, width, precision, in v));
                    return;
                default:
                    // d i u o x X: formatlong, the same text (and errors) as str formatting's.
                    w.AppendAscii(NumberText(conv, left, in op, width, precision, in v));
                    return;
            }
        }

        /// <summary>
        ///     An integer or float conversion's text (ASCII) through <see cref="PrintfFormatter"/>: a NumPy scalar's or a Python
        ///     object's CPython conversion; a text value is refused with the integer conversions' texts.
        /// </summary>
        /// <param name="conv">The conversion character (<c>d i u o x X e E f F g G</c>).</param>
        /// <param name="left">Left-justify.</param>
        /// <param name="op">The spec (its numeric flags).</param>
        /// <param name="width">The minimum width.</param>
        /// <param name="precision">The precision (-1 for none).</param>
        /// <param name="v">The value.</param>
        /// <returns>The field's text.</returns>
        /// <exception cref="TypeError">A value the conversion refuses.</exception>
        /// <exception cref="ValueError">A NaN under an integer conversion.</exception>
        /// <exception cref="OverflowException">An infinity under an integer conversion; <c>precision too large</c>.</exception>
        private static string NumberText(char conv, bool left, in ModFormat.Op op, int width, int precision, in Elem v)
        {
            switch (v.Kind)
            {
                case ElemKind.Number:
                    return PrintfFormatter.ApplyNumPyScalar(conv, left, op.Plus, op.Space, op.Zero, op.Alt, width, precision,
                        BoxNumber(in v), v.Number);
                case ElemKind.Object:
                    return PrintfFormatter.ApplyPyObject(conv, left, op.Plus, op.Space, op.Zero, op.Alt, width, precision, v.Obj);
                default:
                    // A text value: formatlong's TypeError (the float conversions refused it before, with their own text).
                    if (conv is 'x' or 'X' or 'o')
                        throw new TypeError($"%{conv} format: an integer is required, not {TypeName(v)}");
                    throw new TypeError($"%{conv} format: a real number is required, not {TypeName(v)}");
            }
        }

        /// <summary>
        ///     Whether <c>PyFloat_AsDouble</c> succeeds for the value: every NumPy numeric scalar (np.bool and the complex
        ///     scalars too), a Python int that fits a float, a Python float / np.float16 / decimal; not a text value, None, a
        ///     complex object, a container.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <returns>True when the value converts.</returns>
        private static bool HasFloatValue(in Elem v)
        {
            switch (v.Kind)
            {
                case ElemKind.Number:
                    return true;
                case ElemKind.Object:
                    if (PrintfFormatter.TryPyInt(v.Obj, out BigInteger big))
                        return !double.IsInfinity(PythonInt.ToDouble(big));
                    return v.Obj is double or float or Half or decimal;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     bytes formatting's <c>byte_converter</c>: a bytes value of exactly one byte, or an integer (a NumPy integer or a
        ///     Python int / bool, through <c>__index__</c>) in <c>range(256)</c>.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <returns>The byte.</returns>
        /// <exception cref="OverflowException"><c>%c arg not in range(256)</c>.</exception>
        /// <exception cref="TypeError"><c>%c requires an integer in range(256) or a single byte</c>.</exception>
        private static byte ByteConverter(in Elem v)
        {
            switch (v.Kind)
            {
                case ElemKind.NpBytes:
                case ElemKind.Bytes:
                    if (v.Length == 1)
                        return v.Units[0];
                    break;
                case ElemKind.Number:
                    if (IsIntegerNumber(v.Number))
                    {
                        // PyLong_AsLongAndOverflow: an overflow of the C long is reported like any value out of range.
                        var big = IntegerOf(in v);
                        if (big.Sign < 0 || big > 255)
                            throw new OverflowException("%c arg not in range(256)");
                        return (byte)big;
                    }
                    break;   // np.bool, the floats and complex have no __index__
                case ElemKind.Object:
                    if (PrintfFormatter.TryPyInt(v.Obj, out BigInteger value))
                    {
                        if (value.Sign < 0 || value > 255)
                            throw new OverflowException("%c arg not in range(256)");
                        return (byte)value;
                    }
                    break;
            }
            throw new TypeError("%c requires an integer in range(256) or a single byte");
        }

        /// <summary>Whether a NumPy dtype's scalars are integers with an <c>__index__</c> (every integer dtype and NumSharp's
        ///     <c>Char</c>, a <c>uint16</c>; not <c>bool</c>: <c>np.bool.__index__</c> raises in NumPy 2).</summary>
        /// <param name="tc">The dtype.</param>
        /// <returns>True for an integer dtype.</returns>
        private static bool IsIntegerNumber(NPTypeCode tc)
            => tc is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;

        /// <summary>An integer NumPy scalar's value.</summary>
        /// <param name="v">The value (a <see cref="ElemKind.Number"/> of an integer dtype).</param>
        /// <returns>The integer.</returns>
        private static BigInteger IntegerOf(in Elem v) => v.Number switch
        {
            // Each arm converted explicitly: a switch expression first converts its arms to their best common type.
            NPTypeCode.SByte => (BigInteger)(*(sbyte*)v.Raw),
            NPTypeCode.Byte => (BigInteger)(*v.Raw),
            NPTypeCode.Int16 => (BigInteger)(*(short*)v.Raw),
            NPTypeCode.UInt16 => (BigInteger)(*(ushort*)v.Raw),
            NPTypeCode.Char => (BigInteger)(*(ushort*)v.Raw),
            NPTypeCode.Int32 => (BigInteger)(*(int*)v.Raw),
            NPTypeCode.UInt32 => (BigInteger)(*(uint*)v.Raw),
            NPTypeCode.Int64 => (BigInteger)(*(long*)v.Raw),
            NPTypeCode.UInt64 => (BigInteger)(*(ulong*)v.Raw),
            _ => throw new ArgumentOutOfRangeException(nameof(v), v.Number, "not an integer dtype"),
        };

        /// <summary>
        ///     Python's <c>repr()</c> of a value: <c>np.str_('a')</c> / <c>'a'</c>, <c>np.bytes_(b'a')</c> / <c>b'a'</c>, NumPy 2's
        ///     scalar repr (<c>np.int64(5)</c>, <c>np.True_</c>), a Python object's repr.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <returns>The repr.</returns>
        private static string ReprText(in Elem v)
        {
            switch (v.Kind)
            {
                case ElemKind.NpStr:
                    return "np.str_(" + PyRepr.StrCodePoints(new ReadOnlySpan<uint>(v.Units, checked((int)v.Length))) + ")";
                case ElemKind.Str:
                    return PyRepr.StrCodePoints(new ReadOnlySpan<uint>(v.Units, checked((int)v.Length)));
                case ElemKind.NpBytes:
                    return "np.bytes_(" + PyRepr.Bytes(new ReadOnlySpan<byte>(v.Units, checked((int)v.Length))) + ")";
                case ElemKind.Bytes:
                    return PyRepr.Bytes(new ReadOnlySpan<byte>(v.Units, checked((int)v.Length)));
                case ElemKind.Number:
                    return PrintfFormatter.ApplyNumPyScalar('r', false, false, false, false, false, 0, -1, BoxNumber(in v), v.Number);
                default:
                    return NDPyObject.Repr(v.Obj);
            }
        }

        // =============================================================================================================
        // Mappings and the arguments a spec reads
        // =============================================================================================================

        /// <summary>
        ///     PyUnicode_Format's mapping test (<c>PyMapping_Check(args) &amp;&amp; !PyTuple_Check(args) &amp;&amp;
        ///     !PyUnicode_Check(args)</c>): a NumPy scalar (numbers, bool, np.float16), a bytes value, a dict, a list, an ndarray.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <returns>True when the value is a mapping to str formatting.</returns>
        private static bool IsStrMapping(in Elem v) => v.Kind switch
        {
            ElemKind.Number or ElemKind.NpBytes or ElemKind.Bytes => true,
            ElemKind.Object => v.Obj is IDictionary || v.Obj is NDArray || (v.Obj is IList && v.Obj is not ITuple),
            _ => false,
        };

        /// <summary>
        ///     _PyBytes_FormatEx's mapping test (a subscript slot, not a tuple, bytes, str or bytearray): a NumPy numeric scalar,
        ///     a dict, a list, an ndarray.
        /// </summary>
        /// <param name="v">The value.</param>
        /// <returns>True when the value is a mapping to bytes formatting.</returns>
        private static bool IsBytesMapping(in Elem v) => v.Kind switch
        {
            ElemKind.Number => true,
            ElemKind.Object => v.Obj is IDictionary || v.Obj is NDArray || (v.Obj is IList && v.Obj is not ITuple),
            _ => false,
        };

        /// <summary>
        ///     A <c>%(key)</c> lookup of str formatting (<c>PyObject_GetItem(dict, key)</c> with a str key): a dict's value by
        ///     Python equality, or the subscript error of the mapping's type.
        /// </summary>
        /// <param name="mapping">The value (a mapping: <see cref="IsStrMapping"/>).</param>
        /// <param name="key">The key's code points.</param>
        /// <param name="keyLength">Their number.</param>
        /// <param name="w">The call's state (the derived scratch the result lives in).</param>
        /// <param name="result">Receives the value looked up.</param>
        /// <exception cref="IndexError">A NumPy scalar (<c>invalid index to scalar variable.</c>) or an ndarray.</exception>
        /// <exception cref="TypeError">A bytes value (<c>byte indices must be integers or slices, not str</c>) or a list
        ///     (<c>list indices must be integers or slices, not str</c>).</exception>
        /// <exception cref="KeyError">A dict without the key (the text is the key's repr).</exception>
        private static void LookupStr(in Elem mapping, uint* key, int keyLength, ModWalkState w, out Elem result)
        {
            switch (mapping.Kind)
            {
                case ElemKind.Number:
                    throw new IndexError(ScalarIndexText);
                case ElemKind.NpBytes:
                case ElemKind.Bytes:
                    throw new TypeError("byte indices must be integers or slices, not str");
            }
            switch (mapping.Obj)
            {
                case IDictionary dict:
                {
                    // A key holding a unit past U+10FFFF has no .NET string: no C# key can equal it.
                    string text = KeyString(key, keyLength);
                    if (text is not null && TryFindStrKey(dict, text, out object found))
                    {
                        ElemOfObject(found, w, out result);
                        return;
                    }
                    throw new KeyError(PyRepr.StrCodePoints(new ReadOnlySpan<uint>(key, keyLength)));
                }
                case NDArray:
                    throw new IndexError(ArrayIndexText);
                default:
                    throw new TypeError("list indices must be integers or slices, not str");
            }
        }

        /// <summary>
        ///     A <c>%(key)</c> lookup of bytes formatting (the key is a bytes object): a dict's value whose key is equal bytes,
        ///     or the subscript error of the mapping's type.
        /// </summary>
        /// <param name="mapping">The value (a mapping: <see cref="IsBytesMapping"/>).</param>
        /// <param name="key">The key's bytes.</param>
        /// <param name="keyLength">Their number.</param>
        /// <param name="w">The call's state.</param>
        /// <param name="result">Receives the value looked up.</param>
        /// <exception cref="IndexError">A NumPy scalar or an ndarray.</exception>
        /// <exception cref="TypeError">A list (<c>list indices must be integers or slices, not bytes</c>).</exception>
        /// <exception cref="KeyError">A dict without a bytes key equal to the key (the text is its repr, <c>b'a'</c>).</exception>
        private static void LookupBytes(in Elem mapping, byte* key, int keyLength, ModWalkState w, out Elem result)
        {
            if (mapping.Kind == ElemKind.Number)
                throw new IndexError(ScalarIndexText);
            var keySpan = new ReadOnlySpan<byte>(key, keyLength);
            switch (mapping.Obj)
            {
                case IDictionary dict:
                    foreach (DictionaryEntry entry in dict)
                    {
                        // A bytes key equals only bytes with the same contents (Python's bytes.__eq__).
                        if (entry.Key is byte[] candidate && keySpan.SequenceEqual(candidate))
                        {
                            ElemOfObject(entry.Value, w, out result);
                            return;
                        }
                    }
                    throw new KeyError(PyRepr.Bytes(keySpan));
                case NDArray:
                    throw new IndexError(ArrayIndexText);
                default:
                    throw new TypeError("list indices must be integers or slices, not bytes");
            }
        }

        /// <summary>A str key's code points as a .NET string (a lone surrogate kept), or null when a unit is past U+10FFFF.</summary>
        /// <param name="key">The code points.</param>
        /// <param name="length">Their number.</param>
        /// <returns>The string, or null.</returns>
        private static string KeyString(uint* key, int length)
        {
            var sb = new System.Text.StringBuilder(length);
            for (int i = 0; i < length; i++)
            {
                uint u = key[i];
                if (u > 0x10FFFF)
                    return null;
                if (u > 0xFFFF)
                    sb.Append(char.ConvertFromUtf32((int)u));
                else
                    sb.Append((char)u);
            }
            return sb.ToString();
        }

        /// <summary>
        ///     A dict lookup by a str key under Python's equality: a C# string key with the same text (looked up directly, as a
        ///     hashed dictionary does), or a one-character <see cref="char"/> key (a str under the boundary map).
        /// </summary>
        /// <param name="dict">The dict.</param>
        /// <param name="key">The key text.</param>
        /// <param name="value">Receives the value.</param>
        /// <returns>True when the key is present.</returns>
        private static bool TryFindStrKey(IDictionary dict, string key, out object value)
        {
            if (dict.Contains(key))
            {
                value = dict[key];
                return true;
            }
            if (key.Length == 1)
            {
                foreach (DictionaryEntry entry in dict)
                {
                    if (entry.Key is char c && c == key[0])
                    {
                        value = entry.Value;
                        return true;
                    }
                }
            }
            value = null;
            return false;
        }

        /// <summary>
        ///     The element a Python object stands for when a tuple item or a dict value reaches a spec (the boundary map of
        ///     <see cref="Operand"/>'s object reader): a C# string is a str (its code points in the derived scratch), a char a
        ///     one-character str, a <c>byte[]</c> bytes, a <see cref="Half"/> an np.float16, anything else the object itself.
        /// </summary>
        /// <param name="obj">The object (null is None).</param>
        /// <param name="w">The call's state (the derived scratch).</param>
        /// <param name="e">Receives the element (valid until the next derived value of its kind).</param>
        private static void ElemOfObject(object obj, ModWalkState w, out Elem e)
        {
            e = default;
            switch (obj)
            {
                case string s:
                    e.Kind = ElemKind.Str;
                    e.Units = (byte*)w.DerivedPoints(s, out long n);
                    e.Length = n;
                    return;
                case char c:
                    e.Kind = ElemKind.Str;
                    e.Units = (byte*)w.DerivedPoints(c.ToString(), out long one);
                    e.Length = one;
                    return;
                case byte[] b:
                    e.Kind = ElemKind.Bytes;
                    e.Units = w.DerivedBytes(b.Length);
                    b.AsSpan().CopyTo(new Span<byte>(e.Units, b.Length));
                    e.Length = b.Length;
                    return;
                case Half h:
                    e.Kind = ElemKind.Number;
                    e.Number = NPTypeCode.Half;
                    e.Raw = w.DerivedBytes(2);
                    *(Half*)e.Raw = h;
                    return;
                default:
                    e.Kind = ElemKind.Object;
                    e.Obj = obj;
                    return;
            }
        }

        /// <summary>A NumPy scalar's value boxed as its own CLR type (each arm cast to <see cref="object"/>, so no arm widens
        ///     to a common numeric type first).</summary>
        /// <param name="v">The value (a <see cref="ElemKind.Number"/>).</param>
        /// <returns>The boxed scalar.</returns>
        private static object BoxNumber(in Elem v) => v.Number switch
        {
            NPTypeCode.Boolean => (object)(*v.Raw != 0),
            NPTypeCode.SByte => (object)*(sbyte*)v.Raw,
            NPTypeCode.Byte => (object)*v.Raw,
            NPTypeCode.Int16 => (object)*(short*)v.Raw,
            NPTypeCode.UInt16 => (object)*(ushort*)v.Raw,
            NPTypeCode.Char => (object)*(char*)v.Raw,
            NPTypeCode.Int32 => (object)*(int*)v.Raw,
            NPTypeCode.UInt32 => (object)*(uint*)v.Raw,
            NPTypeCode.Int64 => (object)*(long*)v.Raw,
            NPTypeCode.UInt64 => (object)*(ulong*)v.Raw,
            NPTypeCode.Half => (object)*(Half*)v.Raw,
            NPTypeCode.Single => (object)*(float*)v.Raw,
            NPTypeCode.Double => (object)*(double*)v.Raw,
            NPTypeCode.Decimal => (object)*(decimal*)v.Raw,
            NPTypeCode.Complex => (object)*(Complex*)v.Raw,
            _ => throw new ArgumentOutOfRangeException(nameof(v), v.Number, "not a numeric dtype"),
        };

        /// <summary>A bytes value's bytes as a managed array (PrintfFormatter's text conversions take one).</summary>
        /// <param name="v">The value (<see cref="ElemKind.NpBytes"/> / <see cref="ElemKind.Bytes"/>).</param>
        /// <returns>The bytes.</returns>
        private static byte[] BytesArray(in Elem v) => new ReadOnlySpan<byte>(v.Units, checked((int)v.Length)).ToArray();

        /// <summary>A <c>*</c> width: <c>PyLong_Check</c> then <c>PyLong_AsSsize_t</c> — only a Python int (a bool is one);
        ///     no NumPy scalar qualifies, not even np.int64.</summary>
        /// <param name="v">The argument.</param>
        /// <returns>The width (negative left-justifies).</returns>
        /// <exception cref="TypeError"><c>* wants int</c>.</exception>
        /// <exception cref="OverflowException"><c>Python int too large to convert to C ssize_t</c>.</exception>
        private static long StarWidth(in Elem v)
        {
            if (v.Kind != ElemKind.Object || !PrintfFormatter.TryPyInt(v.Obj, out BigInteger big))
                throw new TypeError(StarWantsIntText);
            if (big < long.MinValue || big > long.MaxValue)
                throw new OverflowException(SsizeOverflowText);
            return (long)big;
        }

        /// <summary>A <c>*</c> precision: <c>PyLong_Check</c> then <c>_PyLong_AsInt</c> (a C int).</summary>
        /// <param name="v">The argument.</param>
        /// <returns>The precision (negative means 0).</returns>
        /// <exception cref="TypeError"><c>* wants int</c>.</exception>
        /// <exception cref="OverflowException"><c>Python int too large to convert to C int</c>.</exception>
        private static int StarPrecision(in Elem v)
        {
            if (v.Kind != ElemKind.Object || !PrintfFormatter.TryPyInt(v.Obj, out BigInteger big))
                throw new TypeError(StarWantsIntText);
            if (big < int.MinValue || big > int.MaxValue)
                throw new OverflowException("Python int too large to convert to C int");
            return (int)big;
        }

        /// <summary>The conversion characters str formatting handles: <c>s r a i d u o x X e E f F g G c</c>.</summary>
        /// <param name="c">A code point.</param>
        /// <returns>True when supported.</returns>
        private static bool IsStrConversion(int c)
            => c is 's' or 'r' or 'a' or 'i' or 'd' or 'u' or 'o' or 'x' or 'X' or 'e' or 'E' or 'f' or 'F' or 'g' or 'G'
                or 'c';

        /// <summary>The conversion characters bytes formatting handles: str formatting's and <c>b</c>.</summary>
        /// <param name="c">A byte value.</param>
        /// <returns>True when supported.</returns>
        private static bool IsBytesConversion(int c) => c == 'b' || IsStrConversion(c);

        /// <summary>str formatting's error for another conversion character: the character itself when printable ASCII (31–126),
        ///     else <c>?</c>, then its code point in lowercase hex and its index in code points.</summary>
        /// <param name="c">The conversion code point.</param>
        /// <param name="index">Its index.</param>
        /// <returns><c>ValueError: unsupported format character 'y' (0x79) at index 1</c>.</returns>
        private static ValueError UnsupportedStrConversion(int c, int index)
        {
            char shown = c >= 31 && c <= 126 ? (char)c : '?';
            return new ValueError($"unsupported format character '{shown}' (0x{(uint)c:x}) at index {index}");
        }

        /// <summary>
        ///     bytes formatting's error for another conversion byte: CPython formats the C <c>char</c> (signed) with <c>%c</c>,
        ///     so a byte below 0x80 shows itself (a control character raw) and its hex value, and a byte from 0x80 up is a
        ///     negative ordinal: OverflowError <c>character argument not in range(0x110000)</c>.
        /// </summary>
        /// <param name="c">The conversion byte (0–255).</param>
        /// <param name="index">Its byte index.</param>
        /// <returns>The error.</returns>
        private static Exception UnsupportedBytesConversion(int c, int index)
        {
            if (c >= 0x80)
                return new OverflowException("character argument not in range(0x110000)");
            return new ValueError($"unsupported format character '{(char)c}' (0x{c:x}) at index {index}");
        }
    }
}
