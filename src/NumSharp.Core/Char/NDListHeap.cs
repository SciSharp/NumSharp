using System;
using System.Collections.Generic;
using NumSharp.Backends.Strings;

// =============================================================================
// NDListHeap.cs — the lists an NDListArray family points at: NumPy's object array holds one pointer per element, and the
// Python lists live on the heap, shared by every view and copy of the array
// =============================================================================
//
// NumPy's object array of lists (np.char.split's result) is an array of 8-byte POINTERS: a view or a copy moves pointers,
// an element assignment stores another pointer, and the lists themselves are Python objects shared by every array that
// points at them. NDListArray keeps the same split:
//
//   * NDListArray.Ids — an int64 array of the outer shape, one LIST ID per element (NumPy's pointer); every structural
//     member (transpose, reshape, take, sort, ...) is the same NumSharp operation on that one array, so its view-or-copy
//     decision, byte strides, flags and errors are NumPy's;
//   * NDListHeap      — the lists: list id i holds pieces [ListStart[i], ListEnd[i]), piece k spans
//     Units[Offsets[k] : Offsets[k + 1]] (one code unit per uint8 for bytes pieces, per uint32 for str pieces), and an
//     optional flag per piece says it is an element's own np.str_ scalar (CPython's split returns [self] when the separator
//     is longer than the string, and Python prints that piece as np.str_('...')).
//
// A heap is APPEND-ONLY: a list, once stored, never changes (a Python list from split is never mutated by NumPy), so any
// number of arrays may point at it. New lists are appended (an element assignment, numpy's fill / put), the arrays grow by
// doubling, and no member keeps a pointer into them across an append.
//
// Lifetime: every NDListArray that points at a heap holds one reference (Share / Release); the last Release disposes the
// heap's NumSharp arrays at once. An array dropped without Dispose leaves its reference behind, and the heap's arrays are
// then released by their finalizers when the heap is unreachable.
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     The shared, append-only store of the Python lists an <see cref="NDListArray"/> family points at: every list's range
    ///     of pieces and every piece's code units (see the file header).
    /// </summary>
    /// <remarks>
    ///     Not thread-safe (NumSharp's single-threaded rule): an append may move the arrays, so a caller reads the addresses
    ///     (<see cref="Units"/>, <see cref="Offsets"/>, ...) after its last append, never across one.
    /// </remarks>
    internal sealed unsafe class NDListHeap : IDisposable
    {
        /// <summary>The smallest capacity a growing array is given (in elements).</summary>
        private const long MinCapacity = 16;

        /// <summary>Every piece's code units back to back (1-D uint8 for <c>'S'</c>, uint32 otherwise); its size is the capacity.</summary>
        private NDArray _units;

        /// <summary>The used code units.</summary>
        private long _unitCount;

        /// <summary>1-D int64: piece k spans <c>[_offsets[k], _offsets[k + 1])</c>; <c>_pieceCount + 1</c> entries are used.</summary>
        private NDArray _offsets;

        /// <summary>The stored pieces.</summary>
        private long _pieceCount;

        /// <summary>1-D bool, one flag per piece (an element's own <c>np.str_</c>), or null while no piece has the flag.</summary>
        private NDArray _pieceFlags;

        /// <summary>1-D int64: list i's first piece.</summary>
        private NDArray _listStart;

        /// <summary>1-D int64: one past list i's last piece.</summary>
        private NDArray _listEnd;

        /// <summary>The stored lists.</summary>
        private long _listCount;

        /// <summary>The first piece of the list being appended (between <see cref="BeginList"/> and <see cref="EndList"/>), or
        ///     -1 outside one.</summary>
        private long _openStart = -1;

        /// <summary>The list arrays pointing at this heap (each holds one reference).</summary>
        private int _refs = 1;

        /// <summary>Set once the last reference is released: the arrays are disposed.</summary>
        private bool _released;

        /// <summary>
        ///     The class of the pieces: <c>'S'</c> (Python <c>bytes</c>, from an <c>S</c> array), <c>'U'</c> (Python
        ///     <c>str</c>, from a <c>U</c> array) or <c>'T'</c> (Python <c>str</c>, from a <c>StringDType</c> array). One heap
        ///     holds one class: NumPy's list of str and bytes items has no NumSharp form.
        /// </summary>
        internal char Kind { get; }

        /// <summary>The bytes of one code unit: 1 for bytes pieces, 4 for str pieces.</summary>
        internal int UnitSize => Kind == 'S' ? 1 : 4;

        /// <summary>The number of stored lists (list ids are 0 .. ListCount - 1).</summary>
        internal long ListCount => _listCount;

        /// <summary>The number of stored pieces.</summary>
        internal long PieceCount => _pieceCount;

        /// <summary>The code units (valid until the next append).</summary>
        internal byte* Units => (byte*)_units.Address;

        /// <summary>The piece offsets, <see cref="PieceCount"/> + 1 of them (valid until the next append).</summary>
        internal long* Offsets => (long*)_offsets.Address;

        /// <summary>The per-piece <c>np.str_</c> flags, or null when no piece has one (valid until the next append).</summary>
        internal bool* PieceFlags => _pieceFlags is null ? null : (bool*)_pieceFlags.Address;

        /// <summary>The lists' first pieces (valid until the next append).</summary>
        internal long* ListStart => (long*)_listStart.Address;

        /// <summary>The lists' piece ends (valid until the next append).</summary>
        internal long* ListEnd => (long*)_listEnd.Address;

        /// <summary>
        ///     Creates a heap over arrays a producer built (the caller hands its references over): <paramref name="units"/> and
        ///     <paramref name="offsets"/> hold the pieces, <paramref name="listStart"/> / <paramref name="listEnd"/> the lists
        ///     in id order (any shape, C-contiguous: they are read flat).
        /// </summary>
        /// <param name="kind">The class of the pieces.</param>
        /// <param name="units">The code units (1-D uint8 for <c>'S'</c>, uint32 otherwise); it may be longer than the units
        ///     the pieces use (a producer's capacity).</param>
        /// <param name="offsets">The piece offsets (1-D int64, at least <paramref name="pieceCount"/> + 1 entries; a
        ///     producer's capacity may leave more, uninitialized).</param>
        /// <param name="pieceCount">The number of pieces written: <c>offsets[pieceCount]</c> is the units used, and the next
        ///     append starts there. The arrays' SIZES are capacities, never counts.</param>
        /// <param name="pieceFlags">One flag per piece slot (1-D bool, <c>offsets.size - 1</c> entries), or null.</param>
        /// <param name="listStart">Each list's first piece (C-contiguous int64).</param>
        /// <param name="listEnd">One past each list's last piece (C-contiguous int64).</param>
        internal NDListHeap(char kind, NDArray units, NDArray offsets, long pieceCount, NDArray pieceFlags, NDArray listStart, NDArray listEnd)
        {
            Kind = kind;
            _units = units;
            _offsets = offsets;
            _pieceCount = pieceCount;
            // The units in use end where the last piece ends; the rest of the units array is capacity.
            _unitCount = ((long*)offsets.Address)[pieceCount];
            _pieceFlags = pieceFlags;
            _listStart = listStart;
            _listEnd = listEnd;
            _listCount = listStart.size;
        }

        /// <summary>A heap with no list (the store of an empty result).</summary>
        /// <param name="kind">The class of the pieces.</param>
        /// <returns>The heap (one reference, the caller's).</returns>
        internal static NDListHeap Empty(char kind)
        {
            var units = new NDArray(kind == 'S' ? NPTypeCode.Byte : NPTypeCode.UInt32, Shape.Vector(0), true);
            var offsets = new NDArray(NPTypeCode.Int64, Shape.Vector(1), true);
            var start = new NDArray(NPTypeCode.Int64, Shape.Vector(0), true);
            var end = new NDArray(NPTypeCode.Int64, Shape.Vector(0), true);
            return new NDListHeap(kind, units, offsets, 0, null, start, end);
        }

        /// <summary>
        ///     A heap with no list yet and room for exactly <paramref name="units"/> code units, <paramref name="pieces"/>
        ///     pieces and <paramref name="lists"/> lists: the heap of a bulk builder that counted its whole result first (a
        ///     concatenation, a repetition, a sum, a running sum), whose appends (<see cref="AppendReserved"/>,
        ///     <see cref="EndListReserved"/>) then never grow or move it — so a builder may also read the heap's own lists
        ///     through pointers captured once (a running sum copies its previous list).
        /// </summary>
        /// <param name="kind">The class of the pieces.</param>
        /// <param name="units">The code units to come.</param>
        /// <param name="pieces">The pieces to come.</param>
        /// <param name="lists">The lists to come.</param>
        /// <param name="withFlags">Allocate the per-piece <c>np.str_</c> flags now. Needed when the pieces to come may carry
        ///     flags AND the builder reads THIS heap through pointers captured at the start: <see cref="FlagPiece"/> would
        ///     allocate the flags lazily, after that capture, and the captured pointers would miss every flag.</param>
        /// <returns>The heap (one reference, the caller's).</returns>
        /// <exception cref="OutOfMemoryException">An array cannot be allocated (NumPy's MemoryError).</exception>
        /// <remarks>Each array is allocated once, at its final size, and not filled (every slot is written before it is
        ///     read) — except <c>offsets[0]</c>, the units the first piece starts at, and the flags, whose unwritten slots
        ///     must read false. An <see cref="Empty(char)"/> heap grown to the same size allocated every array twice and
        ///     disposed the first four (~2 us of a 30 us 1K concatenation).</remarks>
        internal static NDListHeap Reserved(char kind, long units, long pieces, long lists, bool withFlags = false)
        {
            NDArray u = null, offsets = null, start = null, end = null, flags = null;
            try
            {
                u = new NDArray(kind == 'S' ? NPTypeCode.Byte : NPTypeCode.UInt32, Shape.Vector(units), false);
                offsets = new NDArray(NPTypeCode.Int64, Shape.Vector(pieces + 1), false);
                ((long*)offsets.Address)[0] = 0;
                start = new NDArray(NPTypeCode.Int64, Shape.Vector(lists), false);
                end = new NDArray(NPTypeCode.Int64, Shape.Vector(lists), false);
                if (withFlags)
                    flags = new NDArray(NPTypeCode.Boolean, Shape.Vector(pieces), true);
            }
            catch
            {
                // A failed allocation (MemoryError) leaves nothing behind.
                u?.Dispose();
                offsets?.Dispose();
                start?.Dispose();
                end?.Dispose();
                flags?.Dispose();
                throw;
            }
            // The constructor counts every list slot as a stored list; a reserved heap stores none yet.
            var heap = new NDListHeap(kind, u, offsets, 0, flags, start, end);
            heap._listCount = 0;
            return heap;
        }

        /// <summary>
        ///     The heap of a producer that built one list per element in C order: the per-element <c>[self]</c> flags become
        ///     flags of the elements' single pieces.
        /// </summary>
        /// <param name="kind">The class of the pieces.</param>
        /// <param name="units">The code units (handed over).</param>
        /// <param name="offsets">The piece offsets (handed over).</param>
        /// <param name="pieceCount">The pieces the producer wrote (the arrays' sizes are its capacities).</param>
        /// <param name="starts">Each element's first piece, in C order (handed over).</param>
        /// <param name="ends">One past each element's last piece (handed over).</param>
        /// <param name="selfElements">Whether each element's single piece is its own <c>np.str_</c> (bool, in C order), or
        ///     null; it is read and disposed here.</param>
        /// <returns>The heap (one reference, the caller's).</returns>
        internal static NDListHeap FromElements(char kind, NDArray units, NDArray offsets, long pieceCount, NDArray starts, NDArray ends, NDArray selfElements)
        {
            NDArray flags = null;
            if (selfElements is not null)
            {
                using (selfElements)
                {
                    long pieces = offsets.size - 1;
                    flags = new NDArray(NPTypeCode.Boolean, Shape.Vector(pieces), true);
                    bool* f = (bool*)flags.Address;
                    bool* s = (bool*)selfElements.Address;
                    long* st = (long*)starts.Address;
                    long n = selfElements.size;
                    // A [self] list holds exactly one piece: its flag goes to that piece.
                    for (long e = 0; e < n; e++)
                        if (s[e])
                            f[st[e]] = true;
                }
            }
            return new NDListHeap(kind, units, offsets, pieceCount, flags, starts, ends);
        }

        // =============================================================================================================
        // References
        // =============================================================================================================

        /// <summary>One more list array points at this heap.</summary>
        /// <returns>This heap (the new reference).</returns>
        /// <exception cref="ObjectDisposedException">The heap was released (a disposed list array's heap).</exception>
        internal NDListHeap Share()
        {
            if (_released)
                throw new ObjectDisposedException(nameof(NDListArray));
            _refs++;
            return this;
        }

        /// <summary>
        ///     Drops ONE reference (the one <see cref="Share"/>, a constructor or <see cref="Empty"/> handed out): the last
        ///     reference disposes the arrays. Each holder calls it once — <see cref="Release"/> under its ownership name.
        /// </summary>
        public void Dispose() => Release();

        /// <summary>A list array no longer points at this heap; the last one disposes its arrays.</summary>
        internal void Release()
        {
            if (_released || --_refs > 0)
                return;
            _released = true;
            _units.Dispose();
            _offsets.Dispose();
            _pieceFlags?.Dispose();
            _listStart.Dispose();
            _listEnd.Dispose();
        }

        // =============================================================================================================
        // Reading
        // =============================================================================================================

        /// <summary>The piece range of list <paramref name="id"/>.</summary>
        /// <param name="id">The list id.</param>
        /// <returns>(first piece, one past the last).</returns>
        internal (long Start, long End) Range(long id) => (ListStart[id], ListEnd[id]);

        /// <summary>Whether piece <paramref name="piece"/> is an element's own <c>np.str_</c>.</summary>
        /// <param name="piece">The piece index.</param>
        /// <returns>The flag.</returns>
        internal bool IsNpStr(long piece) => _pieceFlags is not null && ((bool*)_pieceFlags.Address)[piece];

        /// <summary>The number of pieces of list <paramref name="id"/> (Python's <c>len(list)</c>).</summary>
        /// <param name="id">The list id.</param>
        /// <returns>The length.</returns>
        internal long Length(long id) => ListEnd[id] - ListStart[id];

        /// <summary>One piece's code units.</summary>
        /// <param name="piece">The piece index.</param>
        /// <param name="length">Receives its length in code units.</param>
        /// <returns>Its first unit (valid until the next append).</returns>
        internal byte* Piece(long piece, out long length)
        {
            long* o = Offsets;
            length = o[piece + 1] - o[piece];
            return Units + o[piece] * UnitSize;
        }

        /// <summary>One list as a .NET list of its pieces: <see cref="string"/> for str pieces, <c>byte[]</c> for bytes
        ///     pieces, every piece exact.</summary>
        /// <param name="id">The list id.</param>
        /// <returns>The list.</returns>
        /// <exception cref="SystemError">A str piece holds a code unit above U+10FFFF (it has no .NET form).</exception>
        internal List<object> ToList(long id)
        {
            var (start, end) = Range(id);
            var list = new List<object>(checked((int)(end - start)));
            for (long k = start; k < end; k++)
            {
                byte* p = Piece(k, out long n);
                if (Kind == 'S')
                    list.Add(new ReadOnlySpan<byte>(p, checked((int)n)).ToArray());
                else
                    list.Add(NDCharObjectPath.Ucs4ToString((uint*)p, checked((int)n)));
            }
            return list;
        }

        // =============================================================================================================
        // Appending (a heap never changes a stored list)
        // =============================================================================================================

        /// <summary>Starts appending a new list: the pieces appended next belong to it, until <see cref="EndList"/>.</summary>
        /// <exception cref="InvalidOperationException">A list is already open.</exception>
        internal void BeginList()
        {
            if (_openStart >= 0)
                throw new InvalidOperationException("an NDListHeap list is already open");
            _openStart = _pieceCount;
        }

        /// <summary>Ends the list <see cref="BeginList"/> opened.</summary>
        /// <returns>The new list's id.</returns>
        /// <exception cref="InvalidOperationException">No list is open.</exception>
        internal long EndList()
        {
            if (_openStart < 0)
                throw new InvalidOperationException("no NDListHeap list is open");
            EnsureLists(1);
            long id = _listCount++;
            ListStart[id] = _openStart;
            ListEnd[id] = _pieceCount;
            _openStart = -1;
            return id;
        }

        /// <summary>Appends one piece to the open list.</summary>
        /// <param name="units">The piece's code units (this heap's unit size each); may point into THIS heap's units only
        ///     when the caller has reserved the room first (<see cref="Reserve"/>), so the append cannot move them.</param>
        /// <param name="length">The number of code units.</param>
        /// <param name="npStr">Whether the piece is an element's own <c>np.str_</c>.</param>
        internal void AppendPiece(byte* units, long length, bool npStr)
        {
            EnsureUnits(length);
            EnsurePieces(1);
            long bytes = length * UnitSize;
            Buffer.MemoryCopy(units, Units + _unitCount * UnitSize, bytes, bytes);
            _unitCount += length;
            SetPiece(npStr);
        }

        /// <summary>Appends one str piece from .NET text (its code points: a surrogate pair is one point).</summary>
        /// <param name="text">The text.</param>
        internal void AppendPiece(string text)
        {
            int count = NDStringCodec.CodePointCount(text);
            EnsureUnits(count);
            EnsurePieces(1);
            uint* dst = (uint*)Units + _unitCount;
            int k = 0;
            for (int i = 0; i < text.Length; i++)
            {
                if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                {
                    dst[k++] = (uint)char.ConvertToUtf32(text[i], text[i + 1]);
                    i++;
                }
                else
                {
                    dst[k++] = text[i];
                }
            }
            _unitCount += count;
            SetPiece(false);
        }

        /// <summary>Appends every piece of list <paramref name="id"/> of heap <paramref name="source"/> (this heap included).</summary>
        /// <param name="source">The heap holding the list (its class must be this heap's, <c>'U'</c> and <c>'T'</c> both str).</param>
        /// <param name="id">The list id.</param>
        /// <param name="times">How many times to append them (Python's <c>list * n</c>; 0 or less appends nothing).</param>
        internal void AppendListPieces(NDListHeap source, long id, long times = 1)
        {
            if (times <= 0)
                return;
            var (start, end) = source.Range(id);
            long pieces = end - start;
            if (pieces == 0)
                return;
            long* so = source.Offsets;
            long units = so[end] - so[start];
            // Room first: when the source is this heap, the copy below reads from arrays the growth would move.
            Reserve(units * times, pieces * times);
            so = source.Offsets;
            byte* su = source.Units;
            bool* sf = source.PieceFlags;
            int unit = UnitSize;
            for (long t = 0; t < times; t++)
            {
                long bytes = units * unit;
                Buffer.MemoryCopy(su + so[start] * unit, Units + _unitCount * unit, bytes, bytes);
                long baseUnit = _unitCount - so[start];
                long* o = Offsets;
                for (long k = start; k < end; k++)
                {
                    o[_pieceCount + 1] = so[k + 1] + baseUnit;
                    if (sf is not null && sf[k])
                        FlagPiece(_pieceCount);
                    _pieceCount++;
                }
                _unitCount += units;
            }
        }

        /// <summary>Makes room for <paramref name="units"/> more code units and <paramref name="pieces"/> more pieces, so the
        ///     appends that follow do not move the arrays.</summary>
        /// <param name="units">The code units to come.</param>
        /// <param name="pieces">The pieces to come.</param>
        internal void Reserve(long units, long pieces)
        {
            EnsureUnits(units);
            EnsurePieces(pieces);
        }

        // =============================================================================================================
        // Bulk appending: one exact reservation, then copies without growth checks
        // =============================================================================================================

        /// <summary>
        ///     Appends every piece of list <paramref name="id"/> of <paramref name="source"/>, <paramref name="times"/> times
        ///     (Python's <c>list * n</c>; 0 or less appends nothing), into the room a <see cref="Reserved"/> heap was created
        ///     with: the heap never grows (an append past the reservation throws), the source is read through its raw
        ///     pointers.
        /// </summary>
        /// <param name="source">The source heap's pointers (this heap's own too: a reserved heap's arrays never move).</param>
        /// <param name="id">The list id.</param>
        /// <param name="times">How many times to append the pieces.</param>
        /// <exception cref="InvalidOperationException">The source's unit size differs (a str heap and a bytes heap), or the
        ///     pieces pass the heap's reservation (a builder that under-counted its result).</exception>
        internal void AppendReserved(in ListPointers source, long id, long times = 1)
        {
            if (times <= 0)
                return;
            long start = source.FirstPiece(id), end = source.EndPiece(id);
            if (start == end)
                return;
            int unit = UnitSize;
            // Units are copied as bytes: a str heap and a bytes heap never exchange pieces (their callers refuse it).
            if (source.UnitSize != unit)
                throw new InvalidOperationException("an NDListHeap cannot take pieces of another unit size");
            long* so = source.Offsets;
            long u0 = so[start];
            long units = so[end] - u0;
            // The reservation is the contract: a builder that under-counted would write past the arrays, where the pool's
            // size buckets leave slack that hides the overrun (no crash, right values), so the append refuses instead.
            // The builders bound units * times and pieces * times before reserving (CheckRepeatSize, AddSize).
            if (_unitCount + units * times > _units.size || _pieceCount + (end - start) * times + 1 > _offsets.size)
                throw new InvalidOperationException("an NDListHeap append passed the heap's reservation");
            long bytes = units * unit;
            byte* from = source.Units + u0 * unit;
            byte* du = (byte*)_units.Address;
            long* dof = (long*)_offsets.Address;
            bool* sf = source.Flags;
            for (long t = 0; t < times; t++)
            {
                Buffer.MemoryCopy(from, du + _unitCount * unit, bytes, bytes);
                // The pieces keep their spans, shifted to where their units now start.
                long shift = _unitCount - u0;
                for (long k = start; k < end; k++)
                {
                    dof[_pieceCount + 1] = so[k + 1] + shift;
                    if (sf is not null && sf[k])
                        FlagPiece(_pieceCount);
                    _pieceCount++;
                }
                _unitCount += units;
            }
        }

        /// <summary>Ends the list <see cref="BeginList"/> opened, into the room a <see cref="Reserved"/> heap was created with.</summary>
        /// <returns>The new list's id.</returns>
        /// <exception cref="InvalidOperationException">No list is open, or the lists pass the heap's reservation.</exception>
        internal long EndListReserved()
        {
            if (_openStart < 0)
                throw new InvalidOperationException("no NDListHeap list is open");
            // The list slots too are reserved: one more list than the builder counted would write past them.
            if (_listCount >= _listStart.size)
                throw new InvalidOperationException("an NDListHeap list passed the heap's reservation");
            long id = _listCount++;
            ((long*)_listStart.Address)[id] = _openStart;
            ((long*)_listEnd.Address)[id] = _pieceCount;
            _openStart = -1;
            return id;
        }

        /// <summary>Appends a whole list of .NET pieces (<see cref="string"/> for str, <c>byte[]</c> for bytes).</summary>
        /// <param name="pieces">The pieces (each of this heap's class).</param>
        /// <returns>The new list's id.</returns>
        /// <exception cref="NotSupportedException">A piece of the other class (NumPy's list of str and bytes items has no
        ///     NumSharp form), or an item that is neither.</exception>
        internal long AppendList(IEnumerable<object> pieces)
        {
            BeginList();
            try
            {
                foreach (var p in pieces)
                {
                    switch (p)
                    {
                        case string s when Kind != 'S':
                            AppendPiece(s);
                            break;
                        case byte[] b when Kind == 'S':
                            fixed (byte* pb = b)
                                AppendPiece(pb, b.Length, false);
                            break;
                        default:
                            throw new NotSupportedException(
                                $"an NDListArray of {(Kind == 'S' ? "bytes" : "str")} pieces cannot hold a {NDPyObject.TypeName(p)} item " +
                                "(NumPy's object array would hold the list as it is; an NDListArray holds lists of one piece class)");
                    }
                }
            }
            catch
            {
                // Drop the half-appended list: its pieces stay in the store unreferenced.
                _openStart = -1;
                throw;
            }
            return EndList();
        }

        // =============================================================================================================
        // Growth
        // =============================================================================================================

        /// <summary>Records the piece whose units were just appended: its end offset and flag.</summary>
        /// <param name="npStr">Whether it is an element's own <c>np.str_</c>.</param>
        private void SetPiece(bool npStr)
        {
            Offsets[_pieceCount + 1] = _unitCount;
            if (npStr)
                FlagPiece(_pieceCount);
            _pieceCount++;
        }

        /// <summary>Flags piece <paramref name="piece"/> as an element's own <c>np.str_</c> (the flag array is created on the
        ///     first flag, as wide as the piece capacity).</summary>
        /// <param name="piece">The piece index (within the capacity).</param>
        private void FlagPiece(long piece)
        {
            if (_pieceFlags is null)
                _pieceFlags = new NDArray(NPTypeCode.Boolean, Shape.Vector(_offsets.size - 1), true);
            ((bool*)_pieceFlags.Address)[piece] = true;
        }

        /// <summary>Grows the units so <paramref name="more"/> more fit.</summary>
        /// <param name="more">The code units to come.</param>
        private void EnsureUnits(long more)
        {
            long need = _unitCount + more;
            if (need > _units.size)
                _units = Grow(_units, _unitCount, need);
        }

        /// <summary>Grows the piece offsets (and flags) so <paramref name="more"/> more pieces fit.</summary>
        /// <param name="more">The pieces to come.</param>
        private void EnsurePieces(long more)
        {
            long need = _pieceCount + more + 1;
            if (need > _offsets.size)
            {
                _offsets = Grow(_offsets, _pieceCount + 1, need);
                if (_pieceFlags is not null)
                    _pieceFlags = Grow(_pieceFlags, _pieceCount, _offsets.size - 1, exact: true, zero: true);
            }
        }

        /// <summary>Grows the list ranges so <paramref name="more"/> more lists fit.</summary>
        /// <param name="more">The lists to come.</param>
        private void EnsureLists(long more)
        {
            long need = _listCount + more;
            if (need > _listStart.size)
            {
                _listStart = Grow(_listStart, _listCount, need);
                _listEnd = Grow(_listEnd, _listCount, _listStart.size, exact: true);
            }
        }

        /// <summary>A bigger 1-D array holding the used part of <paramref name="array"/>; the old one is disposed.</summary>
        /// <param name="array">The array.</param>
        /// <param name="used">Its used elements.</param>
        /// <param name="need">The elements needed.</param>
        /// <param name="exact">True to allocate exactly <paramref name="need"/> (a companion array that must match another's
        ///     capacity, or a bulk builder's exact reservation); otherwise the capacity doubles.</param>
        /// <param name="zero">True to zero-fill the new array (the flag array, whose unwritten slots must read false); the
        ///     other arrays are written before they are read, so their fill would only cost a pass over fresh memory.</param>
        /// <returns>The new array.</returns>
        private static NDArray Grow(NDArray array, long used, long need, bool exact = false, bool zero = false)
        {
            long capacity = exact ? need : Math.Max(Math.Max(need, MinCapacity), array.size * 2);
            var bigger = new NDArray(array.typecode, Shape.Vector(capacity), zero);
            long bytes = used * array.dtypesize;
            Buffer.MemoryCopy((void*)array.Address, (void*)bigger.Address, bytes, bytes);
            array.Dispose();
            return bigger;
        }
    }
}
