using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Printing;
using NumSharp.Backends.Strings;

// =============================================================================
// NDListArray.cs — NumPy's object array of Python lists: the result of np.char.split / rsplit / splitlines
// =============================================================================
//
// NumPy returns `_vec_string(a, np.object_, 'split', ...)`: an OBJECT ndarray whose every element is the Python list
// str.split / bytes.split / splitlines returned for it (`array([list(['a', 'b']), list([])], dtype=object)`). NumSharp
// has no object dtype, so this type holds exactly that structure — the owner's decision C-D3 (docs/plans/np-char.md):
//
//   * Ids  — an int64 array of the OUTER shape: one list id per element, NumPy's 8-byte object pointer. Every structural
//            member (basic indexing, transpose, reshape, take, sort, ...) is the same NumSharp operation on this one array,
//            so a view shares the ids as a NumPy view shares the pointers, and a copy copies them (a shallow copy);
//   * Heap — the lists (NDListHeap): each list a range of pieces, each piece its exact code units (a piece that ends in a
//            NUL, 'a\x00' from 'a\x00 b'.split(), keeps it; a fixed-width array of pieces could not), and a flag for a piece
//            that is an element's own np.str_ scalar (CPython's split returns [self] when the separator is longer or of a
//            wider kind than the string; Python prints it as np.str_('...'), and bytes.join can read it as a buffer). The
//            heap is shared by every array pointing into it and only ever appended to (an element assignment stores a new
//            list), like the Python heap the pointers point into.
//
// The layout is NumPy's: the producer's result is C-ordered over the broadcast shape (PyArray_SimpleNewFromDescr),
// whatever the input layout, so its ids are 0 .. n - 1 in C order and the lists follow in that order.
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's object array of Python lists — what <c>np.char.split</c>, <c>rsplit</c> and <c>splitlines</c> return
    ///     (<c>array([list(['a', 'b']), list([])], dtype=object)</c>). NumSharp has no object dtype, so this type holds the
    ///     array's elements as list ids (NumPy's object pointers) into a shared store of the lists: the text of the pieces
    ///     once, each piece with its exact length.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Element access.</b> The indexer is NumPy's: a full integer index (one integer per axis, negative values
    ///     counting from the end) returns the element's LIST as a 0-d <see cref="NDListArray"/> that stands for it
    ///     (<see cref="IsStandIn"/>), every other index a list array over the selected lists (a view for slices). The list's
    ///     pieces come out as .NET values from <see cref="item(long[])"/> and <see cref="tolist"/>, every piece exact.
    ///     Iterating a 1-d list array yields each element's list as a NEW 1-D array of the input's string class —
    ///     <c>S</c> / <c>U</c> as wide as its longest piece (at least 1), or <c>StringDType</c> for a <c>StringDType</c>
    ///     input — which is the closest NumSharp value to NumPy's Python list; a piece that ENDS in a NUL loses it in that
    ///     array, as every fixed-width element does (<c>np.array(['a\x00'])</c> reads <c>'a'</c> in NumPy too).
    ///     </para>
    ///     <para>
    ///     <b>Lifetime.</b> The arrays the type holds are NumSharp arrays: <see cref="Dispose"/> releases them at once (the
    ///     shared lists when the last array pointing at them is disposed), the finalizers release them otherwise. A view
    ///     keeps the memory it reads alive (it counts its own references).
    ///     </para>
    /// </remarks>
    public sealed partial class NDListArray : IEnumerable, IDisposable
    {
        /// <summary>int64 of the outer shape: each element's list id in <see cref="Heap"/> (NumPy's object pointer).</summary>
        internal NDArray Ids { get; }

        /// <summary>The lists the ids point at (shared with every view and copy, one reference held by this array).</summary>
        internal NDListHeap Heap { get; }

        /// <summary>
        ///     The class of the pieces: <c>'S'</c> (Python <c>bytes</c>, from an <c>S</c> array), <c>'U'</c> (Python
        ///     <c>str</c>, from a <c>U</c> array) or <c>'T'</c> (Python <c>str</c>, from a <c>StringDType</c> array).
        /// </summary>
        internal char Kind => Heap.Kind;

        /// <summary>Set by <see cref="Dispose"/>: every member then throws <see cref="ObjectDisposedException"/>.</summary>
        private bool _disposed;

        /// <summary>
        ///     Whether this 0-d array stands for the Python LIST NumPy returned (a full integer index, a reduction to one
        ///     element, a take of one index, a 0-d operator result), not for a 0-d object ARRAY (np.char.split of a 0-d
        ///     text). The two differ where NumPy converts an operand with <c>np.asarray</c>: against an array, a list becomes
        ///     the 1-D array of its ITEMS (<c>r == r[0]</c> compares every element with each str item), while a 0-d array
        ///     stays one element (<c>r == np.char.split('a b')</c> compares every element with the list). A stand-in's own
        ///     members are the 0-d array's.
        /// </summary>
        internal bool IsStandIn { get; private set; }

        /// <summary>Marks this 0-d array as the stand-in of NumPy's returned list (see <see cref="IsStandIn"/>); an array of
        ///     one or more dimensions is left as it is.</summary>
        /// <returns>This array.</returns>
        internal NDListArray MarkStandIn()
        {
            if (Ids.ndim == 0)
                IsStandIn = true;
            return this;
        }

        /// <summary>Creates the list array over its ids and lists (the caller hands over the ids and ONE heap reference).</summary>
        /// <param name="ids">The list ids (int64, the outer shape).</param>
        /// <param name="heap">The lists (a reference the caller took for this array: <see cref="NDListHeap.Share"/>, or a new
        ///     heap's own).</param>
        internal NDListArray(NDArray ids, NDListHeap heap)
        {
            Ids = ids;
            Heap = heap;
        }

        /// <summary>
        ///     The list array of a producer that built one list per element in C order over <paramref name="outer"/>:
        ///     element e points at list e.
        /// </summary>
        /// <param name="units">The pieces' code units (handed over).</param>
        /// <param name="offsets">The piece offsets (handed over).</param>
        /// <param name="pieceCount">The pieces the producer wrote (its arrays are capacity-sized).</param>
        /// <param name="starts">Each element's first piece, C order (handed over).</param>
        /// <param name="ends">One past each element's last piece (handed over).</param>
        /// <param name="selfElements">Each element's <c>[self]</c> flag (bool, C order) or null (read and disposed).</param>
        /// <param name="kind">The class of the pieces.</param>
        /// <param name="outer">The outer shape.</param>
        /// <returns>The list array.</returns>
        internal static NDListArray FromElements(NDArray units, NDArray offsets, long pieceCount, NDArray starts, NDArray ends,
                                                 NDArray selfElements, char kind, Shape outer)
        {
            var heap = NDListHeap.FromElements(kind, units, offsets, pieceCount, starts, ends, selfElements);
            NDArray ids = null;
            try
            {
                // Element e points at list e: the ids are 0 .. n - 1 in C order over the outer shape. The array is relabelled
                // IN PLACE (ndarray.shape's setter), so it stays the owner of its memory, as NumPy's fresh result owns its
                // pointers (a reshape view would report OWNDATA false and a 1-D base).
                ids = np.arange(outer.size);
                ids.Shape = outer.Clean();
                return new NDListArray(ids, heap);
            }
            catch
            {
                ids?.Dispose();
                heap.Release();
                throw;
            }
        }

        // =============================================================================================================
        // The array attributes
        // =============================================================================================================

        /// <summary>NumPy's <c>shape</c> (a copy, NumPy's tuple).</summary>
        public long[] shape => Ids.shape;

        /// <summary>NumPy's <c>ndim</c>: the number of axes.</summary>
        public int ndim => Ids.ndim;

        /// <summary>NumPy's <c>size</c>: the number of elements (lists).</summary>
        public long size => Ids.size;

        /// <summary>The outer shape as a NumSharp <see cref="NumSharp.Shape"/> (dimensions only: the lists have no strides).</summary>
        public Shape Shape => NumSharp.Shape.Adopt(Ids.shape);

        // =============================================================================================================
        // Element access
        // =============================================================================================================

        /// <summary>
        ///     NumPy's indexing of an object array of lists (<c>ndarray.__getitem__</c> / <c>ndarray.__setitem__</c>) as the
        ///     one C# indexer: every index NumPy takes — integers, slices, slice notation (<c>"1:"</c>, <c>"::-1, 0"</c>), index
        ///     arrays, boolean masks and sequences, in any mix.
        /// </summary>
        /// <param name="key">The index tuple, as for <see cref="NDArray.this[object[]]"/> (a lone string holds the whole
        ///     index in slice notation; a <see cref="Slice"/>[] or <c>object[]</c> passed alone is the tuple itself).</param>
        /// <value>
        ///     GET: a list array over the selected lists — a VIEW for an index of integers and slices (the lists are shared,
        ///     as a NumPy view shares the pointers), a new array of the same lists for an index array or a mask (NumPy's
        ///     advanced indexing copies the pointers, not the lists). One integer per axis is NumPy's element access, which
        ///     returns the LIST itself: a 0-d array marked as its stand-in (<see cref="IsStandIn"/>). Read the list's pieces
        ///     with <see cref="item(long[])"/> or <see cref="tolist"/>.
        ///     SET: NumPy's <c>a[index] = value</c> of an object array (see <see cref="SetItem(Slice[], NDListArray)"/>): one
        ///     integer per axis stores the list a 0-d value stands for, a sub-array or an advanced selection receives a list
        ///     array's lists by broadcasting.
        /// </value>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <c>null</c>; SET: a null value (Python's None
        ///     has no NDListArray form).</exception>
        /// <exception cref="IndexError">An integer outside its axis (NumPy's <c>index 5 is out of bounds for axis 0 with size
        ///     2</c>), or an invalid index (NumPy's texts).</exception>
        /// <exception cref="ValueError">SET: a read-only array (<c>assignment destination is read-only</c>, checked before the
        ///     index), or a value that does not broadcast (<c>could not broadcast input array from shape (3,) into shape
        ///     (2,)</c>, or NumPy's <c>shape mismatch</c> text for an advanced index).</exception>
        /// <exception cref="NotSupportedException">SET: an assignment whose NumPy result holds an element that is not a list
        ///     (an array stored into one element, a list's items stored into a sub-array, a list of the other piece
        ///     class).</exception>
        /// <exception cref="ObjectDisposedException">The array or the value was disposed.</exception>
        /// <remarks>One member, NumPy's signature: it replaced <c>this[string]</c>, <c>this[params Slice[]]</c> and the
        ///     element forms <c>NDArray this[params long[]]</c> / <c>this[params int[]]</c>, which returned the element's
        ///     pieces as a new array (iterating a 1-d list array still yields those arrays, see <see cref="GetEnumerator"/>).</remarks>
        public NDListArray this[params object[] key]
        {
            get => Index(key);
            set => SetItem(key, value);
        }

        /// <summary>
        ///     The GET of the indexer: the view of a basic index (<see cref="Index(Slice[])"/>), or a list array over the ids an
        ///     advanced index gathers, marked as the list's stand-in for a full integer index.
        /// </summary>
        /// <param name="key">The index tuple.</param>
        /// <returns>The list array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">The index is invalid for this array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        private NDListArray Index(object[] key)
        {
            ThrowIfDisposed();
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            // An index of integers and slices keeps the old typed indexers' path: one view of the ids.
            if (NDArray.TryBasicIndex(key, out Slice[] slices))
                return Index(slices);
            // An advanced index gathers the ids (NumPy copies the pointers); the gathered array is handed over.
            var view = Over(Ids[key]);
            return IsFullIntegerIndex(key) ? view.MarkStandIn() : view;
        }

        /// <summary>
        ///     Whether a general index is NumPy's full integer index (<c>index_type == HAS_INTEGER</c>): one entry per axis,
        ///     each a C# integer, an integer <see cref="Slice"/>, an integer slice-notation string or a 0-d INTEGER array
        ///     ("a full integer index sees array scalars as part of itself"). Such an index returns the element itself — the
        ///     list.
        /// </summary>
        /// <param name="key">The index tuple (a single C# tuple entry is NumPy's index tuple and is spread).</param>
        /// <returns>True for a full integer index of this array.</returns>
        private bool IsFullIntegerIndex(object[] key)
        {
            if (key.Length == 1 && key[0] is System.Runtime.CompilerServices.ITuple tuple)
                key = PolySequence.Items(tuple);
            if (key.Length != ndim)
                return false;
            foreach (object entry in key)
            {
                switch (entry)
                {
                    case sbyte or byte or short or ushort or int or uint or long or ulong or char or nint:
                        break;
                    case Slice { IsIndex: true, IsEllipsis: false, IsNewAxis: false }:
                        break;
                    // The index already succeeded, so the notation parses; "0" is an integer, "1:" is not.
                    case string notation when new Slice(notation) is { IsIndex: true, IsEllipsis: false, IsNewAxis: false }:
                        break;
                    case NDArray { ndim: 0 } zeroD when zeroD.typecode is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16
                        or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
                        or NPTypeCode.Char:
                        break;
                    default:
                        return false;
                }
            }
            return true;
        }

        /// <summary>
        ///     Basic indexing: the view of the selected lists; one integer per axis is NumPy's element access, which returns
        ///     the LIST itself — the 0-d view is marked as its stand-in (<see cref="IsStandIn"/>).
        /// </summary>
        /// <param name="slices">The parsed index.</param>
        /// <returns>The view.</returns>
        /// <exception cref="IndexError">An integer outside its axis.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        private NDListArray Index(Slice[] slices)
        {
            ThrowIfDisposed();
            slices ??= Array.Empty<Slice>();
            var view = Over(Ids[slices]);
            if (IsElementIndex(slices))
                view.MarkStandIn();
            return view;
        }

        /// <summary>Whether an index is NumPy's element access: one integer per axis (none for a 0-d array).</summary>
        /// <param name="slices">The parsed index.</param>
        /// <returns>True for a full integer index.</returns>
        private bool IsElementIndex(Slice[] slices)
        {
            if (slices.Length != ndim)
                return false;
            foreach (var s in slices)
            {
                if (s is null || !s.IsIndex)
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     <c>np.asarray(list)</c> of one list: the 1-D array of its items — <c>U</c> for str items (a <c>StringDType</c>
        ///     input's lists hold Python strs too), <c>S</c> for bytes, as wide as the longest item (at least 1); an EMPTY list
        ///     is NumPy's float64 <c>(0,)</c> array.
        /// </summary>
        /// <param name="id">The list id.</param>
        /// <returns>The new array.</returns>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF.</exception>
        internal NDArray ItemsArray(long id)
        {
            if (Heap.Length(id) == 0)
                return new NDArray(NPTypeCode.Double, NumSharp.Shape.Vector(0), false);
            return Pieces(id, asarray: true);
        }

        /// <summary>
        ///     NumPy's <c>item(*args)</c>: the list of one element as a .NET list of its pieces (<see cref="string"/> for str
        ///     pieces, <c>byte[]</c> for bytes pieces), every piece exact — no argument for the element of a one-element
        ///     array, ONE integer for a flat index in C order (negative from the end), one integer per axis for a full index.
        /// </summary>
        /// <param name="index">None (<c>lists.item()</c>), a flat index, or one integer per axis; a C# <c>long[]</c> is
        ///     NumPy's index tuple (<c>a.item((1, 2))</c>).</param>
        /// <returns>The list.</returns>
        /// <exception cref="ValueError">No argument and more or fewer than one element (NumPy's <c>can only convert an array
        ///     of size 1 to a Python scalar</c>); neither one integer nor one per axis (<c>incorrect number of indices for
        ///     array</c>).</exception>
        /// <exception cref="IndexError">An index outside the array (<c>index 6 is out of bounds for size 6</c> for a flat
        ///     index, <c>index 5 is out of bounds for axis 1 with size 3</c> for a full one).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF.</exception>
        /// <remarks>One member, NumPy's signature: the <c>params</c> array is NumPy's <c>*args</c>, so the no-argument form is
        ///     this member with an empty array.</remarks>
        public List<object> item(params long[] index)
        {
            ThrowIfDisposed();
            if (index is not { Length: > 0 })
            {
                // array_item with no arguments: the one element of a size-1 array.
                if (size != 1)
                    throw new ValueError("can only convert an array of size 1 to a Python scalar");
                return Heap.ToList(FlatId(0));
            }
            // PyArray_MultiIndexGetItem for one index per axis; a single integer of a 0-d or N-d array is a flat index into
            // the C-order sequence (array_item's `nargs == 1` branch).
            if (index.Length == 1 && ndim != 1)
            {
                long n = size, i = index[0];
                if (i < -n || i >= n)
                    throw new IndexError($"index {i} is out of bounds for size {n}");
                return Heap.ToList(FlatId(i < 0 ? i + n : i));
            }
            if (index.Length != ndim)
                throw new ValueError("incorrect number of indices for array");
            return Heap.ToList(Element(index));
        }

        /// <summary>
        ///     NumPy's <c>tolist()</c>: nested .NET lists of the outer shape whose leaves are the element lists, each a
        ///     <see cref="List{T}"/> of its pieces (<see cref="string"/> for str pieces, <c>byte[]</c> for bytes pieces, every
        ///     piece exact); a 0-d array gives its one list.
        /// </summary>
        /// <returns>The nested lists.</returns>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF (it has no .NET form).</exception>
        public object tolist()
        {
            ThrowIfDisposed();
            // The flat C-order copy of the ids, read in the order the nesting visits them.
            using var ids = Ids.flatten();
            if (ndim == 0)
                return Heap.ToList(ids.GetInt64(0));
            long position = 0;
            return Nest(0, ids, ref position);
        }

        /// <summary>
        ///     Iterates the first axis: for a 1-d array every element's list as a new 1-D array of its pieces (<c>S</c> /
        ///     <c>U</c> as wide as the longest piece, or <c>StringDType</c>), for an N-d array every sub-array as an
        ///     <see cref="NDListArray"/> view.
        /// </summary>
        /// <returns>The enumerator.</returns>
        /// <exception cref="TypeError">A 0-d array (<c>iteration over a 0-d array</c>).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public IEnumerator GetEnumerator()
        {
            ThrowIfDisposed();
            if (ndim == 0)
                throw new TypeError("iteration over a 0-d array");
            return Iterate();
        }

        /// <summary>The enumerator body of <see cref="GetEnumerator"/> (after its eager checks).</summary>
        /// <returns>The items of the first axis.</returns>
        private IEnumerator Iterate()
        {
            long n = Ids.Shape.dimensions[0];
            for (long i = 0; i < n; i++)
            {
                // A 1-d array yields each list's pieces (the indexer's element access returns the list's 0-d stand-in
                // instead); an N-d one yields the sub-array views.
                if (ndim == 1)
                    yield return Pieces(Element(new[] { i }));
                else
                    yield return this[Slice.Index(i)];
            }
        }

        // =============================================================================================================
        // Printing
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>str()</c>: <c>[list(['a', 'b']) list(['c'])]</c>, each element <c>list(</c> + the Python repr of
        ///     its list + <c>)</c>, wrapped and summarized as NumPy's <c>_formatArray</c> does; a 0-d array prints its list
        ///     (<c>['a', 'b']</c>).
        /// </summary>
        /// <returns>The text.</returns>
        public override string ToString() => ToString(false);

        /// <summary>
        ///     NumPy's <c>str()</c> (<paramref name="repr"/> false) or <c>repr()</c> (true): the repr is
        ///     <c>array([list(['a', 'b']), list([])], dtype=object)</c>, with <c>shape=</c> for an empty array of another
        ///     shape than <c>(0,)</c> and for a summarized one.
        /// </summary>
        /// <param name="repr">True for <c>repr()</c>.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public string ToString(bool repr)
        {
            ThrowIfDisposed();
            using var ids = Ids.flatten();
            if (!repr && ndim == 0)
                return ListRepr(ids.GetInt64(0));
            var formatter = new ListReprFormat(this, ids);
            // The layout engine walks an int64 array of the outer shape whose values are the C-order positions; the
            // formatter maps a position to its element's list repr.
            using var flatPositions = np.arange(size);
            using var positions = ndim == 0 ? flatPositions.reshape(NumSharp.Shape.NewScalar()) : flatPositions.reshape(NumSharp.Shape.Adopt(Ids.shape));
            var opts = PrintOptions.Current;
            return repr ? ArrayFormatter.ObjectArrayRepr(positions, formatter, opts) : ArrayFormatter.ObjectArrayStr(positions, formatter, opts);
        }

        /// <summary>Python's <c>repr()</c> of one list: <c>['a', 'b']</c>, an <c>np.str_</c> piece as
        ///     <c>np.str_('ab')</c>, a bytes piece as <c>b'ab'</c>.</summary>
        /// <param name="id">The list id.</param>
        /// <returns>The repr.</returns>
        internal string ListRepr(long id)
        {
            var buffer = new ListReprBuffer();
            buffer.AppendList(new ListPointers(Heap), id);
            return buffer.ToString();
        }

        /// <summary>
        ///     Python's <c>repr()</c> of lists, written into one reused character buffer. A piece of printable ASCII with no
        ///     quote and no backslash (the common piece) is copied between single quotes, exactly as CPython's repr writes
        ///     it (single quotes are its choice when the text holds no <c>'</c>); any other piece goes through
        ///     <see cref="PyRepr"/>, which owns the quote choice and every escape.
        /// </summary>
        /// <remarks>
        ///     A conversion of a million lists makes a million strings: the StringBuilder per list and the string per piece
        ///     the repr used to build cost more than the text (~350 ns a list against NumPy's <c>str(list)</c> at ~300 ns),
        ///     so <see cref="ReprTexts"/> clears and refills one buffer per element.
        /// </remarks>
        private sealed class ListReprBuffer
        {
            /// <summary>The characters written since the last <see cref="Clear"/>.</summary>
            private char[] _chars = new char[64];

            /// <summary>The number of characters written.</summary>
            private int _length;

            /// <summary>Forgets the text (the buffer is kept).</summary>
            internal void Clear() => _length = 0;

            /// <summary>The text written since the last <see cref="Clear"/>.</summary>
            /// <returns>A new string.</returns>
            public override string ToString() => new string(_chars, 0, _length);

            /// <summary>Claims the next <paramref name="count"/> characters, growing the buffer (doubling) when needed.</summary>
            /// <param name="count">The characters to write.</param>
            /// <returns>The claimed characters.</returns>
            private Span<char> Room(int count)
            {
                if (_length + count > _chars.Length)
                    Array.Resize(ref _chars, Math.Max(_chars.Length * 2, _length + count));
                var span = _chars.AsSpan(_length, count);
                _length += count;
                return span;
            }

            /// <summary>Appends a text.</summary>
            /// <param name="text">The text.</param>
            private void Append(string text) => text.AsSpan().CopyTo(Room(text.Length));

            /// <summary>Appends Python's repr of list <paramref name="id"/>: <c>['a', np.str_('b')]</c>, <c>[b'ab']</c>.</summary>
            /// <param name="lists">The heap's lists (read only while the heap does not grow).</param>
            /// <param name="id">The list.</param>
            /// <exception cref="OverflowException">A piece longer than <see cref="int.MaxValue"/> code units.</exception>
            internal unsafe void AppendList(in ListPointers lists, long id)
            {
                Room(1)[0] = '[';
                long first = lists.FirstPiece(id), end = lists.EndPiece(id);
                long* offsets = lists.Offsets;
                bool* flags = lists.Flags;
                for (long k = first; k < end; k++)
                {
                    if (k > first)
                    {
                        var separator = Room(2);
                        separator[0] = ',';
                        separator[1] = ' ';
                    }
                    long o = offsets[k];
                    int n = checked((int)(offsets[k + 1] - o));
                    if (lists.UnitSize == 1)
                    {
                        // A bytes piece: b'...' (bytes have no np.str_ flag).
                        var bytes = new ReadOnlySpan<byte>(lists.Units + o, n);
                        if (bytes.IndexOfAnyExceptInRange((byte)0x20, (byte)0x7E) < 0 && bytes.IndexOfAny((byte)'\'', (byte)'\\') < 0)
                        {
                            var dst = Room(n + 3);
                            dst[0] = 'b';
                            dst[1] = '\'';
                            for (int i = 0; i < n; i++)
                                dst[2 + i] = (char)bytes[i];
                            dst[n + 2] = '\'';
                        }
                        else
                        {
                            Append(PyRepr.Bytes(bytes));
                        }
                        continue;
                    }
                    // A str piece prints through its code points (a unit above U+10FFFF prints as '\U00110000', as in
                    // CPython); an element's own np.str_ prints inside np.str_(...).
                    var units = new ReadOnlySpan<uint>((uint*)lists.Units + o, n);
                    bool npStr = flags != null && flags[k];
                    if (npStr)
                        Append("np.str_(");
                    if (units.IndexOfAnyExceptInRange(0x20u, 0x7Eu) < 0 && units.IndexOfAny((uint)'\'', (uint)'\\') < 0)
                    {
                        var dst = Room(n + 2);
                        dst[0] = '\'';
                        for (int i = 0; i < n; i++)
                            dst[1 + i] = (char)units[i];
                        dst[n + 1] = '\'';
                    }
                    else
                    {
                        Append(PyRepr.StrCodePoints(units));
                    }
                    if (npStr)
                        Room(1)[0] = ')';
                }
                Room(1)[0] = ']';
            }
        }

        // =============================================================================================================
        // Lifetime
        // =============================================================================================================

        /// <summary>Releases the ids this list array holds and its reference on the lists (a view releases only its own).</summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            Ids.Dispose();
            Heap.Dispose();
        }

        /// <summary>Throws when the array was disposed.</summary>
        /// <exception cref="ObjectDisposedException">Always, once <see cref="Dispose"/> ran.</exception>
        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(NDListArray));
        }

        // =============================================================================================================
        // Internals
        // =============================================================================================================

        /// <summary>A list array over <paramref name="ids"/> (a view or a copy of this array's ids) and this array's lists.</summary>
        /// <param name="ids">The new ids (handed over).</param>
        /// <returns>The list array (holding a new reference on the shared lists).</returns>
        /// <exception cref="ObjectDisposedException">The array was disposed (the ids are disposed first).</exception>
        private NDListArray Over(NDArray ids)
        {
            if (_disposed)
            {
                ids.Dispose();
                throw new ObjectDisposedException(nameof(NDListArray));
            }
            // The new array holds its own reference on the lists: a slice must outlive the array it came from.
            return new NDListArray(ids, Heap.Share());
        }

        /// <summary>The list id of the element at a full integer index.</summary>
        /// <param name="index">One integer per axis.</param>
        /// <returns>The id.</returns>
        /// <exception cref="IndexError">An integer outside its axis, or not one integer per axis.</exception>
        private long Element(long[] index)
        {
            ThrowIfDisposed();
            index ??= Array.Empty<long>();
            if (index.Length != ndim)
                throw new IndexError($"an NDListArray element needs one integer per axis ({index.Length} given for {ndim}); " +
                                     "use slice notation, as in lists[\"0\"], for a sub-array");
            var dims = Ids.Shape.dimensions;
            var at = new long[index.Length];
            for (int d = 0; d < index.Length; d++)
            {
                long i = index[d];
                long n = dims[d];
                if (i < -n || i >= n)
                    throw new IndexError($"index {i} is out of bounds for axis {d} with size {n}");
                at[d] = i < 0 ? i + n : i;
            }
            return Ids.GetInt64(at);
        }

        /// <summary>The list id of the element at a C-order position.</summary>
        /// <param name="position">The position (0 .. size - 1).</param>
        /// <returns>The id.</returns>
        private long FlatId(long position)
        {
            ThrowIfDisposed();
            return Ids.GetAtIndex<long>(position);
        }

        /// <summary>The nested lists of <see cref="tolist"/> below axis <paramref name="axis"/>.</summary>
        /// <param name="axis">The axis being expanded.</param>
        /// <param name="ids">The flat C-order ids.</param>
        /// <param name="position">The next C-order position (advanced as elements are read).</param>
        /// <returns>The list for this axis.</returns>
        private List<object> Nest(int axis, NDArray ids, ref long position)
        {
            long n = Ids.Shape.dimensions[axis];
            var list = new List<object>(checked((int)n));
            for (long i = 0; i < n; i++)
            {
                if (axis == ndim - 1)
                    list.Add(Heap.ToList(ids.GetInt64(position++)));
                else
                    list.Add(Nest(axis + 1, ids, ref position));
            }
            return list;
        }

        /// <summary>
        ///     One list's pieces as a new 1-D array of the input's class: <c>S</c> / <c>U</c> as wide as the longest piece
        ///     (at least 1), or <c>StringDType</c>.
        /// </summary>
        /// <param name="id">The list id.</param>
        /// <returns>The array.</returns>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF (for a <c>StringDType</c> array of
        ///     the pieces, which needs .NET strings).</exception>
        private unsafe NDArray Pieces(long id, bool asarray = false)
        {
            var (start, end) = Heap.Range(id);
            long count = end - start;
            // np.asarray of the list makes a U array of Python strs whatever produced them; the element view keeps the
            // StringDType class of a StringDType input.
            if (Kind == 'T' && !asarray)
            {
                var texts = new string[count];
                for (long k = 0; k < count; k++)
                {
                    uint* p = (uint*)Heap.Piece(start + k, out long n);
                    texts[k] = NDCharObjectPath.Ucs4ToString(p, checked((int)n));
                }
                return np.array(texts, np.dtypes.StringDType());
            }

            long width = 1;
            for (long k = start; k < end; k++)
            {
                Heap.Piece(k, out long n);
                width = Math.Max(width, n);
            }
            var descr = Kind == 'S' ? DTypeRegistry.Bytes.Descr(width) : DTypeRegistry.Str.Descr(width);
            // A fresh zeroed array: each piece is copied into the start of its element, the rest stays NUL padding.
            var result = new NDArray(descr, NumSharp.Shape.Adopt(new[] { count }), true);
            int unit = Heap.UnitSize;
            byte* dst = (byte*)result.Address;
            long itemsize = width * unit;
            for (long k = 0; k < count; k++)
            {
                byte* p = Heap.Piece(start + k, out long n);
                long bytes = n * unit;
                Buffer.MemoryCopy(p, dst + k * itemsize, bytes, bytes);
            }
            return result;
        }

        /// <summary>
        ///     The element formatter NumPy's <c>_object_format</c> is for an object array of lists: <c>list(</c> + the
        ///     list's repr + <c>)</c>, keyed by the element's C-order position.
        /// </summary>
        [NDBorrowed] // a formatter over the printed array and its flat ids, which ToString owns and releases
        private sealed class ListReprFormat : IElementFormatter
        {
            /// <summary>The list array printed.</summary>
            private readonly NDListArray _owner;

            /// <summary>The flat C-order ids.</summary>
            private readonly NDArray _ids;

            /// <summary>Creates the formatter over the flat ids.</summary>
            /// <param name="owner">The list array.</param>
            /// <param name="ids">The flat C-order ids.</param>
            internal ListReprFormat(NDListArray owner, NDArray ids)
            {
                _owner = owner;
                _ids = ids;
            }

            /// <summary>Formats the element at a C-order position.</summary>
            /// <param name="value">The position (a boxed int64 from the layout engine's position array).</param>
            /// <returns><c>list([...])</c>.</returns>
            public string Format(object value) => "list(" + _owner.ListRepr(_ids.GetInt64(Convert.ToInt64(value))) + ")";
        }
    }
}
