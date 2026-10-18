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
    ///     <b>Element access.</b> A full integer index (one integer per axis, negative values counting from the end) is the
    ///     element's list as a NEW 1-D array of the input's string class — <c>S</c> / <c>U</c> as wide as its longest piece
    ///     (at least 1), or <c>StringDType</c> for a <c>StringDType</c> input — which is the closest NumSharp value to NumPy's
    ///     Python list. A piece that ENDS in a NUL loses it in that array, as every fixed-width element does
    ///     (<c>np.array(['a\x00'])</c> reads <c>'a'</c> in NumPy too); <see cref="tolist"/> and <see cref="item(long[])"/>
    ///     keep it. Slice notation and <see cref="Slice"/>s give an <see cref="NDListArray"/> view.
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
        /// <param name="starts">Each element's first piece, C order (handed over).</param>
        /// <param name="ends">One past each element's last piece (handed over).</param>
        /// <param name="selfElements">Each element's <c>[self]</c> flag (bool, C order) or null (read and disposed).</param>
        /// <param name="kind">The class of the pieces.</param>
        /// <param name="outer">The outer shape.</param>
        /// <returns>The list array.</returns>
        internal static NDListArray FromElements(NDArray units, NDArray offsets, NDArray starts, NDArray ends, NDArray selfElements,
                                                 char kind, Shape outer)
        {
            var heap = NDListHeap.FromElements(kind, units, offsets, starts, ends, selfElements);
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
        ///     The list of one element (a full integer index: one integer per axis, a negative one counting from the end) as a
        ///     new 1-D array of its pieces — <c>S</c> / <c>U</c> as wide as the longest piece (at least 1), or
        ///     <c>StringDType</c>.
        /// </summary>
        /// <param name="index">One integer per axis.</param>
        /// <returns>The pieces, in order (an empty array for an empty list).</returns>
        /// <exception cref="IndexError">An integer outside its axis (NumPy's <c>index 5 is out of bounds for axis 0 with size
        ///     2</c>), or fewer or more integers than axes (a partial index selects a sub-array: use slice notation, as in
        ///     <c>lists["0"]</c>, for that).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF (it has no .NET form).</exception>
        public NDArray this[params long[] index] => Pieces(Element(index));

        /// <summary>The list of one element as a new 1-D array of its pieces (see <see cref="this[long[]]"/>).</summary>
        /// <param name="index">One integer per axis.</param>
        /// <returns>The pieces.</returns>
        /// <exception cref="IndexError">An integer outside its axis, or not one integer per axis.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF.</exception>
        public NDArray this[params int[] index]
        {
            get
            {
                var wide = new long[index?.Length ?? 0];
                for (int i = 0; i < wide.Length; i++)
                    wide[i] = index[i];
                return this[wide];
            }
        }

        /// <summary>Basic indexing in slice notation (<c>"1:"</c>, <c>"::-1, 0"</c>): a view over the selected lists.</summary>
        /// <param name="slices">NumPy's slice notation.</param>
        /// <returns>The view (a 0-d one for one integer per axis).</returns>
        /// <exception cref="IndexError">An integer outside its axis.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray this[string slices] => Over(Ids[slices]);

        /// <summary>Basic indexing with <see cref="Slice"/>s: a view over the selected lists.</summary>
        /// <param name="slices">One slice (or index) per axis, as for an <see cref="NDArray"/>.</param>
        /// <returns>The view.</returns>
        /// <exception cref="IndexError">An integer outside its axis.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray this[params Slice[] slices] => Over(Ids[slices]);

        /// <summary>
        ///     NumPy's <c>item()</c>: the list of a one-element array as a .NET list of its pieces (<see cref="string"/> for
        ///     str pieces, <c>byte[]</c> for bytes pieces), every piece exact.
        /// </summary>
        /// <returns>The list.</returns>
        /// <exception cref="ValueError">The array has more or fewer than one element (NumPy's <c>can only convert an array of
        ///     size 1 to a Python scalar</c>).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF.</exception>
        public List<object> item()
        {
            ThrowIfDisposed();
            if (size != 1)
                throw new ValueError("can only convert an array of size 1 to a Python scalar");
            return Heap.ToList(FlatId(0));
        }

        /// <summary>
        ///     NumPy's <c>item(*args)</c>: the list of one element, its pieces exact — ONE integer is a flat index in C order
        ///     (negative from the end), one integer per axis a full index.
        /// </summary>
        /// <param name="index">A flat index, or one integer per axis.</param>
        /// <returns>The list.</returns>
        /// <exception cref="IndexError">An index outside the array (<c>index 6 is out of bounds for size 6</c> for a flat
        ///     index, <c>index 5 is out of bounds for axis 1 with size 3</c> for a full one).</exception>
        /// <exception cref="ValueError">Neither one integer nor one per axis (<c>incorrect number of indices for
        ///     array</c>).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF.</exception>
        public List<object> item(params long[] index)
        {
            ThrowIfDisposed();
            if (index is not { Length: > 0 })
                return item();
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
        ///     Iterates the first axis: for a 1-d array every element's list as the array of its pieces (see
        ///     <see cref="this[long[]]"/>), for an N-d array every sub-array as an <see cref="NDListArray"/> view.
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
                if (ndim == 1)
                    yield return this[i];
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
        internal unsafe string ListRepr(long id)
        {
            var sb = new StringBuilder("[");
            var (start, end) = Heap.Range(id);
            for (long k = start; k < end; k++)
            {
                if (k > start)
                    sb.Append(", ");
                byte* p = Heap.Piece(k, out long n);
                if (Kind == 'S')
                {
                    sb.Append(PyRepr.Bytes(new ReadOnlySpan<byte>(p, checked((int)n))));
                    continue;
                }
                // A str piece prints through the code points (a unit above U+10FFFF prints as '\U00110000', as in CPython).
                string text = PyRepr.StrCodePoints(new ReadOnlySpan<uint>((uint*)p, checked((int)n)));
                sb.Append(Heap.IsNpStr(k) ? "np.str_(" + text + ")" : text);
            }
            return sb.Append(']').ToString();
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
            Heap.Release();
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
        private unsafe NDArray Pieces(long id)
        {
            var (start, end) = Heap.Range(id);
            long count = end - start;
            if (Kind == 'T')
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
