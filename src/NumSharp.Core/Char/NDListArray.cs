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
//   * Units   — every piece's code units back to back (uint8 for bytes pieces, uint32 code points for str pieces);
//   * Offsets — piece k is Units[Offsets[k] : Offsets[k + 1]], so a piece keeps its exact length (a piece that ends in a
//               NUL, 'a\x00' from 'a\x00 b'.split(), keeps it; a fixed-width array of pieces could not);
//   * Starts / Ends — int64 arrays of the OUTER shape: element e's list is pieces [Starts[e], Ends[e]). Basic indexing
//               slices only these two (an NDListArray view shares Units and Offsets, as a NumPy view shares the lists);
//   * SelfPiece — for str pieces of a U array: whether the element's single piece IS the element's own np.str_ scalar
//               (CPython's split returns [self] when the separator is longer or of a wider kind than the string). Python
//               would print that piece as np.str_('...'), and bytes.join can read it as a buffer, so the flag is kept.
//
// The layout is NumPy's: the result is C-ordered over the broadcast shape (PyArray_SimpleNewFromDescr), whatever the
// input layout, so Starts / Ends are fresh C-contiguous arrays and the pieces follow in C order.
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's object array of Python lists — what <c>np.char.split</c>, <c>rsplit</c> and <c>splitlines</c> return
    ///     (<c>array([list(['a', 'b']), list([])], dtype=object)</c>). NumSharp has no object dtype, so this type holds the
    ///     array's shape and every element's list of pieces: the text of the pieces once, each piece with its exact length.
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
    ///     <b>Lifetime.</b> The arrays the type holds are NumSharp arrays: <see cref="Dispose"/> releases them at once, the
    ///     finalizers release them otherwise. A view keeps the memory it reads alive (it counts its own references).
    ///     </para>
    /// </remarks>
    public sealed class NDListArray : IEnumerable, IDisposable
    {
        /// <summary>Every piece's code units back to back: 1-D uint8 for bytes pieces, 1-D uint32 code points for str pieces.</summary>
        internal NDArray Units { get; }

        /// <summary>1-D int64 of (piece count + 1) entries: piece k spans <c>Units[Offsets[k] : Offsets[k + 1]]</c>.</summary>
        internal NDArray Offsets { get; }

        /// <summary>int64 of the outer shape: the index of each element's first piece.</summary>
        internal NDArray Starts { get; }

        /// <summary>int64 of the outer shape: one past each element's last piece.</summary>
        internal NDArray Ends { get; }

        /// <summary>
        ///     bool of the outer shape, or null when no element has it: the element's single piece is the element's own
        ///     <c>np.str_</c> scalar (only for the pieces of a <c>U</c> array: <c>'ab'.split('abc')</c> is <c>[self]</c>).
        /// </summary>
        internal NDArray SelfPiece { get; }

        /// <summary>
        ///     The class of the pieces: <c>'S'</c> (Python <c>bytes</c>, from an <c>S</c> array), <c>'U'</c> (Python
        ///     <c>str</c>, from a <c>U</c> array) or <c>'T'</c> (Python <c>str</c>, from a <c>StringDType</c> array).
        /// </summary>
        internal char Kind { get; }

        /// <summary>Set by <see cref="Dispose"/>: every member then throws <see cref="ObjectDisposedException"/>.</summary>
        private bool _disposed;

        /// <summary>Creates the list array over its parts (the caller hands its references over).</summary>
        /// <param name="units">The pieces' code units (1-D uint8 for <c>'S'</c>, uint32 otherwise).</param>
        /// <param name="offsets">The piece offsets (1-D int64, piece count + 1 entries).</param>
        /// <param name="starts">The first piece of each element (int64, the outer shape).</param>
        /// <param name="ends">One past the last piece of each element (int64, the outer shape).</param>
        /// <param name="selfPiece">The self-piece flags (bool, the outer shape) or null.</param>
        /// <param name="kind">The class of the pieces.</param>
        internal NDListArray(NDArray units, NDArray offsets, NDArray starts, NDArray ends, NDArray selfPiece, char kind)
        {
            Units = units;
            Offsets = offsets;
            Starts = starts;
            Ends = ends;
            SelfPiece = selfPiece;
            Kind = kind;
        }

        // =============================================================================================================
        // The array attributes
        // =============================================================================================================

        /// <summary>NumPy's <c>shape</c> (a copy, NumPy's tuple).</summary>
        public long[] shape => Starts.shape;

        /// <summary>NumPy's <c>ndim</c>: the number of axes.</summary>
        public int ndim => Starts.ndim;

        /// <summary>NumPy's <c>size</c>: the number of elements (lists).</summary>
        public long size => Starts.size;

        /// <summary>The outer shape as a NumSharp <see cref="NumSharp.Shape"/> (dimensions only: the lists have no strides).</summary>
        public Shape Shape => NumSharp.Shape.Adopt(Starts.shape);

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
        public NDListArray this[string slices] => View(Starts[slices], Ends[slices], SelfPiece?[slices]);

        /// <summary>Basic indexing with <see cref="Slice"/>s: a view over the selected lists.</summary>
        /// <param name="slices">One slice (or index) per axis, as for an <see cref="NDArray"/>.</param>
        /// <returns>The view.</returns>
        /// <exception cref="IndexError">An integer outside its axis.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray this[params Slice[] slices] => View(Starts[slices], Ends[slices], SelfPiece?[slices]);

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
            if (size != 1)
                throw new ValueError("can only convert an array of size 1 to a Python scalar");
            return ListOf(FlatElement(0));
        }

        /// <summary>NumPy's <c>item(*args)</c>: the list of the element at a full integer index, its pieces exact.</summary>
        /// <param name="index">One integer per axis.</param>
        /// <returns>The list.</returns>
        /// <exception cref="IndexError">An integer outside its axis, or not one integer per axis.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF.</exception>
        public List<object> item(params long[] index) => index is { Length: > 0 } ? ListOf(Element(index)) : item();

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
            // The flat C-order copies of the element ranges, read in the order the nesting visits them.
            using var starts = Starts.flatten();
            using var ends = Ends.flatten();
            long position = 0;
            if (ndim == 0)
                return ListOf((starts.GetInt64(0), ends.GetInt64(0)));
            return Nest(0, starts, ends, ref position);
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
            long n = Starts.Shape.dimensions[0];
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
            using var starts = Starts.flatten();
            using var ends = Ends.flatten();
            using var self = SelfPiece?.flatten();
            var formatter = new ListReprFormat(this, starts, ends, self);
            // The layout engine walks an int64 array of the outer shape whose values are the C-order positions; the
            // formatter maps a position to its element's list repr.
            using var flatPositions = np.arange(size);
            using var positions = ndim == 0 ? flatPositions.reshape(NumSharp.Shape.NewScalar()) : flatPositions.reshape(NumSharp.Shape.Adopt(Starts.shape));
            var opts = PrintOptions.Current;
            if (!repr)
                return ndim == 0 ? StrOfList(starts.GetInt64(0), ends.GetInt64(0), self) : ArrayFormatter.ObjectArrayStr(positions, formatter, opts);
            return ArrayFormatter.ObjectArrayRepr(positions, formatter, opts);
        }

        /// <summary>Python's <c>str()</c> of a list (its repr): <c>['a', np.str_('b')]</c>.</summary>
        /// <param name="start">The element's first piece.</param>
        /// <param name="end">One past its last piece.</param>
        /// <param name="self">The flat self-piece flags, or null.</param>
        /// <returns>The text.</returns>
        private string StrOfList(long start, long end, NDArray self)
            => ListRepr(start, end, self is not null && self.GetBoolean(0));

        /// <summary>Python's <c>repr()</c> of one element's list: <c>['a', 'b']</c>, an <c>np.str_</c> self piece as
        ///     <c>np.str_('ab')</c>.</summary>
        /// <param name="start">The first piece.</param>
        /// <param name="end">One past the last piece.</param>
        /// <param name="selfPiece">Whether the single piece is the element's own <c>np.str_</c>.</param>
        /// <returns>The repr.</returns>
        internal unsafe string ListRepr(long start, long end, bool selfPiece)
        {
            var sb = new StringBuilder("[");
            long* offsets = (long*)Offsets.Address;
            for (long k = start; k < end; k++)
            {
                if (k > start)
                    sb.Append(", ");
                long from = offsets[k], to = offsets[k + 1];
                if (Kind == 'S')
                {
                    sb.Append(PyRepr.Bytes(new ReadOnlySpan<byte>((byte*)Units.Address + from, checked((int)(to - from)))));
                    continue;
                }
                // A str piece prints through the code points (a unit above U+10FFFF prints as '\U00110000', as in CPython).
                string text = PyRepr.StrCodePoints(new ReadOnlySpan<uint>((uint*)Units.Address + from, checked((int)(to - from))));
                sb.Append(selfPiece ? "np.str_(" + text + ")" : text);
            }
            return sb.Append(']').ToString();
        }

        // =============================================================================================================
        // Lifetime
        // =============================================================================================================

        /// <summary>Releases the arrays this list array holds (a view releases only its own references).</summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            Units.Dispose();
            Offsets.Dispose();
            Starts.Dispose();
            Ends.Dispose();
            SelfPiece?.Dispose();
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

        /// <summary>
        ///     A view over selected lists: new references on the shared pieces, the sliced element ranges.
        /// </summary>
        /// <param name="starts">The sliced <see cref="Starts"/>.</param>
        /// <param name="ends">The sliced <see cref="Ends"/>.</param>
        /// <param name="self">The sliced <see cref="SelfPiece"/>, or null.</param>
        /// <returns>The view.</returns>
        private NDListArray View(NDArray starts, NDArray ends, NDArray self)
        {
            ThrowIfDisposed();
            // The view holds its own references on the shared pieces: a slice must outlive the array it came from.
            return new NDListArray(new NDArray(Units.Storage.Alias()) { TensorEngine = Units.TensorEngine },
                                   new NDArray(Offsets.Storage.Alias()) { TensorEngine = Offsets.TensorEngine },
                                   starts, ends, self, Kind);
        }

        /// <summary>The piece range and self flag of the element at a full integer index.</summary>
        /// <param name="index">One integer per axis.</param>
        /// <returns>(first piece, one past the last, the self-piece flag).</returns>
        /// <exception cref="IndexError">An integer outside its axis, or not one integer per axis.</exception>
        private (long Start, long End, bool Self) Element(long[] index)
        {
            ThrowIfDisposed();
            index ??= Array.Empty<long>();
            if (index.Length != ndim)
                throw new IndexError($"an NDListArray element needs one integer per axis ({index.Length} given for {ndim}); " +
                                     "use slice notation, as in lists[\"0\"], for a sub-array");
            var dims = Starts.Shape.dimensions;
            var at = new long[index.Length];
            for (int d = 0; d < index.Length; d++)
            {
                long i = index[d];
                long n = dims[d];
                if (i < -n || i >= n)
                    throw new IndexError($"index {i} is out of bounds for axis {d} with size {n}");
                at[d] = i < 0 ? i + n : i;
            }
            long start = Starts.GetInt64(at);
            long end = Ends.GetInt64(at);
            bool self = SelfPiece is not null && SelfPiece.GetBoolean(at);
            return (start, end, self);
        }

        /// <summary>The piece range and self flag of the element at a C-order position.</summary>
        /// <param name="position">The position (0 .. size - 1).</param>
        /// <returns>(first piece, one past the last, the self-piece flag).</returns>
        private (long Start, long End, bool Self) FlatElement(long position)
        {
            ThrowIfDisposed();
            using var starts = Starts.flatten();
            using var ends = Ends.flatten();
            using var self = SelfPiece?.flatten();
            return (starts.GetInt64(position), ends.GetInt64(position), self is not null && self.GetBoolean(position));
        }

        /// <summary>One element's pieces as a .NET list (<see cref="string"/> / <c>byte[]</c>, exact).</summary>
        /// <param name="range">The element's piece range.</param>
        /// <returns>The list.</returns>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF.</exception>
        private List<object> ListOf((long Start, long End) range) => ListOf((range.Start, range.End, false));

        /// <summary>One element's pieces as a .NET list (<see cref="string"/> / <c>byte[]</c>, exact).</summary>
        /// <param name="element">The element's piece range (the self flag does not change the values).</param>
        /// <returns>The list.</returns>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF.</exception>
        private unsafe List<object> ListOf((long Start, long End, bool Self) element)
        {
            var list = new List<object>(checked((int)(element.End - element.Start)));
            long* offsets = (long*)Offsets.Address;
            for (long k = element.Start; k < element.End; k++)
            {
                long from = offsets[k], to = offsets[k + 1];
                int n = checked((int)(to - from));
                if (Kind == 'S')
                    list.Add(new ReadOnlySpan<byte>((byte*)Units.Address + from, n).ToArray());
                else
                    list.Add(NDCharObjectPath.Ucs4ToString((uint*)Units.Address + from, n));
            }
            return list;
        }

        /// <summary>The nested lists of <see cref="tolist"/> below axis <paramref name="axis"/>.</summary>
        /// <param name="axis">The axis being expanded.</param>
        /// <param name="starts">The flat C-order first pieces.</param>
        /// <param name="ends">The flat C-order ends.</param>
        /// <param name="position">The next C-order position (advanced as elements are read).</param>
        /// <returns>The list for this axis.</returns>
        private List<object> Nest(int axis, NDArray starts, NDArray ends, ref long position)
        {
            long n = Starts.Shape.dimensions[axis];
            var list = new List<object>(checked((int)n));
            for (long i = 0; i < n; i++)
            {
                if (axis == ndim - 1)
                {
                    list.Add(ListOf((starts.GetInt64(position), ends.GetInt64(position))));
                    position++;
                }
                else
                {
                    list.Add(Nest(axis + 1, starts, ends, ref position));
                }
            }
            return list;
        }

        /// <summary>
        ///     One element's pieces as a new 1-D array of the input's class: <c>S</c> / <c>U</c> as wide as the longest piece
        ///     (at least 1), or <c>StringDType</c>.
        /// </summary>
        /// <param name="element">The element's piece range.</param>
        /// <returns>The array.</returns>
        /// <exception cref="SystemError">A <c>U</c> piece holds a code unit above U+10FFFF (for a <c>StringDType</c> array of
        ///     the pieces, which needs .NET strings).</exception>
        private unsafe NDArray Pieces((long Start, long End, bool Self) element)
        {
            long count = element.End - element.Start;
            long* offsets = (long*)Offsets.Address;
            if (Kind == 'T')
            {
                var texts = new string[count];
                for (long k = 0; k < count; k++)
                {
                    long from = offsets[element.Start + k], to = offsets[element.Start + k + 1];
                    texts[k] = NDCharObjectPath.Ucs4ToString((uint*)Units.Address + from, checked((int)(to - from)));
                }
                return np.array(texts, np.dtypes.StringDType());
            }

            long width = 1;
            for (long k = element.Start; k < element.End; k++)
                width = Math.Max(width, offsets[k + 1] - offsets[k]);
            var descr = Kind == 'S' ? DTypeRegistry.Bytes.Descr(width) : DTypeRegistry.Str.Descr(width);
            // A fresh zeroed array: each piece is copied into the start of its element, the rest stays NUL padding.
            var result = new NDArray(descr, NumSharp.Shape.Adopt(new[] { count }), true);
            int unit = Kind == 'S' ? 1 : 4;
            byte* dst = (byte*)result.Address;
            byte* src = (byte*)Units.Address;
            long itemsize = width * unit;
            for (long k = 0; k < count; k++)
            {
                long from = offsets[element.Start + k], to = offsets[element.Start + k + 1];
                long bytes = (to - from) * unit;
                Buffer.MemoryCopy(src + from * unit, dst + k * itemsize, bytes, bytes);
            }
            return result;
        }

        /// <summary>
        ///     The element formatter NumPy's <c>_object_format</c> is for an object array of lists: <c>list(</c> + the
        ///     list's repr + <c>)</c>, keyed by the element's C-order position.
        /// </summary>
        [NDBorrowed] // a formatter over the printed array's own buffers; ToString releases the flat copies it made
        private sealed class ListReprFormat : IElementFormatter
        {
            /// <summary>The list array printed.</summary>
            private readonly NDListArray _owner;

            /// <summary>The flat C-order first pieces.</summary>
            private readonly NDArray _starts;

            /// <summary>The flat C-order ends.</summary>
            private readonly NDArray _ends;

            /// <summary>The flat C-order self-piece flags, or null.</summary>
            private readonly NDArray _self;

            /// <summary>Creates the formatter over the flat element ranges.</summary>
            /// <param name="owner">The list array.</param>
            /// <param name="starts">The flat first pieces.</param>
            /// <param name="ends">The flat ends.</param>
            /// <param name="self">The flat self flags, or null.</param>
            internal ListReprFormat(NDListArray owner, NDArray starts, NDArray ends, NDArray self)
            {
                _owner = owner;
                _starts = starts;
                _ends = ends;
                _self = self;
            }

            /// <summary>Formats the element at a C-order position.</summary>
            /// <param name="value">The position (a boxed int64 from the layout engine's position array).</param>
            /// <returns><c>list([...])</c>.</returns>
            public string Format(object value)
            {
                long p = Convert.ToInt64(value);
                return "list(" + _owner.ListRepr(_starts.GetInt64(p), _ends.GetInt64(p), _self is not null && _self.GetBoolean(p)) + ")";
            }
        }
    }
}
