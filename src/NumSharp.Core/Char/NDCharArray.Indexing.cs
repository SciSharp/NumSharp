using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Generic;

// =============================================================================
// NDCharArray.Indexing.cs — chararray.__getitem__: a scalar strips, an array stays a chararray
// =============================================================================
//
// NumPy's chararray.__getitem__ is two lines:
//
//     val = ndarray.__getitem__(self, obj)
//     if isinstance(val, character): return val.rstrip()
//     return val
//
// ndarray.__getitem__ returns a SCALAR exactly for a full integer index (mapping.c: index_type == HAS_INTEGER): one
// integer per axis — a C# integer, an integer Slice, or a 0-d integer ARRAY ("a full integer index sees array scalars
// as part of itself") — and no ellipsis, new axis, slice, bool or index array. For a 0-d chararray the empty index
// (c[""], c[new Slice[0]], NumPy's c[()]) is that full index. Every other index returns an array, created as the
// subclass: a chararray view (basic index), a chararray view of the gathered array (index array, mask), or an owning
// copy (NumPy's HAS_SCALAR_ARRAY: a 0-d integer array next to slices), each after the finalize rule.
//
// THE C# TYPE. An indexer has one type for get and set. These are typed NDCharArray so that c["1:3"][0] and c2[0][1]
// strip like NumPy's chained indexing, therefore the value assigned through them is an NDCharArray or a C# string /
// byte[] (implicit conversions). Assigning stores through the plain array rule (NumPy's chararray does not override
// __setitem__). Assign any other value (a number, a plain NDArray) with __setitem__.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        #region Indexers

        /// <summary>
        ///     Indexes with slice objects (<c>c[Slice.All, 0]</c>, <c>c[new Slice(1, 3)]</c>) or integers
        ///     (<c>c[0]</c>, <c>c[0, 1]</c> — an <see cref="int"/> converts to an integer <see cref="Slice"/>).
        /// </summary>
        /// <param name="slices">One entry per indexed axis; <see cref="Slice.Ellipsis"/> and <see cref="Slice.NewAxis"/> allowed.</param>
        /// <value>
        ///     For an integer per axis: the element as a read-only 0-d stand-in, its trailing whitespace stripped
        ///     (NumPy's Python <c>str</c> / <c>bytes</c>). Otherwise a chararray VIEW. Assigning stores through the plain
        ///     array rule (truncated to the element width, <c>S</c> encodes ASCII).
        /// </value>
        /// <exception cref="ArgumentNullException"><paramref name="slices"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes (NumPy's texts).</exception>
        /// <exception cref="ValueError">(get) The chararray holds data that is not string data and the index selects an
        ///     array; (set) the chararray is read-only, or the value does not broadcast.</exception>
        /// <exception cref="SystemError">(get) A <c>U</c> element holds a unit above <c>U+10FFFF</c>.</exception>
        public new NDCharArray this[params Slice[] slices]
        {
            get => GetBasic(slices);
            set => base[slices] = value;
        }

        /// <summary>
        ///     Indexes the first axis with one integer (<c>c[0]</c>, <c>c[-1]</c>): the same result as
        ///     <c>c[Slice.Index(index)]</c>, without building the index array. The usual element access of a 1-d chararray.
        /// </summary>
        /// <param name="index">The index along the first axis; negative counts from the end.</param>
        /// <value>
        ///     For a 1-d chararray: the element as a read-only 0-d stand-in, its trailing whitespace stripped (NumPy's Python
        ///     <c>str</c> / <c>bytes</c>). For more axes: a chararray VIEW of the row. Assigning stores through the plain array
        ///     rule (truncated to the element width, <c>S</c> encodes ASCII).
        /// </value>
        /// <exception cref="IndexError"><paramref name="index"/> is out of bounds (NumPy's text), or the chararray is 0-d.</exception>
        /// <exception cref="ValueError">(get) The chararray holds data that is not string data and the result is a row;
        ///     (set) the chararray is read-only, or the value does not broadcast.</exception>
        /// <exception cref="SystemError">(get) A <c>U</c> element holds a unit above <c>U+10FFFF</c>.</exception>
        public NDCharArray this[int index]
        {
            get => ElementOrRow(index);
            set => base[Slice.Index(index)] = value;
        }

        /// <inheritdoc cref="this[int]"/>
        public NDCharArray this[long index]
        {
            get => ElementOrRow(index);
            set => base[Slice.Index(index)] = value;
        }

        /// <summary>
        ///     Indexes with Python slice notation (<c>c["1:3"]</c>, <c>c[":, ::-1"]</c>, <c>c["0, 1"]</c>); the empty string
        ///     is NumPy's empty index <c>c[()]</c>.
        /// </summary>
        /// <param name="slice">The comma-separated entries: integers, <c>start:stop:step</c>, <c>...</c>, <c>np.newaxis</c>.</param>
        /// <value>The stripped 0-d stand-in for an integer per axis (the empty index of a 0-d chararray included); a
        ///     chararray view otherwise.</value>
        /// <exception cref="ArgumentNullException"><paramref name="slice"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes.</exception>
        /// <exception cref="ValueError">(get) The chararray holds data that is not string data and the index selects an
        ///     array; (set) the chararray is read-only, or the value does not broadcast.</exception>
        /// <exception cref="SystemError">(get) A <c>U</c> element holds a unit above <c>U+10FFFF</c>.</exception>
        public new NDCharArray this[string slice]
        {
            get => GetBasic(Slice.ParseSlices(slice));
            set => base[slice] = value;
        }

        /// <summary>
        ///     The general index: integers, slices, slice strings, index arrays, boolean masks and sequences in any mix
        ///     (NumPy's index tuple). A boolean array from a comparison binds here: <c>c[c != ""]</c>.
        /// </summary>
        /// <param name="indices">The index entries. A single C# tuple entry is NumPy's index tuple and is spread.</param>
        /// <value>
        ///     The stripped 0-d stand-in for a full integer index (integers and 0-d integer arrays, one per axis). A chararray
        ///     view for an index of integers and slices; a chararray view of the gathered array (OWNDATA false) for an index
        ///     array or a mask; an owning chararray copy for NumPy's 0-d-integer-array-plus-slices case.
        /// </value>
        /// <exception cref="IndexError">An index is out of bounds, too many entries, or index arrays that do not broadcast
        ///     together.</exception>
        /// <exception cref="ValueError">(get) The chararray holds data that is not string data and the index selects an
        ///     array; (set) the chararray is read-only, or the value does not fit the selection.</exception>
        /// <exception cref="SystemError">(get) A <c>U</c> element holds a unit above <c>U+10FFFF</c>.</exception>
        public new NDCharArray this[params object[] indices]
        {
            // An index of integers and slices takes the view route of the typed indexers (one scalar rule, NumPy's index
            // errors); any other entry goes through the plain array's advanced indexing first, whose result the scope
            // reclaims once the chararray holds its own reference.
            [NDScoped]
            get
            {
                if (indices is not null && TryAsSlices(indices, out Slice[] slices))
                    return GetBasic(slices);
                NDArray result = base[indices];
                return IsFullIntegerIndex(indices)
                    ? ScalarOf(result.Storage, strip: true)
                    : FromIndexing(result, this, IsScalarArrayIndex(indices));
            }
            set => base[indices] = value;
        }

        /// <summary>Selects with a boolean mask (NumPy's <c>c[mask]</c>): the selected elements in C order.</summary>
        /// <param name="mask">The mask.</param>
        /// <value>A chararray VIEW of the gathered elements (OWNDATA false, as NumPy's subclass result). Assigning stores
        ///     into the selected elements.</value>
        /// <exception cref="IndexError">The mask's shape does not match.</exception>
        /// <exception cref="ValueError">(get) The chararray holds data that is not string data; (set) the chararray is
        ///     read-only, or the value count differs from the number of selected elements.</exception>
        public new NDCharArray this[NDArray<bool> mask]
        {
            [NDScoped] get => FromIndexing(base[mask], this, scalarArrayCopy: false);
            set => base[mask] = value;
        }

        /// <summary>Selects with integer index arrays, one per axis (NumPy's <c>c[rows, cols]</c>).</summary>
        /// <param name="selection">The index arrays.</param>
        /// <value>The stripped 0-d stand-in when every array is 0-d and there is one per axis; otherwise a chararray VIEW of
        ///     the gathered elements.</value>
        /// <exception cref="IndexError">An index is out of bounds, or the arrays do not broadcast together.</exception>
        /// <exception cref="ValueError">(get) The chararray holds data that is not string data; (set) the chararray is
        ///     read-only, or the value does not broadcast to the selection.</exception>
        public new NDCharArray this[params NDArray<int>[] selection]
        {
            [NDScoped]
            get
            {
                NDArray result = base[selection];
                return IsFullIntegerIndex(selection)
                    ? ScalarOf(result.Storage, strip: true)
                    : FromIndexing(result, this, scalarArrayCopy: false);
            }
            set => base[selection] = value;
        }

        #endregion

        #region Python's __getitem__

        /// <summary>Python's <c>c.__getitem__(i)</c>: the same as <c>c[i]</c>.</summary>
        /// <param name="index">The index along the first axis; negative counts from the end.</param>
        /// <returns>The stripped 0-d stand-in for a 1-d chararray, a chararray row view otherwise.</returns>
        /// <exception cref="IndexError"><paramref name="index"/> is out of bounds, or the chararray is 0-d.</exception>
        /// <exception cref="ValueError">The chararray holds data that is not string data and the result is a row.</exception>
        public new NDCharArray __getitem__(int index) => ElementOrRow(index);

        /// <inheritdoc cref="__getitem__(int)"/>
        public new NDCharArray __getitem__(long index) => ElementOrRow(index);

        /// <summary>Python's <c>c.__getitem__</c> with slice notation: the same as <c>c[slice]</c>.</summary>
        /// <param name="slice">The slice notation.</param>
        /// <returns>The stripped 0-d stand-in for an integer per axis, a chararray view otherwise.</returns>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes.</exception>
        /// <exception cref="ValueError">The chararray holds data that is not string data and the result is an array.</exception>
        public new NDCharArray __getitem__(string slice) => this[slice];

        /// <summary>Python's <c>c.__getitem__((i, j))</c> with integer entries (the coordinates of one element when there is
        ///     one per axis).</summary>
        /// <param name="indices">One integer per indexed axis.</param>
        /// <returns>The stripped 0-d stand-in for an integer per axis, a chararray view of the sub-array otherwise.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An index is out of bounds, or more entries than axes.</exception>
        /// <exception cref="ValueError">The chararray holds data that is not string data and the result is an array.</exception>
        public new NDCharArray __getitem__(params int[] indices)
        {
            if (indices is null)
                throw new ArgumentNullException(nameof(indices));
            var slices = new Slice[indices.Length];
            for (int i = 0; i < slices.Length; i++)
                slices[i] = Slice.Index(indices[i]);
            return GetBasic(slices);
        }

        #endregion

        #region Cores

        /// <summary>
        ///     The result of an index of integers and slices: the stripped scalar stand-in for a full integer index, else a
        ///     chararray over the selected view. Built on the storage directly: no intermediate array is created.
        /// </summary>
        /// <param name="slices">The index entries.</param>
        /// <returns>The stand-in or the chararray view.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="slices"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An integer is out of bounds, or more entries than axes (the storage's NumPy texts).</exception>
        /// <exception cref="ValueError">This chararray holds data that is not string data and the result is an array.</exception>
        /// <exception cref="SystemError">A <c>U</c> element holds a unit above <c>U+10FFFF</c>.</exception>
        [NDAnchored] // the element's raw address is read after this array's last use: keep it rooted until the copy
        private unsafe NDCharArray GetBasic(Slice[] slices)
        {
            if (slices is null)
                throw new ArgumentNullException(nameof(slices));
            // One element of S / U text: its address straight from the strides (no view of this array is created; a view
            // cost more than the rest of the access).
            if (IsFullIntegerIndex(slices) && Storage.IsFixedString)
                return StandIn(ElementAddress(slices), Storage.StringSlice, strip: true);
            // The index errors come first (NumPy's prepare_index runs before the result exists); a bare view storage holds
            // no counted reference, so dropping it on a finalize error leaks nothing.
            UnmanagedStorage view = Storage.GetView(slices);
            if (IsFullIntegerIndex(slices))
                return ScalarOf(view, strip: true);
            Finalize(dtype);
            // A trivial index may hand the storage itself back; the result must be its own object.
            if (ReferenceEquals(view, Storage))
                view = Storage.Alias();
            return new NDCharArray(view);
        }

        /// <summary>
        ///     The address of the element a full integer index names (<see cref="IsFullIntegerIndex(Slice[])"/> must hold):
        ///     each integer is checked against its axis and wrapped from the end when negative, then the element offset is
        ///     the shape's offset plus the sum of index times stride (strides and offset in elements).
        /// </summary>
        /// <param name="slices">One integer slice per axis.</param>
        /// <returns>The element's first byte in this array's buffer.</returns>
        /// <exception cref="IndexError">An integer outside <c>[-size, size)</c> of its axis: NumPy's
        ///     <c>index 5 is out of bounds for axis 0 with size 3</c>, the integer as given.</exception>
        private unsafe byte* ElementAddress(Slice[] slices)
        {
            ref readonly Shape shape = ref Storage.ShapeReference;
            long offset = shape.offset;
            for (int axis = 0; axis < slices.Length; axis++)
                offset += Wrapped(slices[axis].Start ?? 0, shape.dimensions[axis], axis) * shape.strides[axis];

            var slice = Storage.StringSlice;
            return slice.Address + offset * (long)slice.ItemSize;
        }

        /// <summary>
        ///     <c>c[index]</c> with one integer: for a 1-d S / U chararray the stripped stand-in of the element, its address
        ///     straight from the stride; for any other chararray the general basic index (a row view, the 0-d error, an int8
        ///     element).
        /// </summary>
        /// <param name="index">The index along the first axis; negative counts from the end.</param>
        /// <returns>The stand-in or the row view.</returns>
        /// <exception cref="IndexError"><paramref name="index"/> is out of bounds, or the chararray is 0-d.</exception>
        /// <exception cref="ValueError">The chararray holds data that is not string data and the result is a row.</exception>
        /// <exception cref="SystemError">A <c>U</c> element holds a unit above <c>U+10FFFF</c>.</exception>
        [NDAnchored] // the element's raw address is read after this array's last use: keep it rooted until the copy
        private unsafe NDCharArray ElementOrRow(long index)
        {
            if (ndim != 1 || !Storage.IsFixedString)
                return GetBasic(new[] { Slice.Index(index) });
            ref readonly Shape shape = ref Storage.ShapeReference;
            long offset = shape.offset + Wrapped(index, shape.dimensions[0], 0) * shape.strides[0];
            var slice = Storage.StringSlice;
            return StandIn(slice.Address + offset * (long)slice.ItemSize, slice, strip: true);
        }

        /// <summary>
        ///     One integer of a full integer index, checked against its axis and wrapped from the end when negative — the
        ///     check and text of the view route (<c>UnmanagedStorage.GetView</c>), so the element and view routes fail alike.
        /// </summary>
        /// <param name="index">The integer as given.</param>
        /// <param name="dim">The axis length.</param>
        /// <param name="axis">The axis number, for the message.</param>
        /// <returns>The index in <c>[0, dim)</c>.</returns>
        /// <exception cref="IndexError"><paramref name="index"/> is outside <c>[-dim, dim)</c>: NumPy's
        ///     <c>index 5 is out of bounds for axis 0 with size 3</c>, the integer as given.</exception>
        private static long Wrapped(long index, long dim, int axis)
        {
            if (index < -dim || index >= dim)
                throw new IndexError($"index {index} is out of bounds for axis {axis} with size {dim}");
            return index < 0 ? index + dim : index;
        }

        /// <summary>
        ///     Whether an index of slice entries names ONE element: one integer per axis, no ellipsis, new axis or slice
        ///     (mapping.c's full integer index, <c>index_type == HAS_INTEGER</c>). The empty index of a 0-d chararray is one.
        /// </summary>
        /// <param name="slices">The parsed index.</param>
        /// <returns>True when NumPy's <c>ndarray.__getitem__</c> returns a scalar.</returns>
        private bool IsFullIntegerIndex(Slice[] slices)
        {
            if (slices.Length != ndim)
                return false;
            foreach (var slice in slices)
                if (slice is null || !slice.IsIndex || slice.IsEllipsis || slice.IsNewAxis)
                    return false;
            return true;
        }

        /// <summary>
        ///     Whether a general index names ONE element: one entry per axis, each a C# integer, an integer
        ///     <see cref="Slice"/> or a 0-d integer ARRAY (NumPy's "a full integer index sees array scalars as part of
        ///     itself"). A bool, a slice, a string, a sequence or an array with an axis makes it an array index.
        /// </summary>
        /// <param name="indices">The index entries. A single C# tuple entry is NumPy's index tuple.</param>
        /// <returns>True when NumPy's <c>ndarray.__getitem__</c> returns a scalar.</returns>
        private bool IsFullIntegerIndex(object[] indices)
        {
            if (indices is null)
                return false;
            if (indices.Length == 1 && indices[0] is ITuple tuple)
                indices = PolySequence.Items(tuple);
            if (indices.Length != ndim)
                return false;
            foreach (object entry in indices)
            {
                switch (entry)
                {
                    case sbyte or byte or short or ushort or int or uint or long or ulong:
                        break;
                    case Slice { IsIndex: true }:
                        break;
                    case NDArray { ndim: 0 } zeroD when IsIntegerArray(zeroD):
                        break;
                    default:
                        return false;
                }
            }

            return true;
        }

        /// <summary><see cref="IsFullIntegerIndex(object[])"/> for index ARRAYS: one 0-d integer array per axis.</summary>
        /// <param name="selection">The index arrays.</param>
        /// <returns>True when every array is 0-d and there is one per axis.</returns>
        private bool IsFullIntegerIndex(NDArray<int>[] selection)
        {
            if (selection is null || selection.Length != ndim)
                return false;
            foreach (var array in selection)
                if (array is null || array.ndim != 0)
                    return false;
            return true;
        }

        /// <summary>Whether a 0-d array is an INTEGER scalar to NumPy's indexing (any integer dtype; NumSharp's char too).</summary>
        /// <param name="a">The 0-d array.</param>
        /// <returns>True for an integer dtype; false for bool (a mask), floats and text (refused by the indexing).</returns>
        private static bool IsIntegerArray(NDArray a)
            => a.typecode is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32
                or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char;

        /// <summary>
        ///     Whether a general index consists of integers, slices and slice strings only, and those entries as slices. Such
        ///     an index selects a view (or one element); any other entry makes it an advanced index.
        /// </summary>
        /// <param name="index">The index entries. A single C# tuple entry is NumPy's index tuple and is spread.</param>
        /// <param name="slices">The entries as slices when the result is <c>true</c>.</param>
        /// <returns><c>true</c> for an index of integers and slices.</returns>
        private static bool TryAsSlices(object[] index, out Slice[] slices)
        {
            slices = null;
            if (index.Length == 1 && index[0] is ITuple tuple)
                index = PolySequence.Items(tuple);

            var list = new List<Slice>(index.Length);
            foreach (object entry in index)
            {
                switch (entry)
                {
                    case Slice slice:
                        list.Add(slice);
                        break;
                    case string notation:
                        list.AddRange(Slice.ParseSlices(notation));
                        break;
                    case int i:
                        list.Add(Slice.Index(i));
                        break;
                    case long l:
                        list.Add(Slice.Index(l));
                        break;
                    case sbyte or byte or short or ushort or uint:
                        list.Add(Slice.Index(Convert.ToInt64(entry)));
                        break;
                    default:
                        return false;
                }
            }

            slices = list.ToArray();
            return true;
        }

        /// <summary>
        ///     Whether a general index is NumPy's <c>HAS_SCALAR_ARRAY</c> case: at least one 0-d integer array among entries
        ///     that are otherwise integers and slices (and not one per axis, the full integer index). NumPy treats the 0-d
        ///     array as an integer, selects the view, and returns a COPY of it.
        /// </summary>
        /// <param name="index">The index entries. A single C# tuple entry is NumPy's index tuple.</param>
        /// <returns><c>true</c> when the result is a copy the chararray owns.</returns>
        private static bool IsScalarArrayIndex(object[] index)
        {
            if (index is null)
                return false;
            if (index.Length == 1 && index[0] is ITuple tuple)
                index = PolySequence.Items(tuple);

            bool scalarArray = false;
            foreach (object entry in index)
            {
                switch (entry)
                {
                    case NDArray { ndim: 0 } zeroD when IsIntegerArray(zeroD):
                        scalarArray = true;
                        break;
                    case Slice or string or sbyte or byte or short or ushort or int or uint or long or ulong:
                        break;
                    default:
                        return false;
                }
            }

            return scalarArray;
        }

        #endregion
    }
}
