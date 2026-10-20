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
// THE C# TYPE. An indexer has one type for get and set. The one indexer (NumPy's one __getitem__) is typed NDCharArray so
// that c["1:3"][0] and c2[0][1] strip like NumPy's chained indexing, therefore the value assigned through it is an
// NDCharArray or a C# string / byte[] (implicit conversions). Assigning stores through the plain array rule (NumPy's
// chararray does not override __setitem__). Assign any other value (a number, a plain NDArray) with __setitem__.
//
// STATIC, NOT VIRTUAL. The indexer hides NDArray's (virtual) indexer instead of overriding it: the chararray rule applies
// through an NDCharArray reference, and library code that receives a chararray as an NDArray indexes it as a plain array.
//
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        #region Indexer

        /// <summary>
        ///     NumPy's <c>chararray.__getitem__(obj)</c> as the one C# indexer: every index NumPy takes (integers, slices,
        ///     slice notation, index arrays, boolean masks, sequences, in any mix), with chararray's rule for the result — an
        ///     element is stripped of trailing whitespace, an array stays a chararray. A boolean array from a comparison is an
        ///     index too: <c>c[c != ""]</c>.
        /// </summary>
        /// <param name="key">The index tuple (one entry per C# argument; a <see cref="Slice"/>[] or <c>object[]</c> passed
        ///     alone is the tuple itself, and a single C# tuple entry is spread). A slice-notation string may hold several
        ///     comma-separated entries (<c>c[":, ::-1"]</c>, <c>c["0, 1"]</c>); the empty string is NumPy's empty index
        ///     <c>c[()]</c>.</param>
        /// <value>
        ///     GET: the element as a read-only 0-d stand-in, its trailing whitespace stripped (NumPy's Python <c>str</c> /
        ///     <c>bytes</c>), for a full integer index (one integer or 0-d integer array per axis; for a 0-d chararray the
        ///     empty index); otherwise a chararray VIEW for an index of integers and slices, a chararray view of the gathered
        ///     array (OWNDATA false) for an index array or a mask, and an owning chararray copy for NumPy's
        ///     0-d-integer-array-plus-slices case.
        ///     SET: stores through the plain array rule (NumPy's chararray does not override <c>__setitem__</c>): truncated to
        ///     the element width, <c>S</c> encodes ASCII.
        /// </value>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> is <c>null</c>, or an entry is <c>null</c>.</exception>
        /// <exception cref="IndexError">An index is out of bounds, too many entries, a mask that does not match, or index
        ///     arrays that do not broadcast together (NumPy's texts).</exception>
        /// <exception cref="ValueError">(get) The chararray holds data that is not string data and the index selects an
        ///     array; (set) the chararray is read-only, or the value does not fit the selection.</exception>
        /// <exception cref="SystemError">(get) A <c>U</c> element holds a unit above <c>U+10FFFF</c>.</exception>
        /// <remarks>
        ///     <para>
        ///     <b>The C# type.</b> The indexer has one type for get and set. It is typed <see cref="NDCharArray"/> so that
        ///     <c>c["1:3"][0]</c> and <c>c2[0][1]</c> strip like NumPy's chained indexing, therefore the value assigned through
        ///     it is an <see cref="NDCharArray"/> or a C# <see cref="string"/> / <c>byte[]</c> (implicit conversions). Assign
        ///     any other value (a number, a plain <see cref="NDArray"/>) with <see cref="NDArray.__setitem__(object, object)"/>.
        ///     </para>
        ///     <para>
        ///     <b>Static, not virtual.</b> This indexer HIDES <see cref="NDArray.this[object[]]"/> instead of overriding it:
        ///     chararray's rule applies through an <see cref="NDCharArray"/> reference, and through an <see cref="NDArray"/>
        ///     reference a chararray is indexed as a plain array (the library's own code receives chararrays as
        ///     <see cref="NDArray"/> and relies on that, see <see cref="NDCharArrayBoundary"/>).
        ///     </para>
        ///     <para>
        ///     <b>Dispatch.</b> One integer (a C# integer of any type C# converts to <see cref="long"/>) reads the element
        ///     straight from the strides for a 1-d S / U chararray (<see cref="ElementOrRow"/>, the body of the old
        ///     <c>this[int]</c> / <c>this[long]</c>); an index of integers, slices and slice strings selects its view without
        ///     building an intermediate array (<see cref="GetBasic"/>); any other index goes through the plain array's advanced
        ///     indexing first, whose result the scope reclaims once the chararray holds its own reference.
        ///     </para>
        /// </remarks>
        public new NDCharArray this[params object[] key]
        {
            [NDScoped]
            get
            {
                if (key is null)
                    throw new ArgumentNullException(nameof(key));
                if (key.Length == 1 && TryLoneInteger(key[0], out long index))
                    return ElementOrRow(index);
                if (TryAsSlices(key, out Slice[] slices))
                    return GetBasic(slices);
                NDArray result = base[key];
                return IsFullIntegerIndex(key)
                    ? ScalarOf(result.Storage, strip: true)
                    : FromIndexing(result, this, IsScalarArrayIndex(key));
            }
            // The plain array rule for every key: NDArray's setter sorts the key the way the typed setters this one
            // replaced did (one integer and slice notation as basic views, a lone mask as the mask path).
            set => base[key] = value;
        }

        #endregion

        #region Python's __getitem__

        /// <summary>NumPy's <c>chararray.__getitem__(obj)</c>: the same as <c>c[obj]</c> (see <see cref="this[object[]]"/>).</summary>
        /// <param name="obj">The index as Python passes it: one entry (an integer, slice notation, a <see cref="Slice"/>,
        ///     an index array, a mask) or the whole index tuple — an <c>object[]</c> / <see cref="Slice"/>[] or a C# tuple. A
        ///     typed C# array other than <see cref="Slice"/>[] (<c>int[]</c>, <c>bool[]</c>) is ONE entry, NumPy's list.</param>
        /// <returns>The stripped 0-d stand-in for a full integer index, a chararray (view, gathered view or copy)
        ///     otherwise.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="obj"/> is <c>null</c>.</exception>
        /// <exception cref="IndexError">An index is out of bounds, more entries than axes, or an invalid index.</exception>
        /// <exception cref="ValueError">The chararray holds data that is not string data and the result is an array.</exception>
        /// <exception cref="SystemError">A <c>U</c> element holds a unit above <c>U+10FFFF</c>.</exception>
        /// <remarks>One member, NumPy's signature (it replaced the <c>int</c>, <c>long</c>, <c>string</c> and
        ///     <c>params int[]</c> forms; coordinates are a tuple: <c>c.__getitem__((1, 2))</c>).</remarks>
        public new NDCharArray __getitem__(object obj) => this[ItemKey(obj)];

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

        /// <summary>
        ///     Whether one index entry is a C# integer that the old <c>this[int]</c> / <c>this[long]</c> indexers took: an
        ///     integer of a type C# converts implicitly to <see cref="int"/> or <see cref="long"/> (a <see cref="char"/>
        ///     included, as C#'s <c>char</c> → <c>int</c> conversion bound it there).
        /// </summary>
        /// <param name="entry">The index entry.</param>
        /// <param name="index">The integer when the result is true.</param>
        /// <returns>True for such an integer; false for anything else (a <see cref="ulong"/> has no implicit conversion to
        ///     <see cref="long"/>, so it keeps the general index's handling).</returns>
        private static bool TryLoneInteger(object entry, out long index)
        {
            switch (entry)
            {
                case int i32:
                    index = i32;
                    return true;
                case long i64:
                    index = i64;
                    return true;
                case nint native:
                    index = native;
                    return true;
                case short or ushort or sbyte or byte or uint or char:
                    // Each widens to long without loss (a char to its UTF-16 code unit, as C# did).
                    index = Convert.ToInt64(entry, System.Globalization.CultureInfo.InvariantCulture);
                    return true;
                default:
                    index = 0;
                    return false;
            }
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
