using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Printing;
using NumSharp.Backends.Strings;

// =============================================================================
// NDListArray.Assign.cs — the members of NumPy's object array of lists that WRITE: element and sub-array assignment
// (the indexer setters), fill, put, take / compress with out=, setfield / getfield, resize, setflags, byteswap, choose,
// tobytes and tofile
// =============================================================================
//
// An object array holds POINTERS, so assignment copies pointers: `a[1:3] = b` makes a's elements point at b's lists
// (the same list objects, shared from then on). Here every element is an id into a heap, so an assignment copies ids.
// A list that lives in another heap is first appended to this array's heap (heaps are append-only, so the arrays that
// already share this heap are not disturbed), then its new id is stored; a list from this array's own heap is stored
// as it is, and the assignment is NumPy's array assignment of the int64 ids (np.copyto: its broadcasting, its overlap
// rule, its error texts).
//
// What a VALUE means follows NumPy's setitem (array_assign_subscript, probed on 2.4.2):
//   * One integer per axis (HAS_INTEGER): the value is stored AS IT IS. A 0-d NDListArray stands for the Python list
//     (the house rule: a 0-d array is NumPy's scalar — r["1"], r.max()), so its list is stored. An NDListArray of one or
//     more dimensions would be stored as an ndarray OBJECT, which a list array cannot hold (NotSupportedException).
//   * A sub-array (a view, a 0-d array's `...`): an NDListArray value of one or more dimensions is assigned element by
//     element, broadcasting. A LIST value is first converted as a sequence (PyArray_CopyObject discovers it up to the
//     destination's rank), so it becomes the 1-D array of its items: into a 0-d destination the list itself is stored,
//     into a 1-D slice of matching length its STRINGS are (NotSupportedException — a list array holds lists), and a
//     length that does not broadcast is NumPy's "could not broadcast input array from shape (3,) into shape (2,)".
//   * Every assignment checks the destination's writeability FIRST, before the index is read ("assignment destination
//     is read-only"), as NumPy's setitem does.
// =============================================================================

namespace NumSharp
{
    public sealed partial class NDListArray
    {
        // =============================================================================================================
        // Element and sub-array assignment
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>a[index] = value</c> for an object array of lists (see the file header for what the value means).
        /// </summary>
        /// <param name="slices">The parsed index.</param>
        /// <param name="value">The value: a 0-d list array (one list) or a list array of one or more dimensions.</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null (NumPy would store None).</exception>
        /// <exception cref="ValueError">A read-only array; a value that does not broadcast into the destination.</exception>
        /// <exception cref="IndexError">An index outside the array.</exception>
        /// <exception cref="NotSupportedException">NumPy's result holds an element that is not a list: an array stored into
        ///     one element, a list's items stored into a sub-array, or a list of the other piece class.</exception>
        /// <exception cref="ObjectDisposedException">This array or <paramref name="value"/> was disposed.</exception>
        private void SetItem(Slice[] slices, NDListArray value)
        {
            ThrowIfDisposed();
            if (value is null)
                throw new ArgumentNullException(nameof(value), "None stored into an object array of lists has no NDListArray form");
            value.ThrowIfDisposed();
            // array_assign_subscript: the writeability before the index is even prepared.
            NumSharpException.ThrowIfNotWriteable(Ids.Shape, "assignment destination");
            slices ??= Array.Empty<Slice>();
            bool element = slices.Length == ndim;
            foreach (var s in slices)
                element &= s is not null && s.IsIndex;

            using var target = Ids[slices];
            if (element)
            {
                // HAS_INTEGER: PyArray_Pack of the value itself into the element — a list is stored, an array would be.
                if (!value.IsStandIn)
                    throw new NotSupportedException(
                        "NumPy stores the array object itself in the element (an ndarray inside an object array), which an " +
                        "NDListArray cannot hold; assign a list (lists[\"1\"], one element of a list array) to store a list");
                target.SetAtIndex(MapOne(value.Heap, value.FlatId(0)), 0);
                return;
            }
            if (value.IsStandIn)
            {
                AssignListAsSequence(target, value);
                return;
            }
            // An array (a 0-d one too) is assigned element by element, broadcasting.
            AssignFrom(target, value);
        }

        /// <summary>
        ///     NumPy's <c>a[key] = value</c> for an object array of lists, for ANY index (the indexer's setter): an index of
        ///     integers and slices takes the basic assignment (<see cref="SetItem(Slice[], NDListArray)"/>); an advanced index
        ///     (index arrays, masks, a C# bool, a mix) copies the value's list ids into the selection with NumPy's fancy
        ///     assignment of the int64 ids (its broadcasting and error texts).
        /// </summary>
        /// <param name="key">The index tuple.</param>
        /// <param name="value">The value: a 0-d list array (one list) or a list array of one or more dimensions.</param>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> or <paramref name="value"/> is null.</exception>
        /// <exception cref="ValueError">A read-only array (checked before the index); a value that does not broadcast to the
        ///     selection (NumPy's <c>shape mismatch: value array of shape (3,) could not be broadcast to indexing result of
        ///     shape (2,)</c>, or the boolean-mask texts).</exception>
        /// <exception cref="IndexError">An index outside the array, or an invalid index.</exception>
        /// <exception cref="NotSupportedException">NumPy's result holds an element that is not a list: an array stored into
        ///     one element (a full integer index of 0-d index arrays), a list's items stored into the selection, or a list of
        ///     the other piece class.</exception>
        /// <exception cref="ObjectDisposedException">This array or <paramref name="value"/> was disposed.</exception>
        private void SetItem(object[] key, NDListArray value)
        {
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            if (NDArray.TryBasicIndex(key, out Slice[] slices))
            {
                SetItem(slices, value);
                return;
            }

            ThrowIfDisposed();
            if (value is null)
                throw new ArgumentNullException(nameof(value), "None stored into an object array of lists has no NDListArray form");
            value.ThrowIfDisposed();
            // array_assign_subscript: the writeability before the index is even prepared.
            NumSharpException.ThrowIfNotWriteable(Ids.Shape, "assignment destination");

            if (IsFullIntegerIndex(key))
            {
                // HAS_INTEGER with 0-d index arrays: PyArray_Pack of the value itself into the element, as for the basic
                // full integer index — a list is stored, an array object would be.
                if (!value.IsStandIn)
                    throw new NotSupportedException(
                        "NumPy stores the array object itself in the element (an ndarray inside an object array), which an " +
                        "NDListArray cannot hold; assign a list (lists[\"1\"], one element of a list array) to store a list");
                using var one = NDArray.Scalar(MapOne(value.Heap, value.FlatId(0)));
                Ids[key] = one;
                return;
            }

            if (value.IsStandIn)
            {
                AssignListThroughIndex(key, value);
                return;
            }

            // An array of lists: its ids (in this heap) assigned through the index, as NumPy copies the pointers.
            var ids = IdsInThisHeap(value, out bool owned);
            try
            {
                Ids[key] = ids;
            }
            finally
            {
                if (owned)
                    ids.Dispose();
            }
        }

        /// <summary>
        ///     A Python list assigned through an ADVANCED index (probed on NumPy 2.4.2): NumPy reads it as the SEQUENCE of its
        ///     items, a 1-D object array of the list's str / bytes items, so it is checked like that array and then refused
        ///     wherever it would store items, since a list array holds lists.
        ///     <list type="bullet">
        ///       <item><description>A lone boolean mask of this array's own shape is <c>array_assign_boolean_subscript</c>: the
        ///       items must number 1 or one per true element (NumPy's <c>NumPy boolean array indexing assignment cannot
        ///       assign 3 input values to the 2 output values where the mask is true</c>).</description></item>
        ///       <item><description>Any other advanced index converts the list against the selection, and a length that does
        ///       not broadcast is <c>could not broadcast input array from shape (3,) into shape (2,)</c> — not the MapIter
        ///       <c>shape mismatch</c> text an array value gets.</description></item>
        ///     </list>
        ///     A selection of no element stores nothing, so a list that fits it is a no-op.
        /// </summary>
        /// <param name="key">The advanced index tuple.</param>
        /// <param name="value">The 0-d list array standing for the list.</param>
        /// <exception cref="ValueError">The list's items do not fit the selection (NumPy's texts, above).</exception>
        /// <exception cref="IndexError">The index is invalid for this array.</exception>
        /// <exception cref="NotSupportedException">The items would be stored into a non-empty selection.</exception>
        private void AssignListThroughIndex(object[] key, NDListArray value)
        {
            long length = value.Heap.Length(value.FlatId(0));
            long selected;
            if (key.Length == 1 && key[0] is NDArray { typecode: NPTypeCode.Boolean } mask && HasShapeOf(mask, Ids))
            {
                selected = np.count_nonzero(mask);
                if (length != 1 && length != selected)
                    throw new ValueError(
                        $"NumPy boolean array indexing assignment cannot assign {length} input values to the {selected} output values where the mask is true");
            }
            else
            {
                // The selection's shape (gathering the ids also raises NumPy's index errors), then the 1-D item sequence's
                // broadcast against it: its one axis must equal the selection's last axis or be 1.
                using var selection = Ids[key];
                long[] dims = selection.Shape.dimensions;
                long last = dims.Length == 0 ? 1 : dims[dims.Length - 1];
                if (length != last && length != 1)
                    throw new ValueError($"could not broadcast input array from shape ({length},) into shape {selection.Shape.ToPythonTuple()}");
                selected = selection.size;
            }
            if (selected > 0)
                throw new NotSupportedException(
                    "NumPy assigns a list through an index array as the sequence of its items, so the elements would become the " +
                    $"{(value.Kind == 'S' ? "bytes" : "str")} items themselves, which an NDListArray (a list array) cannot hold");
        }

        /// <summary>Whether two arrays have the same dimensions (NumPy's single-mask assignment test).</summary>
        /// <param name="a">The mask.</param>
        /// <param name="b">The ids.</param>
        /// <returns>True when the ranks and every extent agree.</returns>
        private static bool HasShapeOf(NDArray a, NDArray b)
            => a.Shape.dimensions.AsSpan().SequenceEqual(b.Shape.dimensions);

        /// <summary>
        ///     A Python list assigned to a sub-array: PyArray_CopyObject discovers it as a sequence up to the destination's
        ///     rank, so a 0-d destination stores the list and any other is given the 1-D array of the list's items.
        /// </summary>
        /// <param name="target">The destination ids (a view of this array's ids).</param>
        /// <param name="value">The 0-d list array standing for the list.</param>
        /// <exception cref="ValueError"><c>could not broadcast input array from shape (3,) into shape (2,)</c>.</exception>
        /// <exception cref="NotSupportedException">The items (strings) would be stored into a non-empty destination.</exception>
        private void AssignListAsSequence(NDArray target, NDListArray value)
        {
            if (target.ndim == 0)
            {
                target.SetAtIndex(MapOne(value.Heap, value.FlatId(0)), 0);
                return;
            }
            long length = value.Heap.Length(value.FlatId(0));
            // NumPy's assignment broadcast of the (length,) item array: leading extent-1 axes of the source are stripped
            // (there are none for one axis), then each source axis must equal the destination's or be 1.
            var dims = target.Shape.dimensions;
            long last = dims[dims.Length - 1];
            if (length != last && length != 1)
                throw new ValueError($"could not broadcast input array from shape ({length},) into shape {target.Shape.ToPythonTuple()}");
            if (target.size > 0)
                throw new NotSupportedException(
                    "NumPy assigns a list to a sub-array as the sequence of its items, so the elements would become the " +
                    $"{(value.Kind == 'S' ? "bytes" : "str")} items themselves, which an NDListArray (a list array) cannot hold");
        }

        /// <summary>
        ///     Stores a list array's lists into <paramref name="target"/> (a view of this array's ids): the lists are put into
        ///     this array's heap (when they live elsewhere), then the ids are copied with np.copyto's broadcasting and overlap
        ///     rules.
        /// </summary>
        /// <param name="target">The destination ids.</param>
        /// <param name="value">The source lists.</param>
        /// <exception cref="ValueError">The source does not broadcast into the destination (np.copyto's text).</exception>
        /// <exception cref="NotSupportedException">A non-empty list of the other piece class (str and bytes).</exception>
        private void AssignFrom(NDArray target, NDListArray value)
        {
            var ids = IdsInThisHeap(value, out bool owned);
            try
            {
                np.copyto(target, ids);
            }
            finally
            {
                if (owned)
                    ids.Dispose();
            }
        }

        /// <summary>
        ///     Stores a list array's lists into <paramref name="target"/> (a view of this array's ids, of the same shape) only
        ///     where <paramref name="mask"/> is true — a masked ufunc's output: the other positions keep their lists, and only
        ///     the selected lists are put into this array's heap.
        /// </summary>
        /// <param name="target">The destination ids.</param>
        /// <param name="value">The computed lists (their ids at masked-off positions are never read).</param>
        /// <param name="mask">A bool mask of the target's shape, or null for every position (a ufunc call without where=, or
        ///     with Python's <c>where=True</c>).</param>
        /// <exception cref="NotSupportedException">A selected non-empty list of the other piece class.</exception>
        private unsafe void AssignWhere(NDArray target, NDListArray value, NDArray mask)
        {
            // No mask selects every position: the plain assignment (the walk below reads the mask at every element).
            if (mask is null)
            {
                AssignFrom(target, value);
                return;
            }
            if (ReferenceEquals(value.Heap, Heap))
            {
                np.copyto(target, value.Ids, where: mask);
                return;
            }
            // Map only the selected lists into this heap (a masked-off one is never stored, so it is never copied: its class
            // could even be the other one), then copy those ids under the mask.
            using var mapped = new NDArray(NPTypeCode.Int64, value.Ids.Shape.KeepOrder(), true);
            var memo = new Dictionary<(NDListHeap, long), long>();
            var source = value.Heap;
            var heap = Heap;
            Walk(new[] { value.Ids, mask, mapped }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY },
                (dataptrs, strides, n, aux) =>
                {
                    byte* pi = (byte*)dataptrs[0];
                    byte* pm = (byte*)dataptrs[1];
                    byte* po = (byte*)dataptrs[2];
                    long si = strides[0], sm = strides[1], so = strides[2];
                    for (long k = 0; k < n; k++, pi += si, pm += sm, po += so)
                    {
                        if (*pm != 0)
                            *(long*)po = CopyList(source, *(long*)pi, heap, memo);
                    }
                });
            np.copyto(target, mapped, where: mask);
        }

        /// <summary>
        ///     The ids, in THIS array's heap, of a list array's elements (same shape and layout as its ids): the array's own
        ///     ids when it shares this heap, else a new id array whose lists were appended here (each distinct list once).
        /// </summary>
        /// <param name="value">The list array.</param>
        /// <param name="owned">Receives true when the returned array is new (the caller disposes it).</param>
        /// <returns>The ids.</returns>
        /// <exception cref="NotSupportedException">A non-empty list of the other piece class.</exception>
        private unsafe NDArray IdsInThisHeap(NDListArray value, out bool owned)
        {
            if (ReferenceEquals(value.Heap, Heap))
            {
                owned = false;
                return value.Ids;
            }
            owned = true;
            var mapped = new NDArray(NPTypeCode.Int64, value.Ids.Shape.KeepOrder(), false);
            var memo = new Dictionary<(NDListHeap, long), long>();
            var source = value.Heap;
            var heap = Heap;
            try
            {
                Walk(new[] { value.Ids, mapped }, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY },
                    (dataptrs, strides, n, aux) =>
                    {
                        byte* pi = (byte*)dataptrs[0];
                        byte* po = (byte*)dataptrs[1];
                        long si = strides[0], so = strides[1];
                        for (long k = 0; k < n; k++, pi += si, po += so)
                            *(long*)po = CopyList(source, *(long*)pi, heap, memo);
                    });
            }
            catch
            {
                mapped.Dispose();
                throw;
            }
            return mapped;
        }

        /// <summary>The id, in this array's heap, of one list of <paramref name="source"/> (appended here when needed).</summary>
        /// <param name="source">The list's heap.</param>
        /// <param name="id">The list.</param>
        /// <returns>The id here.</returns>
        /// <exception cref="NotSupportedException">A non-empty list of the other piece class.</exception>
        private long MapOne(NDListHeap source, long id)
            => CopyList(source, id, Heap, new Dictionary<(NDListHeap, long), long>());

        // =============================================================================================================
        // fill / put
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>fill(value)</c>: every element becomes the SAME list (NumPy stores one object in every element).
        /// </summary>
        /// <param name="value">The list: a 0-d <see cref="NDListArray"/> (as <c>lists["1"]</c> or <c>lists.max()</c> give),
        ///     or a C# sequence of <see cref="string"/> pieces (a list of str) or <c>byte[]</c> pieces (a list of bytes).</param>
        /// <exception cref="ValueError">A read-only array (<c>assignment destination is read-only</c>, checked first).</exception>
        /// <exception cref="NotSupportedException">A value that is not one list (NumPy stores any object: an ndarray, a
        ///     number, a str, None), or a list of the other piece class.</exception>
        /// <exception cref="ObjectDisposedException">The array or the value was disposed.</exception>
        public void fill(object value)
        {
            ThrowIfDisposed();
            // PyArray_FillWithScalar: the writeability first, then PyArray_Pack of the object.
            NumSharpException.ThrowIfNotWriteable(Ids.Shape, "assignment destination");
            if (value is NDListArray { IsStandIn: false })
                throw new NotSupportedException(
                    "NumPy's fill stores the array object itself in every element, which an NDListArray cannot hold; fill with a " +
                    "list (lists[\"1\"], one element of a list array) or a C# sequence of pieces");
            if (!TryPythonList(value, Kind, out var heap, out long id))
                throw new NotSupportedException(
                    $"NumPy's fill stores the Python {NDPyObject.TypeName(value)} itself in every element, which an NDListArray " +
                    "(a list array) cannot hold");
            try
            {
                long mine = MapOne(heap, id);
                Ids.fill(mine);
            }
            finally
            {
                heap.Release();
            }
        }

        /// <summary>
        ///     NumPy's <c>put(indices, values, mode='raise')</c>: <c>a.flat[indices[i]] = values.flat[i % len(values)]</c> —
        ///     the lists of <paramref name="values"/>, cycled, stored at the flat positions (NumPy's PyArray_PutTo: a
        ///     non-contiguous array written back through its own layout).
        /// </summary>
        /// <param name="indices">The flat indices: an integer array (NumPy casts an ndarray of indices safely), or one C#
        ///     integer (<c>lists.put(1, values)</c>), which converts to the 0-d array NumPy's <c>np.asarray(1)</c> is.</param>
        /// <param name="values">The lists (a list array of any shape, read in C order). A 0-d list array stands for one
        ///     Python list, which NumPy converts to the array of its ITEMS (NotSupportedException unless it is empty, or no
        ///     index is given).</param>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c> or <c>"clip"</c> (null is <c>"raise"</c>).</param>
        /// <exception cref="ValueError">A bad mode; a read-only array (<c>put: output array is read-only</c>).</exception>
        /// <exception cref="TypeError">Indices that do not cast to int64 under <c>safe</c>.</exception>
        /// <exception cref="IndexError">An index out of bounds under <c>"raise"</c>; indices into an empty array
        ///     (<c>cannot replace elements of an empty array</c>).</exception>
        /// <exception cref="NotSupportedException">A list as <paramref name="values"/> whose items would be stored, or a list
        ///     of the other piece class.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="values"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">The array or the values were disposed.</exception>
        /// <remarks>One overload, NumPy's signature: NumPy's <c>indices</c> is an int or an int array_like, which this
        ///     <see cref="NDArray"/> parameter receives through NumSharp's implicit conversions.</remarks>
        public void put(NDArray indices, NDListArray values, string mode = "raise")
        {
            ThrowIfDisposed();
            if (values is null)
                throw new ArgumentNullException(nameof(values));
            values.ThrowIfDisposed();
            if (values.IsStandIn)
            {
                // np.asarray(list, dtype=object) is the 1-D array of the list's items. NumPy's own checks run first (the mode,
                // the writeability, the indices, an empty array); an empty item array then puts nothing, as does an empty
                // index array.
                using var none = new NDArray(NPTypeCode.Int64, Shape.Vector(0), false);
                Ids.put(indices, none, mode);
                if (values.Heap.Length(values.FlatId(0)) == 0 || (indices is not null && indices.size == 0))
                    return;
                throw new NotSupportedException(
                    "NumPy's put converts a Python list value to the array of its items, so the elements would become the " +
                    $"{(values.Kind == 'S' ? "bytes" : "str")} items themselves, which an NDListArray cannot hold");
            }
            var ids = IdsInThisHeap(values, out bool owned);
            try
            {
                Ids.put(indices, ids, mode);
            }
            finally
            {
                if (owned)
                    ids.Dispose();
            }
        }

        // =============================================================================================================
        // take / compress with out=
        // =============================================================================================================

        /// <summary>
        ///     <c>take</c>'s <c>out=</c>: the taken ids written into <paramref name="out"/> with NumPy's checks in NumPy's order
        ///     (the shape, then the writeability — <c>WRITEBACKIFCOPY base is read-only</c> —, then the indices).
        /// </summary>
        /// <param name="out">The destination list array.</param>
        /// <param name="taking">The take into an id array: (destination ids) → (the destination).</param>
        /// <returns><paramref name="out"/>.</returns>
        private NDListArray TakeInto(NDListArray @out, Func<NDArray, NDArray> taking)
        {
            ThrowIfDisposed();
            @out.ThrowIfDisposed();
            // The same heap (or a read-only destination, which refuses before anything is written): the take writes the
            // destination's ids directly, so its checks and its overlap copy are NumPy's.
            if (ReferenceEquals(@out.Heap, Heap) || !@out.Ids.Shape.IsWriteable)
            {
#pragma warning disable NDW012 // not a leak: np.take(out=) returns the destination itself, which @out owns
                taking(@out.Ids);
#pragma warning restore NDW012
                return @out;
            }
            // Another heap: the ids of THIS heap are taken into a scratch of the destination's shape (the same checks), then
            // their lists are put into the destination's heap and stored.
            using var scratch = new NDArray(NPTypeCode.Int64, @out.Ids.Shape.KeepOrder(), false);
#pragma warning disable NDW012 // not a leak: np.take(out=) returns the destination itself, the scratch disposed below
            taking(scratch);
#pragma warning restore NDW012
            using var taken = new NDListArray(scratch.view(), Heap.Share());
            @out.AssignFrom(@out.Ids, taken);
            return @out;
        }

        // =============================================================================================================
        // getfield / setfield
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>getfield(dtype, offset=0)</c> of an object array: only the object dtype at offset 0 is a field of
        ///     it, and NumSharp has no object dtype, so every dtype it can spell is <c>_getfield_is_safe</c>'s TypeError.
        /// </summary>
        /// <param name="dtype">The field's dtype.</param>
        /// <param name="offset">The field's byte offset.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="TypeError"><c>Cannot get/set field of an object array</c>.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray getfield(DType dtype, int offset = 0)
        {
            ThrowIfDisposed();
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            throw new TypeError("Cannot get/set field of an object array");
        }

        /// <summary>NumPy's <c>flat</c>, as NumSharp spells it for an <see cref="NDArray"/>: the elements as a 1-D list array
        ///     in C order (a view when the layout allows, else a copy of the pointers; the lists are shared).</summary>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray flat => ravel();

        /// <summary>
        ///     NumPy's <c>setfield(val, dtype, offset=0)</c> of an object array: the writeability first (<c>assignment
        ///     destination is read-only</c>), then <see cref="getfield"/>'s TypeError for every dtype NumSharp can spell.
        /// </summary>
        /// <param name="value">The value (never stored).</param>
        /// <param name="dtype">The field's dtype.</param>
        /// <param name="offset">The field's byte offset.</param>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="ValueError">A read-only array.</exception>
        /// <exception cref="TypeError"><c>Cannot get/set field of an object array</c>.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public void setfield(object value, DType dtype, int offset = 0)
        {
            ThrowIfDisposed();
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            NumSharpException.ThrowIfNotWriteable(Ids.Shape, "assignment destination");
            // PyArray_SetField's getfield: _getfield_is_safe refuses every dtype NumSharp can spell.
            throw new TypeError("Cannot get/set field of an object array");
        }

        // =============================================================================================================
        // resize / setflags / byteswap / choose
        // =============================================================================================================

        /// <summary>
        ///     NumPy's in-place <c>resize(new_shape, refcheck=True)</c>: the array keeps its first lists (in its memory order:
        ///     an F-contiguous array keeps its column-major prefix) and is relabelled with the new shape. NumPy fills grown
        ///     elements with the int 0, which a list array cannot hold, so a GROWING resize is refused — after NumPy's own
        ///     checks, in its order: one segment, no negative dimension, ownership, other references.
        /// </summary>
        /// <param name="new_shape">The new shape (a 0-d shape for a scalar).</param>
        /// <param name="refcheck">Refuse when another array shares the elements (default true).</param>
        /// <exception cref="IncorrectShapeException">NumPy's ValueErrors (house type): <c>resize only works on single-segment
        ///     arrays</c>, <c>negative dimensions not allowed</c>, <c>cannot resize this array: it does not own its data</c>,
        ///     <c>cannot resize an array that references or is referenced ...</c>.</exception>
        /// <exception cref="OutOfMemoryException">A size whose product overflows (NumPy's MemoryError).</exception>
        /// <exception cref="NotSupportedException">A growing resize (NumPy's new elements are ints).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>
        ///     NumPy's <c>resize(*new_shape, refcheck=True)</c> takes the shape as one tuple OR as separate integers, and the
        ///     keyword-only <c>refcheck</c> after them. C# has no keyword-only parameter after a <c>params</c> array, so the
        ///     two spellings are two members: this one (a <see cref="Shape"/>, which a C# <c>long[]</c> or tuple converts to,
        ///     with <paramref name="refcheck"/>) and <see cref="resize(long[])"/> (separate integers, NumPy's default
        ///     <c>refcheck</c>).
        /// </remarks>
        public void resize(Shape new_shape, bool refcheck = true)
        {
            ThrowIfDisposed();
            long newSize = ResizeGuards(new_shape, refcheck);
            if (newSize > size)
                throw new NotSupportedException(
                    $"NumPy fills the {newSize - size} new elements of a growing resize with the int 0, which an NDListArray (a list " +
                    "array) cannot hold; np.resize repeats the lists instead");
            Ids.resize(new_shape, refcheck);
        }

        /// <summary>NumPy's <c>resize(*new_shape)</c> with separate integers, NumPy's default <c>refcheck=True</c> (see
        ///     <see cref="resize(Shape, bool)"/>, the spelling that takes <c>refcheck</c>).</summary>
        /// <param name="new_shape">The new dimensions (none: a no-op, NumPy's <c>a.resize()</c>).</param>
        /// <exception cref="IncorrectShapeException">See <see cref="resize(Shape, bool)"/>.</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="resize(Shape, bool)"/>.</exception>
        /// <exception cref="NotSupportedException">A growing resize.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public void resize(params long[] new_shape)
        {
            ThrowIfDisposed();
            if (new_shape is not { Length: > 0 })
                return;
            resize(new Shape(new_shape), refcheck: true);
        }

        /// <summary>
        ///     The checks of NumPy's <c>PyArray_Resize_int</c>, in its order, as <see cref="NDArray.resize(Shape, bool)"/>
        ///     makes them (the same texts and types), without resizing.
        /// </summary>
        /// <param name="new_shape">The requested shape.</param>
        /// <param name="refcheck">The reference check flag.</param>
        /// <returns>The new element count.</returns>
        private long ResizeGuards(Shape new_shape, bool refcheck)
        {
            var shape = Ids.Shape;
            if (shape.NDim != 0 && !shape.IsContiguous && !shape.IsFContiguous)
                throw new IncorrectShapeException("resize only works on single-segment arrays");
            var dims = new_shape.dimensions ?? Array.Empty<long>();
            long newSize = 1;
            for (int i = 0; i < dims.Length; i++)
            {
                if (dims[i] == 0)
                {
                    newSize = 0;
                    break;
                }
                if (dims[i] < 0)
                    throw new IncorrectShapeException("negative dimensions not allowed");
                if (dims[i] > long.MaxValue / newSize)
                    throw new OutOfMemoryException();
                newSize *= dims[i];
            }
            if (newSize > long.MaxValue / 8)
                throw new OutOfMemoryException();
            if (newSize != size)
            {
                if (!Ids.Storage.OwnsData)
                    throw new IncorrectShapeException("cannot resize this array: it does not own its data");
                if (refcheck && !Ids.Storage.InternalArray.IsUniquelyReferenced)
                    throw new IncorrectShapeException(
                        "cannot resize an array that references or is referenced\n" +
                        "by another array in this way.\n" +
                        "Use the np.resize function or refcheck=False");
            }
            return newSize;
        }

        /// <summary>NumPy's <c>setflags(write=None, align=None, uic=None)</c>: the flags of the array of pointers (the id
        ///     array), with NumPy's rules and texts.</summary>
        /// <param name="write">Set or clear WRITEABLE (null: unchanged).</param>
        /// <param name="align">Set or clear ALIGNED (null: unchanged).</param>
        /// <param name="uic">NumPy's WRITEBACKIFCOPY flag (only false is accepted).</param>
        /// <exception cref="ValueError">NumPy's refusals (making a view of a read-only base writeable, …).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public void setflags(bool? write = null, bool? align = null, bool? uic = null)
        {
            ThrowIfDisposed();
            Ids.setflags(write, align, uic);
        }

        /// <summary>
        ///     NumPy's <c>byteswap(inplace=False)</c> of an object array: pointers are not swapped, so it is a copy in
        ///     <c>'A'</c> order (an F-contiguous array stays F), or — in place — the array itself.
        /// </summary>
        /// <param name="inplace">Swap in place (returns this array).</param>
        /// <returns>The copy, or this array.</returns>
        /// <exception cref="ValueError">In place on a read-only array (<c>array to be byte-swapped is read-only</c>).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray byteswap(bool inplace = false)
        {
            ThrowIfDisposed();
            if (!inplace)
                return copy('A');
            if (!Ids.Shape.IsWriteable)
                throw new ValueError("array to be byte-swapped is read-only");
            return this;
        }

        /// <summary>
        ///     NumPy's <c>choose(choices, out=None, mode='raise')</c> with this array as the INDEX array: an object array does
        ///     not cast to intp safely, so after the mode check it is always NumPy's TypeError.
        /// </summary>
        /// <param name="choices">The choice arrays (never read).</param>
        /// <param name="out">The destination (never written).</param>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c> or <c>"clip"</c>.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError">A bad mode.</exception>
        /// <exception cref="TypeError"><c>Cannot cast array data from dtype('O') to dtype('int64') according to the rule
        ///     'safe'</c>.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray choose(object choices, NDArray @out = null, string mode = "raise")
        {
            ThrowIfDisposed();
            np.ClipmodeConverter(mode);
            // PyArray_FromArray's cast text calls a 0-d array a scalar.
            throw new TypeError($"Cannot cast {(ndim == 0 ? "scalar" : "array data")} from dtype('O') to dtype('int64') according to the rule 'safe'");
        }

        // =============================================================================================================
        // tobytes / tofile
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>tobytes()</c> of an object array is the bytes of its <c>PyObject*</c> pointers — addresses in one
        ///     Python process, meaningless as data — so it is not offered.
        /// </summary>
        /// <param name="order">The byte order NumPy would read the elements in.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always.</exception>
        public byte[] tobytes(char order = 'C')
            => throw new NotSupportedException(
                "the bytes of an object array are its PyObject* pointers, which have no meaning outside one Python process; " +
                "an NDListArray has no such bytes");

        /// <summary>
        ///     NumPy's <c>tofile(fid, sep='', format='%s')</c> of an object array: binary mode is NumPy's OSError; text mode
        ///     writes each list as <c>format % (list,)</c> (<c>""</c>: its repr), the separator between elements, in C order,
        ///     as ASCII.
        /// </summary>
        /// <param name="fid">NumPy's <c>fid</c>: a file name — a <see cref="string"/>, a <c>byte[]</c> (Python <c>bytes</c>,
        ///     UTF-8 as Windows' file-system encoding) or a <see cref="FileSystemInfo"/> (Python's <c>os.PathLike</c>) — which
        ///     is created or truncated first, as NumPy opens it with <c>"wb"</c>; or an open <see cref="Stream"/> (NumPy's
        ///     file object), written from its position and left open.</param>
        /// <param name="sep">The separator; <c>""</c> is binary mode. Null is Python's <c>None</c>, which NumPy refuses.</param>
        /// <param name="format">CPython's <c>%</c> format applied to each list (<c>%s</c> / <c>%r</c> its repr, <c>%a</c> its
        ///     ascii(), width and precision as for a str). Null is Python's <c>None</c>, which NumPy refuses.</param>
        /// <exception cref="AttributeError"><paramref name="fid"/> is neither a file name nor a stream: NumPy calls its
        ///     <c>flush()</c> first (<c>'NoneType' object has no attribute 'flush'</c> for null).</exception>
        /// <exception cref="OSError"><c>cannot write object arrays to a file in binary mode</c> (an empty array too), after the
        ///     file is created, as in NumPy.</exception>
        /// <exception cref="ValueError">A NUL in <paramref name="sep"/> or <paramref name="format"/> (checked before the file
        ///     is opened); a malformed format.</exception>
        /// <exception cref="TypeError">A null <paramref name="sep"/> or <paramref name="format"/> (<c>tofile() argument 2 must
        ///     be str, not None</c>, argument 3 for the format; checked before the file is opened), a conversion a list does
        ///     not take (<c>%d format: a real number is required, not list</c>), or the wrong number of conversions.</exception>
        /// <exception cref="UnicodeEncodeError">A list whose text is not ASCII (NumPy 2.4.2 crashes the process there); a
        ///     lone surrogate in <paramref name="sep"/> or <paramref name="format"/>.</exception>
        /// <exception cref="IOException">The file cannot be created (.NET's exception for NumPy's OSError).</exception>
        /// <exception cref="UnauthorizedAccessException">The name is a directory or is not writable (NumPy's
        ///     PermissionError).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>One overload, NumPy's signature: NumPy's <c>fid</c> is "an open file object, or a string containing a
        ///     filename", so a name and a stream are two cases of one parameter.</remarks>
        public void tofile(object fid, string sep = "", string format = "%s")
        {
            ThrowIfDisposed();
            // array_tofile parses sep and format (its arguments 2 and 3) with the "s" converter before it touches the file.
            NDArray.CheckToFileArgument(sep, 2);
            NDArray.CheckToFileArgument(format, 3);
            switch (fid)
            {
                case Stream stream:
                    WriteText(stream, sep, format);
                    return;
                case string or byte[] or FileSystemInfo:
                    // NpyPath_PathlikeToFspath + npy_PyFile_OpenFile(file, "wb"): the file is created (or truncated) before
                    // the binary-mode check, so a binary write leaves an empty file behind, as NumPy's does.
                    string path = fid switch
                    {
                        string s => s,
                        byte[] b => Encoding.UTF8.GetString(b),
                        _ => ((FileSystemInfo)fid).FullName,
                    };
                    using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
                        WriteText(fs, sep, format);
                    return;
                default:
                    // npy_PyFile_Dup2 calls fid.flush() first: any other object has no such method.
                    throw new AttributeError($"'{NDPyObject.TypeName(fid)}' object has no attribute 'flush'");
            }
        }

        /// <summary>The text-mode write of <see cref="tofile"/> into an open stream (its arguments checked by the caller).</summary>
        /// <param name="stream">The stream (written from its position, flushed, left open).</param>
        /// <param name="sep">The separator; <c>""</c> is binary mode.</param>
        /// <param name="format">The <c>%</c> format, or <c>""</c> for each list's repr.</param>
        /// <exception cref="OSError">Binary mode.</exception>
        /// <exception cref="ValueError">A malformed format.</exception>
        /// <exception cref="TypeError">A conversion a list does not take.</exception>
        /// <exception cref="UnicodeEncodeError">A list whose text is not ASCII.</exception>
        private void WriteText(Stream stream, string sep, string format)
        {
            if (string.IsNullOrEmpty(sep))
                throw new OSError("cannot write object arrays to a file in binary mode");
            if (size == 0)
                return;
            // PyArray_ToFile's text mode: an empty format writes repr(obj), any other `format % (obj,)`.
            var program = string.IsNullOrEmpty(format) ? null : PrintfFormatter.Compile(format);
            byte[] sepBytes = Encoding.UTF8.GetBytes(sep);
            var args = new object[1];
            var kinds = new[] { PercentFormat.PyObject };
            var item = new StringBuilder(64);
            using var flat = Ids.flatten();
            try
            {
                for (long i = 0; i < flat.size; i++)
                {
                    long id = flat.GetInt64(i);
                    string text;
                    if (program is null)
                    {
                        text = ListRepr(id);
                    }
                    else
                    {
                        item.Clear();
                        args[0] = PythonListObject(id);
                        program.FormatInto(item, args, kinds);
                        text = item.ToString();
                    }
                    NDStringCodec.ThrowIfNotAscii(text.AsSpan());
                    var bytes = Encoding.ASCII.GetBytes(text);
                    stream.Write(bytes, 0, bytes.Length);
                    if (i != flat.size - 1)
                        stream.Write(sepBytes, 0, sepBytes.Length);
                }
            }
            catch (PrintfArgumentException e)
            {
                // CPython's `format % (obj,)` TypeError, unwrapped as NumPy leaves it (NDArray.tofile's rule).
                throw new TypeError(e.Message, e);
            }
            finally
            {
                stream.Flush();
            }
        }

        /// <summary>
        ///     One list as the Python object CPython's <c>%</c> formats: a .NET list of its pieces (<see cref="string"/>,
        ///     <c>byte[]</c>), an element's own <c>np.str_</c> piece as an object whose repr is <c>np.str_('ab')</c>.
        /// </summary>
        /// <param name="id">The list id.</param>
        /// <returns>The list.</returns>
        private List<object> PythonListObject(long id)
        {
            var pieces = Heap.ToList(id);
            var (start, _) = Heap.Range(id);
            for (int k = 0; k < pieces.Count; k++)
            {
                if (Heap.IsNpStr(start + k))
                    pieces[k] = new NpStrPiece((string)pieces[k]);
            }
            return pieces;
        }

        /// <summary>A list piece that is an element's own <c>np.str_</c>: inside a list's repr it prints as NumPy 2 prints
        ///     the scalar (<c>np.str_('ab')</c>).</summary>
        private sealed class NpStrPiece
        {
            /// <summary>The piece's text.</summary>
            private readonly string _text;

            /// <summary>Wraps a piece.</summary>
            /// <param name="text">The piece's text.</param>
            internal NpStrPiece(string text) => _text = text;

            /// <summary>The repr CPython's list repr calls (NDPyObject's opaque-object rule prints ToString()).</summary>
            /// <returns><c>np.str_('ab')</c>.</returns>
            public override string ToString() => "np.str_(" + PyRepr.Str(_text) + ")";
        }
    }
}
