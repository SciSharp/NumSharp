using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return the array as an (possibly nested) list.
        /// </summary>
        /// <returns>
        ///     The possibly nested list of array elements.
        ///     - For 0-d arrays (scalars): returns the scalar value itself
        ///     - For 1-d arrays: returns List&lt;object&gt; of elements
        ///     - For n-d arrays: returns nested List&lt;object&gt; structures
        /// </returns>
        /// <exception cref="SystemError">A <c>U</c> element holds a unit above <c>U+10FFFF</c> (NumPy raises the same).</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> element's cell is corrupt (NumPy's <c>Failed to load
        ///     string in StringDType getitem</c>) or its text is too long for a .NET string, or an axis is longer than a .NET
        ///     list can hold (<see cref="System.Array.MaxLength"/> elements).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
        ///
        ///     Copy of the array data as a (nested) Python list. Data items are converted
        ///     to the nearest compatible builtin Python type, via the item function.
        ///
        ///     If a.ndim is 0, then since the depth of the nested list is 0, it will not
        ///     be a list at all, but a simple Python scalar.
        ///
        ///     <para>
        ///     NumPy's <c>PyArray_ToList</c> / <c>recursive_tolist</c>: the walk follows the strides from the first element
        ///     (any layout — strided, transposed, reversed, broadcast — with no copy and no per-element index arithmetic), one
        ///     list per axis position, and each innermost row is read by one <see cref="IArraySlice.GetItems"/> call into the
        ///     row's list. Each element is what <see cref="item(long)"/> returns for it (the storage's <c>getitem</c>): the
        ///     boxed CLR value of a numeric element, a <see cref="string"/> / <c>byte[]</c> of a <c>U</c> / <c>S</c> element,
        ///     the text or the NA object of a <c>StringDType</c> element. An element that cannot be read raises at the first
        ///     one in C order, and the half-built lists are dropped, as NumPy drops its own.
        ///     </para>
        ///     <para>
        ///     The walk replaced an <see cref="item(long)"/> call per element (a bounds check, a dtype switch and, for a
        ///     non-contiguous array, a coordinate array per element) and a sub-array view per outer index: over 100K text
        ///     elements of 20–60 characters a <c>tolist()</c> went 46.5 → 25.2 ns per element for <c>S</c>, 90.9 → 60.0 for
        ///     <c>U</c> and 79.1 → 46.3 for <c>StringDType</c> (NumPy 58.3 / 92.8 / 100.2), at 4–12 characters 34.0 → 23.9,
        ///     38.7 → 29.7 and 49.6 → 30.8 (NumPy 39.7 / 54.2 / 74.1). What remains is the object every element becomes: a
        ///     text of 300–500 characters stays at or below NumPy (S / U / StringDType about 0.97 / 0.83 / 0.69 of its speed,
        ///     pinned), because
        ///     allocating 100K such .NET strings (UTF-16, two bytes a character) and keeping them alive through the gen0 /
        ///     gen1 collections that copy them costs about what NumPy's whole <c>tolist</c> costs (its compact one-byte
        ///     <c>str</c>, reference-counted, is never copied).
        ///     </para>
        ///     <para>
        ///     <see cref="NDAnchoredAttribute"/> (lost-anchor site LT29): each innermost row is read through a raw pointer
        ///     while one object per element is allocated, and nothing but this array keeps the buffer alive, so the receiver
        ///     is rooted to the end. Without it <c>a.copy().tolist()</c> left the copy reachable from nothing during the
        ///     walk: a collection could finalize its memory block and the rest of the row read freed memory.
        ///     </para>
        /// </remarks>
        [NDAnchored]
        public object tolist()
        {
            // 0-d array (scalar): return the scalar value directly
            if (ndim == 0 || Shape.IsScalar)
                return Storage.GetAtIndex(0);

            // n-d array: NumPy's recursive_tolist from the array's first element (its offset into the storage).
            return ToListRecursive(Storage.InternalArray, Shape.dimensions, Shape.strides, 0, Shape.offset);
        }

        /// <summary>
        ///     NumPy's <c>recursive_tolist</c> at axis <paramref name="axis"/>: the list of the positions along that axis,
        ///     starting at storage element <paramref name="start"/> — the values themselves at the last axis, the sub-lists of
        ///     the next axis otherwise.
        /// </summary>
        /// <param name="slice">The storage's elements (indexed as <see cref="IArraySlice.GetIndex(long)"/> takes them).</param>
        /// <param name="dims">The array's dimensions.</param>
        /// <param name="strides">The array's strides, in elements (0 on a broadcast axis, negative on a reversed one).</param>
        /// <param name="axis">The axis this list walks (0 = outermost).</param>
        /// <param name="start">The storage index of the list's first element.</param>
        /// <returns>The list (its length is the axis' extent; an empty axis gives an empty list).</returns>
        /// <exception cref="OutOfMemoryException">The axis is longer than <see cref="System.Array.MaxLength"/>, the most elements a
        ///     .NET list holds; or an element read raises it (see <see cref="tolist"/>).</exception>
        /// <exception cref="SystemError">An element read raises it (see <see cref="tolist"/>).</exception>
        private static List<object> ToListRecursive(IArraySlice slice, long[] dims, long[] strides, int axis, long start)
        {
            long n = dims[axis];
            if (n > System.Array.MaxLength)
                throw new OutOfMemoryException($"An axis of {n} elements is longer than a .NET list can hold ({System.Array.MaxLength}).");
            // The list is sized once and filled through its span (CollectionsMarshal): no Add per element, and a row's
            // values land in place by one GetItems call.
            var list = new List<object>((int)n);
            CollectionsMarshal.SetCount(list, (int)n);
            var items = CollectionsMarshal.AsSpan(list);
            long stride = strides[axis];
            if (axis == dims.Length - 1)
            {
                // An empty row never reaches the slice: an empty array's storage may hold no element at its offset.
                if (items.Length != 0)
                    slice.GetItems(start, stride, items);
            }
            else
            {
                for (int i = 0; i < items.Length; i++)
                    items[i] = ToListRecursive(slice, dims, strides, axis + 1, start + i * stride);
            }
            return list;
        }
    }
}
