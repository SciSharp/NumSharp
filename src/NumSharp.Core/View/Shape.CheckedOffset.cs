using System.Globalization;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public partial struct Shape
    {
        /// <summary>
        ///     The storage offset of the element at <paramref name="indices"/>, checked and wrapped by NumPy's coordinate
        ///     rule — the offset every PUBLIC element member (<c>item(i, j, …)</c>, <c>GetValue</c>, <c>GetDouble</c> …,
        ///     <c>SetValue</c>, <c>SetDouble</c> …, <c>itemset</c>) addresses, so no coordinate can read or write outside
        ///     the array.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis. A negative coordinate counts from the end of its axis
        ///     (NumPy's wrap); fewer coordinates than dimensions address the first element of the axes left out (NumSharp's
        ///     documented "the rest are filled with 0"); a 0-d shape takes none, or the single flat index <c>0</c> / <c>-1</c>
        ///     of its only element.</param>
        /// <returns>The element's offset in the storage buffer (the shape's base offset included).</returns>
        /// <exception cref="IndexError">A coordinate outside its axis (<c>index 5 is out of bounds for axis 1 with size
        ///     3</c>, NumPy's text, the coordinate as given), more coordinates than dimensions (<c>too many indices for array:
        ///     array is 2-dimensional, but 3 were indexed</c>), or a 0-d shape given any index but its element's
        ///     (<c>index 1 is out of bounds for size 1</c>).</exception>
        /// <remarks>
        ///     The unchecked <see cref="GetOffset(long[])"/> stays the kernels' and the iterators' addressing: they compute
        ///     coordinates that are in range by construction, and the per-axis test would cost them in every inner loop. A
        ///     public member hands a caller's numbers to the address arithmetic, where an out-of-range coordinate used to
        ///     land in another row or outside the buffer in a Release build (parity QA-33).
        /// </remarks>
        [MethodImpl(Inline)]
        internal readonly long GetOffsetChecked(long[] indices)
        {
            int ndim = dimensions.Length;
            if (ndim == 0)
                return ScalarOffsetChecked(indices.Length == 0 ? 0 : indices[0], indices.Length);
            if (indices.Length > ndim)
                throw new IndexError(string.Create(CultureInfo.InvariantCulture, $"too many indices for array: array is {ndim}-dimensional, but {indices.Length} were indexed"));

            long off = offset;
            for (int ax = 0; ax < indices.Length; ax++)
                off += AxisCoordinateChecked(indices[ax], ax) * strides[ax];
            return off;
        }

        /// <summary>
        ///     <see cref="GetOffsetChecked(long[])"/> for <see cref="int"/> coordinates (the <c>int[]</c> overloads of the
        ///     public element members).
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (see <see cref="GetOffsetChecked(long[])"/>).</param>
        /// <returns>The element's offset in the storage buffer.</returns>
        /// <exception cref="IndexError">As <see cref="GetOffsetChecked(long[])"/>.</exception>
        [MethodImpl(Inline)]
        internal readonly long GetOffsetChecked(int[] indices)
        {
            int ndim = dimensions.Length;
            if (ndim == 0)
                return ScalarOffsetChecked(indices.Length == 0 ? 0 : indices[0], indices.Length);
            if (indices.Length > ndim)
                throw new IndexError(string.Create(CultureInfo.InvariantCulture, $"too many indices for array: array is {ndim}-dimensional, but {indices.Length} were indexed"));

            long off = offset;
            for (int ax = 0; ax < indices.Length; ax++)
                off += AxisCoordinateChecked(indices[ax], ax) * strides[ax];
            return off;
        }

        /// <summary>
        ///     The storage offset of the element at logical flat index <paramref name="index"/> (C order), checked and
        ///     wrapped by NumPy's flat rule (<c>a.item(i)</c>, <c>a.flat[i]</c>) — the checked form of
        ///     <see cref="TransformOffset(long)"/> behind the public <c>GetAtIndex</c> / <c>SetAtIndex</c>.
        /// </summary>
        /// <param name="index">The flat index; a negative one counts from the end of the array.</param>
        /// <returns>The element's offset in the storage buffer.</returns>
        /// <exception cref="IndexError"><paramref name="index"/> outside the array: NumPy's item texts, <c>index 10 is out
        ///     of bounds for axis 0 with size 3</c> for a 1-D shape and <c>index 10 is out of bounds for size 6</c> for any
        ///     other rank (the index as given).</exception>
        [MethodImpl(Inline)]
        internal readonly long TransformOffsetChecked(long index)
        {
            long n = size;
            long i = index < 0 ? index + n : index;
            // One unsigned compare tests both ends: a still-negative index becomes a huge unsigned value.
            if ((ulong)i >= (ulong)n)
                throw new IndexError(dimensions.Length == 1
                    ? string.Create(CultureInfo.InvariantCulture, $"index {index} is out of bounds for axis 0 with size {n}")
                    : string.Create(CultureInfo.InvariantCulture, $"index {index} is out of bounds for size {n}"));
            return TransformOffset(i);
        }

        /// <summary>
        ///     One coordinate of axis <paramref name="axis"/>, wrapped and checked (NumPy's <c>check_and_adjust_index</c>).
        /// </summary>
        /// <param name="coordinate">The coordinate as given.</param>
        /// <param name="axis">The axis it addresses.</param>
        /// <returns>The coordinate in <c>[0, dimensions[axis])</c>.</returns>
        /// <exception cref="IndexError">The coordinate is outside the axis (an empty axis has none inside).</exception>
        [MethodImpl(Inline)]
        private readonly long AxisCoordinateChecked(long coordinate, int axis)
        {
            long n = dimensions[axis];
            long i = coordinate < 0 ? coordinate + n : coordinate;
            if ((ulong)i >= (ulong)n)
                throw new IndexError(string.Create(CultureInfo.InvariantCulture, $"index {coordinate} is out of bounds for axis {axis} with size {n}"));
            return i;
        }

        /// <summary>
        ///     The offset of a 0-d shape's only element for the indices a public member was given: none, or the flat index of
        ///     that element (internal callers address a 0-d array as <c>GetValue(0)</c>).
        /// </summary>
        /// <param name="index">The single index given (0 when none was).</param>
        /// <param name="count">How many indices were given.</param>
        /// <returns>The element's offset.</returns>
        /// <exception cref="IndexError">More than one index, or one other than <c>0</c> / <c>-1</c> (NumPy's item text for a
        ///     size-1 array).</exception>
        private readonly long ScalarOffsetChecked(long index, int count)
        {
            if (count > 1)
                throw new IndexError(string.Create(CultureInfo.InvariantCulture, $"too many indices for array: array is 0-dimensional, but {count} were indexed"));
            if (index != 0 && index != -1)
                throw new IndexError(string.Create(CultureInfo.InvariantCulture, $"index {index} is out of bounds for size 1"));
            return offset;
        }
    }
}
