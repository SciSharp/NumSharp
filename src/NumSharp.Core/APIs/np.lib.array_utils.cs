using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class lib
        {
            /// <summary>
            ///     NumPy's <c>numpy.lib.array_utils</c> (NumPy 2.0's public home of three helpers that lived in
            ///     <c>numpy.lib.utils</c> / <c>numpy.core.multiarray</c>): <see cref="byte_bounds"/>,
            ///     <see cref="normalize_axis_index"/> and <see cref="normalize_axis_tuple"/>, over the same statements every
            ///     NumSharp axis-taking function runs (<see cref="Shape"/>'s axis checks), so their texts are NumPy's.
            /// </summary>
            [ModuleName("np.lib.array_utils")]
            public static class array_utils
            {
                /// <summary>
                ///     NumPy's <c>byte_bounds(a)</c>: the first byte address <paramref name="a"/> reaches and the address one
                ///     past its last byte - the memory range its elements occupy, whatever its strides (a reversed axis moves
                ///     the low end, a broadcast axis adds nothing).
                /// </summary>
                /// <param name="a">The array (any layout, a view included).</param>
                /// <returns>The <c>(low, high)</c> byte addresses: <c>high - low</c> is the span (<c>size * itemsize</c> for a
                ///     C-contiguous array, 0 for an empty one).</returns>
                /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
                /// <remarks>
                ///     NumPy's statements over <c>a.__array_interface__</c>: a C-contiguous array reports no strides, so its
                ///     high end is <c>data + size * itemsize</c>; any other array adds <c>(n - 1) * stride</c> per axis to the
                ///     low end (a negative stride) or the high end, then one item. The addresses are this process's, as
                ///     NumPy's are, and stay valid only while the array's buffer lives.
                /// </remarks>
                public static unsafe (long low, long high) byte_bounds(NDArray a)
                {
                    if (a is null)
                        throw new ArgumentNullException(nameof(a));

                    var shape = a.Shape;
                    long itemsize = a.dtypesize;
                    // __array_interface__['data'][0]: the address of the view's first element (the buffer start plus the
                    // view's offset, which counts items).
                    long low = (long)((byte*)a.Storage.Address + shape.offset * itemsize);
                    long high = low;

                    if (shape.IsContiguous)
                    {
                        // The interface's strides are None for a C-contiguous array: NumPy's `a_high += a.size * bytes_a`.
                        high += shape.size * itemsize;
                        return (low, high);
                    }

                    long[] dims = shape.dimensions ?? System.Array.Empty<long>();
                    long[] strides = shape.strides ?? System.Array.Empty<long>();
                    for (int i = 0; i < dims.Length; i++)
                    {
                        // NumSharp's strides count items; the interface's count bytes.
                        long reach = (dims[i] - 1) * strides[i] * itemsize;
                        if (strides[i] < 0)
                            low += reach;
                        else
                            high += reach;
                    }
                    high += itemsize;
                    return (low, high);
                }

                /// <summary>
                ///     NumPy's <c>normalize_axis_index(axis, ndim, msg_prefix=None)</c>: <paramref name="axis"/> wrapped into
                ///     <c>[0, ndim)</c>, or NumPy's AxisError naming it.
                /// </summary>
                /// <param name="axis">The axis; negative values count from the end.</param>
                /// <param name="ndim">The rank the axis addresses.</param>
                /// <param name="msg_prefix">A prefix NumPy prints before the text (<c>"axes"</c> gives <c>axes: axis 3 is out
                ///     of bounds for array of dimension 3</c>), or null for none.</param>
                /// <returns>The non-negative axis.</returns>
                /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-ndim, ndim)</c>: <c>axis 3 is out of
                ///     bounds for array of dimension 3</c>.</exception>
                public static int normalize_axis_index(int axis, int ndim, string msg_prefix = null)
                    => Shape.check_and_adjust_axis(ndim, axis, msg_prefix);

                /// <summary>
                ///     NumPy's <c>normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False)</c>: every axis of
                ///     <paramref name="axis"/> wrapped into <c>[0, ndim)</c>, then NumPy's repeated-axis refusal unless
                ///     <paramref name="allow_duplicate"/>.
                /// </summary>
                /// <param name="axis">One axis or a tuple of axes (NumPy's int or iterable of ints).</param>
                /// <param name="ndim">The rank the axes address.</param>
                /// <param name="argname">The argument name: the AxisError's prefix and the repeat text's <c>repeated axis in
                ///     `axes` argument</c>; null gives the plain texts.</param>
                /// <param name="allow_duplicate">When false (NumPy's default) a repeated axis raises; when true repeats are
                ///     kept in order.</param>
                /// <returns>The normalized axes, one per input entry, in input order (NumPy's tuple).</returns>
                /// <exception cref="AxisError">An axis is out of range (every range is checked before any repeat).</exception>
                /// <exception cref="ValueError">A repeated axis and <paramref name="allow_duplicate"/> false (<c>repeated
                ///     axis</c>).</exception>
                public static int[] normalize_axis_tuple(NDAxis axis, int ndim, string argname = null, bool allow_duplicate = false)
                    => Shape.normalize_axis_tuple(axis.ToArray(), ndim, argname, allow_duplicate);

                /// <summary>
                ///     <see cref="normalize_axis_tuple(NDAxis, int, string, bool)"/> with one <c>int</c> axis: the overload a
                ///     C# integer binds (the owner's kept integer form beside every NumPy-axis member).
                /// </summary>
                /// <param name="axis">The axis; negative values count from the end.</param>
                /// <param name="ndim">The rank the axis addresses.</param>
                /// <param name="argname">The argument name for the AxisError's prefix, or null.</param>
                /// <param name="allow_duplicate">Unused for one axis (it cannot repeat); NumPy's signature.</param>
                /// <returns>The one-element tuple of the normalized axis.</returns>
                /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
                public static int[] normalize_axis_tuple(int axis, int ndim, string argname = null, bool allow_duplicate = false)
                    => normalize_axis_tuple((NDAxis)axis, ndim, argname, allow_duplicate);

                /// <summary>
                ///     <see cref="normalize_axis_tuple(NDAxis, int, string, bool)"/> with an <c>int[]</c> of axes (the owner's
                ///     kept array form beside every NumPy-axis member).
                /// </summary>
                /// <param name="axis">The axes (not null; never modified).</param>
                /// <param name="ndim">The rank the axes address.</param>
                /// <param name="argname">The argument name for the texts, or null.</param>
                /// <param name="allow_duplicate">Whether a repeated axis is kept.</param>
                /// <returns>The normalized axes in input order.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="axis"/> is null.</exception>
                /// <exception cref="AxisError">An axis is out of range.</exception>
                /// <exception cref="ValueError">A repeated axis and <paramref name="allow_duplicate"/> false.</exception>
                public static int[] normalize_axis_tuple(int[] axis, int ndim, string argname = null, bool allow_duplicate = false)
                {
                    if (axis is null)
                        throw new ArgumentNullException(nameof(axis));
                    return normalize_axis_tuple((NDAxis)axis, ndim, argname, allow_duplicate);
                }
            }
        }
    }
}
