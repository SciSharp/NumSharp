using System;
using System.Globalization;
using System.Linq;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>numpy.lib</c> package: a namespace for its public submodules (<see cref="stride_tricks"/>). It holds
        ///     no functions of its own here.
        /// </summary>
        public static partial class lib
        {
            /// <summary>
            ///     NumPy's <c>numpy.lib.stride_tricks</c>: views that re-read an array's memory with other strides.
            ///     <see cref="sliding_window_view(NDArray, Shape, NDAxis?, bool)"/> is the safe one (every window lies inside
            ///     the array); NumPy's <c>as_strided</c>, which reads whatever memory the given strides reach, is not part of
            ///     NumSharp (a bounds-checked form would be a decision of its own).
            /// </summary>
            [ModuleName("np.lib.stride_tricks")]
            public static class stride_tricks
            {
                /// <summary>
                ///     A sliding-window VIEW of <paramref name="x"/> — NumPy's
                ///     <c>sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False)</c>: every window
                ///     of <paramref name="window_shape"/> over the chosen axes, as trailing dimensions, without copying.
                /// </summary>
                /// <param name="x">The array to take windows of (any layout: transposed, strided, broadcast, 0-d).</param>
                /// <param name="window_shape">The window size per windowed axis: an int (one axis) or a tuple; one entry
                ///     per axis in <paramref name="axis"/> (every axis of <paramref name="x"/> when <paramref name="axis"/>
                ///     is null). A 0 is legal (an empty window); a negative entry is NumPy's ValueError.</param>
                /// <param name="axis">The axes to slide over: null (NumPy's None) is every axis in order; an int or a tuple
                ///     (negative axes count from the end). An axis may repeat — NumPy windows it again.</param>
                /// <param name="writeable">False (the default) returns a READ-ONLY view (a write raises NumPy's
                ///     <c>assignment destination is read-only</c>); true leaves it writeable when <paramref name="x"/> is,
                ///     so writes reach <paramref name="x"/> — and, as windows overlap, one element may appear in several
                ///     windows: NumPy's warning that writing through such a view is rarely what one wants applies.</param>
                /// <returns>A view of <paramref name="x"/>'s buffer, of shape <c>x.shape</c> with each windowed axis trimmed
                ///     to <c>n - w + 1</c>, followed by <paramref name="window_shape"/>; its strides are
                ///     <paramref name="x"/>'s followed by the windowed axes' strides. A plain array (NumPy's
                ///     <c>subok=False</c>).</returns>
                /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
                /// <exception cref="ValueError">In NumPy's order: a negative <paramref name="window_shape"/> entry
                ///     (<c>`window_shape` cannot contain negative values</c>); with a null <paramref name="axis"/>, a
                ///     <paramref name="window_shape"/> length other than <c>x.ndim</c> (<c>Since axis is `None`, must provide
                ///     window_shape for all dimensions of `x`; got 1 window_shape elements and `x.ndim` is 2.</c>); a length
                ///     other than the axes' (<c>Must provide matching length window_shape and axis; ...</c>); a window longer
                ///     than its axis (<c>window shape cannot be larger than input array shape</c>).</exception>
                /// <exception cref="AxisError">An axis out of range (<c>axis 2 is out of bounds for array of dimension
                ///     2</c>).</exception>
                /// <remarks>
                ///     https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
                ///     <para>NumPy's statements: the window checks, <c>normalize_axis_tuple(axis, x.ndim,
                ///     allow_duplicate=True)</c>, <c>out_strides = x.strides + tuple(x.strides[ax] for ax in axis)</c>, the
                ///     trimmed shape, then <c>as_strided(x, strides=out_strides, shape=out_shape, writeable=writeable)</c> —
                ///     here a <see cref="Shape"/> over <paramref name="x"/>'s own buffer window (its offset and buffer size),
                ///     so every window stays inside the memory <paramref name="x"/> may read. The view holds its own
                ///     reference to that buffer, so it outlives a disposed <paramref name="x"/>.</para>
                ///     <para>A moving reduction is a reduction over the window axes:
                ///     <c>np.mean(sliding_window_view(x, 3), axis: -1)</c> is the 3-point moving average. Added in API-parity
                ///     batch 7 with the module facade.</para>
                /// </remarks>
                public static NDArray sliding_window_view(NDArray x, Shape window_shape, NDAxis? axis = null, bool writeable = false)
                {
                    if (x is null)
                        throw new ArgumentNullException(nameof(x));

                    // `window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)`, then
                    // `if np.any(window_shape_array < 0)` — before anything about the axes.
                    long[] window = window_shape.dimensions ?? Array.Empty<long>();
                    foreach (long w in window)
                        if (w < 0)
                            throw new ValueError("`window_shape` cannot contain negative values");

                    var src = x.Shape;
                    int ndim = src.NDim;
                    int[] axes;
                    if (axis is null)
                    {
                        // axis=None: every axis in order, and one window entry per axis.
                        axes = Enumerable.Range(0, ndim).ToArray();
                        if (window.Length != axes.Length)
                            throw new ValueError(string.Create(CultureInfo.InvariantCulture,
                                $"Since axis is `None`, must provide window_shape for all dimensions of `x`; got {window.Length} window_shape elements and `x.ndim` is {ndim}."));
                    }
                    else
                    {
                        // normalize_axis_tuple(axis, x.ndim, allow_duplicate=True): an out-of-range axis is AxisError; a
                        // repeated axis is allowed and windows that axis again.
                        axes = Shape.normalize_axis_tuple(axis.Value.ToArray(), ndim, allow_duplicate: true);
                        if (window.Length != axes.Length)
                            throw new ValueError(string.Create(CultureInfo.InvariantCulture,
                                $"Must provide matching length window_shape and axis; got {window.Length} window_shape elements and {axes.Length} axes elements."));
                    }

                    // x_shape_trimmed: each windowed axis keeps the n - w + 1 window starts (a repeated axis is trimmed
                    // again, cumulatively, as NumPy's loop does); a window longer than the (remaining) axis is refused.
                    long[] trimmed = src.dimensions is null ? Array.Empty<long>() : (long[])src.dimensions.Clone();
                    for (int i = 0; i < axes.Length; i++)
                    {
                        int ax = axes[i];
                        if (trimmed[ax] < window[i])
                            throw new ValueError("window shape cannot be larger than input array shape");
                        trimmed[ax] -= window[i] - 1;
                    }

                    // out_shape = trimmed + window_shape; out_strides = x.strides + the windowed axes' strides (element
                    // strides here, NumPy's byte strides divided by the itemsize — the same walk).
                    var outDims = new long[ndim + window.Length];
                    var outStrides = new long[ndim + window.Length];
                    Array.Copy(trimmed, outDims, ndim);
                    Array.Copy(window, 0, outDims, ndim, window.Length);
                    if (ndim > 0)
                        Array.Copy(src.strides, outStrides, ndim);
                    for (int i = 0; i < axes.Length; i++)
                        outStrides[ndim + i] = src.strides[axes[i]];

                    // as_strided over x's own buffer window: x's offset and buffer size, so the view reads only memory x may
                    // read. The alias holds its own counted reference to the buffer.
                    var viewShape = new Shape(outDims, outStrides, src.offset, src.bufferSize > 0 ? src.bufferSize : src.size);
                    var view = new NDArray(x.Storage.Alias(viewShape)) { TensorEngine = x.TensorEngine };

                    // as_strided: `if view.flags.writeable and not writeable: view.flags.writeable = False` — the view starts
                    // as writeable as x (a read-only x stays read-only even for writeable=True).
                    if (!writeable && view.Shape.IsWriteable)
                        view.setflags(write: false);
                    return view;
                }

                /// <summary>
                ///     <see cref="sliding_window_view(NDArray, Shape, NDAxis?, bool)"/> with the axis as one <c>int</c>: the
                ///     overload a C# integer binds (the owner's kept integer form beside every NumPy-axis member),
                ///     forwarding to the NDAxis form, so every answer and error is that overload's.
                /// </summary>
                /// <param name="x">The array to take windows of.</param>
                /// <param name="window_shape">The window size (one entry: one axis).</param>
                /// <param name="axis">The axis to slide over (negative counts from the end).</param>
                /// <param name="writeable">False (the default): a read-only view.</param>
                /// <returns>The windowed view.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
                /// <exception cref="ValueError">The NDAxis form's window checks.</exception>
                /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
                public static NDArray sliding_window_view(NDArray x, Shape window_shape, int axis, bool writeable = false)
                    => sliding_window_view(x, window_shape, NDAxis.FromNullable(axis), writeable);

                /// <summary>
                ///     <see cref="sliding_window_view(NDArray, Shape, NDAxis?, bool)"/> with the axes as an <c>int[]</c>: the
                ///     overload a C# array (and a bare <c>null</c>, NumPy's None) binds — the owner's kept integer form,
                ///     forwarding to the NDAxis form.
                /// </summary>
                /// <param name="x">The array to take windows of.</param>
                /// <param name="window_shape">The window size per axis.</param>
                /// <param name="axis">The axes to slide over (a repeat windows an axis again); null is every axis.</param>
                /// <param name="writeable">False (the default): a read-only view.</param>
                /// <returns>The windowed view.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
                /// <exception cref="ValueError">The NDAxis form's window checks.</exception>
                /// <exception cref="AxisError">An axis is out of range.</exception>
                public static NDArray sliding_window_view(NDArray x, Shape window_shape, int[] axis, bool writeable = false)
                    => sliding_window_view(x, window_shape, (NDAxis?)axis, writeable);
            }
        }
    }
}
