using System;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        // ============================== np.pad ==============================
        // Pad an N-D array along every axis. Eleven string modes + user callable.
        //
        // NumPy 2.4.2 reference: numpy/lib/_arraypad_impl.py. The statements below the facades are the array object's
        // (NDArray.ArrayPad.cs, P3's batch 7: NDArray.Pad for a named mode, NDArray.PadCallable for a callable); these
        // overloads keep the numpy.matrix region and call them.
        //
        // Architecture mirrors NumPy's _arraypad_impl.py — the helper functions
        // (_AsPairs, _PadSimple, _SliceAtAxis, _ViewRoi, _SetPadArea, _GetEdges,
        // _GetLinearRamps, _GetStats, _SetReflectBoth, _SetWrapBoth) have a 1:1
        // correspondence with their underscore-prefixed Python counterparts.
        // This is intentional — the corner-propagation logic that turns iterative
        // axis-padding into N-D padding is load-bearing and brittle, so the port
        // tracks the upstream structure exactly.
        //
        // Performance: every bulk operation routes through existing IL kernels —
        //   * np.empty for allocation (single pointer alloc)
        //   * IArraySlice.Fill for uniform-value PadSimple (InitBlockUnaligned/unrolled scalar)
        //   * padded[band] = src routes through NDIter.Copy → StridedCastKernel
        //     (cpblk contig path, per-row memcpy strided path, "convert once +
        //      memcpy" broadcast fast path for 1-thick stripe and scalar fills)
        //   * stat modes use existing axis-reduction IL kernels (np.max/min/mean/median)
        //
        // No new IL emission is required for any mode.

        public delegate void PadFunc(NDArray vector, (int before, int after) padWidth, int axis, object kwargs);

        // ---------------------------- public overloads ----------------------------

        /// <summary>
        ///     Pad an array. Scalar <paramref name="pad_width"/> applies <c>(pad_width, pad_width)</c>
        ///     to every axis.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.pad.html</remarks>
        public static NDArray pad(NDArray array, int pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.Pad(array, (object)pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        public static NDArray pad(NDArray array, int[] pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.Pad(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        public static NDArray pad(NDArray array, int[,] pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.Pad(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        public static NDArray pad(NDArray array, (int before, int after) pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.Pad(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        /// <summary>
        ///     Pad an array by a per-axis dict (NumPy 2.3+): each key names an axis (negative counts from the end), each
        ///     value pads it by an integer on both sides or by a <c>(before, after)</c> tuple; an axis without a key is not
        ///     padded.
        /// </summary>
        /// <param name="array">The array to pad.</param>
        /// <param name="pad_width">The per-axis widths, read in the dictionary's order (a later key for the same axis wins).</param>
        /// <param name="mode">The pad mode, as for the other overloads.</param>
        /// <param name="constant_values">The fill of <c>'constant'</c>.</param>
        /// <param name="end_values">The ends of <c>'linear_ramp'</c>.</param>
        /// <param name="stat_length">The window of the statistic modes.</param>
        /// <param name="reflect_type"><c>'even'</c> or <c>'odd'</c> for <c>'reflect'</c> / <c>'symmetric'</c>.</param>
        /// <returns>The padded array.</returns>
        /// <exception cref="AssertionError">A value that is neither an integer nor a 2-tuple of integers (NumPy's
        ///     <c>typing.assert_never</c>; a typed C# array is an ndarray, which NumPy refuses too).</exception>
        /// <exception cref="IndexError">A key outside <c>[-ndim, ndim)</c> ("list assignment index out of range", NumPy's
        ///     list assignment - not an AxisError). Every key of a 0-d array is one.</exception>
        /// <exception cref="TypeError">The widths are no integer table: a 0-d array's empty dict, every axis given bools,
        ///     or a width outside int64.</exception>
        /// <exception cref="ValueError">A negative width (checked after every key, as NumPy's <c>_as_pairs</c> does).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.pad.html</remarks>
        public static NDArray pad(NDArray array, IDictionary<int, object> pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.Pad(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        // Callable mode (overloads matching each pad_width shape).
        public static NDArray pad(NDArray array, int pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.PadCallable(array, (object)pad_width, mode, kwargs);
        }

        public static NDArray pad(NDArray array, int[] pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.PadCallable(array, pad_width, mode, kwargs);
        }

        public static NDArray pad(NDArray array, int[,] pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.PadCallable(array, pad_width, mode, kwargs);
        }

        public static NDArray pad(NDArray array, (int before, int after) pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.PadCallable(array, pad_width, mode, kwargs);
        }

        /// <summary>
        ///     Pad an array by a per-axis dict with a callable mode: the dict is read as in the string-mode dict overload,
        ///     then <paramref name="mode"/> fills each 1-D vector's pad area.
        /// </summary>
        /// <param name="array">The array to pad.</param>
        /// <param name="pad_width">The per-axis widths, read in the dictionary's order (a later key for the same axis wins).</param>
        /// <param name="mode">The callable that writes each vector's pad area in place.</param>
        /// <param name="kwargs">Passed to <paramref name="mode"/> unchanged.</param>
        /// <returns>The padded array.</returns>
        /// <exception cref="AssertionError">A value that is neither an integer nor a 2-tuple of integers.</exception>
        /// <exception cref="IndexError">A key outside <c>[-ndim, ndim)</c> ("list assignment index out of range").</exception>
        /// <exception cref="TypeError">The widths are no integer table (a 0-d array's empty dict, every axis given bools,
        ///     a width outside int64).</exception>
        /// <exception cref="ValueError">A negative width.</exception>
        public static NDArray pad(NDArray array, IDictionary<int, object> pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return NDArray.PadCallable(array, pad_width, mode, kwargs);
        }
    }
}
