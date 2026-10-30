using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Take elements from an array along an axis. Equivalent to fancy
        ///     indexing along the specified axis.
        /// </summary>
        /// <param name="a">Source array.</param>
        /// <param name="indices">Integer array of indices to take (any layout; read in C order).</param>
        /// <param name="axis">
        ///     Axis along which to take. <c>null</c> (default) flattens <paramref name="a"/>
        ///     and treats <paramref name="indices"/> as flat indices. A 0-d <paramref name="a"/> is flattened too, so its
        ///     axis must be valid for one dimension (0 or -1).
        /// </param>
        /// <param name="out">
        ///     Optional destination array. When supplied, its shape must match the
        ///     natural take output; values are cast to <paramref name="out"/>'s dtype
        ///     via <see cref="np.copyto"/> with unsafe casting and the method returns
        ///     <paramref name="out"/> itself. When <c>null</c> (default), a fresh
        ///     array is allocated with <paramref name="a"/>'s dtype.
        /// </param>
        /// <param name="mode">
        ///     Boundary mode: <c>"raise"</c> (default — throw on OOB), <c>"wrap"</c>
        ///     (modulo with sign correction), or <c>"clip"</c> (saturate); <c>null</c> is NumPy's <c>None</c> (raise).
        /// </param>
        /// <returns>
        ///     New array with shape:
        ///     <list type="bullet">
        ///       <item><c>axis=None</c>: same as <paramref name="indices"/>.</item>
        ///       <item><c>axis=k</c>: <c>a.shape[:k] + indices.shape + a.shape[k+1:]</c>.</item>
        ///     </list>
        ///     Dtype matches <paramref name="a"/> (or <paramref name="out"/>'s dtype
        ///     when <paramref name="out"/> is supplied).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result; a 0-d result without
        ///     <paramref name="out"/> is the element's UNSTRIPPED NumPy scalar (<c>PyArray_Return</c>), NumSharp's
        ///     read-only stand-in.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="indices"/> is null.</exception>
        /// <exception cref="ValueError">A bad <paramref name="mode"/> (<see cref="ClipModes.ClipmodeConverter"/>), checked first;
        ///     an <paramref name="out"/> of another shape (<c>output array does not match result of ndarray.take</c>) or a
        ///     read-only one (<c>WRITEBACKIFCOPY base is read-only</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">The index dtype does not cast to int64 under <c>same_kind</c> (float, complex),
        ///     an empty index array included; or <paramref name="out"/>'s dtype does not cast to <paramref name="a"/>'s
        ///     under <c>safe</c>.</exception>
        /// <exception cref="IndexError">A non-empty result from an empty axis (<c>cannot do a non-empty take from an empty
        ///     axes.</c>), or an index out of bounds under <c>mode="raise"</c> (<c>index 9 is out of bounds for axis 0 with
        ///     size 4</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.take.html
        ///     <para>The work is <see cref="UnmanagedStorage.TakeFrom"/>, NumPy's <c>PyArray_TakeFrom</c>, which also fixes the order of the
        ///     checks above.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // reclaims the int64 index copy, the C-order source copy and the natural-result temp on the out= path
        public static NDArray take(NDArray a, NDArray indices, int? axis = null, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, indices, axis, @out, mode), static s => take(s.a, s.indices, s.axis, s.@out, s.mode), @out);
            // a numpy.char.chararray operand comes back as a chararray (a 0-d result: the element's unstripped scalar), as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.KeptTake((a, indices, axis, @out, mode), static s => take(s.a, s.indices, s.axis, s.@out, s.mode), @out);
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            // array_take converts the mode (and parses the axis) with its arguments, before PyArray_TakeFrom runs.
            int modeInt = ClipModes.ClipmodeConverter(mode);
            return UnmanagedStorage.TakeFrom(a, indices, axis, @out, modeInt);
        }

        /// <summary>
        ///     Scalar convenience overload — take a single element by flat index.
        /// </summary>
        /// <param name="a">Source array.</param>
        /// <param name="index">The index (a 0-d index: the result has the shape of the taken element).</param>
        /// <param name="axis">As in <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</param>
        /// <param name="out">As in <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</param>
        /// <param name="mode">As in <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</param>
        /// <returns>The taken element(s).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result; a 0-d result without
        ///     <paramref name="out"/> is the element's UNSTRIPPED NumPy scalar (<c>PyArray_Return</c>), NumSharp's
        ///     read-only stand-in.</returns>
        /// <exception cref="ValueError">See <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</exception>
        /// <exception cref="AxisError">See <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</exception>
        /// <exception cref="IndexError">See <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // the 0-d index wrapper is built in THIS frame, before the array overload's scope opens
        public static NDArray take(NDArray a, long index, int? axis = null, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, index, axis, @out, mode), static s => take(s.a, s.index, s.axis, s.@out, s.mode), @out);
            // a numpy.char.chararray operand comes back as a chararray (a 0-d result: the element's unstripped scalar), as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.KeptTake((a, index, axis, @out, mode), static s => take(s.a, s.index, s.axis, s.@out, s.mode), @out);
            // Wrap the scalar in a 0-d NDArray so the array overload's shape-preserving
            // logic emits a 0-d result (NumPy semantic for scalar input).
            var idxArr = NDArray.Scalar(index);
            return take(a, idxArr, axis, @out, mode);
        }
    }
}
