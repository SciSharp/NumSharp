using System;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Remove axes of length one from <paramref name="a"/> (NumPy's <c>np.squeeze(a, axis=None)</c>): all of them,
        ///     or only the ones <paramref name="axis"/> names.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) removes every axis of length one; one
        ///     axis removes only that axis; a tuple (<c>(0, 2)</c> or <c>new[] { 0, 2 }</c>) removes each axis it names, and
        ///     the empty tuple removes none. Every named axis must have length one; negative axes count from the end. The
        ///     integer form is NumPy's backward-compatibility case on a 0-d input (axis 0 and -1 are accepted and remove
        ///     nothing); a tuple there has no axis to name.</param>
        /// <returns>A view of <paramref name="a"/> with the selected dimensions of length 1 removed: a fresh view object
        ///     every time, <c>a.view()</c> when there is nothing to remove (NumPy returns <c>a</c> itself there; a fresh
        ///     view keeps disposing the result from disposing the input - the plan's D15). A
        ///     <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the subclass),
        ///     owning its data exactly when the plain result would.</returns>
        /// <remarks>
        ///     A pure VIEW like NumPy's <c>PyArray_Squeeze</c>: the length-one axes are dropped from the
        ///     dims AND strides while offset/buffer stay — never a reshape (which rebuilds C-strides and
        ///     so lost F-contiguity, and MATERIALIZED non-contiguous inputs where NumPy shares memory).
        ///     Probed: <c>np.asfortranarray(zeros((3,1,4))).squeeze()</c> is F-contiguous, a transposed
        ///     input stays a strided view, and a broadcast input keeps stride-0 (read-only).
        ///     With an axis: NumPy's <c>PyArray_ConvertMultiAxis</c> then <c>PyArray_SqueezeSelected</c>. Its integer
        ///     branch lets axis 0 and -1 through for a 0-d input (no axis is selected, so nothing is removed) and checks every
        ///     other axis with <c>normalize_axis_index</c>; its tuple branch checks each axis in order - the range, then
        ///     whether an earlier item named it (<c>duplicate value in 'axis'</c>), so <c>(0, 0, 5)</c> reports the
        ///     repeat and <c>(5, 0, 0)</c> the range. Then every selected axis must have length one.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
        ///     <para>One overload, NumPy's signature (2026-11-11): <c>squeeze(a)</c> and <c>squeeze(a, int axis)</c> were
        ///     two overloads, so <c>np.squeeze(a, axis: null)</c> did not compile; the NumSharp-only
        ///     <c>Shape squeeze(Shape)</c> had no caller and went. The axis is an <see cref="NDAxis"/> since API-parity
        ///     batch 4 (2026-11-12): NumPy's tuple of axes (the axis-type decision, OD1); an <c>int?</c> argument
        ///     converts.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="AxisError">An axis is out of range ("axis … is out of bounds for array of dimension …", the
        ///     axis as given).</exception>
        /// <exception cref="ValueError">A selected axis does not have length one ("cannot select an axis to squeeze out
        ///     which has size not equal to one"), or a tuple names an axis twice ("duplicate value in 'axis'").</exception>
        public static NDArray squeeze(NDArray a, NDAxis? axis = null)
        {
            // An integer axis is PyArray_ConvertMultiAxis' integer branch (the 0-d special case); a tuple its tuple branch.
            if (axis is { IsTuple: false } single)
                return SqueezeAxis(a, single[0]);
            if (axis is { } tuple)
                return SqueezeAxes(a, tuple);

            var dims = a.Shape.dimensions;
            int keep = 0;
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] != 1)
                    keep++;
            // Nothing to remove: NumPy's PyArray_Squeeze returns the array itself (Py_INCREF(self)); NumSharp returns
            // a.view(), a fresh view object over the same memory and engine, so disposing the result never disposes the
            // input (D15; the view-flags oracle carves the identity difference).
            if (keep == dims.Length)
                return a.view();

            return SqueezeView(a, keep, onlyAxis: -1);
        }

        /// <summary>
        ///     <see cref="squeeze"/>'s tuple branch: removes every axis <paramref name="axes"/> names.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axes">The axes to remove (a tuple, possibly empty); each must have length one.</param>
        /// <returns>A view of <paramref name="a"/> without the named axes (a fresh view object; <c>a.view()</c> when the
        ///     tuple is empty, which removes nothing).</returns>
        /// <remarks>
        ///     NumPy's <c>PyArray_ConvertMultiAxis</c> (its tuple branch: each item through <c>check_and_adjust_axis</c>, then
        ///     the repeat test against the items before it, in order) then <c>PyArray_SqueezeSelected</c> (every selected axis
        ///     of length one, else the ValueError; nothing selected returns the array itself, here <c>a.view()</c>, D15). A 0-d
        ///     input has no backward-compatibility case in the tuple branch: <c>(0,)</c> is an AxisError there. A subclass
        ///     operand gets its class back from <see cref="SqueezeSelectedView"/>'s <c>[NewSubtype]</c>.
        /// </remarks>
        /// <exception cref="AxisError">An axis is out of range (the first one in order, as given).</exception>
        /// <exception cref="ValueError">An axis repeats an earlier one (<c>duplicate value in 'axis'</c>), or a selected
        ///     axis does not have length one (<c>cannot select an axis to squeeze out which has size not equal to
        ///     one</c>).</exception>
        private static NDArray SqueezeAxes(NDArray a, NDAxis axes)
        {
            int ndim = a.ndim;
            var selected = new bool[ndim];
            // PyArray_ConvertMultiAxis' tuple loop: the range check and the repeat check of each item before the next item
            // is read, so a repeat that comes before an out-of-range item is the error NumPy reports.
            for (int i = 0; i < axes.Count; i++)
            {
                int axis = Shape.check_and_adjust_axis(ndim, axes[i]);
                if (selected[axis])
                    throw new ValueError("duplicate value in 'axis'");
                selected[axis] = true;
            }

            // PyArray_SqueezeSelected: every selected axis must have length one, checked from the first axis on.
            var dims = a.Shape.dimensions;
            int removed = 0;
            for (int d = 0; d < ndim; d++)
            {
                if (!selected[d])
                    continue;
                if (dims[d] != 1)
                    throw new ValueError("cannot select an axis to squeeze out which has size not equal to one");
                removed++;
            }

            // Nothing selected: NumPy returns the array itself; a.view() is a fresh view object (see squeeze, D15).
            if (removed == 0)
                return a.view();

            return SqueezeSelectedView(a, selected, ndim - removed);
        }

        /// <summary>
        ///     <see cref="squeeze"/>'s integer-axis branch: removes only <paramref name="axis"/>.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The one axis to remove; it must have length one.</param>
        /// <returns>A view of <paramref name="a"/> without the named axis (a fresh view object; <c>a.view()</c> for the 0-d
        ///     case, which removes nothing).</returns>
        /// <remarks>
        ///     NumPy's <c>PyArray_ConvertMultiAxis</c> (its integer branch) then <c>PyArray_SqueezeSelected</c>: a 0-d input
        ///     lets axis 0 and -1 through (no axis is selected, so nothing is removed), every other axis must be in range
        ///     (<c>normalize_axis_index</c>), and the selected axis must have length one. A subclass operand gets its class
        ///     back from <see cref="SqueezeView"/>'s <c>[NewSubtype]</c>, as in the no-axis branch.
        /// </remarks>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range ("axis … is out of bounds for array of
        ///     dimension …").</exception>
        /// <exception cref="ValueError">The selected axis does not have length one ("cannot select an axis to squeeze out
        ///     which has size not equal to one").</exception>
        private static NDArray SqueezeAxis(NDArray a, int axis)
        {
            // The backward-compatibility special case of PyArray_ConvertMultiAxis: axis 0 / -1 of a 0-d input selects no
            // axis. NumPy then returns the array itself; NumSharp returns a.view(), a fresh view object (as squeeze(NDArray)
            // does for an array with nothing to remove), so disposing the result never disposes the input.
            if (a.ndim == 0 && (axis == 0 || axis == -1))
                return a.view();

            // normalize_axis_index on the axis as given. The wrap-then-test loop this replaces never ended for a negative
            // axis of a 0-d input, and wrapped -ndim - 1 to the last axis.
            axis = Shape.check_and_adjust_axis(a.ndim, axis);

            if (a.Shape.dimensions[axis] != 1)
                throw new ValueError("cannot select an axis to squeeze out which has size not equal to one");

            // Stride-dropping VIEW (see squeeze(NDArray)) removing ONLY the named axis.
            return SqueezeView(a, a.ndim - 1, onlyAxis: axis);
        }

        /// <summary>
        ///     Builds the squeezed VIEW: dims and strides minus the removed length-one axes (all of them,
        ///     or only <paramref name="onlyAxis"/>), offset and buffer preserved — NumPy's
        ///     <c>PyArray_Squeeze</c>/<c>SelectedSqueeze</c>. Writeability/read-onlyness inherit through
        ///     <see cref="UnmanagedStorage.Alias(Shape)"/>.
        /// </summary>
        [NewSubtype("a")]
        private static NDArray SqueezeView(NDArray a, int keep, int onlyAxis)
        {
            var dims = a.Shape.dimensions;
            var strides = a.Shape.strides;

            if (keep == 0)
            {
                var scalar = new Shape(Array.Empty<long>(), Array.Empty<long>(), a.Shape.offset, a.Shape.BufferSize);
                return new NDArray(a.Storage.Alias(scalar));
            }

            var newDims = new long[keep];
            var newStrides = new long[keep];
            int j = 0;
            for (int i = 0; i < dims.Length; i++)
            {
                if (onlyAxis >= 0 ? i == onlyAxis : dims[i] == 1)
                    continue;
                newDims[j] = dims[i];
                newStrides[j] = strides[i];
                j++;
            }

            return new NDArray(a.Storage.Alias(new Shape(newDims, newStrides, a.Shape.offset, a.Shape.BufferSize)));
        }

        /// <summary>
        ///     Builds the squeezed VIEW for <see cref="SqueezeAxes"/>: dims and strides minus the axes
        ///     <paramref name="selected"/> marks (each of length one), offset and buffer preserved — NumPy's
        ///     <c>PyArray_SqueezeSelected</c>. Writeability/read-onlyness inherit through
        ///     <see cref="UnmanagedStorage.Alias(Shape)"/>.
        /// </summary>
        /// <param name="a">The input (its selected axes already checked to have length one).</param>
        /// <param name="selected">One flag per axis of <paramref name="a"/>: true for an axis to remove.</param>
        /// <param name="keep">The number of axes left (<c>a.ndim</c> minus the selected ones), 0 for a 0-d result.</param>
        /// <returns>A new view over <paramref name="a"/>'s buffer; NumSharp.Build gives a subclass operand its class back
        ///     (the <c>[NewSubtype]</c> rule, as for <see cref="SqueezeView"/>).</returns>
        [NewSubtype("a")]
        private static NDArray SqueezeSelectedView(NDArray a, bool[] selected, int keep)
        {
            var dims = a.Shape.dimensions;
            var strides = a.Shape.strides;

            // Every axis removed: the 0-d view of the one element, at the same offset.
            if (keep == 0)
            {
                var scalar = new Shape(Array.Empty<long>(), Array.Empty<long>(), a.Shape.offset, a.Shape.BufferSize);
                return new NDArray(a.Storage.Alias(scalar));
            }

            var newDims = new long[keep];
            var newStrides = new long[keep];
            int j = 0;
            for (int i = 0; i < dims.Length; i++)
            {
                if (selected[i])
                    continue;
                newDims[j] = dims[i];
                newStrides[j] = strides[i];
                j++;
            }

            return new NDArray(a.Storage.Alias(new Shape(newDims, newStrides, a.Shape.offset, a.Shape.BufferSize)));
        }

        /// <summary>
        ///     Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater than one, an error is raised.</param>
        /// <returns>The input array, but with all or a subset of the dimensions of length 1 removed. This is always a itself or a view into a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html</remarks>
        /// <exception cref="IncorrectShapeException">If axis is not None, and an axis being squeezed is not of length 1</exception>
        [MethodImpl(Inline)]
        internal static NDArray squeeze_fast(NDArray a, int axis)
        {
            return a.reshape(squeeze_fast(a.Shape, axis));
        }

        /// <summary>
        ///     Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater than one, an error is raised.</param>
        /// <returns>The input array, but with all or a subset of the dimensions of length 1 removed. This is always a itself or a view into a.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html</remarks>
        /// <exception cref="IncorrectShapeException">If axis is not None, and an axis being squeezed is not of length 1</exception>
        [MethodImpl(Inline)]
        internal static Shape squeeze_fast(Shape a, int axis)
        {
            var r = a.dimensions.RemoveAt(axis);
            // NumPy squeeze(axis) removes ONLY the named axis. Only collapse to 0-D when that was the
            // last remaining axis (r.Length == 0); a remaining length-1 dimension must be kept (e.g.
            // squeeze([1,1], axis=0) -> [1], not scalar) — over-collapsing it diverges from NumPy and
            // breaks the matmul 1-D-promotion squeeze.
            if (r.Length == 0)
                return Shape.Scalar;

            return new Shape(r, copy: false);
        }
    }
}
