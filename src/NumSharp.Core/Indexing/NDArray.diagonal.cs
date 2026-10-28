using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return specified diagonals. Refer to <see cref="np.diagonal(NDArray, int, int, int)"/> for full documentation.
        /// </summary>
        /// <param name="offset">Offset of the diagonal from the main diagonal. Can be positive or negative. Defaults to 0.</param>
        /// <param name="axis1">Axis to be used as the first axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to 0.</param>
        /// <param name="axis2">Axis to be used as the second axis of the 2-D sub-arrays from which the diagonals should be taken. Defaults to 1.</param>
        /// <returns>A view onto the requested diagonal(s).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.diagonal.html</remarks>
        public NDArray diagonal(int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return np.diagonal(this, offset, axis1, axis2);
        }

        /// <summary>
        ///     The read-only view <c>np.diagonal</c> returns: <paramref name="a"/>'s storage under the layout
        ///     <see cref="Shape.Diagonal(int, int, int)"/> computes. Library code that needs a diagonal calls this (np.trace's
        ///     engine member does); it applies no class boundary, so a <c>numpy.char.chararray</c> or <c>numpy.matrix</c>
        ///     operand gives a plain view, where the public <see cref="np.diagonal(NDArray, int, int, int)"/> keeps the
        ///     class.
        /// </summary>
        /// <param name="a">The array (at least 2-D).</param>
        /// <param name="offset">The diagonal's offset from the main one: positive above it (along
        ///     <paramref name="axis2"/>), negative below (along <paramref name="axis1"/>).</param>
        /// <param name="axis1">The first axis of the 2-D sub-arrays (negative counts from the end).</param>
        /// <param name="axis2">The second axis of the 2-D sub-arrays (negative counts from the end).</param>
        /// <returns>A new array that shares <paramref name="a"/>'s storage and engine and never owns its data, read-only:
        ///     <paramref name="a"/>'s shape without the two axes, the diagonal (of length 0 when the offset leaves none) as
        ///     the last axis. No element is read or copied.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="a"/> has fewer than two dimensions ("diag requires an array of at
        ///     least two dimensions"), or the two axes are the same axis ("axis1 and axis2 cannot be the same").</exception>
        /// <exception cref="AxisError">An axis is out of range: "axis1: axis …" / "axis2: axis …" (axis1 checked
        ///     first).</exception>
        /// <remarks>NumPy's <c>PyArray_Diagonal</c>, the C function behind np.diagonal and ndarray.diagonal. It was the
        ///     engine member <c>TensorEngine.Diagonal</c> (the np-layer dtype plan's P1b) until phase PE's E7 returned it to
        ///     the array object before 0.71.0 shipped: building a view computes no value, so an engine has nothing to
        ///     override.</remarks>
        internal static NDArray Diagonal(NDArray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            // Alias builds a storage over a's block under the diagonal's layout, its base chained to a's owner (so the
            // view never owns its data); the new NDArray takes its own counted reference on that block, and computes with
            // a's engine, as every view does.
            return new NDArray(a.Storage.Alias(a.Shape.Diagonal(offset, axis1, axis2))) { TensorEngine = a.TensorEngine };
        }
    }
}
