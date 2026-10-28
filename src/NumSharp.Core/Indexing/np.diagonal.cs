using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return specified diagonals of <paramref name="a"/>. For a 2-D
        ///     array, returns the diagonal as a 1-D array. For an N-D array,
        ///     the last two axes (default <paramref name="axis1"/>=0,
        ///     <paramref name="axis2"/>=1) define the 2-D sub-arrays from which
        ///     diagonals are taken; the diagonal is appended as the last axis
        ///     of the returned array.
        /// </summary>
        /// <param name="a">Source array. Must have at least 2 dimensions.</param>
        /// <param name="offset">
        ///     Offset of the diagonal from the main diagonal. Positive values
        ///     refer to diagonals above the main, negative below. Default 0.
        /// </param>
        /// <param name="axis1">
        ///     First axis of the 2-D sub-array. Default 0.
        /// </param>
        /// <param name="axis2">
        ///     Second axis of the 2-D sub-array. Default 1.
        /// </param>
        /// <returns>
        ///     A <strong>read-only view</strong> sharing storage with
        ///     <paramref name="a"/>. Shape: <c>a.shape</c> with
        ///     <paramref name="axis1"/> and <paramref name="axis2"/> removed
        ///     and the diagonal appended as the last axis.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html
        ///     <para>
        ///     Mirrors NumPy's <c>PyArray_Diagonal</c> (item_selection.c). The
        ///     view trick: combining the two strides into one
        ///     <c>stride[axis1] + stride[axis2]</c> walks the diagonal in one
        ///     step. Read-only by NumPy contract (the writeable-by-default
        ///     change pencilled in for NumPy 1.10 hasn't shipped in 2.x).
        ///     </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="a"/> has fewer than two dimensions ("diag requires an array of at
        ///     least two dimensions"), or the two axes are the same axis ("axis1 and axis2 cannot be the same").</exception>
        /// <exception cref="AxisError">An axis is out of range: "axis1: axis …" / "axis2: axis …" (axis1 checked
        ///     first).</exception>
        public static NDArray diagonal(NDArray a, int offset = 0, int axis1 = 0, int axis2 = 1)
        {
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((a, offset, axis1, axis2), static s => diagonal(s.a, s.offset, s.axis1, s.axis2));
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (a is null) throw new ArgumentNullException(nameof(a));

            // NumPy's diagonal IS PyArray_Diagonal, a read-only view built from the strides: the array object's
            // (NDArray.Diagonal over Shape.Diagonal's layout; a view computes no value, so it is no engine member).
            return NDArray.Diagonal(a, offset, axis1, axis2);
        }
    }
}
