using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Take values from the input array by matching 1-D index and data slices.
        ///     Iterates over matching 1-D slices oriented along <paramref name="axis"/> in the
        ///     index and data arrays, using the former to look up values in the latter. These
        ///     slices can be different lengths. Functions returning an index along an axis, like
        ///     <see cref="argsort(NDArray, int?, string, object, bool?)"/> and <c>argmax</c>/<c>argmin</c> with
        ///     <c>keepdims</c>, produce suitable indices.
        /// </summary>
        /// <param name="arr">Source array <c>(Ni..., M, Nk...)</c>.</param>
        /// <param name="indices">
        ///     Integer index array <c>(Ni..., J, Nk...)</c>. Must match the dimension count of
        ///     <paramref name="arr"/>; the non-axis dimensions <c>Ni</c>/<c>Nk</c> only need to
        ///     broadcast against <paramref name="arr"/>.
        /// </param>
        /// <param name="axis">
        ///     The axis to take 1-D slices along (default <c>-1</c>, matching NumPy 2.3+). When
        ///     <c>null</c> the source is treated as if first flattened to 1-D in C-order, for
        ///     consistency with <c>sort</c>/<c>argsort</c>; then <paramref name="indices"/> must
        ///     be 1-D.
        /// </param>
        /// <returns>
        ///     A fresh C-contiguous array of shape <c>(Ni..., J, Nk...)</c> (the broadcast of the
        ///     non-axis dimensions, with <c>J = indices.shape[axis]</c>) and dtype of
        ///     <paramref name="arr"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray take_along_axis(NDArray arr, NDArray indices, int? axis = -1)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            // NumPy's numpy.take_along_axis: the statements are the array object's (NDArray.ShapeBase.cs)
            return NDArray.take_along_axis(arr, indices, axis);
        }
    }
}
