using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Put values into the destination array by matching 1-D index and data slices — the
        ///     setter twin of <see cref="take_along_axis(NDArray,NDArray,int?)"/>. Iterates over
        ///     matching 1-D slices oriented along <paramref name="axis"/> in the index and data
        ///     arrays and uses the former to place <paramref name="values"/> into the latter.
        ///     In-place: <paramref name="arr"/> is modified and nothing is returned. Functions
        ///     returning an index along an axis, like <c>argsort</c> and <c>argmax</c>/<c>argmin</c>
        ///     with <c>keepdims</c>, produce suitable indices.
        /// </summary>
        /// <param name="arr">
        ///     Destination array <c>(Ni..., M, Nk...)</c>, modified in place. Must be writeable.
        /// </param>
        /// <param name="indices">
        ///     Integer index array <c>(Ni..., J, Nk...)</c>. Must match the dimension count of
        ///     <paramref name="arr"/>; the non-axis dimensions <c>Ni</c>/<c>Nk</c> need only
        ///     broadcast against <paramref name="arr"/>. <c>J</c> need not equal <c>M</c>.
        /// </param>
        /// <param name="values">
        ///     Values to write, cast to <paramref name="arr"/>'s dtype (assignment cast: floats
        ///     truncate toward zero). Broadcast — NOT cycled — to the indexing result shape
        ///     <c>(Ni..., J, Nk...)</c>: right-aligned, with extra leading size-1 dimensions
        ///     stripped. Where several index positions collapse onto one <paramref name="arr"/>
        ///     element (a size-1 <paramref name="arr"/> dimension, or duplicate indices in a slice),
        ///     the last write in C-order wins. <c>null</c> is Python's <c>None</c>, converted into
        ///     <paramref name="arr"/>'s dtype (NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c>, a <c>StringDType</c>
        ///     NA object; int()'s TypeError for an integer array, before the indices' bounds are checked).
        /// </param>
        /// <param name="axis">
        ///     The axis to place 1-D slices along (no default — required, as in NumPy). When
        ///     <c>null</c> the destination is treated as its C-order flattening: a <b>C-contiguous</b>
        ///     <paramref name="arr"/> is written back through that flat view (matching NumPy's
        ///     <c>np.array(arr.flat)</c>, which aliases contiguous storage), while a
        ///     <b>non-contiguous</b> <paramref name="arr"/> raises read-only — NumPy's
        ///     <c>np.array(arr.flat)</c> yields a read-only copy there, so the assignment fails
        ///     (probed 2.4.2). <paramref name="indices"/> must then be 1-D.
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="indices"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="values"/> is null (<c>None</c>) and <paramref name="arr"/> is an integer
        ///     (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError">
        ///     <paramref name="axis"/> is <c>null</c> and <paramref name="indices"/> is not 1-D; the
        ///     dimension counts of <paramref name="arr"/> and <paramref name="indices"/> differ; or
        ///     <paramref name="values"/> cannot broadcast to the indexing result shape.
        /// </exception>
        /// <exception cref="IndexError">
        ///     <paramref name="indices"/> is not an integer array; a non-axis dimension of
        ///     <paramref name="indices"/> conflicts with <paramref name="arr"/>; or an index is out
        ///     of bounds for the axis (after a single negative wrap). On out-of-bounds NOTHING is
        ///     written — every index is validated before the first store (NumPy's atomicity).
        /// </exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="arr"/>'s rank.</exception>
        /// <exception cref="ValueError"><paramref name="arr"/> (or the non-contiguous <c>axis=null</c> flat copy) is read-only — NumPy's "assignment destination is read-only".</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.put_along_axis.html</remarks>
        [NDScoped]
        public static void put_along_axis(NDArray arr, NDArray indices, NDArray values, int? axis)
        {
            using var plain = NDMatrixBoundary.PlainFor(arr, indices, values);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (indices is null) throw new ArgumentNullException(nameof(indices));
            // A null `values` is Python's None, converted where NumPy's fancy assignment converts values (NDArray.PutAlongAxisCore).

            // NumPy's numpy.put_along_axis: the statements are the array object's (NDArray.ShapeBase.cs)
            NDArray.put_along_axis(arr, indices, values, axis);
        }
    }
}
