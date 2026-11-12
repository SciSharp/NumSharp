using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the arithmetic mean along the specified axis, ignoring NaNs: NumPy's <c>numpy.nanmean</c>, statement
        ///     for statement. The mean is the sum of the non-NaN elements divided by their count, per slice.
        /// </summary>
        /// <param name="a">The array whose mean is desired; any dtype and layout.</param>
        /// <param name="axis">The axis along which the means are computed (negative counts from the end); null (the
        /// default) computes the mean of the flattened array. A tuple (<c>(0, 2)</c>) reduces every axis it names
        ///     together, in one pass (NumPy's tuple axis; the empty tuple reduces none, and a tuple item gets no 0-d
        ///     allowance).</param>
        /// <param name="dtype">The type of the sum (NumPy's <c>dtype=</c>). For a float or complex input it must be inexact;
        /// null uses the input's dtype (np.mean's float64 for an integer or bool input).</param>
        /// <param name="out">NumPy's <c>out=</c>: the destination of the sum and the division (returned); for a float or
        /// complex input it must be inexact.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one.</param>
        /// <param name="where">NumPy's <c>where=</c>: the bool mask of the elements to include (the count and the sum read
        /// it), or null for all.</param>
        /// <returns>The mean, in <paramref name="dtype"/> or the input's dtype (float64 for an integer or bool input); a 0-d
        /// array for a flat reduction without <paramref name="keepdims"/>. A slice with only NaNs is NaN.</returns>
        /// <exception cref="TypeError">A float or complex <paramref name="a"/> with a non-inexact <paramref name="dtype"/>:
        /// "If a is inexact, then dtype must be inexact"; or with a non-inexact <paramref name="out"/>: "If a is inexact, then
        /// out must be inexact".</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <paramref name="a"/>'s dimensions. A 0-d float or
        /// complex array takes axis 0 and -1 as a flat reduction (NumPy's nanmean reads the axis through np.sum).</exception>
        /// <exception cref="ValueError">A tuple that names an axis twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     An array that cannot hold NaN (integer, bool, char, decimal) has no NaN to ignore: the result is
        ///     <c>np.mean</c>'s, keywords included. For a float or complex array the arithmetic is NumPy's
        ///     <c>_nanfunctions_impl.nanmean</c> (see <c>TensorEngine.NanMean</c>): the NaN-free sum in the input's
        ///     dtype (a float16 or float32 input sums in its own dtype, not in double), then the division by the int64 count,
        ///     which for complex is NumPy's complex division (an infinite part turns the other part into NaN).
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html</para>
        ///     <para>One signature in NumPy's order (API-parity batch 3, 2026-11-11): it replaces <c>nanmean(a, axis, keepdims, dtype)</c>, so <c>dtype</c> is the third
        ///     argument and a positional <c>keepdims</c>
        ///     is written <c>keepdims: true</c> (a bool in the dtype slot does not compile). A tuple axis waits for the
        ///     owner's axis type (OD1, decided: <c>NDAxis</c>).</para>
        /// </remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>, <c>initial=</c> or
        ///     <c>where=</c>: NumPy calls the matrix method, which has no such keyword (<c>matrix.sum() got an unexpected
        ///     keyword argument 'where'</c>).</exception>
        [NDScoped]
        public static NDArray nanmean(NDArray a, NDAxis? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false,
            NDArray @where = null)
            => a.TensorEngine.NanMean(a, axis, dtype, keepdims, @out, @where);
    }
}
