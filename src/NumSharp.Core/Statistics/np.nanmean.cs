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
        /// default) computes the mean of the flattened array.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one.</param>
        /// <param name="dtype">The type of the sum (NumPy's <c>dtype=</c>). For a float or complex input it must be inexact;
        /// null uses the input's dtype (np.mean's float64 for an integer or bool input).</param>
        /// <returns>The mean, in <paramref name="dtype"/> or the input's dtype (float64 for an integer or bool input); a 0-d
        /// array for a flat reduction without <paramref name="keepdims"/>. A slice with only NaNs is NaN.</returns>
        /// <exception cref="TypeError">A float or complex <paramref name="a"/> with a non-inexact <paramref name="dtype"/>:
        /// "If a is inexact, then dtype must be inexact".</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <paramref name="a"/>'s dimensions. A 0-d float or
        /// complex array takes axis 0 and -1 as a flat reduction (NumPy's nanmean reads the axis through np.sum).</exception>
        /// <remarks>
        ///     <para>
        ///     An array that cannot hold NaN (integer, bool, char, decimal) has no NaN to ignore: the result is
        ///     <c>np.mean</c>'s, keywords included. For a float or complex array the arithmetic is NumPy's
        ///     <c>_nanfunctions_impl.nanmean</c> (see <c>DefaultEngine.NanMeanNumPy</c>): the NaN-free sum in the input's
        ///     dtype (a float16 or float32 input sums in its own dtype, not in double), then the division by the int64 count,
        ///     which for complex is NumPy's complex division (an infinite part turns the other part into NaN).
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html</para>
        /// </remarks>
        [NDScoped]
        public static NDArray nanmean(NDArray a, int? axis = null, bool keepdims = false, DType dtype = null)
            => DefaultEngine.NanMeanNumPy(a, axis, keepdims, dtype);
    }
}
