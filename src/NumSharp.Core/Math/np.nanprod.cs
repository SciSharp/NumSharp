namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.
        /// </summary>
        /// <param name="a">Array containing numbers whose product is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the product is computed. The default is to compute the product of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the product, with NaN values treated as one.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nanprod.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.prod (no NaN values possible).
        /// </remarks>
        public static NDArray nanprod(NDArray a, int? axis = null, bool keepdims = false)
            => a.TensorEngine.NanProd(a, axis, keepdims);
    }
}
