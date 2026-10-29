using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Sum of all array elements.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <returns>A scalar (0-d) sum of the flattened array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sum.html</remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static NDArray sum(NDArray a)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced(a, static s => sum(s), null, false, "sum")
            : a.TensorEngine.Sum(a, axis: null, dtype: null, keepdims: false);

        /// <summary>
        ///     Sum of array elements over the given axis.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis along which a sum is performed (negative counts from the last axis).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sum.html</remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static NDArray sum(NDArray a, int axis)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis), static s => sum(s.a, s.axis), axis, false, "sum")
            : a.TensorEngine.Sum(a, axis: axis, dtype: null, keepdims: false);

        /// <summary>
        ///     Sum of all array elements, optionally keeping the reduced dimensions.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as size-one dimensions.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sum.html</remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static NDArray sum(NDArray a, bool keepdims)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, keepdims), static s => sum(s.a, s.keepdims), null, keepdims, "sum")
            : a.TensorEngine.Sum(a, axis: null, dtype: null, keepdims: keepdims);

        /// <summary>
        ///     Sum of array elements over the given axis, optionally keeping the reduced dimensions.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis along which a sum is performed (null sums the flattened array).</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as size-one dimensions.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sum.html</remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        [NDScoped]
        public static NDArray sum(NDArray a, int? axis, bool keepdims)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims), static s => sum(s.a, s.axis, s.keepdims), axis, keepdims, "sum")
            : a.TensorEngine.Sum(a, axis: axis, dtype: null, keepdims: keepdims);

        /// <summary>
        ///     Sum of array elements in a given <paramref name="dtype"/> (the accumulator/return type).
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="dtype">The <see cref="DType"/> of the accumulator/return (a C# <see cref="System.Type"/>, an <see cref="NPTypeCode"/> or a NumPy dtype string all convert implicitly). By default NumPy's NEP50 accumulator rules apply (e.g. int32 → int64).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sum.html</remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static NDArray sum(NDArray a, DType dtype)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, dtype), static s => sum(s.a, s.dtype), null, false, "sum")
            : a.TensorEngine.Sum(a, axis: null, dtype: dtype, keepdims: false);

        /// <summary>
        ///     Sum of array elements over the given axis in a given <paramref name="dtype"/>.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis along which a sum is performed (null sums the flattened array).</param>
        /// <param name="dtype">The <see cref="DType"/> of the accumulator/return (implicit from <see cref="System.Type"/> / <see cref="NPTypeCode"/> / NumPy dtype string).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sum.html</remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static NDArray sum(NDArray a, int? axis, DType dtype)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, dtype), static s => sum(s.a, s.axis, s.dtype), axis, false, "sum")
            : a.TensorEngine.Sum(a, axis: axis, dtype: dtype, keepdims: false);

        /// <summary>
        ///     Sum of array elements over the given axis in a given <paramref name="dtype"/>, optionally keeping the reduced dimensions.
        /// </summary>
        /// <param name="a">Elements to sum.</param>
        /// <param name="axis">Axis along which a sum is performed (null sums the flattened array).</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as size-one dimensions.</param>
        /// <param name="dtype">The <see cref="DType"/> of the accumulator/return (implicit from <see cref="System.Type"/> / <see cref="NPTypeCode"/> / NumPy dtype string).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sum.html</remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>.</exception>
        public static NDArray sum(NDArray a, int? axis, bool keepdims, DType dtype)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims, dtype), static s => sum(s.a, s.axis, s.keepdims, s.dtype), axis, keepdims, "sum")
            : a.TensorEngine.Sum(a, axis: axis, dtype: dtype, keepdims: keepdims);
    }
}
