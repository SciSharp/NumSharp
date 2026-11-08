using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the weighted average along the specified axis.
        ///     Equivalent to <c>sum(a * weights) / sum(weights)</c>. When <paramref name="weights"/>
        ///     is null this reduces to <see cref="mean(NDArray, int?, DType, NDArray, bool, NDArray)"/> over the same axes.
        /// </summary>
        /// <param name="a">The values to average.</param>
        /// <param name="axis">The axis to average along (a negative axis counts from the end), or null for every axis.</param>
        /// <param name="weights">The weights, or null for equal weights (the mean): of <paramref name="a"/>'s shape, or 1-D
        ///     of the averaged axis' length (then <paramref name="axis"/> is required). A <c>numpy.matrix</c> of
        ///     weights beside a plain operand computes as a plain array, where NumPy returns a matrix (its weight sum
        ///     is the matrix method).</param>
        /// <param name="keepdims">Keep the averaged axis with length 1, so the result broadcasts against
        ///     <paramref name="a"/>.</param>
        /// <returns>The average: float64 for an integer or bool operand, else the common dtype of the operand and the weights;
        ///     0-d without an axis.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html (NumPy's <c>average</c>, whose body
        ///     <see cref="TensorEngine.Average"/> runs).</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds (<c>axis: axis 2 is out of bounds for array
        ///     of dimension 2</c>).</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero, or an empty average without weights (NumPy's
        ///     ZeroDivisionError).</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType: <c>StringDType</c> beside a
        ///     number or beside S.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); weights not of the averaged axis' length (<c>Shape of weights must be consistent with shape of a along specified axis.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>. Weights of another shape without an axis (<c>Axis must be specified when shapes of a and weights differ.</c>). Text weights beside a number, or a text operand: NumPy's refusal of the weight sum's text result dtype, or of a text mean.</exception>
        public static NDArray average(NDArray a, int? axis = null, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // NumPy's statements are the array object's (a matrix operand's method reduction is its declared rule).
            return NDArray.average(a, axisArr, weights, keepdims);
        }

        /// <summary>
        ///     Compute the weighted average along a tuple of axes. Equivalent to
        ///     <c>np.average(a, axis, weights, keepdims)</c> in NumPy with a tuple axis.
        /// </summary>
        /// <param name="a">The values to average.</param>
        /// <param name="axis">The axes to average over (negative ones count from the end; an empty array averages over no
        ///     axis), or null for every axis.</param>
        /// <param name="weights">The weights, or null for equal weights (the mean): of <paramref name="a"/>'s shape, or of
        ///     the averaged axes' extents in the order of <paramref name="axis"/> (NumPy transposes an unsorted tuple's
        ///     weights to the axes' order). A <c>numpy.matrix</c> of weights beside a plain operand computes as a plain
        ///     array, where NumPy returns a matrix (its weight sum is the matrix method).</param>
        /// <param name="keepdims">Keep the averaged axes with length 1, so the result broadcasts against
        ///     <paramref name="a"/>.</param>
        /// <returns>The average: float64 for an integer or bool operand, else the common dtype of the operand and the weights;
        ///     0-d over every axis.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html (NumPy's <c>average</c>, whose body
        ///     <see cref="TensorEngine.Average"/> runs).</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of bounds (<c>axis: axis 9 is out of bounds for array of dimension
        ///     2</c>).</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero, or an empty average without weights (NumPy's
        ///     ZeroDivisionError).</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType: <c>StringDType</c> beside a
        ///     number or beside S.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); an axis that appears twice (<c>repeated axis in `axis` argument</c>); weights not of the averaged axes' extents (<c>Shape of weights must be consistent with shape of a along specified axis.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>. Weights of another shape without an axis (<c>Axis must be specified when shapes of a and weights differ.</c>). Text weights beside a number, or a text operand: NumPy's refusal of the weight sum's text result dtype, or of a text mean.</exception>
        public static NDArray average(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's statements are the array object's (a matrix operand's method reduction is its declared rule).
            return NDArray.average(a, axis, weights, keepdims);
        }

        /// <summary>
        ///     Compute the weighted average and return a tuple <c>(avg, sum_of_weights)</c>.
        ///     Equivalent to <c>numpy.average(..., returned=True)</c>. When <paramref name="weights"/>
        ///     is null, <c>sum_of_weights</c> is the number of elements per output cell
        ///     (broadcast to the average's shape).
        /// </summary>
        /// <param name="a">The values to average.</param>
        /// <param name="axis">The axis to average along (a negative axis counts from the end), or null for every axis.</param>
        /// <param name="weights">The weights, or null for equal weights (the mean): of <paramref name="a"/>'s shape, or 1-D
        ///     of the averaged axis' length (then <paramref name="axis"/> is required). A <c>numpy.matrix</c> of
        ///     weights beside a plain operand computes as a plain array, where NumPy returns a matrix (its weight sum
        ///     is the matrix method).</param>
        /// <param name="keepdims">Keep the averaged axis with length 1, so the results broadcast against
        ///     <paramref name="a"/>.</param>
        /// <returns>The average (see <see cref="average(NDArray, int?, NDArray, bool)"/>) and the sum of weights in the
        ///     average's dtype and shape (the element count per average without weights).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html (NumPy's <c>average</c> with
        ///     <c>returned=True</c>, whose body <see cref="TensorEngine.Average"/> runs).</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds (<c>axis: axis 2 is out of bounds for array
        ///     of dimension 2</c>).</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero, or an empty average without weights (NumPy's
        ///     ZeroDivisionError).</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType: <c>StringDType</c> beside a
        ///     number or beside S.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); weights not of the averaged axis' length (<c>Shape of weights must be consistent with shape of a along specified axis.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>. Weights of another shape without an axis (<c>Axis must be specified when shapes of a and weights differ.</c>). Text weights beside a number, or a text operand: NumPy's refusal of the weight sum's text result dtype, or of a text mean.</exception>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int? axis = null, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // NumPy's statements are the array object's (a matrix operand's method reduction is its declared rule).
            return NDArray.average_returned(a, axisArr, weights, keepdims);
        }

        /// <summary>
        ///     Tuple-axis overload of <see cref="average_returned(NDArray, int?, NDArray, bool)"/>: the weighted average over
        ///     several axes and the sum of weights in the average's shape.
        /// </summary>
        /// <param name="a">The values to average.</param>
        /// <param name="axis">The axes to average over (negative ones count from the end; an empty array averages over no
        ///     axis), or null for every axis.</param>
        /// <param name="weights">The weights, or null for equal weights (the mean): of <paramref name="a"/>'s shape, or of
        ///     the averaged axes' extents in the order of <paramref name="axis"/>. A <c>numpy.matrix</c> of weights
        ///     beside a plain operand computes as a plain array, where NumPy returns a matrix (its weight sum is the
        ///     matrix method).</param>
        /// <param name="keepdims">Keep the averaged axes with length 1, so the results broadcast against
        ///     <paramref name="a"/>.</param>
        /// <returns>The average (see <see cref="average(NDArray, int[], NDArray, bool)"/>) and the sum of weights in the
        ///     average's dtype and shape (the element count per average without weights).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html (NumPy's <c>average</c> with
        ///     <c>returned=True</c>, whose body <see cref="TensorEngine.Average"/> runs).</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of bounds (<c>axis: axis 9 is out of bounds for array of dimension
        ///     2</c>).</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero, or an empty average without weights (NumPy's
        ///     ZeroDivisionError).</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType: <c>StringDType</c> beside a
        ///     number or beside S.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); an axis that appears twice (<c>repeated axis in `axis` argument</c>); weights not of the averaged axes' extents (<c>Shape of weights must be consistent with shape of a along specified axis.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>. Weights of another shape without an axis (<c>Axis must be specified when shapes of a and weights differ.</c>). Text weights beside a number, or a text operand: NumPy's refusal of the weight sum's text result dtype, or of a text mean.</exception>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's statements are the array object's (a matrix operand's method reduction is its declared rule).
            return NDArray.average_returned(a, axis, weights, keepdims);
        }
    }
}
