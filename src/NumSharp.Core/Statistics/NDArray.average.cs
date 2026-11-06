using System;

namespace NumSharp
{
    // ============================== lib/_function_base_impl.py: average ==============================
    // NumPy's np.average as members of the array object: the facades (np.average, np.average_returned) check their
    // arguments and call these, and the subclass rule of a matrix operand (the matrix METHOD's reduction) is declared here.
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>np.average(a, axis, weights, keepdims)</c>: <c>a.mean(axis)</c> without weights, else
        ///     <c>np.multiply(a, wgt).sum(axis) / wgt.sum(axis)</c>, over the given axes (<c>null</c>: every axis).
        /// </summary>
        /// <param name="a">The values (not null).</param>
        /// <param name="axis">The axes to average over, or <c>null</c> for every axis.</param>
        /// <param name="weights">The weights, or <c>null</c> for equal weights.</param>
        /// <param name="keepdims">Keep the averaged axes with length 1.</param>
        /// <returns>A new array: the average (0-d over every axis).</returns>
        /// <exception cref="AxisError">An axis out of bounds.</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero, or an empty average without weights.</exception>
        /// <exception cref="ValueError">A repeated axis; weights inconsistent with the averaged axes.</exception>
        /// <exception cref="TypeError">Weights of another shape without an axis; text the reduction refuses.</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType.</exception>
        /// <remarks>
        ///     A matrix operand's <c>mean</c> / <c>sum</c> is the matrix METHOD, which reduces with <c>keepdims=True</c> and
        ///     then <c>_collapse(axis)</c>: the plain view standing for a matrix (<see cref="NDSubclass.StandsForMatrix"/>)
        ///     keeps the reduced axes whenever an axis is given, and the computation's rule
        ///     (<see cref="NDSubclassRules.Average"/>) makes the class - the scalar without an axis, the kept axes as a matrix
        ///     otherwise; <see cref="NDSubclassRules.AverageCheck"/> refuses <c>keepdims</c> as the method does.
        /// </remarks>
        [SubclassRule(typeof(NDSubclassRules), nameof(NDSubclassRules.Average), Check = nameof(NDSubclassRules.AverageCheck))]
        internal static NDArray average(NDArray a, int[] axis, NDArray weights, bool keepdims)
        {
            bool kept = keepdims || axis is not null && NDSubclass.StandsForMatrix(a);
            return a.TensorEngine.Average(a, axis, weights, returned: false, keepdims: kept).avg;
        }

        /// <summary>
        ///     NumPy's <c>np.average(..., returned=True)</c>: the average of <see cref="average(NDArray, int[], NDArray, bool)"/>
        ///     and the sum of weights in the average's shape (the element count per average without weights).
        /// </summary>
        /// <param name="a">The values (not null).</param>
        /// <param name="axis">The axes to average over, or <c>null</c> for every axis.</param>
        /// <param name="weights">The weights, or <c>null</c> for equal weights.</param>
        /// <param name="keepdims">Keep the averaged axes with length 1.</param>
        /// <returns>The average and the sum of weights, both new arrays.</returns>
        /// <exception cref="AxisError">An axis out of bounds.</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero, or an empty average without weights.</exception>
        /// <exception cref="ValueError">A repeated axis; weights inconsistent with the averaged axes.</exception>
        /// <exception cref="TypeError">Weights of another shape without an axis; text the reduction refuses.</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType.</exception>
        /// <remarks>
        ///     A matrix operand reduces with <c>keepdims=True</c> as in <see cref="average(NDArray, int[], NDArray, bool)"/>, so
        ///     the sum of weights already has the matrix average's shape - NumPy's
        ///     <c>np.broadcast_to(scl, avg_as_array.shape).copy()</c>, a plain array; the rule
        ///     (<see cref="NDSubclassRules.AverageReturned"/>) gives the average its class and leaves the sum plain.
        /// </remarks>
        [SubclassRule(typeof(NDSubclassRules), nameof(NDSubclassRules.AverageReturned), Check = nameof(NDSubclassRules.AverageCheck))]
        internal static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int[] axis, NDArray weights, bool keepdims)
        {
            bool kept = keepdims || axis is not null && NDSubclass.StandsForMatrix(a);
            return a.TensorEngine.Average(a, axis, weights, returned: true, keepdims: kept);
        }
    }
}
