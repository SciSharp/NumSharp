using System;

namespace NumSharp
{
    // ============================== lib/_function_base_impl.py: average ==============================
    // NumPy's np.average as a member of the array object: the facade (np.average) checks its arguments and calls it, and
    // the subclass rule of a matrix operand (the matrix METHOD's reduction) is declared here.
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>np.average(a, axis, weights, returned, keepdims)</c>: <c>a.mean(axis)</c> without weights, else
        ///     <c>np.multiply(a, wgt).sum(axis) / wgt.sum(axis)</c>, over the given axes (<c>null</c>: every axis), and with
        ///     <paramref name="returned"/> the sum of weights in the average's shape (the element count per average
        ///     without weights).
        /// </summary>
        /// <param name="a">The values (not null).</param>
        /// <param name="axis">The axes to average over, or <c>null</c> for every axis.</param>
        /// <param name="weights">The weights, or <c>null</c> for equal weights.</param>
        /// <param name="returned">Keep the sum of weights in the result (NumPy's tuple return).</param>
        /// <param name="keepdims">Keep the averaged axes with length 1.</param>
        /// <returns>The average (0-d over every axis) and, with <paramref name="returned"/>, the sum of weights - new
        ///     arrays.</returns>
        /// <exception cref="AxisError">An axis out of bounds.</exception>
        /// <exception cref="ZeroDivisionError">Weights that sum to zero, or an empty average without weights.</exception>
        /// <exception cref="ValueError">A repeated axis; weights inconsistent with the averaged axes.</exception>
        /// <exception cref="TypeError">Weights of another shape without an axis; text the reduction refuses.</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType.</exception>
        /// <remarks>
        ///     <para>A matrix operand's <c>mean</c> / <c>sum</c> is the matrix METHOD, which reduces with <c>keepdims=True</c>
        ///     and then <c>_collapse(axis)</c>: the plain view standing for a matrix (<see cref="NDSubclass.StandsForMatrix"/>)
        ///     keeps the reduced axes whenever an axis is given, and the computation's rule
        ///     (<see cref="NDSubclassRules.Average"/>) makes the class of the average - the scalar without an axis, the kept
        ///     axes as a matrix otherwise - and leaves the sum of weights plain (NumPy's
        ///     <c>np.broadcast_to(scl, avg_as_array.shape).copy()</c>), which the same keepdims already shaped like the
        ///     average; <see cref="NDSubclassRules.AverageCheck"/> refuses <c>keepdims</c> as the method does.</para>
        ///     <para>API-parity batch 11: one computation for both NumPy return shapes (it replaces <c>average</c> and
        ///     <c>average_returned</c>); a sum of weights the caller did not ask for is released here.</para>
        /// </remarks>
        [SubclassRule(typeof(NDSubclassRules), nameof(NDSubclassRules.Average), Check = nameof(NDSubclassRules.AverageCheck))]
        internal static np.AverageResult average(NDArray a, int[] axis, NDArray weights, bool returned, bool keepdims)
        {
            bool kept = keepdims || axis is not null && NDSubclass.StandsForMatrix(a);
            var (avg, sumOfWeights) = a.TensorEngine.Average(a, axis, weights, returned: returned, keepdims: kept);
            if (returned)
                return new np.AverageResult(avg, sumOfWeights);
            // The engine may hand back a sum of weights even when not asked (its own intermediate): only the average leaves.
            if (sumOfWeights is not null && !ReferenceEquals(sumOfWeights, avg))
                sumOfWeights.Dispose();
            return new np.AverageResult(avg, null);
        }
    }
}
