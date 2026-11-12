using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.average(a, axis=None, weights=None, returned=False, *, keepdims=False)</c> without
        ///     <c>returned</c>: the weighted average along the given axes, <c>sum(a * weights) / sum(weights)</c>. When
        ///     <paramref name="weights"/> is null this is the mean over the same axes.
        /// </summary>
        /// <param name="a">The values to average.</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) averages every axis; one axis (negative
        ///     counts from the end) or a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) averages those axes, and the empty
        ///     tuple averages none (a float copy of every element).</param>
        /// <param name="weights">The weights, or null for equal weights (the mean): of <paramref name="a"/>'s shape, or of
        ///     the averaged axes' extents in the order of <paramref name="axis"/> (NumPy transposes an unsorted tuple's
        ///     weights to the axes' order; then <paramref name="axis"/> is required). A <c>numpy.matrix</c> of weights beside
        ///     a plain operand computes as a plain array, where NumPy returns a matrix (its weight sum is the matrix
        ///     method).</param>
        /// <param name="keepdims">Keep the averaged axes with length 1, so the result broadcasts against
        ///     <paramref name="a"/>.</param>
        /// <returns>The average: float64 for an integer or bool operand, else the common dtype of the operand and the weights;
        ///     0-d over every axis. A <c>numpy.matrix</c> operand gives the matrix method's answer: the scalar without an
        ///     axis, otherwise a matrix that keeps the averaged axes with length 1 (a tuple's included).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of bounds (<c>axis: axis 2 is out of bounds for array of dimension
        ///     2</c>; NumPy names the argument, and checks every axis before the repeat check).</exception>
        /// <exception cref="ZeroDivisionError">Weights that sum to zero, or an empty average without weights (NumPy's
        ///     ZeroDivisionError).</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType: <c>StringDType</c> beside a
        ///     number or beside S.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); an axis that appears twice (<c>repeated axis in `axis` argument</c>); weights not of the averaged axes' extents (<c>Shape of weights must be consistent with shape of a along specified axis.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>. Weights of another shape without an axis (<c>Axis must be specified when shapes of a and weights differ.</c>). Text weights beside a number, or a text operand: NumPy's refusal of the weight sum's text result dtype, or of a text mean.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html (NumPy's <c>average</c>, whose body
        ///     <see cref="TensorEngine.Average"/> runs; <c>returned=True</c> is <see cref="average_returned"/>, C#'s typed
        ///     pair).
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces the <c>int?</c> and <c>int[]</c>
        ///     overloads, which made <c>np.average(a, null)</c> ambiguous (CS0121) and had no tuple literal.</para></remarks>
        public static NDArray average(NDArray a, NDAxis? axis = null, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's statements are the array object's (a matrix operand's method reduction is its declared rule, which
            // reads the axis as given: None or not). The axes go as an array, an int as NumPy's one-element tuple.
            return NDArray.average(a, axis?.ToArray(), weights, keepdims);
        }

        /// <summary>
        ///     <see cref="average(NDArray, NDAxis?, NDArray, bool)"/> with the axis as one <c>int</c>: the overload a
        ///     C# integer binds (its identity conversion beats NDAxis' implicit one), forwarding to the NDAxis form, so
        ///     every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="average(NDArray, NDAxis?, NDArray, bool)"/>
        public static NDArray average(NDArray a, int axis, NDArray weights = null, bool keepdims = false)
            => average(a, (NDAxis?)axis, weights, keepdims);

        /// <summary>
        ///     <see cref="average(NDArray, NDAxis?, NDArray, bool)"/> with the axes as an <c>int[]</c>: the overload an
        ///     <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to NDAxis),
        ///     forwarding to the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple (<c>new
        ///     int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="average(NDArray, NDAxis?, NDArray, bool)"/>
        public static NDArray average(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
            => average(a, (NDAxis?)axis, weights, keepdims);

        /// <summary>
        ///     NumPy's <c>np.average(..., returned=True)</c>: the weighted average along the given axes and the sum of the
        ///     weights, as the tuple <c>(avg, sum_of_weights)</c>. When <paramref name="weights"/> is null,
        ///     <c>sum_of_weights</c> is the number of elements per output cell (broadcast to the average's shape).
        /// </summary>
        /// <param name="a">The values to average.</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) averages every axis; one axis (negative
        ///     counts from the end) or a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) averages those axes, and the empty
        ///     tuple averages none (a float copy of every element).</param>
        /// <param name="weights">The weights, or null for equal weights (the mean): of <paramref name="a"/>'s shape, or of
        ///     the averaged axes' extents in the order of <paramref name="axis"/> (NumPy transposes an unsorted tuple's
        ///     weights to the axes' order; then <paramref name="axis"/> is required). A <c>numpy.matrix</c> of weights beside
        ///     a plain operand computes as a plain array, where NumPy returns a matrix (its weight sum is the matrix
        ///     method).</param>
        /// <param name="keepdims">Keep the averaged axes with length 1, so the results broadcast against
        ///     <paramref name="a"/>.</param>
        /// <returns>The average (see <see cref="average(NDArray, NDAxis?, NDArray, bool)"/>) and the sum of weights in the
        ///     average's dtype and shape (the element count per average without weights), a plain array for a matrix
        ///     operand too.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of bounds (<c>axis: axis 2 is out of bounds for array of dimension
        ///     2</c>; NumPy names the argument, and checks every axis before the repeat check).</exception>
        /// <exception cref="ZeroDivisionError">Weights that sum to zero, or an empty average without weights (NumPy's
        ///     ZeroDivisionError).</exception>
        /// <exception cref="DTypePromotionError">An operand and weights with no common DType: <c>StringDType</c> beside a
        ///     number or beside S.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>); an axis that appears twice (<c>repeated axis in `axis` argument</c>); weights not of the averaged axes' extents (<c>Shape of weights must be consistent with shape of a along specified axis.</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>: NumPy calls the matrix method, which has no such keyword. The text names that method, as in <c>matrix.sum() got an unexpected keyword argument 'keepdims'</c> for <c>np.sum</c>. Weights of another shape without an axis (<c>Axis must be specified when shapes of a and weights differ.</c>). Text weights beside a number, or a text operand: NumPy's refusal of the weight sum's text result dtype, or of a text mean.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html (NumPy's <c>average</c> with
        ///     <c>returned=True</c>, whose body <see cref="TensorEngine.Average"/> runs).
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces the <c>int?</c> and <c>int[]</c>
        ///     overloads, which made <c>np.average_returned(a, null)</c> ambiguous (CS0121) and had no tuple
        ///     literal.</para></remarks>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, NDAxis? axis = null, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's statements are the array object's (a matrix operand's method reduction is its declared rule).
            return NDArray.average_returned(a, axis?.ToArray(), weights, keepdims);
        }

        /// <summary>
        ///     <see cref="average_returned(NDArray, NDAxis?, NDArray, bool)"/> with the axis as one <c>int</c>: the
        ///     overload a C# integer binds (its identity conversion beats NDAxis' implicit one), forwarding to the
        ///     NDAxis form, so every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="average_returned(NDArray, NDAxis?, NDArray, bool)"/>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int axis, NDArray weights = null, bool keepdims = false)
            => average_returned(a, (NDAxis?)axis, weights, keepdims);

        /// <summary>
        ///     <see cref="average_returned(NDArray, NDAxis?, NDArray, bool)"/> with the axes as an <c>int[]</c>: the
        ///     overload an <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to
        ///     NDAxis), forwarding to the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple
        ///     (<c>new int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="average_returned(NDArray, NDAxis?, NDArray, bool)"/>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
            => average_returned(a, (NDAxis?)axis, weights, keepdims);
    }
}
