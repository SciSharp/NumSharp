using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the weighted average along the specified axis.
        ///     Equivalent to <c>sum(a * weights) / sum(weights)</c>. When <paramref name="weights"/>
        ///     is null this reduces to <see cref="mean(NDArray)"/> over the same axes.
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
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return AverageOfMatrix(a, axis, weights, keepdims);
            if (a is null) throw new ArgumentNullException(nameof(a));
            using var plain = NDMatrixBoundary.PlainFor(weights);   // matrix weights compute plain (NDMatrixBoundary)
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return a.TensorEngine.Average(a, axisArr, weights, returned: false, keepdims: keepdims).avg;
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
            // a numpy.matrix operand: the matrix with every reduced axis kept, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return AverageOfMatrix(a, axis, weights, keepdims);
            if (a is null) throw new ArgumentNullException(nameof(a));
            using var plain = NDMatrixBoundary.PlainFor(weights);   // matrix weights compute plain (NDMatrixBoundary)
            return a.TensorEngine.Average(a, axis, weights, returned: false, keepdims: keepdims).avg;
        }

        /// <summary>
        ///     <c>np.average</c> of a <c>numpy.matrix</c> along one axis or none: NumPy calls <c>a.mean(axis)</c>
        ///     (<c>np.multiply(a, w).sum(axis)</c> with weights), the matrix method, so the result is the scalar
        ///     without an axis, a 1 x N row for axis 0 and an N x 1 column for axis 1.
        /// </summary>
        /// <param name="a">The matrix operand.</param>
        /// <param name="axis">The axis, or <c>null</c>.</param>
        /// <param name="weights">The weights, or <c>null</c>.</param>
        /// <param name="keepdims">The caller's keepdims; <c>true</c> is refused as NumPy's matrix method refuses it.</param>
        /// <returns>The average with NumPy's class and shape.</returns>
        /// <exception cref="TypeError"><paramref name="keepdims"/> is <c>true</c>: <c>matrix.mean() got an
        ///     unexpected keyword argument 'keepdims'</c> (<c>matrix.sum()</c> with weights).</exception>
        private static NDArray AverageOfMatrix(NDArray a, int? axis, NDArray weights, bool keepdims)
            => NDMatrixBoundary.Reduced((a, axis, weights, keepdims), static s => average(s.a, s.axis, s.weights, s.keepdims),
                axis, keepdims, weights is null ? "mean" : "sum");

        /// <summary>
        ///     <c>np.average</c> of a <c>numpy.matrix</c> over a tuple of axes: a matrix reduction over a TUPLE
        ///     keeps every reduced axis with length 1 (<c>matrix._collapse</c> returns the kept result for any axis
        ///     that is not None; probed 2.4.2: <c>np.average(m, axis=(0, 1))</c> is a 1 x 1 matrix). Without axes
        ///     it is the scalar of the single-axis form.
        /// </summary>
        /// <param name="a">The matrix operand.</param>
        /// <param name="axis">The axes, or <c>null</c>.</param>
        /// <param name="weights">The weights, or <c>null</c>.</param>
        /// <param name="keepdims">The caller's keepdims; <c>true</c> is refused as NumPy's matrix method refuses it.</param>
        /// <returns>The average with NumPy's class and shape.</returns>
        /// <exception cref="TypeError"><paramref name="keepdims"/> is <c>true</c>: <c>matrix.mean() got an
        ///     unexpected keyword argument 'keepdims'</c> (<c>matrix.sum()</c> with weights).</exception>
        private static NDArray AverageOfMatrix(NDArray a, int[] axis, NDArray weights, bool keepdims)
        {
            string method = weights is null ? "mean" : "sum";
            if (axis is null)
                return NDMatrixBoundary.Reduced((a, weights, keepdims), static s => average(s.a, (int[])null, s.weights, s.keepdims), null, keepdims, method);
            if (keepdims)
                throw NDMatrixBoundary.KeywordError(method, "keepdims");
            return NDMatrixBoundary.Wrapped((a, axis, weights), static s => average(s.a, s.axis, s.weights, keepdims: true));
        }

        /// <summary>
        ///     The pair <c>np.average(m, ..., returned=True)</c> gives for a <c>numpy.matrix</c> operand: the average
        ///     NumPy's matrix rule returned, and the sum of weights as a PLAIN array of the average's shape
        ///     (NumPy: <c>if scl.shape != avg.shape: scl = np.broadcast_to(scl, avg.shape).copy()</c>; probed
        ///     2.4.2: <c>np.average(m, axis=0, returned=True)</c> is a 1 x N matrix and a 1 x N ndarray).
        /// </summary>
        /// <param name="avg">The average, already computed through <c>np.average</c>'s matrix rule (a matrix, or a
        ///     0-d scalar without an axis). Returned as it is.</param>
        /// <param name="a">The matrix operand.</param>
        /// <param name="axis">The axes, or <c>null</c>.</param>
        /// <param name="weights">The weights, or <c>null</c>.</param>
        /// <returns>The average and the sum of weights.</returns>
        /// <remarks>
        ///     The sum of weights comes from a second, plain computation (the matrix path pays one extra pass; the
        ///     plain path is untouched). A matrix <paramref name="weights"/> makes NumPy's sum of weights a matrix
        ///     as well; here it is always a plain array. It raises nothing of its own: the plain computation repeats,
        ///     with the same operand, axes and weights, the checks that the pass producing <paramref name="avg"/>
        ///     already passed.
        /// </remarks>
        private static (NDArray avg, NDArray sumOfWeights) AverageReturnedOfMatrix(NDArray avg, NDArray a, int[] axis, NDArray weights)
        {
            using var plain = NDMatrixBoundary.Inside();   // the calls below compute plain: this method shapes the pair itself
            var (plainAvg, scl) = a.TensorEngine.Average(a, axis, weights, returned: true, keepdims: false);
            plainAvg.Dispose();   // only the sum of weights is wanted from this pass
            if (scl.Shape.Equals(avg.Shape) || SameDimensions(scl.Shape, avg.Shape))
                return (avg, scl);

            // Same elements in the average's shape: a reshape when the counts agree (a row or a column of sums),
            // else NumPy's broadcast (the element count repeated for every output cell).
            Shape target = Shape.Adopt((long[])avg.Shape.dimensions.Clone());
            NDArray view = scl.size == avg.size ? scl.reshape(target) : np.broadcast_to(scl, target);
            NDArray shaped = view.copy();
            if (!ReferenceEquals(view, scl))
                view.Dispose();
            scl.Dispose();
            return (avg, shaped);
        }

        /// <summary>Whether two shapes have the same dimensions (layout and flags apart).</summary>
        /// <param name="x">The first shape.</param>
        /// <param name="y">The second shape.</param>
        /// <returns>True when rank and every extent agree.</returns>
        private static bool SameDimensions(Shape x, Shape y)
        {
            long[] dx = x.dimensions ?? Array.Empty<long>(), dy = y.dimensions ?? Array.Empty<long>();
            if (dx.Length != dy.Length)
                return false;
            for (int i = 0; i < dx.Length; i++)
                if (dx[i] != dy[i])
                    return false;
            return true;
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
            if (NDMatrixBoundary.Sees(a))   // a numpy.matrix operand: np.average's matrix rule, then NumPy's sum of weights
#pragma warning disable NDW012 // not a leak: AverageReturnedOfMatrix returns its first argument as the pair's average
                return AverageReturnedOfMatrix(AverageOfMatrix(a, axis, weights, keepdims), a, axis.HasValue ? new[] { axis.Value } : null, weights);
#pragma warning restore NDW012
            if (a is null) throw new ArgumentNullException(nameof(a));
            using var plain = NDMatrixBoundary.PlainFor(weights);   // matrix weights compute plain (NDMatrixBoundary)
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return a.TensorEngine.Average(a, axisArr, weights, returned: true, keepdims: keepdims);
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
            if (NDMatrixBoundary.Sees(a))   // a numpy.matrix operand: np.average's matrix rule, then NumPy's sum of weights
#pragma warning disable NDW012 // not a leak: AverageReturnedOfMatrix returns its first argument as the pair's average
                return AverageReturnedOfMatrix(AverageOfMatrix(a, axis, weights, keepdims), a, axis, weights);
#pragma warning restore NDW012
            if (a is null) throw new ArgumentNullException(nameof(a));
            using var plain = NDMatrixBoundary.PlainFor(weights);   // matrix weights compute plain (NDMatrixBoundary)
            return a.TensorEngine.Average(a, axis, weights, returned: true, keepdims: keepdims);
        }
    }
}
