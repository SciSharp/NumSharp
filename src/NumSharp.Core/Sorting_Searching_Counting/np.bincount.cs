using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Count the number of occurrences of each non-negative integer in <paramref name="x"/>.
        ///     <para/>
        ///     With no <paramref name="weights"/>, <c>bincount(x)[i]</c> is the number of times the
        ///     value <c>i</c> appears in <paramref name="x"/> (dtype <c>int64</c>). With
        ///     <paramref name="weights"/>, <c>bincount(x, w)[i]</c> is the sum of <c>w[j]</c> for all
        ///     <c>j</c> where <c>x[j] == i</c> (dtype <c>float64</c>). The output length is
        ///     <c>max(x.max() + 1, minlength)</c>.
        /// </summary>
        /// <param name="x">
        ///     1-D array of non-negative integers. Integer/bool/char dtypes are cast to <c>int64</c>;
        ///     a float/complex/decimal input raises (matching NumPy's behaviour for an actual
        ///     <c>ndarray</c> of a non-integer dtype).
        /// </param>
        /// <param name="weights">
        ///     Optional 1-D array of weights, same length as <paramref name="x"/>, cast to
        ///     <c>float64</c>. When present the result is a weighted sum per bin instead of a count.
        /// </param>
        /// <param name="minlength">Minimum length of the output array. Must be non-negative.</param>
        /// <returns>
        ///     A 1-D array: <c>int64</c> counts when <paramref name="weights"/> is null, otherwise
        ///     <c>float64</c> weighted sums.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="x"/>, or with points the weights, is not 1-D (NumPy's
        ///     depth check: <c>object of too small depth for desired array</c>, <c>object too deep for desired array</c>;
        ///     its ValueError under the house dimension-error type).</exception>
        /// <exception cref="InvalidCastException"><paramref name="x"/> is not an integer, bool or char array, or with points
        ///     the weights are complex or text (NumPy's safe-cast TypeError text).</exception>
        /// <exception cref="ValueError">NumPy's three ValueErrors, in its order: a negative <paramref name="minlength"/>
        ///     (<c>'minlength' must not be negative</c>), a negative element (<c>'list' argument must have no negative
        ///     elements</c>), weights of another length (<c>The weights and list don't have the same length.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.bincount.html <para>The body is
        ///     <see cref="TensorEngine.Bincount"/> (the np-layer dtype plan); this facade computes a <c>numpy.matrix</c>
        ///     operand as a plain array.</para></remarks>
        [NDScoped]
        public static NDArray bincount(NDArray x, NDArray weights = null, int minlength = 0)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            using var plain = NDMatrixBoundary.PlainFor(x, weights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return x.TensorEngine.Bincount(x, weights, minlength);
        }
    }
}
