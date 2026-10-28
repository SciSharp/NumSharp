using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Estimate a covariance matrix, given data and weights.
        ///
        ///     Covariance indicates the level to which two variables vary together. Given
        ///     N-dimensional samples <c>X = [x_1, x_2, ..., x_N]^T</c>, the covariance matrix
        ///     element <c>C_ij</c> is the covariance of <c>x_i</c> and <c>x_j</c>; <c>C_ii</c> is
        ///     the variance of <c>x_i</c>.
        /// </summary>
        /// <param name="m">
        ///     A 1-D or 2-D array containing multiple variables and observations. Each row
        ///     represents a variable, each column a single observation of all those variables
        ///     (see <paramref name="rowvar"/>).
        /// </param>
        /// <param name="y">An additional set of variables and observations, same form as <paramref name="m"/>.</param>
        /// <param name="rowvar">
        ///     If true (default) each row is a variable with observations in the columns.
        ///     Otherwise the relationship is transposed: each column is a variable, rows are observations.
        /// </param>
        /// <param name="bias">
        ///     Default normalization (false) is by <c>(N - 1)</c> (unbiased). If true, normalization
        ///     is by <c>N</c>. Overridden by <paramref name="ddof"/>.
        /// </param>
        /// <param name="ddof">
        ///     If not null, overrides the default implied by <paramref name="bias"/>. <c>ddof=1</c>
        ///     returns the unbiased estimate even when weights are given; <c>ddof=0</c> returns the
        ///     simple average.
        /// </param>
        /// <param name="fweights">1-D array of integer frequency weights (number of times each observation is repeated).</param>
        /// <param name="aweights">1-D array of observation vector weights (relative importance of each observation).</param>
        /// <param name="dtype">Data-type of the result. By default the result has at least float64 precision.</param>
        /// <returns>The covariance matrix of the variables.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="m"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="m"/> or <paramref name="y"/> has more than 2 dimensions; text variables
        ///     (or a text <paramref name="dtype"/> request) whose <paramref name="m"/> and <paramref name="y"/> have different
        ///     observation counts (concatenate's text); a negative weight; text that does not parse into a numeric
        ///     <paramref name="dtype"/>; a <c>StringDType</c> request over no observations (add.reduce's "no
        ///     identity").</exception>
        /// <exception cref="IncorrectShapeException">Numeric variables of <paramref name="m"/> and <paramref name="y"/> with
        ///     different observation counts (concatenate's text; NumPy's ValueError).</exception>
        /// <exception cref="TypeError">A non-integral frequency weight; text the mean refuses (add.reduce's text, the weighted
        ///     mean's class-only refusal of a sized string descriptor, the division of <c>StringDType</c> text by the
        ///     count).</exception>
        /// <exception cref="RuntimeError">A weight array that is multidimensional or does not have one weight per
        ///     observation.</exception>
        /// <exception cref="DTypePromotionError"><c>StringDType</c> variables without <paramref name="dtype"/>
        ///     (<c>result_type(m, y, float64)</c>), or a <c>StringDType</c> request with weights.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="dtype"/>: NumPy's in-place
        ///     <c>X -= avg[:, None]</c> cannot cast the float64 mean's difference back (its UFuncTypeError text).</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero (the weighted mean's "Weights sum to zero, can't
        ///     be normalized"; NumPy's ZeroDivisionError).</exception>
        /// <exception cref="OverflowException">An integer text outside an integer <paramref name="dtype"/>, or decimal
        ///     arithmetic out of decimal's range.</exception>
        /// <exception cref="UnicodeEncodeError">A U → S conversion of non-ASCII text.</exception>
        /// <exception cref="UnicodeDecodeError">An S → U conversion of non-ASCII bytes.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.cov.html</remarks>
        [NDScoped]
        public static NDArray cov(NDArray m, NDArray y = null, bool rowvar = true, bool bias = false,
            int? ddof = null, NDArray fweights = null, NDArray aweights = null, DType dtype = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(m, y, fweights);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (m is null) throw new ArgumentNullException(nameof(m));
            return m.TensorEngine.Cov(m, y, rowvar, bias, ddof, fweights, aweights, dtype);
        }
    }
}
