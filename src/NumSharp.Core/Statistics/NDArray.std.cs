using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Backends;

namespace NumSharp
{
    [SuppressMessage("ReSharper", "ParameterHidesMember")]
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>ndarray.std(axis=None, dtype=None, out=None, ddof=0, *, keepdims=&lt;no value&gt;,
        ///     where=&lt;no value&gt;, mean=&lt;no value&gt;)</c> (<c>_methods._std</c>): the standard deviation over
        ///     <paramref name="axis"/>. This is the only overload: NumPy's whole signature (np-layer dtype plan P10, D10); the
        ///     parameters behave as <see cref="np.std"/>'s. There is no <c>correction=</c>: NumPy 2.4.2's method refuses it;
        ///     <see cref="np.std"/> takes it.
        /// </summary>
        /// <param name="axis">The axis to compute along (negative counts from the end); <c>null</c> uses every element.</param>
        /// <param name="dtype">The loop of both sums, or <c>null</c> for NumPy's default.</param>
        /// <param name="out">The float or complex array the result is written into and returned. Footgun: the THIRD positional
        ///     parameter, as in NumPy; write <c>ddof:</c> and <c>keepdims:</c> by name.</param>
        /// <param name="ddof">Delta degrees of freedom: the variance divides by <c>max(N - ddof, 0)</c>.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="where">A bool mask of the elements each output takes, or <c>null</c>.</param>
        /// <param name="mean">The mean to subtract as given (see <see cref="np.var"/>), or <c>null</c> to compute it.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) when the
        ///     result has no dimensions.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="dtype"/> or <paramref name="out"/> with a
        ///     result that has dimensions (NumPy's sqrt output-cast error); a non-bool <paramref name="where"/>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/>, <paramref name="where"/> or
        ///     <paramref name="mean"/> does not broadcast.</exception>
        /// <exception cref="TypeError">A boolean subtraction; a C# scalar passed as <paramref name="out"/>; NumPy's text
        ///     refusals.</exception>
        /// <remarks>The member keeps the plain-array meaning for a numpy.matrix object reached through an
        ///     <see cref="NDArray"/> reference.</remarks>
        public NDArray std(int? axis = null, DType dtype = null, NDArray @out = null, int ddof = 0, bool keepdims = false,
            NDArray @where = null, NDArray mean = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return TensorEngine.ReduceStd(this, axis, keepdims, ddof, dtype, @out, @where, mean);
        }
    }
}
