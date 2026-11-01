using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>ndarray.sum(axis=None, dtype=None, out=None, *, keepdims=&lt;no value&gt;, initial=&lt;no value&gt;,
        ///     where=&lt;no value&gt;)</c> (<c>_methods._sum</c>, i.e. <c>np.add.reduce</c>): the sum of the elements over
        ///     <paramref name="axis"/>. This is the only overload: NumPy's whole signature (np-layer dtype plan P10, D10); the
        ///     parameters behave as <see cref="np.sum"/>'s.
        /// </summary>
        /// <param name="axis">The axis to sum over (negative counts from the end); <c>null</c> sums every element.</param>
        /// <param name="dtype">The accumulator and result dtype, or <c>null</c> for NumPy's default (NEP 50's accumulator;
        ///     <c>promote_types(out, this)</c> with an out).</param>
        /// <param name="out">The array the sum is written into and returned. Footgun: the THIRD positional parameter, as in
        ///     NumPy, so write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every sum starts from instead of 0 (see <see cref="np.sum"/>), or <c>null</c>.</param>
        /// <param name="where">A bool mask of the elements to sum, or <c>null</c> (the C# <c>true</c> literal is NumPy's
        ///     default).</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) for a sum
        ///     over every axis without <paramref name="keepdims"/>.</returns>
        /// <exception cref="TypeError">A C# scalar passed as <paramref name="out"/>; an <paramref name="initial"/> the loop
        ///     cannot take; NumPy's text refusals.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>; a mask with more axes; an
        ///     unconvertible <paramref name="initial"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> or <paramref name="where"/> does not broadcast.</exception>
        /// <exception cref="OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>The member keeps the plain-array meaning for a numpy.matrix object reached through an
        ///     <see cref="NDArray"/> reference (an <see cref="NDMatrix"/> reference binds <see cref="NDMatrix.sum(int?, DType, NDMatrix)"/>).</remarks>
        public NDArray sum(int? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false,
            object initial = null, NDArray @where = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return TensorEngine.ReduceAdd(this, axis, keepdims, dtype, @out, initial, @where);
        }
    }
}
