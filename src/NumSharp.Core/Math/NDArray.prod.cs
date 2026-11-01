using System;
using System.Diagnostics.CodeAnalysis;

namespace NumSharp
{
    [SuppressMessage("ReSharper", "ParameterHidesMember")]
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>ndarray.prod(axis=None, dtype=None, out=None, *, keepdims=&lt;no value&gt;, initial=&lt;no value&gt;,
        ///     where=&lt;no value&gt;)</c> (<c>_methods._prod</c>, i.e. <c>np.multiply.reduce</c>): the product of the
        ///     elements over <paramref name="axis"/>. This is the only overload: NumPy's whole signature (np-layer dtype plan
        ///     P10, D10); the parameters behave as <see cref="np.prod"/>'s.
        /// </summary>
        /// <param name="axis">The axis to multiply over (negative counts from the end); <c>null</c> multiplies every
        ///     element.</param>
        /// <param name="dtype">The accumulator and result dtype, or <c>null</c> for NumPy's default.</param>
        /// <param name="out">The array the product is written into and returned. Footgun: the THIRD positional parameter, as
        ///     in NumPy, so write <c>keepdims: true</c> by name (a positional bool there is a C# scalar passed as out and
        ///     raises).</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every product starts from instead of 1, or <c>null</c>.</param>
        /// <param name="where">A bool mask of the elements to multiply, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) for a
        ///     product over every axis without <paramref name="keepdims"/>.</returns>
        /// <exception cref="TypeError">A C# scalar passed as <paramref name="out"/>; an <paramref name="initial"/> the loop
        ///     cannot take; text (NumPy's refusal).</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>; a mask with more axes; an
        ///     unconvertible <paramref name="initial"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> or <paramref name="where"/> does not broadcast.</exception>
        /// <exception cref="OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>The member keeps the plain-array meaning for a numpy.matrix object reached through an
        ///     <see cref="NDArray"/> reference.</remarks>
        public NDArray prod(int? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false,
            object initial = null, NDArray @where = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return TensorEngine.ReduceProduct(this, axis, keepdims, dtype, @out, initial, @where);
        }
    }
}
