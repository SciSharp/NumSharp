using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Numerical positive, element-wise: NumPy's <c>np.positive(x)</c> (the unary <c>+x</c> operator), a fresh copy of
        ///     every element in the input's dtype.
        /// </summary>
        /// <returns>A fresh array of the input's dtype and shape (never the input itself), laid out as NumPy's ufunc allocates
        ///     it (an F-ordered input gives an F-ordered result, a broadcast one a C-ordered result).</returns>
        /// <exception cref="TypeError">A bool array: positive has identity loops for every numeric dtype EXCEPT bool
        ///     (NumPy's <c>ufunc 'positive' did not contain a loop with signature matching types &lt;class
        ///     'numpy.dtypes.BoolDType'&gt; -&gt; None</c>); or text, which has no string loop either (the class-named
        ///     text).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.positive.html — the whole call is the engine's
        ///     <see cref="TensorEngine.Positive"/>, as for <c>np.positive</c> and the unary <c>+</c> operator (the np-layer
        ///     dtype plan, P1c; the member checked bool and text itself and returned a Clone).</remarks>
        public NDArray positive() => TensorEngine.Positive(this);
    }
}
