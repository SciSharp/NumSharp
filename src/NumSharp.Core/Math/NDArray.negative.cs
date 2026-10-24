using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Numerical negative, element-wise: NumPy's <c>np.negative(x)</c> (the <c>-x</c> operator), a fresh array of
        ///     every element negated in the input's dtype. Unsigned integers wrap (<c>-1</c> of uint8 is 255), as NumPy's
        ///     loops do.
        /// </summary>
        /// <returns>A fresh array of the input's dtype and shape, laid out as NumPy's ufunc allocates it (an F-ordered
        ///     input gives an F-ordered result).</returns>
        /// <exception cref="TypeError">A bool array (NumPy's "The numpy boolean negative, the `-` operator, is not
        ///     supported, ..." text), or text: negative has no string loop (NumPy's class-named UFuncTypeError).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.negative.html — the whole call is the engine's
        ///     <see cref="TensorEngine.Negate"/>, as for <c>np.negative</c> and the unary <c>-</c> operator. Until the
        ///     np-layer dtype plan's P1c this member negated a C-ordered copy with its own per-dtype loop, which threw
        ///     <see cref="NotSupportedException"/> for the unsigned integers and char where NumPy wraps.
        /// </remarks>
        public NDArray negative() => TensorEngine.Negate(this);
    }
}
