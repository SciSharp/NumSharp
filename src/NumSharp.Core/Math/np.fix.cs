namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Round to nearest integer towards zero, element-wise. The rounded values keep the input's
        /// data-type (int/bool are returned unchanged; floats are truncated toward zero).
        /// </summary>
        /// <param name="x">Array to be rounded.</param>
        /// <param name="out">
        ///     A location into which the result is stored (NumPy's positional <c>out</c>). If provided it
        ///     must have a shape the input broadcasts to and a dtype the truncated result can be cast to
        ///     under <c>same_kind</c> (a float→int <c>out</c> raises, matching NumPy); the same instance
        ///     is returned. If <c>null</c>, a freshly allocated array is returned.
        /// </param>
        /// <returns>The element-wise truncation of <paramref name="x"/> (or <paramref name="out"/>).</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.fix.html
        /// <para>
        /// Port of NumPy's <c>numpy.fix</c> (<c>numpy/lib/_ufunclike_impl.py</c>), which in 2.4.2 is a
        /// pure delegation to <c>numpy.trunc(x, out=out)</c> — so <see cref="fix"/> is <see cref="trunc"/>
        /// by another name and inherits its behaviour exactly: dtype preservation, NaN/±inf pass-through,
        /// the sign of <c>-0.0</c>, and the error texts (a complex input and an incompatible <c>out</c>
        /// dtype both surface the underlying <c>trunc</c> ufunc — which is what NumPy's <c>fix</c> leaks
        /// too, since it also delegates). Unlike a true ufunc it exposes NO <c>where</c>/<c>dtype</c>
        /// parameters (NumPy's <c>fix</c> signature is only <c>fix(x, out=None)</c>).
        /// </para>
        /// </remarks>
        public static NDArray fix(NDArray x, NDArray @out = null)
            => trunc(x, @out);
    }
}
