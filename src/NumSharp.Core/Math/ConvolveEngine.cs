namespace NumSharp
{
    /// <summary>
    ///     NumPy's <c>np.convolve</c> (numpy/_core/numeric.py), the operation behind <see cref="np.convolve"/> and
    ///     <see cref="NDArray.convolve"/>: NumPy's Python statements, then <c>multiarray.correlate(a, v[::-1], mode)</c>,
    ///     which is the engine member <see cref="Backends.TensorEngine.Correlate"/>.
    /// </summary>
    /// <remarks>
    ///     NumPy writes np.convolve in Python, so its body is an internal operation class (the np-layer dtype plan, D1); the
    ///     dtype logic it reaches (the mode, the text answers, the promotion, the kernels) is the C function's, in the
    ///     engine. Moved from <c>NDArray.convolve</c> in P2.
    /// </remarks>
    internal static class ConvolveEngine
    {
        /// <summary>
        ///     NumPy's np.convolve statements: the 0-d promotion of <c>array(x, copy=None, ndmin=1)</c>, the two emptiness
        ///     checks, the swap that makes the longer sequence the data (convolution is commutative, so no reversal of the
        ///     result follows), and the correlation of the data with the reversed kernel.
        /// </summary>
        /// <param name="a">The first sequence (any dtype, any layout; a 0-d one is one element).</param>
        /// <param name="v">The second sequence (likewise).</param>
        /// <param name="mode"><c>full</c>, <c>same</c> or <c>valid</c>.</param>
        /// <returns>A fresh 1-D array of the operands' common dtype: the discrete linear convolution.</returns>
        /// <exception cref="ValueError"><c>a cannot be empty</c> / <c>v cannot be empty</c> (on the operands as given,
        ///     before the swap), then the correlation's own errors (<see cref="Backends.TensorEngine.Correlate"/>: the mode
        ///     comes AFTER these checks here, as NumPy parses it inside <c>multiarray.correlate</c>).</exception>
        /// <exception cref="System.ArgumentException">An unknown <paramref name="mode"/>.</exception>
        /// <exception cref="System.NullReferenceException">A null <paramref name="mode"/>.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> operand's refusal or first-product error.</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy
        ///     computes.</exception>
        /// <remarks>The 0-d promotions and the reversed view are temporaries the [NDScoped] weaver releases when the call
        ///     ends; the operands themselves are never tracked, and the correlation's result is returned.</remarks>
        [NDScoped]
        internal static NDArray Convolve(NDArray a, NDArray v, string mode)
        {
            // a, v = array(a, copy=None, ndmin=1), array(v, copy=None, ndmin=1): a 0-d operand becomes one element.
            if (a.ndim == 0)
                a = a.reshape(1);
            if (v.ndim == 0)
                v = v.reshape(1);

            // if len(a) == 0 / if len(v) == 0: NumPy convolve's own texts (distinct from the correlation's), on the
            // operands AS GIVEN, so an empty first operand reports `a` even when the second is longer.
            if (a.Shape.dimensions[0] == 0)
                throw new ValueError("a cannot be empty");
            if (v.Shape.dimensions[0] == 0)
                throw new ValueError("v cannot be empty");

            // if len(v) > len(a): a, v = v, a — by len(), the FIRST dimension.
            if (v.Shape.dimensions[0] > a.Shape.dimensions[0])
                (a, v) = (v, a);

            // return multiarray.correlate(a, v[::-1], mode): the mode, the one-dimension check and every dtype decision
            // happen there.
            return a.TensorEngine.Correlate(a, v["::-1"], mode);
        }
    }
}
