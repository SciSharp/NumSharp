// =============================================================================
// Default.Convolve.cs - DefaultEngine's np.convolve (numpy/_core/numeric.py)
// =============================================================================
//
// DefaultEngine's override of TensorEngine.Convolve (TensorEngine.Numeric.cs): NumPy's Python statements, then
// multiarray.correlate(a, v[::-1], mode), which is the engine member TensorEngine.Correlate. The dtype logic it reaches
// (the mode, the text answers, the promotion, the kernels) is the C function's, in Correlate.
//
// History: NumPy writes np.convolve in Python; the np-layer dtype plan's P2 moved its statements from NDArray.convolve
// into the internal operation class ConvolveEngine, and its phase PE (E1) made that class this DefaultEngine partial.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.Convolve"/>: NumPy's np.convolve statements, the 0-d promotion of
        ///     <c>array(x, copy=None, ndmin=1)</c>, the two emptiness checks, the swap that makes the longer sequence the data
        ///     (convolution is commutative, so no reversal of the result follows), and the correlation of the data with the
        ///     reversed kernel.
        /// </summary>
        /// <param name="a">The first sequence (any dtype, any layout; a 0-d one is one element).</param>
        /// <param name="v">The second sequence (likewise).</param>
        /// <param name="mode"><c>full</c>, <c>same</c> or <c>valid</c>.</param>
        /// <returns>A fresh 1-D array of the operands' common dtype: the discrete linear convolution.</returns>
        /// <exception cref="ValueError"><c>a cannot be empty</c> / <c>v cannot be empty</c> (on the operands as given,
        ///     before the swap), then the correlation's own errors (<see cref="TensorEngine.Correlate"/>: an unknown
        ///     mode comes AFTER these checks here, as NumPy parses it inside <c>multiarray.correlate</c>).</exception>
        /// <exception cref="UnicodeEncodeError">A <paramref name="mode"/> holding a lone surrogate (after the emptiness
        ///     checks).</exception>
        /// <exception cref="TypeError">A null <paramref name="mode"/> (NumPy's <c>None</c>, after the emptiness checks), or
        ///     a <c>StringDType</c> operand's refusal or first-product error.</exception>
        /// <exception cref="System.NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy
        ///     computes.</exception>
        /// <remarks>The 0-d promotions and the reversed view are temporaries the [NDScoped] weaver releases when the call
        ///     ends; the operands themselves are never tracked (the weaver roots them at this entry, so neither is finalized
        ///     while the kernels read it), and the correlation's result is returned.</remarks>
        [NDScoped]
        public override NDArray Convolve(NDArray a, NDArray v, string mode = "full")
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
