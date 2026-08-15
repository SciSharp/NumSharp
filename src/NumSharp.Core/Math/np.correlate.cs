using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Cross-correlation of two 1-dimensional sequences.
        ///     <para><c>c_k = sum_n a_{n+k} * conj(v_n)</c> — the signal-processing convention.</para>
        /// </summary>
        /// <param name="a">First one-dimensional input sequence.</param>
        /// <param name="v">Second one-dimensional input sequence (complex-conjugated internally).</param>
        /// <param name="mode">'valid' (default), 'same', or 'full'. Note the default is 'valid', unlike <see cref="convolve"/> which defaults to 'full'.</param>
        /// <returns>Discrete cross-correlation of <paramref name="a"/> and <paramref name="v"/>.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.correlate.html
        ///
        /// Port of NumPy 2.4.2's <c>PyArray_Correlate2</c> + <c>_pyarray_correlate</c> +
        /// <c>small_correlate</c>. correlate is non-commutative: when <c>len(a) &lt; len(v)</c> the
        /// operands are swapped and the output time-reversed, and a complex <paramref name="v"/> is
        /// conjugated — so <c>correlate(v, a)</c> is the time-reversed conjugate of
        /// <c>correlate(a, v)</c>.
        ///
        /// <para>
        /// The output dtype is <c>result_type(a, v)</c>. The sliding multiply-accumulate is
        /// outer-vectorized over output positions with SIMD for the 10 <c>Vector&lt;T&gt;</c>-capable
        /// dtypes (byte/sbyte/int/uint families + float32/float64; char via ushort) and scalar for
        /// Half/Complex/Decimal/Boolean, sharing one engine with <see cref="convolve"/>.
        /// </para>
        ///
        /// <para>
        /// <b>Float bit-parity.</b> Each output accumulates <c>sum_t a[i+t]*k[t]</c> in t-order with
        /// separate multiply-then-add (never FMA), matching NumPy's <c>small_correlate</c> exactly —
        /// so the result is byte-identical to NumPy for the small-kernel regime. For LONG float
        /// kernels NumPy routes the middle/ramps through cblas <c>sdot</c>/<c>ddot</c> (a different
        /// summation order); NumSharp.Core has no BLAS, so those cases carry a bounded-ULP
        /// divergence — a pre-existing property of the scalar sum, NOT introduced by the SIMD path
        /// (the SIMD kernel is bit-identical to the scalar sum for every dtype and size). Integer,
        /// bool, complex, decimal and Half outputs are exact.
        /// </para>
        /// </remarks>
        public static NDArray correlate(NDArray a, NDArray v, string mode = "valid")
            => a.correlate(v, mode);
    }
}
