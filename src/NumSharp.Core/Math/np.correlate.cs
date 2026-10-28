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
        /// Half/Complex/Decimal/Boolean, except that a LONG float32 / float64 / complex128 / float16
        /// product runs a blocked SIMD kernel on a host with AVX and FMA
        /// (<c>Backends/Default/Math/Sliding/SlidingDot.Long.cs</c>); one engine serves
        /// <see cref="convolve"/> too.
        /// </para>
        ///
        /// <para>
        /// <b>Float bit-parity.</b> Each output accumulates <c>sum_t a[i+t]*k[t]</c> in t-order with
        /// separate multiply-then-add (never FMA), matching NumPy's <c>small_correlate</c> exactly —
        /// so the result is byte-identical to NumPy for the small-kernel regime. Where NumPy reduces
        /// a position with its dotfunc instead, NumSharp reproduces each dotfunc's scalar regime
        /// byte for byte (<c>ddot</c> below 16 terms, <c>sdot</c> below 32, <c>zdotu</c> below 8,
        /// float16's <c>HALF_dot</c> at every length); a float32 / float64 / complex128 position at or
        /// past the dot's vector block carries a bounded-ULP divergence, because NumPy sums it in
        /// OpenBLAS's order — which <c>NumSharp.Interop.OpenBLAS</c> reproduces byte for byte when it
        /// is referenced. Integer, bool and decimal outputs are exact at every size (modular / exact
        /// arithmetic does not depend on the order).
        /// </para>
        ///
        /// <para>The body is NumPy's C function <c>multiarray.correlate2</c>, the engine member
        /// <see cref="Backends.TensorEngine.Correlate2"/>. A <c>numpy.matrix</c> operand is two-dimensional, so it is
        /// refused as too deep, as in NumPy.</para>
        /// </remarks>
        /// <exception cref="ArgumentException">An unknown <paramref name="mode"/> (NumPy's two mode texts).</exception>
        /// <exception cref="NullReferenceException">A null <paramref name="mode"/>.</exception>
        /// <exception cref="ValueError"><c>object of too small depth for desired array</c> (a 0-d operand) or <c>object too
        ///     deep for desired array</c> (two or more dimensions), <c>first array argument cannot be empty</c> /
        ///     <c>second array argument cannot be empty</c>, or — for a fixed-width string operand, after every check
        ///     above — <c>data type must provide an itemsize</c>.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> operand: NumPy's legacy-only descriptor refusal, or Python's error
        ///     from the first product of the OBJECT fallback.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy computes (an
        ///     object array, which NumSharp does not have).</exception>
        public static NDArray correlate(NDArray a, NDArray v, string mode = "valid")
        {
            using var plain = NDMatrixBoundary.PlainFor(a, v);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return a.TensorEngine.Correlate2(a, v, mode);
        }
    }
}
