using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Cross-correlation of two 1-dimensional sequences:
        /// <c>c_k = sum_n a_{n+k} * conj(v_n)</c> (NumPy signal-processing convention).
        /// </summary>
        /// <param name="v">The second one-dimensional input array.</param>
        /// <param name="mode">'valid', 'same', or 'full'. Default is 'valid' (unlike convolve, which defaults to 'full').</param>
        /// <returns>Discrete cross-correlation of a and v.</returns>
        /// <remarks>
        /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.correlate.html
        ///
        /// Port of NumPy's <c>PyArray_Correlate2</c> (numpy/_core/src/multiarray/multiarraymodule.c), the engine member
        /// <see cref="Backends.TensorEngine.Correlate2"/>: the second argument is complex-conjugated first (real inputs
        /// unchanged); the operands are then swapped when <c>len(a) &lt; len(v)</c> — correlate is NOT commutative — and the
        /// output is reversed in that case (<c>_pyarray_revert</c>). The kernel is read forward. Float bit-parity notes are
        /// in <c>np.correlate.cs</c>.
        /// </remarks>
        /// <exception cref="ValueError">First an unknown <paramref name="mode"/> (NumPy's two texts, the bad argument in
        ///     Python's <c>repr</c>); then <c>object of too small depth for desired array</c> (a 0-d operand) or <c>object too
        ///     deep for desired array</c> (two or more dimensions), <c>first array argument cannot be empty</c> /
        ///     <c>second array argument cannot be empty</c>, or — for a fixed-width string operand, after every check
        ///     above — <c>data type must provide an itemsize</c>.</exception>
        /// <exception cref="UnicodeEncodeError">A <paramref name="mode"/> holding a lone surrogate.</exception>
        /// <exception cref="TypeError">A null <paramref name="mode"/> (NumPy's <c>None</c>): <c>convolve/correlate mode not
        ///     understood</c>. A <c>StringDType</c> operand: NumPy's legacy-only descriptor refusal (a <c>StringDType</c>
        ///     first operand, or a <c>U</c> one beside it), or Python's error from the first product of the OBJECT fallback
        ///     (<c>can't multiply sequence by non-int of type 'float'</c>).</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy computes (an
        ///     object array, which NumSharp does not have).</exception>
        // Scope: the facade keeps its scope over the call, so the weaver roots this array and v at the public entry (the
        // lifetime oracle's gap LT22: an unrooted operand could be finalized while the kernels read its buffer); the
        // engine's correlation reclaims its own temporaries.
        [NDScoped]
        public NDArray correlate(NDArray v, string mode = "valid")
            => TensorEngine.Correlate2(this, v, mode);
    }
}
