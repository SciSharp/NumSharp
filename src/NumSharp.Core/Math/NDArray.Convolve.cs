using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Returns the discrete, linear convolution of two one-dimensional sequences.
        ///
        /// The convolution operator is often seen in signal processing, where it models the effect of a linear time-invariant system on a signal[1]. In probability theory, the sum of two independent random variables is distributed according to the convolution of their individual distributions.
        ///
        /// If v is longer than a, the arrays are swapped before computation.
        /// </summary>
        /// <param name="v">The second one-dimensional input array.</param>
        /// <param name="mode">'full', 'same', or 'valid'. Default is 'full'.</param>
        /// <returns>Discrete, linear convolution of a and v.</returns>
        /// <remarks>
        /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
        ///
        /// convolve is <c>correlate(a, v[::-1], mode)</c> (numpy/_core/numeric.py): NumPy's Python statements run in this
        /// array's <see cref="TensorEngine.Convolve"/>, which reverses the kernel and calls the engine's correlation
        /// (<see cref="TensorEngine.Correlate"/>, the shared sliding multiply-accumulate kernels).
        /// Unlike correlate it does NOT conjugate a complex kernel and is commutative, so the
        /// "swap if v longer" needs no output reversal. See <c>np.correlate.cs</c> for the
        /// float bit-parity notes (both share the same kernel).
        /// </remarks>
        /// <exception cref="ValueError"><c>a cannot be empty</c> / <c>v cannot be empty</c> (on the operands as given, before
        ///     the length swap), then an unknown <paramref name="mode"/> (NumPy's two texts, the bad argument in Python's
        ///     <c>repr</c>), <c>object too deep for desired array</c> for an operand of two or more dimensions, or — for a
        ///     fixed-width string operand, after every check above — <c>data type must provide an itemsize</c> (NumPy's
        ///     result allocation for the string type number).</exception>
        /// <exception cref="UnicodeEncodeError">A <paramref name="mode"/> holding a lone surrogate.</exception>
        /// <exception cref="TypeError">A null <paramref name="mode"/> (NumPy's <c>None</c>): <c>convolve/correlate mode not
        ///     understood</c>. A <c>StringDType</c> operand (after the emptiness checks and the swap): NumPy's legacy-only
        ///     descriptor refusal, or Python's error from the first product of the OBJECT fallback.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy computes (an
        ///     object array, which NumSharp does not have).</exception>
        // Scope: the facade keeps its scope over the call, so the weaver roots this array and v at the public entry (the
        // lifetime oracle's gap LT22: an unrooted operand could be finalized while the kernels read its buffer);
        // the engine member (DefaultEngine's Convolve, [NDScoped]) reclaims its own temporaries.
        [NDScoped]
        public NDArray convolve(NDArray v, string mode = "full")
            => TensorEngine.Convolve(this, v, mode);
    }
}
