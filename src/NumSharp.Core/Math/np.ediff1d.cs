using System;
using System.Collections.Generic;
using NumSharp.Backends;

namespace NumSharp
{
    // ============================== np.ediff1d ==============================
    // The differences between consecutive elements of a (flattened) array.
    //
    //   ediff1d(ary) == ary.ravel()[1:] - ary.ravel()[:-1]
    //
    // NumPy 2.4.2 reference: numpy/lib/_arraysetops_impl.py::ediff1d
    //
    // Differences from np.diff:
    //   * Always operates on the C-order flattened (1-D) input.
    //   * The result is always 1-D.
    //   * The result dtype is forced to the input dtype; to_begin / to_end are
    //     cast to it under the NumPy `same_kind` rule (else a TypeError).
    //   * Unlike diff, ediff1d uses subtract for every dtype with NO not_equal
    //     special case, so a boolean input raises (matching NumPy, which rejects
    //     boolean subtraction).
    //
    // The subtract is backed by NumSharp's SIMD IL kernel (TensorEngine.Subtract);
    // ediff1d itself contains no per-element loop.
    public static partial class np
    {
        /// <summary>
        ///     The differences between consecutive elements of an array. The
        ///     input is flattened first; the result is always 1-D.
        /// </summary>
        /// <param name="ary">Input array (flattened before differencing).</param>
        /// <param name="to_end">
        ///     Number(s) to append to the end of the returned differences.
        ///     <c>null</c> means none. Cast to <paramref name="ary"/>'s dtype
        ///     under the <c>same_kind</c> casting rule.
        /// </param>
        /// <param name="to_begin">
        ///     Number(s) to prepend to the beginning of the returned differences.
        ///     <c>null</c> means none. Cast to <paramref name="ary"/>'s dtype
        ///     under the <c>same_kind</c> casting rule.
        /// </param>
        /// <returns>
        ///     1-D array of consecutive differences (input dtype), optionally
        ///     bracketed by <paramref name="to_begin"/> and <paramref name="to_end"/>.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="ary"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="to_begin"/> or <paramref name="to_end"/> cannot be
        ///     cast to <paramref name="ary"/>'s dtype under <c>same_kind</c>.</exception>
        /// <exception cref="TypeError"><paramref name="ary"/> is boolean (NumPy's boolean-subtract TypeError).</exception>
        /// <exception cref="TypeError"><paramref name="ary"/> holds fixed-width strings: NumPy's subtract has no string
        ///     loop (<c>ufunc 'subtract' did not contain a loop with signature matching types (dtype('&lt;U2'),
        ///     dtype('&lt;U2')) -&gt; None</c>), raised after the <paramref name="to_begin"/> / <paramref name="to_end"/>
        ///     checks, as NumPy raises it.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ediff1d.html
        ///     <para>Scoped by <c>[NDScoped]</c>: the ravel, the prepared <paramref name="to_begin"/> /
        ///     <paramref name="to_end"/> copies, the slice views of the fallback path and the middle differences
        ///     that a concatenation supersedes are released on return, exceptions included, and the result is
        ///     yielded. The body no longer repeats that bookkeeping with a hand-written try/finally.</para>
        /// </remarks>
        [NDScoped]
        public static NDArray ediff1d(NDArray ary, object to_end = null, object to_begin = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(ary);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return np.EngineOf(ary).Ediff1d(ary, to_end, to_begin);
        }
    }
}
