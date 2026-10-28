// The functions of numpy/_core/numeric.py that compute: isclose, allclose, argwhere, flatnonzero, count_nonzero
// and correlate (with NumPy's multiarray.correlate / correlate2 behind it).

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>multiarray.correlate(a, v, mode)</c> — <c>PyArray_Correlate</c>, the correlation <c>np.convolve</c>
        ///     ends with (<c>multiarray.correlate(a, v[::-1], mode)</c>) — the whole call: the mode, the operands'
        ///     descriptor lookup (a <c>StringDType</c> operand's answers), their one-dimension check,
        ///     <c>_pyarray_correlate</c>'s emptiness checks, the result allocation (a fixed-width string type number has no
        ///     item size) and the sliding product <c>c_k = Σ_n a[n + k] · v[n]</c> in the operands' common dtype. Unlike
        ///     <see cref="Correlate2"/> it neither conjugates <paramref name="v"/> nor reverses the result when the operands
        ///     are swapped (the longer one is always the data).
        /// </summary>
        /// <param name="a">The first one-dimensional sequence (any dtype, any layout).</param>
        /// <param name="v">The second one-dimensional sequence, read forward: np.convolve passes its kernel reversed.</param>
        /// <param name="mode"><c>valid</c> (the default), <c>same</c> or <c>full</c>, matched exactly.</param>
        /// <returns>A fresh 1-D array of the operands' common dtype, of <c>max(n1, n2) - min(n1, n2) + 1</c>,
        ///     <c>max(n1, n2)</c> or <c>n1 + n2 - 1</c> elements.</returns>
        /// <exception cref="ValueError">First an unknown <paramref name="mode"/>, with NumPy's two texts: <c>Use one of
        ///     'valid', 'same', or 'full' for convolve/correlate mode</c> for a text whose first letter names a mode,
        ///     <c>mode must be one of 'valid', 'same', or 'full' (got 'x')</c> with the argument in Python's <c>repr</c> for
        ///     any other. Then an operand that is not one-dimensional (<c>object of too small depth for desired array</c> for
        ///     a 0-d one, <c>object too deep for desired array</c> for two or more dimensions, the first operand first), an
        ///     empty one (<c>first array argument cannot be empty</c> / <c>second array argument cannot be empty</c>), or a
        ///     fixed-width string operand after those checks (<c>data type must provide an itemsize</c>).</exception>
        /// <exception cref="UnicodeEncodeError">A <paramref name="mode"/> holding a lone surrogate: CPython's UTF-8
        ///     encoding error, raised before the mode is read.</exception>
        /// <exception cref="TypeError">A null <paramref name="mode"/> (NumPy's <c>None</c>): <c>convolve/correlate mode not
        ///     understood</c>. A <c>StringDType</c> operand: NumPy's legacy-only descriptor refusal, or Python's error from
        ///     the first product of its OBJECT fallback.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy computes (an
        ///     object array, which NumSharp does not have).</exception>
        /// <remarks>NumPy's function is C (<c>multiarraymodule.c</c>), so the call is an engine member; np.convolve's Python
        ///     statements (the 0-d promotion, its own emptiness checks, the swap by length, the reversal) run before it in
        ///     <c>ConvolveEngine</c>.</remarks>
        public abstract NDArray Correlate(NDArray a, NDArray v, string mode = "valid");

        /// <summary>
        ///     NumPy's <c>multiarray.correlate2(a, v, mode)</c> — <c>PyArray_Correlate2</c>, the whole body of
        ///     <c>np.correlate</c> — the whole call: <see cref="Correlate"/>'s statements, with the second operand conjugated
        ///     for a complex common dtype before the product (<c>c_k = Σ_n a[n + k] · conj(v[n])</c>) and, when the operands
        ///     are swapped because <paramref name="a"/> is the shorter, the result reversed (<c>_pyarray_revert</c>), so
        ///     <c>correlate2(v, a)</c> is the reversed conjugate of <c>correlate2(a, v)</c>.
        /// </summary>
        /// <param name="a">The first one-dimensional sequence (any dtype, any layout).</param>
        /// <param name="v">The second one-dimensional sequence (conjugated for a complex common dtype).</param>
        /// <param name="mode"><c>valid</c> (the default), <c>same</c> or <c>full</c>, matched exactly.</param>
        /// <returns>A fresh 1-D array of the operands' common dtype (lengths as for <see cref="Correlate"/>).</returns>
        /// <exception cref="ValueError">An unknown <paramref name="mode"/>, then an operand that is not one-dimensional, an
        ///     empty one, or a fixed-width string one (the texts and order of <see cref="Correlate"/>).</exception>
        /// <exception cref="UnicodeEncodeError">A <paramref name="mode"/> holding a lone surrogate.</exception>
        /// <exception cref="TypeError">A null <paramref name="mode"/> (<c>convolve/correlate mode not understood</c>), or a
        ///     <c>StringDType</c> operand's refusal or first-product error.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy computes.</exception>
        /// <remarks>NumPy's function is C, so the call is an engine member; the numpy.matrix boundary stays with
        ///     <c>np.correlate</c>.</remarks>
        public abstract NDArray Correlate2(NDArray a, NDArray v, string mode = "valid");

        public abstract bool AllClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);
        public abstract NDArray<bool> IsClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false);

        public abstract NDArray<long> FlatNonZero(NDArray a);

        public abstract NDArray Argwhere(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.count_nonzero(a)</c> (<c>multiarray.count_nonzero</c>), the whole call: the number of elements
        ///     whose truth value is True (a text element counts when non-empty), 0 for an array with no element.
        /// </summary>
        /// <param name="a">The array (any dtype, any layout).</param>
        /// <returns>The count.</returns>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <remarks>The public <c>np.count_nonzero</c> only opens the matrix boundary's plain region and calls this (the
        ///     np-layer dtype plan, P1b).</remarks>
        public abstract long CountNonZero(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.count_nonzero(a, axis, keepdims=)</c>, the whole call: <c>a_bool.sum(axis=axis, dtype=intp,
        ///     keepdims=keepdims)</c> over the truth values (text through <c>a != ''</c>), int64 zeros of the reduced shape for
        ///     an array with no element.
        /// </summary>
        /// <param name="a">The array (any dtype, any layout).</param>
        /// <param name="axis">The counted axis (negative counts from the end; a 0-d array accepts 0 and -1).</param>
        /// <param name="keepdims">Whether the counted axis stays with length one.</param>
        /// <returns>The int64 counts, laid out as NumPy's sum allocates them.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range (the axis as given).</exception>
        /// <remarks>The public <c>np.count_nonzero</c> only opens the matrix boundary's plain region and calls this (the
        ///     np-layer dtype plan, P1b).</remarks>
        public abstract NDArray CountNonZero(NDArray a, int axis, bool keepdims = false);
    }
}
