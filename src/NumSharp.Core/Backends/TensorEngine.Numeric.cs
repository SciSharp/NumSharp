// The functions of numpy/_core/numeric.py that compute: isclose, allclose, argwhere, flatnonzero, count_nonzero,
// array_equal, array_equiv, convolve and correlate (with NumPy's multiarray.correlate / correlate2 behind those two),
// and cross.

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
        ///     <see cref="Convolve"/>.</remarks>
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

        /// <summary>
        ///     NumPy's <c>np.array_equal(a1, a2, equal_nan=False)</c>: whether two arrays have the same shape and equal
        ///     elements. No broadcasting (that is <see cref="ArrayEquiv"/>).
        /// </summary>
        /// <param name="a1">The first array, or null (NumSharp's null convention: two nulls are equal, a null and an array
        ///     are not).</param>
        /// <param name="a2">The second array, or null.</param>
        /// <param name="equal_nan">
        ///     Count NaNs at the same places as equal (a complex value is NaN when either part is; a <c>StringDType</c>
        ///     NaN-like NA is a NaN). Without it NaN equals nothing, so an array holding a NaN is not equal to itself.
        /// </param>
        /// <returns>True when the shapes are equal and every element compares equal.</returns>
        /// <exception cref="TypeError">
        ///     With <paramref name="equal_nan"/>, a fixed-width string operand and two different arrays of one shape (NumPy's
        ///     <c>isnan</c> has no string loop); two <c>StringDType</c> operands whose NA objects differ (their comparison
        ///     refuses the pair).
        /// </exception>
        /// <remarks>
        ///     NumPy writes array_equal in Python (<c>numpy/_core/numeric.py</c>); DefaultEngine's override runs those
        ///     statements for every dtype, text included, with a fused early-exiting equality scan for two dense operands of
        ///     one dtype. <c>np.array_equal</c> and the member <see cref="NDArray.array_equal(NDArray, bool)"/> call it after
        ///     the matrix boundary's plain region, on the engine of the first operand that is not null; an override replaces
        ///     both for every array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal
        ///     class <c>ArrayEqualEngine</c>.
        /// </remarks>
        public abstract bool ArrayEqual(NDArray a1, NDArray a2, bool equal_nan = false);

        /// <summary>
        ///     NumPy's <c>np.array_equiv(a1, a2)</c>: whether two arrays broadcast together and every element of the
        ///     broadcast comparison is equal.
        /// </summary>
        /// <param name="a1">The first array, or null (two nulls are equivalent, a null and an array are not).</param>
        /// <param name="a2">The second array, or null.</param>
        /// <returns>
        ///     True when the shapes broadcast and every element compares equal (NaN never does: there is no
        ///     <c>equal_nan</c>). An empty broadcast is equivalent.
        /// </returns>
        /// <exception cref="TypeError">Two <c>StringDType</c> operands whose NA objects differ.</exception>
        /// <remarks>
        ///     NumPy writes array_equiv in Python (<c>numpy/_core/numeric.py</c>). <c>np.array_equiv</c> calls this after the
        ///     matrix boundary's plain region, on the engine of the first operand that is not null; an override replaces it
        ///     for every array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal
        ///     class <c>ArrayEqualEngine</c>.
        /// </remarks>
        public abstract bool ArrayEquiv(NDArray a1, NDArray a2);

        /// <summary>
        ///     NumPy's <c>np.convolve(a, v, mode='full')</c>: the discrete linear convolution of two one-dimensional
        ///     sequences, <c>multiarray.correlate(a, v[::-1], mode)</c> after NumPy's Python statements.
        /// </summary>
        /// <param name="a">The first sequence (any dtype, any layout; a 0-d one is one element, NumPy's <c>ndmin=1</c>).</param>
        /// <param name="v">The second sequence (likewise). The longer of the two is the data, the other is reversed.</param>
        /// <param name="mode"><c>full</c> (the default), <c>same</c> or <c>valid</c>, matched exactly.</param>
        /// <returns>A fresh 1-D array of the operands' common dtype.</returns>
        /// <exception cref="ValueError">
        ///     <c>a cannot be empty</c> / <c>v cannot be empty</c> (on the operands as given, before the swap), then
        ///     <see cref="Correlate"/>'s errors in its order: an unknown <paramref name="mode"/>, an operand of two or more
        ///     dimensions, a fixed-width string operand.
        /// </exception>
        /// <exception cref="UnicodeEncodeError">A <paramref name="mode"/> holding a lone surrogate (after the emptiness
        ///     checks).</exception>
        /// <exception cref="TypeError">A null <paramref name="mode"/> (after the emptiness checks), or a <c>StringDType</c>
        ///     operand's refusal or first-product error.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> operand whose OBJECT product NumPy computes.</exception>
        /// <remarks>
        ///     NumPy writes convolve in Python (<c>numpy/_core/numeric.py</c>); its last statement is the C function behind
        ///     <see cref="Correlate"/>, whose kernels reproduce NumPy's dot products bit for bit, except long float products
        ///     without the OpenBLAS backend (a bounded ULP difference). Unlike <see cref="Correlate2"/>
        ///     a complex kernel is not conjugated. <c>np.convolve</c> and the member <see cref="NDArray.convolve"/> call it;
        ///     an override replaces both for every array whose engine it is (the engine of <paramref name="a"/>). New in
        ///     0.71.0 (the np-layer dtype plan, phase PE); it was the internal class <c>ConvolveEngine</c>.
        /// </remarks>
        public abstract NDArray Convolve(NDArray a, NDArray v, string mode = "full");

        /// <summary>
        ///     NumPy's <c>np.cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None)</c>: the cross product of (arrays of)
        ///     2- or 3-vectors taken along <paramref name="axisa"/> / <paramref name="axisb"/>, the other axes broadcast; a
        ///     2-vector's missing third component is zero, and two 2-vectors give only the z-component.
        /// </summary>
        /// <param name="a">The first vector(s), at least 1-D.</param>
        /// <param name="b">The second vector(s), at least 1-D.</param>
        /// <param name="axisa">The axis of <paramref name="a"/> that holds the vectors (negative counts from the end).</param>
        /// <param name="axisb">The axis of <paramref name="b"/> that holds the vectors.</param>
        /// <param name="axisc">The axis of the result that holds the 3-vectors; ignored when both inputs are 2-vectors.</param>
        /// <param name="axis">When given, the one axis of both inputs and the result; it overrides the other three.</param>
        /// <returns>A new array in <c>promote_types(a.dtype, b.dtype)</c> over the broadcast leading axes, plus the 3-vector
        ///     axis at <paramref name="axisc"/> unless both inputs are 2-vectors.</returns>
        /// <exception cref="NullReferenceException"><paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="ValueError">A 0-D input (<c>At least one array has zero dimension</c>), or a vector length
        ///     other than 2 or 3 (<c>incompatible dimensions for cross product</c>).</exception>
        /// <exception cref="AxisError">An axis out of range for its operand, reported with its <c>axisa</c> / <c>axisb</c> /
        ///     <c>axisc</c> prefix; <c>axisc</c> is checked against the result's rank right after the broadcast.</exception>
        /// <exception cref="IncorrectShapeException">Leading axes that do not broadcast (<c>np.broadcast</c>'s text),
        ///     raised before the dtype promotion.</exception>
        /// <exception cref="TypeError">A text operand: multiply has no string loop (raised after the shape checks).</exception>
        /// <remarks>DefaultEngine composes NumPy's statements over multiply, negative, subtract and moveaxis (no kernel of its
        ///     own). The 2-vector forms are deprecated in NumPy 2.0; the warning is not modelled.</remarks>
        public abstract NDArray Cross(NDArray a, NDArray b, int axisa = -1, int axisb = -1, int axisc = -1, int? axis = null);
    }
}
