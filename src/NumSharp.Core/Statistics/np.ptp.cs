using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.ptp(a, axis=None, out=None, keepdims=False)</c>: the range of values (maximum - minimum) along
        ///     the given axes, every axis by default. Equivalent to <c>np.amax(a, axis) - np.amin(a, axis)</c>; the dtype is
        ///     preserved, so unsigned/signed integer overflow wraps the same way NumPy does
        ///     (e.g. <c>ptp(uint8[0,255]) == 255</c>, <c>ptp(int8[-128,127]) == -1</c>).
        /// </summary>
        /// <param name="a">The array (numbers or StringDType; fixed-width text is refused as NumPy refuses it).</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) reduces every axis; one axis (negative
        ///     counts from the end; a 0-d array accepts 0 and -1) or a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) reduces
        ///     those axes, and the empty tuple reduces none (the result is <c>a - a</c>).</param>
        /// <param name="out">An optional output of the result's shape; the result is written into it and it is
        ///     returned.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise the range, a fresh array (read-only when 0-d:
        ///     NumPy's scalar); a <c>numpy.matrix</c> operand gives a matrix.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ptp.html — NumPy's whole function
        ///     (<c>_methods._ptp</c>: <c>subtract(maximum.reduce(a, axis, out, keepdims), minimum.reduce(a, axis, keepdims),
        ///     out)</c>) is the engine's <see cref="TensorEngine.Ptp(NDArray, int?, NDArray, bool)"/> for None or one axis and
        ///     <see cref="TensorEngine.Ptp(NDArray, int[], NDArray, bool)"/> for a tuple (the np-layer dtype plan, P1b and
        ///     PE).
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces the <c>int?</c> and <c>int[]</c>
        ///     overloads, which made <c>np.ptp(a, null)</c> ambiguous (CS0121) and had no tuple literal.</para></remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range (a 0-d input has no axis for a tuple to name).</exception>
        /// <exception cref="TypeError">Fixed-width text, or a text out (NumPy's maximum.reduce refuses the loop).</exception>
        /// <exception cref="ValueError">An <paramref name="out"/> that _ptp's <c>maximum.reduce(a, axis, out=out)</c>
        ///     refuses in its prologue, before anything is computed: of the wrong rank (<c>output parameter for reduction
        ///     operation maximum has the wrong number of dimensions: Found 2 but expected 1</c>) or with a length-1 axis
        ///     where the result is longer (parity QA-31); a tuple that names an axis twice (<c>duplicate value in
        ///     'axis'</c>); a read-only out; an empty reduction (no identity); a <c>numpy.matrix</c> operand whose result
        ///     cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="IncorrectShapeException">An <paramref name="out"/> whose axes do not broadcast with the
        ///     result's (<c>operands could not be broadcast together with remapped shapes [original-&gt;remapped]: (3,)-&gt;(3)
        ///     (3,4) </c>), NumPy's ValueError under the house type of a broadcast failure.</exception>
        [NDScoped]
        public static NDArray ptp(NDArray a, NDAxis? axis = null, NDArray @out = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's ptp is _methods._ptp: subtract(maximum.reduce(a, axis, out, keepdims), minimum.reduce(a, axis,
            // keepdims), out). The whole function, its text routes included, is the engine's: a tuple takes the tuple
            // member (maximum.reduce's tuple checks first), None and an integer the single-axis member (a 0-d operand
            // accepts the integer 0 / -1). Both declare the same class rule (__array_wrap__ of a), so a matrix operand
            // gets its matrix either way.
            return axis is { IsTuple: true } tuple
                ? a.TensorEngine.Ptp(a, tuple.ToArray(), @out, keepdims)
                : a.TensorEngine.Ptp(a, axis is { } single ? single[0] : (int?)null, @out, keepdims);
        }
    }
}
