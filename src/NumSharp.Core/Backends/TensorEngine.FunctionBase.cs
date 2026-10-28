// The functions of numpy/lib/_function_base_impl.py that compute, as TensorEngine members: NumPy writes them in
// Python, DefaultEngine runs those statements in NumPy's order, and an engine that derives from DefaultEngine can
// replace one for every array bound to it. The array-object functions of the same NumPy file (copy, place, insert,
// delete, flip, ...) are not engine members (docs/plans/np-layer-dtype-logic.md, section 1 "Which O", Appendix D).

using System;
using NumSharp.Backends;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.angle</c>: the counterclockwise angle of each element from the positive real axis, in
        ///     radians in (-pi, pi], or in degrees when <paramref name="deg"/> is set. A complex element gives
        ///     <c>arctan2(z.imag, z.real)</c>; a real one gives NumPy's <c>arctan2(0, z)</c>: 0 for a positive value or
        ///     +0, pi for a negative value or -0, NaN for NaN.
        /// </summary>
        /// <param name="z">
        ///     The input, a plain array: <c>np.angle</c> decides the <c>numpy.matrix</c> class before it calls this
        ///     member, so an override never sees a matrix operand.
        /// </param>
        /// <param name="deg">Return degrees (the radians multiplied by <c>180 / pi</c> in the result's own dtype).</param>
        /// <returns>
        ///     A new array of <paramref name="z"/>'s shape: float64 for complex128; for a real input the float tier of
        ///     NumPy's <c>arctan2(0, z)</c> with a weak 0 (float16 for int8 / uint8 / float16, float32 for int16 /
        ///     uint16 / char / float32, float64 for bool, the wider integers, float64 and decimal).
        /// </returns>
        /// <exception cref="TypeError">
        ///     <paramref name="z"/> holds text (<c>S</c>, <c>U</c> or <c>StringDType</c>): NumPy's <c>arctan2</c> has no
        ///     string loop, and the text names <c>arctan2</c> as NumPy's does.
        /// </exception>
        /// <remarks>
        ///     <para>NumPy writes <c>angle</c> in Python (<c>numpy/lib/_function_base_impl.py</c>); DefaultEngine's
        ///     override runs those statements. An override replaces <c>np.angle</c> for every array whose engine it is,
        ///     so it must keep NumPy's dtype tiers and the real-input values above.</para>
        /// </remarks>
        public abstract NDArray Angle(NDArray z, bool deg = false);
    }
}
