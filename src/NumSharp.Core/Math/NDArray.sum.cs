using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>ndarray.sum(axis=None, dtype=None, out=None, *, keepdims=&lt;no value&gt;, initial=&lt;no value&gt;,
        ///     where=&lt;no value&gt;)</c> (<c>_methods._sum</c>, i.e. <c>np.add.reduce</c>): the sum of the elements over
        ///     <paramref name="axis"/>. This is the only overload: NumPy's whole signature (np-layer dtype plan P10, D10); the
        ///     parameters behave as <see cref="np.sum"/>'s.
        /// </summary>
        /// <param name="axis">The axis to sum over (negative counts from the end); <c>null</c> sums every element. A
        ///     tuple (<c>(0, 2)</c>) reduces every axis it names together, in one pass (NumPy's tuple axis; the empty
        ///     tuple reduces none, and a tuple item gets no 0-d allowance).</param>
        /// <param name="dtype">The accumulator and result dtype, or <c>null</c> for NumPy's default (NEP 50's accumulator;
        ///     <c>promote_types(out, this)</c> with an out).</param>
        /// <param name="out">The array the sum is written into and returned. Footgun: the THIRD positional parameter, as in
        ///     NumPy, so write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every sum starts from instead of 0 (see <see cref="np.sum"/>), or <c>null</c>.</param>
        /// <param name="where">A bool mask of the elements to sum, or <c>null</c> (the C# <c>true</c> literal is NumPy's
        ///     default).</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) for a sum
        ///     over every axis without <paramref name="keepdims"/>.</returns>
        /// <exception cref="TypeError">A C# scalar passed as <paramref name="out"/>; an <paramref name="initial"/> the loop
        ///     cannot take; NumPy's text refusals.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>; a mask with more axes; an
        ///     unconvertible <paramref name="initial"/>. Also a tuple that names an axis twice (<c>duplicate value in
        ///     'axis'</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> or <paramref name="where"/> does not broadcast.</exception>
        /// <exception cref="OverflowError"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>The member keeps the plain-array meaning for a numpy.matrix object reached through an
        ///     <see cref="NDArray"/> reference (an <see cref="NDMatrix"/> reference binds <see cref="NDMatrix.sum(int?, DType, NDMatrix)"/>).</remarks>
        [PlainReceiver]
        public NDArray sum(NDAxis? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false,
            object initial = null, NDArray @where = null)
        {
            return TensorEngine.ReduceAdd(this, axis, keepdims, dtype, @out, initial, @where);
        }

        /// <summary>
        ///     <see cref="sum(NDAxis?, DType, NDArray, bool, object, NDArray)"/> with the axis as one <c>int</c>: the
        ///     overload a C# integer binds (its identity conversion beats NDAxis' implicit one), forwarding to the
        ///     NDAxis form, so every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="sum(NDAxis?, DType, NDArray, bool, object, NDArray)"/>
        public NDArray sum(int axis, DType dtype = null, NDArray @out = null, bool keepdims = false, object initial = null, NDArray @where = null)
            => sum((NDAxis?)axis, dtype, @out, keepdims, initial, @where);

        /// <summary>
        ///     <see cref="sum(NDAxis?, DType, NDArray, bool, object, NDArray)"/> with the axes as an <c>int[]</c>: the
        ///     overload an <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to
        ///     NDAxis), forwarding to the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple
        ///     (<c>new int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="sum(NDAxis?, DType, NDArray, bool, object, NDArray)"/>
        public NDArray sum(int[] axis, DType dtype = null, NDArray @out = null, bool keepdims = false, object initial = null, NDArray @where = null)
            => sum((NDAxis?)axis, dtype, @out, keepdims, initial, @where);
    }
}
