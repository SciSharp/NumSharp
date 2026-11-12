using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>ndarray.mean(axis=None, dtype=None, out=None, *, keepdims=&lt;no value&gt;, where=&lt;no value&gt;)</c>
        ///     (<c>_methods._mean</c>): the arithmetic mean over <paramref name="axis"/>. This is the only overload: NumPy's
        ///     whole signature (np-layer dtype plan P10, D10); the parameters behave as <see cref="np.mean"/>'s.
        /// </summary>
        /// <param name="axis">The axis to average over (negative counts from the end); <c>null</c> averages every
        ///     element. A tuple (<c>(0, 2)</c>) reduces every axis it names together, in one pass (NumPy's tuple axis;
        ///     the empty tuple reduces none, and a tuple item gets no 0-d allowance).</param>
        /// <param name="dtype">The dtype the sum is computed in and the mean returned in, or <c>null</c> for NumPy's default
        ///     (float64 for bool / integer input, float32 summing a float16 input into a float16 result).</param>
        /// <param name="out">The array the mean is written into and returned (the sum is stored into it and divided in place,
        ///     with unsafe casts). Footgun: the THIRD positional parameter, as in NumPy, so write <c>keepdims: true</c> by
        ///     name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="where">A bool mask of the elements each output averages, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) for a mean
        ///     over every axis without <paramref name="keepdims"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>. Also a tuple that names an
        ///     axis twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> or <paramref name="where"/> does not broadcast.</exception>
        /// <exception cref="TypeError">A C# scalar passed as <paramref name="out"/>; NumPy's text refusals.</exception>
        /// <remarks>The member keeps the plain-array meaning for a numpy.matrix object reached through an
        ///     <see cref="NDArray"/> reference.</remarks>
        [PlainReceiver]
        public NDArray mean(NDAxis? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
        {
            return TensorEngine.ReduceMean(this, axis, keepdims, dtype, @out, @where);
        }

        /// <summary>
        ///     <see cref="mean(NDAxis?, DType, NDArray, bool, NDArray)"/> with the axis as one <c>int</c>: the overload
        ///     a C# integer binds (its identity conversion beats NDAxis' implicit one), forwarding to the NDAxis form,
        ///     so every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="mean(NDAxis?, DType, NDArray, bool, NDArray)"/>
        public NDArray mean(int axis, DType dtype = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
            => mean((NDAxis?)axis, dtype, @out, keepdims, @where);

        /// <summary>
        ///     <see cref="mean(NDAxis?, DType, NDArray, bool, NDArray)"/> with the axes as an <c>int[]</c>: the
        ///     overload an <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to
        ///     NDAxis), forwarding to the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple
        ///     (<c>new int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="mean(NDAxis?, DType, NDArray, bool, NDArray)"/>
        public NDArray mean(int[] axis, DType dtype = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
            => mean((NDAxis?)axis, dtype, @out, keepdims, @where);
    }
}
