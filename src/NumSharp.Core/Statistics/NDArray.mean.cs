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
        /// <param name="axis">The axis to average over (negative counts from the end); <c>null</c> averages every element.</param>
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
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> or <paramref name="where"/> does not broadcast.</exception>
        /// <exception cref="TypeError">A C# scalar passed as <paramref name="out"/>; NumPy's text refusals.</exception>
        /// <remarks>The member keeps the plain-array meaning for a numpy.matrix object reached through an
        ///     <see cref="NDArray"/> reference.</remarks>
        public NDArray mean(int? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return TensorEngine.ReduceMean(this, axis, keepdims, dtype, @out, @where);
        }
    }
}
