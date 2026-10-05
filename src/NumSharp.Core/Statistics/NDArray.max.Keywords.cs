using System.Runtime.CompilerServices;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return the maximum along an axis — NumPy's <c>ndarray.max(axis=None, out=None, keepdims=False, initial=&lt;no
        ///     value&gt;, where=True)</c>, i.e. <see cref="np.max(NDArray, int?, NDArray, bool, object, NDArray)"/> on this
        ///     array.
        /// </summary>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned (any dtype; a narrowing one is lossy, as in
        ///     NumPy — see <see cref="np.max(NDArray, int?, NDArray, bool, object, NDArray)"/>).</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, <paramref name="where"/> without
        ///     <paramref name="initial"/>, an empty reduced axis without <paramref name="initial"/>, or an unconvertible
        ///     <paramref name="initial"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="TypeError"><paramref name="initial"/> is complex for a real loop, or of an unconvertible type.</exception>
        /// <remarks>
        ///     NumSharp's earlier forms stay next to this one: <c>max(int, bool, DType)</c> / <c>max(DType)</c> (the dtype
        ///     extension) at a lower overload priority, and <see cref="max(int?, bool)"/> for a positional keepdims
        ///     (<c>a.max(1, true)</c>). https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html
        /// </remarks>
        [OverloadResolutionPriority(1)]
        public NDArray max(int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null, NDArray @where = null)
            => np.max(this, axis, @out, keepdims, initial, @where);

        /// <summary>
        ///     <c>a.max(axis, keepdims)</c> with keepdims in the second position — the positional form NumSharp callers wrote
        ///     before NumPy's keywords existed (NumPy's second parameter is <c>out</c>).
        /// </summary>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <returns>The maximum (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="System.ArgumentException">An empty reduced axis (NumPy's "zero-size array" text).</exception>
        [OverloadResolutionPriority(1)]
        public NDArray max(int? axis, bool keepdims)
            => np.max(this, axis, keepdims);

        /// <summary>
        ///     Return the minimum along an axis — NumPy's <c>ndarray.min(axis=None, out=None, keepdims=False, initial=&lt;no
        ///     value&gt;, where=True)</c>, i.e. <see cref="np.min(NDArray, int?, NDArray, bool, object, NDArray)"/> on this
        ///     array.
        /// </summary>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, <paramref name="where"/> without
        ///     <paramref name="initial"/>, an empty reduced axis without <paramref name="initial"/>, or an unconvertible
        ///     <paramref name="initial"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="TypeError"><paramref name="initial"/> is complex for a real loop, or of an unconvertible type.</exception>
        /// <remarks>
        ///     As <see cref="max(int?, NDArray, bool, object, NDArray)"/>: the dtype extension and the positional keepdims form
        ///     (<see cref="min(int?, bool)"/>) stay. https://numpy.org/doc/stable/reference/generated/numpy.ndarray.min.html
        /// </remarks>
        [OverloadResolutionPriority(1)]
        public NDArray min(int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null, NDArray @where = null)
            => np.min(this, axis, @out, keepdims, initial, @where);

        /// <summary><c>a.min(axis, keepdims)</c> — the positional keepdims form (see <see cref="max(int?, bool)"/>).</summary>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <returns>The minimum.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="System.ArgumentException">An empty reduced axis.</exception>
        [OverloadResolutionPriority(1)]
        public NDArray min(int? axis, bool keepdims)
            => np.min(this, axis, keepdims);
    }
}
