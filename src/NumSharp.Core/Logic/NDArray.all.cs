using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>ndarray.all(axis=None, out=None, keepdims=False, *, where=True)</c>: whether every element evaluates
        ///     to True along the given axes. Refer to <see cref="np.all(NDArray, NDAxis?, NDArray, bool, NDArray)"/> for the
        ///     full contract.
        /// </summary>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) reduces every axis; one axis or a tuple
        ///     (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) reduces those axes; the empty tuple reduces none.</param>
        /// <param name="out">An array of the result's shape and any dtype to store the result in (and return); null for a
        ///     fresh bool array. A C# scalar here is NumPy's TypeError (<c>return arrays must be of ArrayType</c>).</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one.</param>
        /// <param name="where">A bool mask of the elements to include (it broadcasts to this array's shape); null includes
        ///     all.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh bool array, read-only when 0-d (NumPy's
        ///     scalar).</returns>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ValueError">A tuple names an axis twice (<c>duplicate value in 'axis'</c>), or a read-only or
        ///     misshapen <paramref name="out"/> / <paramref name="where"/> (NumPy's reduction texts).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast
        ///     against this array.</exception>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.all.html. NumSharp's static-typing rule
        ///     (<c>[PlainReceiver]</c>): called on a <c>numpy.matrix</c> through an <see cref="NDArray"/> reference it
        ///     computes as the plain array (the matrix's own <c>all</c> is <see cref="NDMatrix.all(NDAxis?, NDArray)"/>).
        ///     The axis is an <see cref="NDAxis"/> since API-parity batch 4 (2026-11-12); an <c>int?</c> argument converts.
        /// </remarks>
        [PlainReceiver]
        public NDArray all(NDAxis? axis = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
        {
            return np.all(this, axis, @out, keepdims, @where);
        }

        /// <summary>
        ///     <see cref="all(NDAxis?, NDArray, bool, NDArray)"/> with the axis as one <c>int</c>: the overload a C#
        ///     integer binds (its identity conversion beats NDAxis' implicit one), forwarding to the NDAxis form, so
        ///     every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="all(NDAxis?, NDArray, bool, NDArray)"/>
        public NDArray all(int axis, NDArray @out = null, bool keepdims = false, NDArray @where = null)
            => all((NDAxis?)axis, @out, keepdims, @where);

        /// <summary>
        ///     <see cref="all(NDAxis?, NDArray, bool, NDArray)"/> with the axes as an <c>int[]</c>: the overload an
        ///     <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to NDAxis),
        ///     forwarding to the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple (<c>new
        ///     int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="all(NDAxis?, NDArray, bool, NDArray)"/>
        public NDArray all(int[] axis, NDArray @out = null, bool keepdims = false, NDArray @where = null)
            => all((NDAxis?)axis, @out, keepdims, @where);
    }
}
