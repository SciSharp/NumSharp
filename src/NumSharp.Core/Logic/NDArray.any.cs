using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>ndarray.any(axis=None, out=None, keepdims=False, *, where=True)</c>: whether any element evaluates
        ///     to True along the given axes. Refer to <see cref="np.any(NDArray, NDAxis?, NDArray, bool, NDArray)"/> for the
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
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.any.html. NumSharp's static-typing rule
        ///     (<c>[PlainReceiver]</c>): called on a <c>numpy.matrix</c> through an <see cref="NDArray"/> reference it
        ///     computes as the plain array (the matrix's own <c>any</c> is <see cref="NDMatrix.any(NDAxis?, NDArray)"/>).
        ///     The axis is an <see cref="NDAxis"/> since API-parity batch 4 (2026-11-12); an <c>int?</c> argument converts.
        /// </remarks>
        [PlainReceiver]
        public NDArray any(NDAxis? axis = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
        {
            return np.any(this, axis, @out, keepdims, @where);
        }
    }
}
