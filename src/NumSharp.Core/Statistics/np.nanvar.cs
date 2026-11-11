using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the variance along the specified axis, while ignoring NaNs: NumPy's <c>numpy.nanvar</c>, statement for
        ///     statement. NaNs count neither in the mean nor in the divisor <c>N - ddof</c>, where <c>N</c> is the number of
        ///     non-NaN elements of each slice.
        /// </summary>
        /// <param name="a">The array whose variance is desired; any dtype and layout.</param>
        /// <param name="axis">The axis along which the variance is computed (negative counts from the end); null (the
        /// default) computes the variance of the flattened array.</param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is <c>N - ddof</c>. Default 0.</param>
        /// <param name="dtype">The type used to compute the variance (both sums run in it). For a float or complex input it
        /// must be inexact; null uses the input's dtype (np.var's float64 for an integer or bool input).</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one.</param>
        /// <param name="out">NumPy's <c>out=</c>: the destination (returned); for a float or complex input it must be
        /// inexact.</param>
        /// <param name="where">NumPy's <c>where=</c>: the bool mask of the elements to include, or null for all.</param>
        /// <param name="mean">NumPy's <c>mean=</c>: the mean to use instead of computing it, shaped as with
        /// <c>keepdims=True</c>, or null.</param>
        /// <param name="correction">NumPy's <c>correction=</c>, the Array-API name of <paramref name="ddof"/>, or null; a
        /// non-zero <paramref name="ddof"/> with it raises.</param>
        /// <returns>The variance, in <paramref name="dtype"/> or the input's float dtype (float64 for a complex input and for
        /// an integer or bool one); a 0-d array for a flat reduction without <paramref name="keepdims"/>. A slice with no
        /// degrees of freedom left (all NaN, or <paramref name="ddof"/> at or past its non-NaN count) is NaN.</returns>
        /// <exception cref="TypeError">A float or complex <paramref name="a"/> with a non-inexact <paramref name="dtype"/>:
        /// "If a is inexact, then dtype must be inexact"; or with a non-inexact <paramref name="out"/>: "If a is inexact, then
        /// out must be inexact".</exception>
        /// <exception cref="ValueError">A non-zero <paramref name="ddof"/> with <paramref name="correction"/>: "ddof and
        /// correction can't be provided simultaneously.".</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <paramref name="a"/>'s dimensions (a 0-d array
        /// included).</exception>
        /// <remarks>
        ///     <para>
        ///     An array that cannot hold NaN (integer, bool, char, decimal) has no NaN to ignore: the result is
        ///     <see cref="var(NDArray, int?, DType, NDArray, int, bool, NDArray, NDArray, int?)"/>'s, keywords included. For a float or complex array the
        ///     arithmetic is NumPy's <c>_nanfunctions_impl.nanvar</c> (see <c>TensorEngine.NanVar</c>), which
        ///     differs from np.var in three observable ways: the deviations are rounded into the input's dtype before they
        ///     are squared (with <c>dtype=float64</c>, a float32 input squares float32 deviations), a complex square is the
        ///     real part of the complex product <c>x * conj(x)</c>, and a slice without degrees of freedom is NaN (np.var
        ///     gives +inf there).
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.nanvar.html</para>
        ///     <para>One signature in NumPy's order (API-parity batch 3, 2026-11-11): it replaces <c>nanvar(a, axis, keepdims, ddof, dtype)</c>, so <c>dtype</c> is the third
        ///     argument, <c>ddof</c> the fifth, and a positional <c>keepdims</c>
        ///     is written <c>keepdims: true</c> (a bool in the dtype slot does not compile). A tuple axis waits for the
        ///     owner's axis type (OD1, decided: <c>NDAxis</c>).</para>
        /// </remarks>
        /// <exception cref="TypeError">A <c>numpy.matrix</c> operand with <c>keepdims: true</c>, <c>initial=</c> or
        ///     <c>where=</c>: NumPy calls the matrix method, which has no such keyword (<c>matrix.sum() got an unexpected
        ///     keyword argument 'where'</c>).</exception>
        [NDScoped]
        public static NDArray nanvar(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null, int ddof = 0,
            bool keepdims = false, NDArray @where = null, NDArray mean = null, int? correction = null)
            => a.TensorEngine.NanVar(a, axis, dtype, ddof, keepdims, @out, @where, mean, correction);
    }
}
