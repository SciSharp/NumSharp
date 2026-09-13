using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the absolute values element-wise, in the FLOATING-POINT domain.
        ///     Mirrors NumPy's ufunc signature: <c>fabs(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">
        ///     Input array. Any dtype EXCEPT complex — <see cref="fabs"/> has no complex loop
        ///     (use <see cref="absolute(NDArray,NDArray,NDArray,DType)"/> for the complex magnitude).
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored (joins the broadcast without being stretched,
        ///     must be same_kind-castable from the resolved float loop dtype; returned as-is).
        /// </param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">
        ///     Explicit loop dtype (NumPy ufunc dtype=): float kinds only. A bool/integer/char or complex
        ///     request raises the no-loop error; a complex input with a float dtype= raises the input-cast error.
        /// </param>
        /// <returns>
        ///     A float array (or <paramref name="out"/>) holding the absolute value of each element. Unlike
        ///     <see cref="absolute(NDArray)"/> this ALWAYS returns a float: integer/bool input is promoted
        ///     (NEP50 tier — bool/int8/uint8→float16, int16/uint16→float32, int32/uint32/int64/uint64→float64),
        ///     while float16/float32/float64 are preserved.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fabs.html
        ///     <para>
        ///     Port of NumPy 2.4.2's <c>fabs</c> ufunc. It is the float-only counterpart of
        ///     <see cref="absolute(NDArray)"/>: the operation is identical on the float loops (clear the
        ///     IEEE sign bit — so <c>-0.0 → +0.0</c>, <c>±inf → +inf</c>, and a negative NaN's sign bit is
        ///     cleared while its payload is preserved), but fabs promotes integer/bool to float and refuses
        ///     complex input (NumPy: "ufunc 'fabs' not supported for the input types").
        ///     </para>
        /// </remarks>
        public static NDArray fabs(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Fabs(x, dtype, @out, where);
    }
}
