using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the distance between each value of <paramref name="x"/> and the nearest adjacent
        ///     representable number in the direction AWAY from zero — i.e. one unit in the last place (ULP)
        ///     at that value. <c>spacing(1.0) == np.finfo(x.dtype).eps</c>.
        ///     Mirrors NumPy's ufunc signature: <c>spacing(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input array (float family; integer/bool inputs promote to a float per NumPy's width rule, complex is rejected).</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=); masked-off slots of <paramref name="out"/> keep their prior contents.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): must be a FLOAT loop — an integer/bool or complex request raises NumPy's "No loop matching" error.</param>
        /// <returns>Array of the same shape as <paramref name="x"/> holding each element's spacing. This is a scalar if <paramref name="x"/> is a scalar.</returns>
        /// <remarks>
        ///     <para>
        ///         The result carries the SIGN of the input for float32/float64 (<c>spacing(1.0) = +eps</c>,
        ///         <c>spacing(-1.0) = -eps</c>) and is <c>+minsubnormal</c> at <c>±0</c>, <c>±inf</c> and NaN
        ///         both map to NaN, and the largest finite value overflows to <c>+inf</c>. float16 follows
        ///         NumPy's separate <c>npy_half_spacing</c> routine, which is always non-negative and halves
        ///         the ULP at negative power-of-2 boundaries — a NumPy inconsistency reproduced for parity.
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.spacing.html</para>
        /// </remarks>
        public static NDArray spacing(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x.TensorEngine.Spacing(x, dtype, @out, where);
    }
}
