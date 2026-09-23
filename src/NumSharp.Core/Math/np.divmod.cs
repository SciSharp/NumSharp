using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return element-wise quotient and remainder simultaneously:
        /// <c>divmod(x1, x2)</c> is equivalent to <c>(floor_divide(x1, x2), remainder(x1, x2))</c> but
        /// computed in a single fused pass. Item1 (<c>Quotient</c>) is the floored quotient, Item2
        /// (<c>Remainder</c>) the floored remainder (same sign as the divisor). Integer inputs stay
        /// integer (NEP50; bool -> int8); float ÷0 yields (±inf/nan, nan). Complex is not supported
        /// (NumPy has no complex divmod loop).
        ///
        /// Mirrors NumPy's two-output ufunc <c>divmod(x1, x2[, out1, out2], / [, out=(None, None)],
        /// *, where=True, dtype=None)</c>. In C# the two outputs are a value tuple; pass an
        /// <paramref name="out"/> tuple to write into existing arrays.
        /// </summary>
        /// <param name="x1">Dividend array.</param>
        /// <param name="x2">Divisor array.</param>
        /// <param name="out">Optional pair of output arrays (NumPy ufunc out=); each is written and
        /// returned as-is (dtype must be same_kind-castable from the loop dtype).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): the loop computes in this dtype.</param>
        /// <returns>(quotient, remainder). Both are scalars if x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.divmod.html</remarks>
        public static (NDArray Quotient, NDArray Remainder) divmod(NDArray x1, NDArray x2,
            (NDArray Quotient, NDArray Remainder) @out = default, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.DivMod(x1, x2, dtype, @out, where);

        /// <summary>
        /// Scalar / array-like divisor convenience: <c>np.divmod(a, 3)</c>. The divisor is coerced via
        /// <see cref="asanyarray"/> (the a // obj / a % obj analog).
        /// </summary>
        // Scope: reclaims the np.asanyarray(x2) temp; the two result arrays are detached through the
        // tuple carrier (same as np.modf's [NDScoped] two-array return).
        [NDScoped]
        public static (NDArray Quotient, NDArray Remainder) divmod(NDArray x1, object x2)
            => x1.TensorEngine.DivMod(x1, np.asanyarray(x2));
    }
}
