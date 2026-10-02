using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     First array elements raised to powers from second array, element-wise, at a MINIMUM
        ///     precision of float64 (np.float_power).
        ///
        ///     Unlike <see cref="power(NDArray,NDArray)"/>, float_power promotes every real input
        ///     (bool/integer/float16/float32/decimal/char) to float64 and a complex operand to
        ///     complex128 — the result is therefore always an inexact float. Because there is no
        ///     integer loop, a negative integer exponent is legal:
        ///     <c>np.float_power(2, -1) == 0.5</c> where <see cref="power(NDArray,NDArray)"/> raises.
        ///     The computation is otherwise bit-identical to <see cref="power(NDArray,NDArray)"/> on
        ///     its float64/complex128 loops.
        /// </summary>
        /// <param name="x1">The bases (any dtype; promoted to the float loop).</param>
        /// <param name="x2">The exponents (any dtype; promoted to the float loop).</param>
        /// <param name="@out">A location into which the result is stored (joins the broadcast without
        ///     being stretched, must be same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc dtype=): only float64 or complex128
        ///     select a loop — any other request raises "No loop matching the specified signature and
        ///     casting was found for ufunc float_power".</param>
        /// <returns>The bases in x1 raised to the exponents in x2 at float64/complex128 precision (or
        ///     <paramref name="@out"/>). A scalar NDArray if both x1 and x2 are scalars.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.float_power.html</remarks>
        public static NDArray float_power(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.FloatPower(x1, x2, dtype, @out, where);

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise, at a minimum
        ///     precision of float64 (np.float_power). Scalar or array-like exponent version.
        /// </summary>
        /// <param name="x1">The bases.</param>
        /// <param name="x2">The exponents (scalar or array-like).</param>
        /// <returns>The bases in x1 raised to the exponents in x2 at float64/complex128 precision.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.float_power.html</remarks>
        // Scope: the object overload mints an np.asanyarray(x2) temp for a scalar/array-like exponent
        // (the same leftover the power(x1, object) overload carries); [NDScoped] reclaims it while an
        // NDArray-input passthrough stays untracked. The (NDArray, NDArray) overload above owns no
        // temp and stays unscoped.
        [NDScoped]
        public static NDArray float_power(NDArray x1, object x2)
            => x1.TensorEngine.FloatPower(x1, np.asanyarray(x2));
    }
}
