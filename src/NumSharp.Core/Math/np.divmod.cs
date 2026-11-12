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
        /// integer (NEP50; bool -> int8); float ÷0 yields (±inf/nan, nan). A complex operand raises NumPy's
        /// TypeError: NumPy has no complex divmod loop.
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.divmod.html
        ///     <para>One signature, NumPy's ufunc form (2026-11-11): the former overload that took the second operand as an
        ///     <c>object</c> read a boxed number as a STRONG array (parity QA-QB30); a C# literal converts to a 0-d array
        ///     marked as the Python scalar it stands for, so NEP 50's weak-scalar rule holds without it, and a value held in
        ///     an <c>object</c> variable goes through <see cref="np.asanyarray(in object, DType, char?, string, bool?)"/>.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="TypeError">A complex operand with no <paramref name="dtype"/>: divmod has no complex loop,
        ///     so NumPy's <c>ufunc 'divmod' not supported for the input types, ...</c> (parity QA-27); under a real
        ///     <paramref name="dtype"/>, an input or an out the loop cannot reach by a same_kind cast (NumPy's
        ///     UFuncTypeError texts); a bool or complex <paramref name="dtype"/> (no loop).</exception>
        public static (NDArray Quotient, NDArray Remainder) divmod(NDArray x1, NDArray x2,
            (NDArray Quotient, NDArray Remainder) @out = default, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.DivMod(x1, x2, dtype, @out, where);
    }
}
