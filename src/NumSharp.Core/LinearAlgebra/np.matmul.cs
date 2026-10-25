using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Matrix product of two arrays — the gufunc <c>(n?,k),(k,m?)-&gt;(n?,m?)</c>, with NumPy's
        ///     full keyword surface.
        /// </summary>
        /// <param name="x1">Lhs input array, scalars not allowed.</param>
        /// <param name="x2">Rhs input array, scalars not allowed.</param>
        /// <param name="out">
        ///     Where to deposit the answer; returned as-is when given. Like a ufunc's <c>out</c> (and
        ///     unlike <see cref="dot"/>'s strict one) it may be strided and takes a cast from the
        ///     product dtype under <paramref name="casting"/>.
        /// </param>
        /// <param name="axes">
        ///     Which axes carry the core dimensions, per operand: <c>{x1, x2, out}</c>. A 2-D operand
        ///     names TWO axes, a 1-D operand ONE (its optional core dim is absent); the output entry
        ///     may be omitted only for the 1-D·1-D product, whose result has no core axes. Cannot be
        ///     combined with <paramref name="axis"/>.
        /// </param>
        /// <param name="axis">
        ///     Present for signature parity only. NumPy raises <c>TypeError</c> for any value, because
        ///     <c>matmul</c>'s signature has three DISTINCT core dimensions — use <paramref name="axes"/>.
        /// </param>
        /// <param name="keepdims">
        ///     Present for signature parity only. NumPy raises <c>TypeError</c> for ANY value (True OR
        ///     False), because its output has core dimensions. Modelled with a <c>bool?</c> sentinel so
        ///     that, like NumPy's <c>np._NoValue</c> default, an explicit <c>false</c> also rejects.
        /// </param>
        /// <param name="dtype">Selects the LOOP: the product runs at this dtype, not merely the result.</param>
        /// <param name="casting">
        ///     Casting rule (default <c>"same_kind"</c>, the ufunc default) gating BOTH the input→loop
        ///     cast a <paramref name="dtype"/> forces and the product→<paramref name="out"/> cast.
        /// </param>
        /// <param name="order">Memory layout of the result — <c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        ///     <para>
        ///     The product itself is unchanged — <c>TensorEngine.Matmul</c>, which routes
        ///     float32/float64/complex128 through OpenBLAS when <c>NumSharp.Interop.OpenBLAS</c> is
        ///     referenced (byte-identical to NumPy) and through the managed GEMM otherwise. A
        ///     <paramref name="dtype"/> request is applied by casting the operands to that dtype BEFORE
        ///     the product, so the loop — and thus the backend route — follows it.
        ///     </para>
        ///     <para>
        ///     Like the five sibling product gufuncs, NumPy's remaining ufunc keywords <c>subok</c> and
        ///     <c>signature</c> are not modelled (<c>signature</c> is what <paramref name="dtype"/>
        ///     does; <c>subok</c> concerns ndarray subclasses NumSharp does not have). A
        ///     <paramref name="out"/> with a wrong CORE dim reports NumPy's verbatim core-dimension
        ///     message; extra leading LOOP dims broadcast (replicating the product, as NumPy does), and
        ///     the rarer genuine loop-dimension mismatch raises <c>copyto</c>'s broadcast
        ///     <c>ValueError</c> whose wording differs from NumPy's leaked iterator text (the same
        ///     latitude the siblings take).
        ///     </para>
        ///     <para>
        ///     The whole call is the engine member
        ///     <c>TensorEngine.Matmul(x1, x2, out, axes, axis, keepdims, dtype, casting, order)</c> since the np-layer
        ///     dtype plan's P2; this function keeps the numpy.matrix boundary.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <returns>The matrix product, or <paramref name="out"/> when one is given (a text <paramref name="out"/> receives
        ///     each product as its scalar's <c>str()</c>).</returns>
        /// <exception cref="TypeError">Text operands or a string <paramref name="dtype"/> (no matmul loop), <c>keepdims</c>
        ///     or <c>axis</c> (NumPy's matmul refusals).</exception>
        /// <exception cref="INumSharpException">A read-only <paramref name="out"/> (NumPy's ValueError "output array is
        ///     read-only").</exception>
        /// <exception cref="ArgumentException">A cast <paramref name="casting"/> refuses (NumPy's UFuncTypeError text).</exception>
        /// <exception cref="NullReferenceException"><paramref name="x1"/> or <paramref name="x2"/> is null.</exception>
        [NDScoped]
        public static NDArray matmul(NDArray x1, NDArray x2, NDArray @out = null, int[][] axes = null,
            int? axis = null, bool? keepdims = null, DType dtype = null, string casting = "same_kind",
            char order = 'K')
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x1, x2))
                return NDMatrixBoundary.Wrapped((x1, x2, @out, axes, axis, keepdims, dtype, casting, order), static s => matmul(s.x1, s.x2, s.@out, s.axes, s.axis, s.keepdims, s.dtype, s.casting, s.order), @out);
            return x1.TensorEngine.Matmul(x1, x2, @out, axes, axis, keepdims, dtype, casting, order);
        }
    }
}
