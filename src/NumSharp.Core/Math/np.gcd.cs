using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Returns the element-wise greatest common divisor of the absolute values of <paramref name="x1"/>
        /// and <paramref name="x2"/> (<c>np.gcd</c>). <c>gcd(0, 0) == 0</c> and <c>gcd(0, x) == |x|</c>.
        ///
        /// INTEGER-ONLY, exactly like NumPy: a bool/half/single/double/decimal/complex operand — or a
        /// <c>uint64</c> paired with a signed integer (they promote to <c>float64</c> under NEP50) — has no
        /// loop and raises the no-loop <see cref="TypeError"/>. Both inputs and the output share ONE
        /// promoted integer dtype (NumPy's uniform resolver). The result is non-negative EXCEPT where the
        /// magnitude wraps the signed range: <c>gcd(int8 -128, -128) == -128</c> (no positive int8 holds
        /// 128), matching NumPy bit-for-bit. A C# integer literal binds as a weak scalar
        /// (<c>np.gcd(int8_arr, 3)</c> stays int8), matching NEP50.
        /// Mirrors NumPy's ufunc signature <c>gcd(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">First input array (integer dtype).</param>
        /// <param name="x2">Second input array (integer dtype); broadcasts against <paramref name="x1"/>.</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc <c>out=</c>); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc <c>where=</c>); masked-off slots keep their prior value.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc <c>dtype=</c>): computation runs in this dtype; must name an integer loop.</param>
        /// <returns>The element-wise gcd. A scalar (0-d) when both inputs are scalars.</returns>
        /// <exception cref="TypeError">The inputs (or <paramref name="dtype"/>) name no integer gcd loop.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is non-boolean, an input is not same_kind-castable to <paramref name="dtype"/>, or <paramref name="out"/> is incompatible.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.gcd.html</remarks>
        public static NDArray gcd(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Gcd(x1, x2, dtype, @out, where);

        /// <summary>
        /// Returns the element-wise lowest common multiple of the absolute values of <paramref name="x1"/>
        /// and <paramref name="x2"/> (<c>np.lcm</c>). <c>lcm(x, 0) == 0</c>; otherwise <c>|x1| / gcd * |x2|</c>
        /// (divide-before-multiply, and the product WRAPS the dtype on overflow — e.g.
        /// <c>lcm(int16 21000, 14000) == -23536</c> — matching NumPy exactly).
        ///
        /// Shares every rule with <see cref="gcd(NDArray, NDArray, NDArray, NDArray, DType)"/>: integer-only
        /// loops (bool/float/complex/decimal and uint64+signed raise the no-loop <see cref="TypeError"/>),
        /// one shared promoted integer dtype, and NEP50 weak-scalar binding for C# integer literals.
        /// Mirrors NumPy's ufunc signature <c>lcm(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">First input array (integer dtype).</param>
        /// <param name="x2">Second input array (integer dtype); broadcasts against <paramref name="x1"/>.</param>
        /// <param name="out">A location into which the result is stored (NumPy ufunc <c>out=</c>); returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc <c>where=</c>); masked-off slots keep their prior value.</param>
        /// <param name="dtype">Explicit loop dtype (NumPy ufunc <c>dtype=</c>): computation runs in this dtype; must name an integer loop.</param>
        /// <returns>The element-wise lcm. A scalar (0-d) when both inputs are scalars.</returns>
        /// <exception cref="TypeError">The inputs (or <paramref name="dtype"/>) name no integer lcm loop.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is non-boolean, an input is not same_kind-castable to <paramref name="dtype"/>, or <paramref name="out"/> is incompatible.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.lcm.html</remarks>
        public static NDArray lcm(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Lcm(x1, x2, dtype, @out, where);
    }
}
