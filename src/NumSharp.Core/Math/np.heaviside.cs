using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the Heaviside step function, element-wise. <br></br>
        ///     <c>heaviside(x1, x2)</c> is <c>0</c> where <c>x1 &lt; 0</c>, <c>1</c> where <c>x1 &gt; 0</c>,
        ///     and <c>x2</c> exactly where <c>x1 == 0</c> (so <c>x2</c> — often <c>0.5</c> — is the value AT
        ///     the step). A NaN in <c>x1</c> yields NaN. <c>x1</c> and <c>x2</c> broadcast to a common shape.
        ///     Mirrors NumPy's ufunc signature: <c>heaviside(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">Input values; the sign of each element selects the branch (negative → 0,
        /// positive → 1, zero → <paramref name="x2"/>, NaN → NaN).</param>
        /// <param name="x2">The value of the function where <paramref name="x1"/> is 0. If shapes differ they
        /// must broadcast to a common shape. NOT interchangeable with <paramref name="x1"/> — heaviside is
        /// not commutative.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being
        /// stretched, same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (float-family only; int/bool/complex raise NumPy's "No loop
        /// matching" error, since heaviside has only the ee/ff/dd/gg loops — an integer/bool INPUT is cast to
        /// its float tier exactly as arctan2 does).</param>
        /// <returns>The element-wise Heaviside step function of <paramref name="x1"/>. A scalar if both
        /// <paramref name="x1"/> and <paramref name="x2"/> are scalars.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html
        ///     <para>
        ///     Bit-identical to NumPy 2.4.2 for every dtype: the step is a port of <c>npy_heaviside</c>. Two
        ///     NaN behaviours are load-bearing — a NaN <paramref name="x1"/> gives the POSITIVE canonical NaN
        ///     (its own sign discarded), while the <c>x1 == 0</c> case returns <paramref name="x2"/>'s EXACT
        ///     bits (a NaN or <c>-0.0</c> <paramref name="x2"/> keeps its sign). Faster than NumPy (which has
        ///     no SIMD heaviside on any platform): the contiguous / scalar-broadcast float32/float64 cases run
        ///     a branchless vector kernel. See <see cref="Utilities.NDHeavisideMath"/>.
        ///     </para>
        /// </remarks>
        public static NDArray heaviside(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Heaviside(x1, x2, dtype, @out, where);
    }
}
