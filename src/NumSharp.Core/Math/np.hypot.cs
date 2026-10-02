using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Given the "legs" of a right triangle, return its hypotenuse. <br></br>
        ///     Computes <c>sqrt(x1**2 + x2**2)</c>, element-wise, without the intermediate overflow or
        ///     underflow a naive square-and-add would suffer (so <c>hypot(1e200, 1e200)</c> is
        ///     <c>1.41e200</c>, not <c>inf</c>). Both scalars and arrays are accepted; if their shapes
        ///     differ they must broadcast to a common shape.
        ///     Mirrors NumPy's ufunc signature: <c>hypot(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">Leg of the triangle(s).</param>
        /// <param name="x2">Leg of the triangle(s). If shapes differ they must broadcast to a common shape.</param>
        /// <param name="out">A location into which the result is stored (joins the broadcast without being stretched, same_kind-castable from the loop dtype; returned as-is).</param>
        /// <param name="where">Boolean mask: only mask-true elements are computed/written (NumPy ufunc where=).</param>
        /// <param name="dtype">Explicit loop dtype (float-family only; int/bool/complex raise NumPy's "No loop matching" error).</param>
        /// <returns>The hypotenuse of the triangle(s). This is a scalar if both x1 and x2 are scalars.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.hypot.html
        ///     <para>
        ///     float32/float16 results are bit-identical to NumPy; float64 is correctly-rounded, so it
        ///     matches NumPy on ~91% of inputs and is within 1 ULP (more accurate) on the rest — NumPy
        ///     calls the platform (UCRT) <c>hypot</c>, which is only faithfully-rounded. See
        ///     <see cref="Utilities.NDHypotMath"/>.
        ///     </para>
        /// </returns>
        public static NDArray hypot(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
            => x1.TensorEngine.Hypot(x1, x2, dtype, @out, where);
    }
}
