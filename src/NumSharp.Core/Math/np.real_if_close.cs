using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     If the input is complex with EVERY imaginary part close to zero, return the (float64) real
        ///     parts; otherwise return the input unchanged. "Close to zero" is <c>tol</c> multiples of the
        ///     machine epsilon of the complex type (float64 eps here) when <paramref name="tol"/> &gt; 1,
        ///     and the plain absolute value of <paramref name="tol"/> when <paramref name="tol"/> &lt;= 1.
        /// </summary>
        /// <param name="a">
        ///     Input array. A real / integer / boolean array is returned UNCHANGED (the same instance) —
        ///     the collapse only applies to complex input; NumPy does the same (its <c>asanyarray(a)</c>
        ///     is already satisfied by the <see cref="NDArray"/> parameter, and scalars / C# arrays convert
        ///     implicitly at the call site).
        /// </param>
        /// <param name="tol">
        ///     Tolerance for the imaginary part. When &gt; 1 it is interpreted in machine epsilons
        ///     (<c>tol * eps</c>); when &lt;= 1 it is used as an absolute tolerance directly. FOOTGUN
        ///     matching NumPy: with <c>tol &lt;= 0</c> NOTHING collapses (a magnitude <c>|imag| &gt;= 0</c>
        ///     is never strictly <c>&lt; 0</c>), even for an exactly-zero imaginary part. The default
        ///     <c>100</c> resolves to <c>100 * 2.22e-16 ≈ 2.22e-14</c>.
        /// </param>
        /// <returns>
        ///     For a COMPLEX input whose imaginary parts are all strictly within <paramref name="tol"/> of
        ///     zero: a float64 VIEW onto the real lane (shares memory with <paramref name="a"/>, exactly
        ///     like <see cref="real(NDArray)"/> / NumPy's <c>a.real</c>). An EMPTY complex array always
        ///     collapses (an empty <c>all</c> is vacuously true), yielding an empty float64 view. Otherwise
        ///     (a real/int/bool input, or a complex input with any imaginary part &gt;= <paramref name="tol"/>,
        ///     NaN or infinite): <paramref name="a"/> itself, unchanged.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html
        ///     <para>
        ///     Port of NumPy 2.4.2 <c>numpy/lib/_type_check_impl.py::real_if_close</c>, whose body is
        ///     <c>if all(absolute(a.imag) &lt; tol): a = a.real</c>. The whole-array decision
        ///     (<see cref="ImagCloseScan.AllImagWithinTol"/>) fuses NumPy's <c>absolute</c> temp, the
        ///     boolean <c>&lt;</c> temp and the <c>all</c> reduce into ONE early-exit streaming pass over
        ///     the imaginary lane (no intermediate allocation), so it beats the three-pass NumPy sequence
        ///     on every layout. NaN / ±inf imaginary parts prevent the collapse (the comparison with
        ///     <c>tol</c> is false), and the comparison is STRICT, so an imaginary part exactly equal to
        ///     <paramref name="tol"/> also prevents it — all probed against NumPy 2.4.2. The collapse
        ///     companion <see cref="real(NDArray)"/> keeps the real lane's writeability, so — like NumPy's
        ///     <c>a.real</c> — writing through the returned view writes through to <paramref name="a"/>.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray real_if_close(NDArray a, double tol = 100)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, tol), static s => real_if_close(s.a, s.tol));
            return np.EngineOf(a).RealIfClose(a, tol);
        }
    }
}
