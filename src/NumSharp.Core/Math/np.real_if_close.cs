using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The float64 machine epsilon (2^-52), i.e. <c>np.finfo(np.complex128).eps</c> ==
        ///     <c>np.finfo(np.float64).eps</c>. NumSharp has only complex128, so this is the ONLY epsilon
        ///     <see cref="real_if_close(NDArray, double)"/> ever multiplies <c>tol</c> by — matching
        ///     NumPy's <c>getlimits.finfo(a.dtype.type).eps</c> for the sole complex dtype.
        /// </summary>
        private const double Float64Eps = 2.220446049250313e-16;

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
        public static NDArray real_if_close(NDArray a, double tol = 100)
        {
            // NumPy's `asanyarray(a)` is already satisfied by the NDArray parameter. A non-complex dtype
            // is returned unchanged (the real part of a real number is itself) — the same instance, as
            // NumPy returns `a` untouched for a non-complexfloating type.
            if (a.typecode != NPTypeCode.Complex)
                return a;

            // tol > 1 is interpreted in machine epsilons of the complex type (float64 eps for complex128);
            // tol <= 1 stays an absolute tolerance. IEEE-identical to NumPy's `f.eps * tol` double multiply.
            if (tol > 1)
                tol = Float64Eps * tol;

            // Collapse to the real lane iff EVERY imaginary part is strictly within tol of zero. The scan
            // returns true for an empty array (vacuous all -> empty complex collapses) and false the moment
            // it sees a magnitude >= tol, a NaN, or an infinity.
            if (ImagCloseScan.AllImagWithinTol(a, tol))
                return np.real(a); // float64 view onto the real lane (shares memory), == NumPy's a.real

            // Some imaginary part is out of band (or NaN/inf): the array stays complex, unchanged.
            return a;
        }
    }
}
