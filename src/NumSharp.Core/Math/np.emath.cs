using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Mathematical functions with <b>automatic domain</b> — the port of Python's
        ///     <c>numpy.emath</c> module (a.k.a. <c>numpy.lib.scimath</c>). Accessed exactly like Python:
        ///     <c>np.emath.sqrt(-1)</c>, <c>np.emath.log(-1)</c>, <c>np.emath.arccos(2)</c>, etc. Mirrors
        ///     the <see cref="fft"/> / <see cref="random"/> facade shape (a lowercase property returning a
        ///     module object whose methods are the functions).
        /// </summary>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/routines.emath.html
        ///     <para>
        ///     These are the branch-cut-aware siblings of the ordinary <c>np.*</c> functions: where
        ///     <see cref="sqrt(NDArray, NDArray, NDArray, DType)"/> returns <c>nan</c> for a negative real
        ///     input, <see cref="EmathModule.sqrt"/> returns the mathematically valid <b>complex</b> value
        ///     (<c>np.emath.sqrt(-1) == 1j</c>). The promotion is a WHOLE-ARRAY decision — if ANY element is
        ///     out of the real-valued domain, the ENTIRE array is promoted to complex128 before the standard
        ///     ufunc runs — matching NumPy's <c>_fix_real_lt_zero</c> / <c>_fix_real_abs_gt_1</c> exactly.
        ///     </para>
        /// </remarks>
        public static EmathModule emath { get; } = new EmathModule();
    }

    /// <summary>
    ///     The <c>numpy.emath</c> (<c>numpy.lib.scimath</c>) module surface, reachable as
    ///     <see cref="np.emath"/>. Holds the nine automatic-domain functions
    ///     (<see cref="sqrt"/>, <see cref="log"/>, <see cref="log2"/>, <see cref="log10"/>,
    ///     <see cref="logn"/>, <see cref="power"/>, <see cref="arccos"/>, <see cref="arcsin"/>,
    ///     <see cref="arctanh"/>).
    ///
    ///     <para>
    ///     Every function is a PURE COMPOSITION over the existing <c>np.*</c> ufuncs — a domain scan
    ///     (<c>np.any</c> of a comparison), an optional cast to complex128, then the standard real/complex
    ///     ufunc — a line-for-line port of NumPy 2.4.2 <c>numpy/lib/_scimath_impl.py</c>. Because the
    ///     numerics ARE the already-validated <c>np.sqrt</c>/<c>np.log</c>/<c>np.arccos</c>/… kernels,
    ///     <b>emath introduces no divergence of its own</b> — every difference from NumPy traces to a
    ///     documented property of the underlying ufunc (validated by an 897-case layout-aware
    ///     differential, 775 bit-exact + 122 accounted). Results are <b>bit-identical</b> to NumPy on
    ///     every REAL path and on the bit-exact COMPLEX paths (complex <c>sqrt</c>/<c>log</c>/
    ///     <c>log10</c>/<c>log2</c>); the remaining complex paths inherit the underlying complex ufunc's
    ///     precision: <see cref="arccos"/>/<see cref="arcsin"/>/<see cref="arctanh"/> ride the
    ///     complex-unary ≤~1-ULP envelope (a property of <c>np.arccos</c> etc. itself — e.g. plain
    ///     <c>np.arccos(-0.5+0j)</c> is already 1 ULP from NumPy — not something emath adds), and
    ///     <see cref="power"/> with a non-integer exponent is <c>allclose</c> (the F5 complex-<c>power</c>
    ///     divergence, see that method).
    ///     </para>
    ///     <para>
    ///     <b>ONE dtype-only divergence (F1):</b> NumSharp has a single complex type (complex128) and no
    ///     complex64, so a triggering <c>float32</c>/<c>float16</c>/small-int input returns complex128 where
    ///     NumPy returns complex64. The VALUES are bit-identical; only the dtype width differs (the
    ///     library-wide "no complex64" divergence, issue #569).
    ///     </para>
    /// </summary>
    /// <remarks>https://numpy.org/doc/stable/reference/routines.emath.html</remarks>
    [ModuleName("np.emath")]
    public partial class EmathModule
    {
        internal EmathModule() { }

        /// <summary>
        ///     Compute the square root of <paramref name="x"/>, returning a <b>complex</b> result for
        ///     negative real inputs (unlike <see cref="np.sqrt(NDArray, NDArray, NDArray, DType)"/>, which
        ///     returns <c>nan</c>). Port of NumPy 2.4.2 <c>numpy.emath.sqrt</c>.
        /// </summary>
        /// <param name="x">The input value(s). A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>sqrt(x)</c>. If any real element is <c>&lt; 0</c> (or the input is already complex) the
        ///     result is complex128; otherwise it is the ordinary real result (integer/bool promote to a
        ///     float per <see cref="np.sqrt(NDArray, NDArray, NDArray, DType)"/>). A scalar input yields a
        ///     0-D array. <c>-0.0</c> does NOT trigger the complex promotion (<c>-0.0 &lt; 0</c> is false),
        ///     matching NumPy.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.sqrt.html</remarks>
        [NDScoped]
        public NDArray sqrt(NDArray x) => np.sqrt(FixRealLtZero(x));

        /// <summary>
        ///     Compute the natural logarithm of <paramref name="x"/>, returning the complex principal value
        ///     for negative real inputs (unlike <see cref="np.log(NDArray)"/>, which returns <c>nan</c>).
        ///     Port of NumPy 2.4.2 <c>numpy.emath.log</c>.
        /// </summary>
        /// <param name="x">The value(s) whose natural log is required. A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>log(x)</c>. <c>log(0)</c> is <c>-inf</c> and <c>log(inf)</c> is <c>inf</c> (both stay
        ///     real); a negative real element promotes the whole array to complex128
        ///     (<c>log(-e) == 1 + pi·i</c>). A 0-D array for a scalar input.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.log.html</remarks>
        [NDScoped]
        public NDArray log(NDArray x) => np.log(FixRealLtZero(x));

        /// <summary>
        ///     Compute the base-10 logarithm of <paramref name="x"/>, returning the complex principal value
        ///     for negative real inputs. Port of NumPy 2.4.2 <c>numpy.emath.log10</c>.
        /// </summary>
        /// <param name="x">The value(s) whose base-10 log is required. A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>log10(x)</c>. <c>log10(0)</c> is <c>-inf</c>; a negative real element promotes the whole
        ///     array to complex128. A 0-D array for a scalar input.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.log10.html</remarks>
        [NDScoped]
        public NDArray log10(NDArray x) => np.log10(FixRealLtZero(x));

        /// <summary>
        ///     Compute the base-2 logarithm of <paramref name="x"/>, returning the complex principal value
        ///     for negative real inputs. Port of NumPy 2.4.2 <c>numpy.emath.log2</c>.
        /// </summary>
        /// <param name="x">The value(s) whose base-2 log is required. A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>log2(x)</c>. <c>log2(0)</c> is <c>-inf</c>; a negative real element promotes the whole
        ///     array to complex128. A 0-D array for a scalar input.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.log2.html</remarks>
        [NDScoped]
        public NDArray log2(NDArray x) => np.log2(FixRealLtZero(x));

        /// <summary>
        ///     Take the log base <paramref name="n"/> of <paramref name="x"/>, computed and returned in the
        ///     complex domain when either argument has a negative real element. Port of NumPy 2.4.2
        ///     <c>numpy.emath.logn</c> — literally <c>log(x) / log(n)</c> after both are domain-fixed.
        /// </summary>
        /// <param name="n">The base(s) of the logarithm. A scalar or C# array converts implicitly.</param>
        /// <param name="x">The value(s) whose log base <paramref name="n"/> is required.</param>
        /// <returns>
        ///     <c>log(x) / log(n)</c>. A negative real element in EITHER operand promotes that operand to
        ///     complex128 (so the quotient is complex); otherwise the result is real (float64). <paramref name="n"/>
        ///     and <paramref name="x"/> broadcast against each other via the division. Note the argument
        ///     order is <c>(n, x)</c>, matching NumPy.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.logn.html</remarks>
        [NDScoped]
        public NDArray logn(NDArray n, NDArray x)
        {
            // NumPy fixes BOTH operands independently, then divides. A negative element in n (e.g. logn(-2, 8))
            // makes log(n) complex, so real log(x) / complex log(n) promotes the quotient to complex — exactly
            // NumPy's `nx.log(x) / nx.log(n)`.
            x = FixRealLtZero(x);
            n = FixRealLtZero(n);
            return np.log(x) / np.log(n);
        }

        /// <summary>
        ///     Return <paramref name="x"/> raised to the power <paramref name="p"/> (<c>x**p</c>), promoting
        ///     to the complex domain when <paramref name="x"/> has a negative real element. Port of NumPy
        ///     2.4.2 <c>numpy.emath.power</c>.
        /// </summary>
        /// <param name="x">The base value(s). A scalar or C# array converts implicitly.</param>
        /// <param name="p">
        ///     The exponent(s). Conceptually integers, but a negative exponent is legal here: unlike
        ///     <see cref="np.power(NDArray, NDArray)"/> (which raises "Integers to negative integer powers"
        ///     for an integer base), a negative real element in <paramref name="p"/> promotes <paramref name="p"/>
        ///     to a float first (NumPy's <c>_fix_int_lt_zero</c> = <c>p * 1.0</c>).
        /// </param>
        /// <returns>
        ///     <c>power(x, p)</c> with <paramref name="x"/> and <paramref name="p"/> broadcast together. A
        ///     negative base promotes the result to complex128; a negative exponent promotes <paramref name="p"/>
        ///     to float (so <c>power([2,4], -2) == [0.25, 0.0625]</c>). <b>Divergence (F5):</b> a negative base
        ///     with a NON-integer exponent takes the complex-<c>power</c> path, which is <c>allclose</c> to
        ///     NumPy but not bit-exact (the documented <c>Complex.Pow</c> vs <c>npy_cpow</c> ULP divergence);
        ///     an integer exponent is bit-exact.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.power.html</remarks>
        [NDScoped]
        public NDArray power(NDArray x, NDArray p)
        {
            // Base: negative real element -> complex128. Exponent: negative real element -> float (so a
            // negative integer exponent does not hit np.power's integer-domain "negative integer power" error).
            x = FixRealLtZero(x);
            p = FixIntLtZero(p);
            return np.power(x, p);
        }

        /// <summary>
        ///     Compute the inverse cosine of <paramref name="x"/>, returning the complex principal value for
        ///     real inputs with <c>|x| &gt; 1</c> (unlike <see cref="np.arccos(NDArray, NDArray, NDArray, DType)"/>,
        ///     which returns <c>nan</c> there). Port of NumPy 2.4.2 <c>numpy.emath.arccos</c>.
        /// </summary>
        /// <param name="x">The value(s) whose arccos is required. A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>arccos(x)</c>. For <c>|x| &lt;= 1</c> a real result in <c>[0, pi]</c>; any real element with
        ///     <c>|x| &gt; 1</c> (or an infinite element, whose <c>|x| &gt; 1</c>) promotes the whole array to
        ///     complex128. A <c>nan</c> element does NOT trigger the promotion (<c>|nan| &gt; 1</c> is false).
        ///     A 0-D array for a scalar input.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.arccos.html</remarks>
        [NDScoped]
        public NDArray arccos(NDArray x) => np.arccos(FixRealAbsGt1(x));

        /// <summary>
        ///     Compute the inverse sine of <paramref name="x"/>, returning the complex principal value for
        ///     real inputs with <c>|x| &gt; 1</c>. Port of NumPy 2.4.2 <c>numpy.emath.arcsin</c>.
        /// </summary>
        /// <param name="x">The value(s) whose arcsin is required. A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>arcsin(x)</c>. For <c>|x| &lt;= 1</c> a real result in <c>[-pi/2, pi/2]</c>; any real element
        ///     with <c>|x| &gt; 1</c> promotes the whole array to complex128. A 0-D array for a scalar input.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.arcsin.html</remarks>
        [NDScoped]
        public NDArray arcsin(NDArray x) => np.arcsin(FixRealAbsGt1(x));

        /// <summary>
        ///     Compute the inverse hyperbolic tangent of <paramref name="x"/>, returning the complex
        ///     principal value for real inputs with <c>|x| &gt; 1</c>. Port of NumPy 2.4.2
        ///     <c>numpy.emath.arctanh</c>.
        /// </summary>
        /// <param name="x">The value(s) whose arctanh is required. A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>arctanh(x)</c>. For <c>|x| &lt; 1</c> a real result; any real element with <c>|x| &gt; 1</c>
        ///     promotes the whole array to complex128. The boundary <c>|x| == 1</c> stays REAL and yields
        ///     <c>±inf</c> (the promotion is strictly <c>|x| &gt; 1</c>, matching NumPy). A 0-D array for a
        ///     scalar input.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.arctanh.html</remarks>
        [NDScoped]
        public NDArray arctanh(NDArray x) => np.arctanh(FixRealAbsGt1(x));

        // ── Domain-fix helpers (ports of numpy/lib/_scimath_impl.py) ──────────────────────────────
        // Each returns the input UNCHANGED unless a whole-array complex/float promotion is required, in
        // which case a fresh copy is returned (matching NumPy's `_tocomplex`/`x*1.0`, both of which copy).
        // A COMPLEX input never enters the scan: NumPy's `isreal(x) & (x<0)` is false for a complex element
        // (imag != 0), and where it IS real-and-negative the `_tocomplex` re-copy is dtype-idempotent for an
        // already-complex128 array — so the standard complex ufunc applied to the untouched input produces
        // the identical result. Skipping the scan for complex input is therefore an exact, cheaper shortcut.

        /// <summary>
        ///     NumPy's <c>_fix_real_lt_zero</c>: convert <paramref name="x"/> to complex128 if it is a real
        ///     array with ANY element <c>&lt; 0</c>; otherwise return it unchanged. Drives
        ///     <see cref="sqrt"/>/<see cref="log"/>/<see cref="log10"/>/<see cref="log2"/>/<see cref="logn"/>
        ///     and the base of <see cref="power"/>.
        /// </summary>
        /// <param name="x">The array to inspect (any real dtype, or already complex).</param>
        /// <returns>
        ///     <paramref name="x"/> unchanged when it is complex, empty, or has no negative real element; a
        ///     complex128 copy when any real element is strictly negative. The scan is
        ///     <c>np.any(x &lt; 0)</c> — a NaN never triggers (<c>nan &lt; 0</c> is false) and <c>-0.0</c>
        ///     never triggers, both matching NumPy's element-wise <c>x &lt; 0</c>.
        /// </returns>
        [NDScoped]
        private static NDArray FixRealLtZero(NDArray x)
        {
            if (x.typecode == NPTypeCode.Complex)
                return x; // complex in -> complex ufunc out; the scan/_tocomplex would be a value-identical no-op
            // Fused early-exit scan for the hot contiguous float32/float64 case (no abs/bool temp, one pass);
            // null => not the fast path, so use the composition, whose `x < 0` boolean is layout- and
            // promotion-invariant (an element is negative or it is not) and equals `any(isreal(x) & (x<0))`
            // for a real dtype (where isreal is all-true). np.any early-exits at the first true lane.
            bool neg = Backends.Kernels.EmathDomainScan.AnyNegative(x) ?? np.any(x < (NDArray)0);
            return neg ? x.astype(np.complex128) : x;
        }

        /// <summary>
        ///     NumPy's <c>_fix_real_abs_gt_1</c>: convert <paramref name="x"/> to complex128 if it is a real
        ///     array with ANY element whose magnitude exceeds 1; otherwise return it unchanged. Drives
        ///     <see cref="arccos"/>/<see cref="arcsin"/>/<see cref="arctanh"/>.
        /// </summary>
        /// <param name="x">The array to inspect (any real dtype, or already complex).</param>
        /// <returns>
        ///     <paramref name="x"/> unchanged when it is complex, empty, or every element has <c>|x| &lt;= 1</c>;
        ///     a complex128 copy when any real element has <c>|x| &gt; 1</c>. The comparison is strict, so
        ///     <c>|x| == 1</c> stays real; a <c>nan</c> never triggers (<c>|nan| &gt; 1</c> is false) while
        ///     an infinity does (<c>|inf| &gt; 1</c> is true) — all matching NumPy's <c>abs(x) &gt; 1</c>.
        /// </returns>
        [NDScoped]
        private static NDArray FixRealAbsGt1(NDArray x)
        {
            if (x.typecode == NPTypeCode.Complex)
                return x;
            bool big = Backends.Kernels.EmathDomainScan.AnyAbsGreaterThanOne(x) ?? np.any(np.abs(x) > (NDArray)1);
            return big ? x.astype(np.complex128) : x;
        }

        /// <summary>
        ///     NumPy's <c>_fix_int_lt_zero</c>: promote <paramref name="p"/> to a float (via <c>p * 1.0</c>)
        ///     if it is a real array with ANY element <c>&lt; 0</c>; otherwise return it unchanged. Used for
        ///     the EXPONENT of <see cref="power"/> so that a negative integer exponent does not hit
        ///     <c>np.power</c>'s integer-domain error.
        /// </summary>
        /// <param name="p">The exponent array to inspect (any real dtype, or already complex).</param>
        /// <returns>
        ///     <paramref name="p"/> unchanged when it is complex or has no negative real element; otherwise
        ///     <c>p * 1.0</c> — a WEAK multiply (NEP50), so an integer <paramref name="p"/> becomes float64
        ///     while a <c>float32</c>/<c>float16</c> <paramref name="p"/> keeps its width (identity for finite
        ///     values, <c>-0.0</c>/<c>±inf</c>/<c>nan</c> preserved). NOT a cast to complex — only the base
        ///     promotes to complex.
        /// </returns>
        [NDScoped]
        private static NDArray FixIntLtZero(NDArray p)
        {
            if (p.typecode == NPTypeCode.Complex)
                return p;
            bool neg = Backends.Kernels.EmathDomainScan.AnyNegative(p) ?? np.any(p < (NDArray)0);
            return neg ? p * 1.0 : p;
        }
    }
}
