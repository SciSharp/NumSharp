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
    ///     Each function's body is its engine member (<see cref="TensorEngine.EmathSqrt"/> and its siblings
    ///     in <c>TensorEngine.Scimath.cs</c>), which <c>DefaultEngine</c> implements as a PURE COMPOSITION over
    ///     the existing <c>np.*</c> ufuncs — a domain scan (<c>np.any</c> of a comparison), an optional cast to
    ///     complex128, then the standard real/complex ufunc — a line-for-line port of NumPy 2.4.2
    ///     <c>numpy/lib/_scimath_impl.py</c>. The methods here keep the signatures, the class-boundary
    ///     statements and one call. Because the
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
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.sqrt.html</remarks>
        [NDScoped]
        public NDArray sqrt(NDArray x)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((self: this, x), static s => s.self.sqrt(s.x))
            : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((self: this, x), static s => s.self.sqrt(s.x))
            : x.TensorEngine.EmathSqrt(x);

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
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.log.html</remarks>
        [NDScoped]
        public NDArray log(NDArray x)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((self: this, x), static s => s.self.log(s.x))
            : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((self: this, x), static s => s.self.log(s.x))
            : x.TensorEngine.EmathLog(x);

        /// <summary>
        ///     Compute the base-10 logarithm of <paramref name="x"/>, returning the complex principal value
        ///     for negative real inputs. Port of NumPy 2.4.2 <c>numpy.emath.log10</c>.
        /// </summary>
        /// <param name="x">The value(s) whose base-10 log is required. A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>log10(x)</c>. <c>log10(0)</c> is <c>-inf</c>; a negative real element promotes the whole
        ///     array to complex128. A 0-D array for a scalar input.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.log10.html</remarks>
        [NDScoped]
        public NDArray log10(NDArray x)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((self: this, x), static s => s.self.log10(s.x))
            : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((self: this, x), static s => s.self.log10(s.x))
            : x.TensorEngine.EmathLog10(x);

        /// <summary>
        ///     Compute the base-2 logarithm of <paramref name="x"/>, returning the complex principal value
        ///     for negative real inputs. Port of NumPy 2.4.2 <c>numpy.emath.log2</c>.
        /// </summary>
        /// <param name="x">The value(s) whose base-2 log is required. A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>log2(x)</c>. <c>log2(0)</c> is <c>-inf</c>; a negative real element promotes the whole
        ///     array to complex128. A 0-D array for a scalar input.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.log2.html</remarks>
        [NDScoped]
        public NDArray log2(NDArray x)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((self: this, x), static s => s.self.log2(s.x))
            : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((self: this, x), static s => s.self.log2(s.x))
            : x.TensorEngine.EmathLog2(x);

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
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.logn.html</remarks>
        [NDScoped]
        public NDArray logn(NDArray n, NDArray x)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(n, x))
                return NDMatrixBoundary.Plain((self: this, n, x), static s => s.self.logn(s.n, s.x));
            // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(n, x))
                return NDCharArrayBoundary.Plain((self: this, n, x), static s => s.self.logn(s.n, s.x));
            return n.TensorEngine.EmathLogn(n, x);
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
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.power.html</remarks>
        [NDScoped]
        public NDArray power(NDArray x, NDArray p)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(x, p))
                return NDMatrixBoundary.Plain((self: this, x, p), static s => s.self.power(s.x, s.p));
            // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(x, p))
                return NDCharArrayBoundary.Plain((self: this, x, p), static s => s.self.power(s.x, s.p));
            return x.TensorEngine.EmathPower(x, p);
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
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.arccos.html</remarks>
        [NDScoped]
        public NDArray arccos(NDArray x)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((self: this, x), static s => s.self.arccos(s.x))
            : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((self: this, x), static s => s.self.arccos(s.x))
            : x.TensorEngine.EmathArccos(x);

        /// <summary>
        ///     Compute the inverse sine of <paramref name="x"/>, returning the complex principal value for
        ///     real inputs with <c>|x| &gt; 1</c>. Port of NumPy 2.4.2 <c>numpy.emath.arcsin</c>.
        /// </summary>
        /// <param name="x">The value(s) whose arcsin is required. A scalar or C# array converts implicitly.</param>
        /// <returns>
        ///     <c>arcsin(x)</c>. For <c>|x| &lt;= 1</c> a real result in <c>[-pi/2, pi/2]</c>; any real element
        ///     with <c>|x| &gt; 1</c> promotes the whole array to complex128. A 0-D array for a scalar input.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.arcsin.html</remarks>
        [NDScoped]
        public NDArray arcsin(NDArray x)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((self: this, x), static s => s.self.arcsin(s.x))
            : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((self: this, x), static s => s.self.arcsin(s.x))
            : x.TensorEngine.EmathArcsin(x);

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
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.emath.arctanh.html</remarks>
        [NDScoped]
        public NDArray arctanh(NDArray x)
            => NDMatrixBoundary.Sees(x)   // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            ? NDMatrixBoundary.Plain((self: this, x), static s => s.self.arctanh(s.x))
            : NDCharArrayBoundary.Sees(x)   // a numpy.char.chararray operand too: asarray drops either class
            ? NDCharArrayBoundary.Plain((self: this, x), static s => s.self.arctanh(s.x))
            : x.TensorEngine.EmathArctanh(x);
    }
}
