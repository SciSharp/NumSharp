// NumPy's numpy/lib/_scimath_impl.py (numpy.emath) as DefaultEngine's members: each function's statements in NumPy's
// order, over the ufuncs' engine members (each np.* ufunc's statement: this engine, or the operand's engine in a static
// helper). The facade is Math/np.emath.cs.

using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped] // reclaims the complex copy _fix_real_lt_zero may make
        public override NDArray EmathSqrt(NDArray x)
            => Sqrt(FixRealLtZero(x), null, null, null);

        /// <inheritdoc />
        [NDScoped] // reclaims the complex copy _fix_real_lt_zero may make
        public override NDArray EmathLog(NDArray x)
            => Log(FixRealLtZero(x));

        /// <inheritdoc />
        [NDScoped] // reclaims the complex copy _fix_real_lt_zero may make
        public override NDArray EmathLog10(NDArray x)
            => Log10(FixRealLtZero(x));

        /// <inheritdoc />
        [NDScoped] // reclaims the complex copy _fix_real_lt_zero may make
        public override NDArray EmathLog2(NDArray x)
            => Log2(FixRealLtZero(x));

        /// <inheritdoc />
        [NDScoped] // reclaims the two converted operands and the two logarithms
        public override NDArray EmathLogn(NDArray n, NDArray x)
        {
            // NumPy fixes x first, then n (an error in x wins), each on its own, then divides: a negative element in n
            // alone (logn(-2, 8)) makes log(n) complex, and the real log(x) over it promotes the quotient to complex.
            x = FixRealLtZero(x);
            n = FixRealLtZero(n);
            return Log(x) / Log(n);
        }

        /// <inheritdoc />
        [NDScoped] // reclaims the converted base and exponent
        public override NDArray EmathPower(NDArray x, NDArray p)
        {
            // Base: a negative real element -> complex. Exponent: a negative real element -> float, so a negative
            // integer exponent does not reach np.power's integer-domain "negative integer power" error.
            x = FixRealLtZero(x);
            p = FixIntLtZero(p);
            return Power(x, p);   // np.power(x1, x2)'s statement (its null dtype)
        }

        /// <inheritdoc />
        [NDScoped] // reclaims the complex copy _fix_real_abs_gt_1 may make
        public override NDArray EmathArccos(NDArray x)
            => ACos(FixRealAbsGt1(x), null, null, null);

        /// <inheritdoc />
        [NDScoped] // reclaims the complex copy _fix_real_abs_gt_1 may make
        public override NDArray EmathArcsin(NDArray x)
            => ASin(FixRealAbsGt1(x), null, null, null);

        /// <inheritdoc />
        [NDScoped] // reclaims the complex copy _fix_real_abs_gt_1 may make
        public override NDArray EmathArctanh(NDArray x)
            => ATanh(FixRealAbsGt1(x), null, null, null);

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
        ///     <see cref="EmathSqrt"/>/<see cref="EmathLog"/>/<see cref="EmathLog10"/>/<see cref="EmathLog2"/>/<see cref="EmathLogn"/>
        ///     and the base of <see cref="EmathPower"/>.
        /// </summary>
        /// <param name="x">The array to inspect (any real dtype, or already complex).</param>
        /// <returns>
        ///     <paramref name="x"/> unchanged when it is complex, empty, or has no negative real element; a
        ///     complex128 copy when any real element is strictly negative. The scan is
        ///     <c>np.any(x &lt; 0)</c> — a NaN never triggers (<c>nan &lt; 0</c> is false) and <c>-0.0</c>
        ///     never triggers, both matching NumPy's element-wise <c>x &lt; 0</c>.
        /// </returns>
        /// <exception cref="TypeError">Text (<see cref="LessThanZeroTextError"/>).</exception>
        [NDScoped]
        private static NDArray FixRealLtZero(NDArray x)
        {
            if (x.typecode == NPTypeCode.Complex)
                return x; // complex in -> complex ufunc out; the scan/_tocomplex would be a value-identical no-op
            // Text: NumPy's `x < 0` has no loop for text and a Python int, whatever the size (an empty array included,
            // which the fast scan below would skip).
            if (NDStringUfuncs.IsText(x))
                throw LessThanZeroTextError(x);
            // Fused early-exit scan for the hot contiguous float32/float64 case (no abs/bool temp, one pass);
            // null => not the fast path, so use the composition, whose `x < 0` boolean is layout- and
            // promotion-invariant (an element is negative or it is not) and equals `any(isreal(x) & (x<0))`
            // for a real dtype (where isreal is all-true). np.any early-exits at the first true lane.
            // np.any's statement on the comparison: the engine of x (the comparison's own) answers Any.
            bool neg = EmathDomainScan.AnyNegative(x) ?? x.TensorEngine.Any(x < (NDArray)0);
            return neg ? x.astype(np.complex128) : x;
        }

        /// <summary>
        ///     NumPy's <c>_fix_real_abs_gt_1</c>: convert <paramref name="x"/> to complex128 if it is a real
        ///     array with ANY element whose magnitude exceeds 1; otherwise return it unchanged. Drives
        ///     <see cref="EmathArccos"/>/<see cref="EmathArcsin"/>/<see cref="EmathArctanh"/>.
        /// </summary>
        /// <param name="x">The array to inspect (any real dtype, or already complex).</param>
        /// <returns>
        ///     <paramref name="x"/> unchanged when it is complex, empty, or every element has <c>|x| &lt;= 1</c>;
        ///     a complex128 copy when any real element has <c>|x| &gt; 1</c>. The comparison is strict, so
        ///     <c>|x| == 1</c> stays real; a <c>nan</c> never triggers (<c>|nan| &gt; 1</c> is false) while
        ///     an infinity does (<c>|inf| &gt; 1</c> is true) — all matching NumPy's <c>abs(x) &gt; 1</c>.
        /// </returns>
        /// <exception cref="TypeError">Text: NumPy's <c>abs(x)</c> has no text loop (<c>ufunc 'absolute' did not contain a
        ///     loop with signature matching types &lt;class 'numpy.dtypes.StrDType'&gt; -&gt; None</c>), even for an empty
        ///     array.</exception>
        [NDScoped]
        private static NDArray FixRealAbsGt1(NDArray x)
        {
            if (x.typecode == NPTypeCode.Complex)
                return x;
            if (NDStringUfuncs.IsText(x))
                throw new TypeError($"ufunc 'absolute' did not contain a loop with signature matching types {x.dtype.Meta} -> None");
            bool big = EmathDomainScan.AnyAbsGreaterThanOne(x) ?? x.TensorEngine.Any(x.TensorEngine.Abs(x) > (NDArray)1);
            return big ? x.astype(np.complex128) : x;
        }

        /// <summary>
        ///     NumPy's <c>_fix_int_lt_zero</c>: promote <paramref name="p"/> to a float (via <c>p * 1.0</c>)
        ///     if it is a real array with ANY element <c>&lt; 0</c>; otherwise return it unchanged. Used for
        ///     the EXPONENT of <see cref="EmathPower"/> so that a negative integer exponent does not hit
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
        /// <exception cref="TypeError">Text (<see cref="LessThanZeroTextError"/>).</exception>
        [NDScoped]
        private static NDArray FixIntLtZero(NDArray p)
        {
            if (p.typecode == NPTypeCode.Complex)
                return p;
            if (NDStringUfuncs.IsText(p))
                throw LessThanZeroTextError(p);
            bool neg = EmathDomainScan.AnyNegative(p) ?? p.TensorEngine.Any(p < (NDArray)0);
            return neg ? p * 1.0 : p;
        }

        /// <summary>
        ///     NumPy's refusal of the domain scans' <c>x &lt; 0</c> for text: the comparison promoter names the text class and
        ///     the Python int's abstract class, <c>ufunc 'less' did not contain a loop with signature matching types
        ///     (&lt;class 'numpy.dtypes.StrDType'&gt;, &lt;class 'numpy.dtypes._PyLongDType'&gt;) -&gt; None</c> (C#'s own
        ///     <c>(NDArray)0</c> is a strong int32 array and would name <c>Int32DType</c>).
        /// </summary>
        /// <param name="x">The text array.</param>
        /// <returns>The exception (not thrown).</returns>
        private static TypeError LessThanZeroTextError(NDArray x)
            => new TypeError($"ufunc 'less' did not contain a loop with signature matching types ({x.dtype.Meta}, {DTypeRegistry.PyLong}) -> None");
    }
}
