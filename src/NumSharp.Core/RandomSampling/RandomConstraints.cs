using System;

namespace NumSharp
{
    /// <summary>
    ///     The parameter constraints NumPy's random samplers enforce before drawing — a 1-to-1 port of
    ///     <c>numpy/random/_common.pxd</c>'s <c>ConstraintType</c>. Each member names the rule its
    ///     <see cref="RandomConstraints.Check(double, string, ConstraintType)"/> branch applies.
    /// </summary>
    /// <remarks>
    ///     The member names are NumPy's own (<c>CONS_*</c>) so a sampler's constraint list reads exactly like the
    ///     <c>cont(...)</c>/<c>disc(...)</c> call it ports in <c>mtrand.pyx</c> / <c>_generator.pyx</c>.
    /// </remarks>
    internal enum ConstraintType
    {
        /// <summary>No check (NumPy's <c>CONS_NONE</c>).</summary>
        CONS_NONE,

        /// <summary><c>val &gt;= 0</c> by SIGN BIT for doubles — <c>-0.0</c> is rejected, NaN passes (<c>"{name} &lt; 0"</c>).</summary>
        CONS_NON_NEGATIVE,

        /// <summary><c>val &gt; 0</c>; NaN passes (<c>"{name} &lt;= 0"</c>).</summary>
        CONS_POSITIVE,

        /// <summary><c>val &gt; 0</c> and not NaN (<c>"{name} must not be NaN"</c> / <c>"{name} &lt;= 0"</c>).</summary>
        CONS_POSITIVE_NOT_NAN,

        /// <summary><c>0 &lt;= val &lt;= 1</c>, NaN rejected (<c>"{name} &lt; 0, {name} &gt; 1 or {name} is NaN"</c>).</summary>
        CONS_BOUNDED_0_1,

        /// <summary><c>0 &lt; val &lt;= 1</c>, NaN rejected (<c>"{name} &lt;= 0, {name} &gt; 1 or {name} contains NaNs"</c>).</summary>
        CONS_BOUNDED_GT_0_1,

        /// <summary><c>0 &lt;= val &lt; 1</c>, NaN rejected (<c>"{name} &lt; 0, {name} &gt;= 1 or {name} is NaN"</c>).</summary>
        CONS_BOUNDED_LT_0_1,

        /// <summary><c>val &gt; 1</c>, NaN rejected (<c>"{name} &lt;= 1 or {name} is NaN"</c>).</summary>
        CONS_GT_1,

        /// <summary><c>val &gt;= 1</c>, NaN rejected (<c>"{name} &lt; 1 or {name} is NaN"</c>).</summary>
        CONS_GTE_1,

        /// <summary>
        ///     <c>0 &lt;= val &lt;= POISSON_LAM_MAX</c> (the <see cref="Generator"/> poisson bound), NaN rejected
        ///     (<c>"{name} &lt; 0 or {name} is NaN"</c> / <c>"{name} value too large"</c>).
        /// </summary>
        CONS_POISSON,

        /// <summary>
        ///     The legacy <c>RandomState.poisson</c> bound, <c>0 &lt;= val &lt;= LEGACY_POISSON_LAM_MAX</c> — sized for C
        ///     <c>long</c> (see <see cref="RandomConstraints.LegacyPoissonLamMax"/>).
        /// </summary>
        LEGACY_CONS_POISSON,

        /// <summary>
        ///     The legacy integer-count bound, <c>0 &lt;= val &lt;= LONG_MAX</c> (<c>"{name} &lt; 0"</c> /
        ///     <c>"{name} is out of bounds for long, consider using the new generator API for 64bit integers."</c>).
        /// </summary>
        LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG,
    }

    /// <summary>
    ///     Port of <c>numpy/random/_common.pyx</c>'s scalar <c>check_constraint</c>: the validation every random sampler
    ///     runs on its parameters BEFORE the output is allocated or a single value is drawn.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The comparisons are written exactly as NumPy writes them — <c>not (val &gt;= 0)</c> rather than
    ///     <c>val &lt; 0</c> — because the difference is observable: the negated forms reject NaN, the plain
    ///     <c>val &lt;= 0</c> of <see cref="ConstraintType.CONS_POSITIVE"/> lets NaN through (NumPy then samples a NaN), and
    ///     <see cref="ConstraintType.CONS_NON_NEGATIVE"/> tests the SIGN BIT so <c>-0.0</c> is rejected while NaN is not.
    ///     </para>
    ///     <para>
    ///     Every failure is a <see cref="ValueError"/> carrying NumPy's message verbatim, so the error surface of a ported
    ///     sampler matches NumPy's text as well as its type.
    ///     </para>
    /// </remarks>
    internal static class RandomConstraints
    {
        /// <summary>
        ///     NumPy's <c>POISSON_LAM_MAX</c> = <c>&lt;double&gt;np.iinfo('int64').max - np.sqrt(np.iinfo('int64').max) * 10</c>
        ///     (<c>0x1.ffffffe3b73a0p+62</c>): the largest lambda whose Poisson draws stay (with overwhelming probability)
        ///     inside int64.
        /// </summary>
        internal const double PoissonLamMax = 9.223372006484771e+18;

        /// <summary>
        ///     NumPy's <c>LEGACY_POISSON_LAM_MAX</c>, the same formula over C <c>long</c> (<c>np.iinfo('l')</c>). NumSharp's
        ///     legacy integer samplers return int64 — NumPy's LP64 (Linux/macOS) shape — so C <c>long</c> is 64-bit here and
        ///     the bound equals <see cref="PoissonLamMax"/>.
        /// </summary>
        /// <remarks>
        ///     NumPy's win-amd64 build has a 32-bit <c>long</c> and a bound of <c>2147020237.4999895</c>; there
        ///     <c>RandomState.poisson(9.2e18)</c> raises <c>lam value too large</c> where NumSharp (like NumPy on Linux) draws.
        /// </remarks>
        internal const double LegacyPoissonLamMax = PoissonLamMax;

        /// <summary>
        ///     C <c>LONG_MAX</c> as the double NumPy compares against in
        ///     <see cref="ConstraintType.LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG"/> — <c>(double)long.MaxValue</c>, i.e. 2^63,
        ///     under the LP64 model <see cref="LegacyPoissonLamMax"/> documents.
        /// </summary>
        internal const double LegacyLongMax = long.MaxValue;

        /// <summary>
        ///     The dtype gate of NumPy's array-parameter conversions (<c>PyArray_FROM_OTF</c> / <c>PyArray_FROMANY</c> without
        ///     <c>NPY_ARRAY_FORCECAST</c>): the array must reach <paramref name="to"/> under the <c>'safe'</c> casting rule,
        ///     or the conversion raises before any value is read.
        /// </summary>
        /// <param name="a">The array parameter as the caller passed it.</param>
        /// <param name="to">The dtype NumPy converts the parameter to (<c>int64</c> for counts, <c>float64</c> for
        ///     probabilities and concentrations).</param>
        /// <exception cref="TypeError">The cast is not safe — a float or uint64 count, a complex probability. The text is
        ///     NumPy's, including its <c>scalar</c> wording for a 0-d array:
        ///     <c>Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'</c>.</exception>
        /// <remarks>
        ///     Decimal (no NumPy dtype) is let through to <c>float64</c> as a NumSharp extension: its <c>can_cast</c> row says
        ///     "unsafe", but its NumPy-facing name is <c>float64</c>, so NumPy's message would read "cannot cast float64 to
        ///     float64". Toward <c>int64</c> it is refused like the float it stands in for.
        /// </remarks>
        internal static void CheckSafeCast(NDArray a, NPTypeCode to)
        {
            var from = a.typecode;
            if (from == to || np.can_cast(from, to, "safe") || (from == NPTypeCode.Decimal && to == NPTypeCode.Double))
                return;
            throw new TypeError($"Cannot cast {(a.ndim == 0 ? "scalar" : "array data")} from dtype('{from.AsNumpyDtypeName()}') "
                                + $"to dtype('{to.AsNumpyDtypeName()}') according to the rule 'safe'");
        }

        /// <summary>
        ///     Validates one scalar parameter the way NumPy's <c>check_constraint(double val, name, cons)</c> does.
        /// </summary>
        /// <param name="val">The parameter value (integer parameters are passed as the double NumPy casts them to).</param>
        /// <param name="name">The parameter name as it appears in NumPy's message (<c>"scale"</c>, <c>"p"</c>, …).</param>
        /// <param name="cons">The rule to apply; <see cref="ConstraintType.CONS_NONE"/> accepts everything.</param>
        /// <exception cref="ValueError">The value violates <paramref name="cons"/>; the message is NumPy's, verbatim.</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="cons"/> is not a defined <see cref="ConstraintType"/> (a caller bug).</exception>
        internal static void Check(double val, string name, ConstraintType cons)
        {
            switch (cons)
            {
                case ConstraintType.CONS_NONE:
                    return;

                case ConstraintType.CONS_NON_NEGATIVE:
                    // signbit, not `< 0`: -0.0 is rejected ("scale < 0" for normal(0, -0.0)) while NaN — whatever its sign
                    // bit — passes and is sampled.
                    if (!double.IsNaN(val) && double.IsNegative(val))
                        throw new ValueError($"{name} < 0");
                    return;

                case ConstraintType.CONS_POSITIVE:
                case ConstraintType.CONS_POSITIVE_NOT_NAN:
                    if (cons == ConstraintType.CONS_POSITIVE_NOT_NAN && double.IsNaN(val))
                        throw new ValueError($"{name} must not be NaN");
                    // Plain `<= 0`: NaN compares false and is let through for CONS_POSITIVE (NumPy then samples NaN).
                    if (val <= 0)
                        throw new ValueError($"{name} <= 0");
                    return;

                case ConstraintType.CONS_BOUNDED_0_1:
                    if (!(val >= 0) || !(val <= 1))
                        throw new ValueError($"{name} < 0, {name} > 1 or {name} is NaN");
                    return;

                case ConstraintType.CONS_BOUNDED_GT_0_1:
                    if (!(val > 0) || !(val <= 1))
                        throw new ValueError($"{name} <= 0, {name} > 1 or {name} contains NaNs");
                    return;

                case ConstraintType.CONS_BOUNDED_LT_0_1:
                    if (!(val >= 0) || !(val < 1))
                        throw new ValueError($"{name} < 0, {name} >= 1 or {name} is NaN");
                    return;

                case ConstraintType.CONS_GT_1:
                    if (!(val > 1))
                        throw new ValueError($"{name} <= 1 or {name} is NaN");
                    return;

                case ConstraintType.CONS_GTE_1:
                    if (!(val >= 1))
                        throw new ValueError($"{name} < 1 or {name} is NaN");
                    return;

                case ConstraintType.CONS_POISSON:
                    // The sign/NaN test runs first, so poisson(nan) reports "is NaN", never "too large".
                    if (!(val >= 0))
                        throw new ValueError($"{name} < 0 or {name} is NaN");
                    if (!(val <= PoissonLamMax))
                        throw new ValueError($"{name} value too large");
                    return;

                case ConstraintType.LEGACY_CONS_POISSON:
                    if (!(val >= 0))
                        throw new ValueError($"{name} < 0 or {name} is NaN");
                    if (!(val <= LegacyPoissonLamMax))
                        throw new ValueError($"{name} value too large");
                    return;

                case ConstraintType.LEGACY_CONS_NON_NEGATIVE_INBOUNDS_LONG:
                    // NumPy assumes an integral value here (the caller cast an integer), so NaN is not tested.
                    if (val < 0)
                        throw new ValueError($"{name} < 0");
                    if (val > LegacyLongMax)
                        throw new ValueError($"{name} is out of bounds for long, consider using the new generator API for 64bit integers.");
                    return;

                default:
                    throw new ArgumentOutOfRangeException(nameof(cons), cons, "Unknown random-parameter constraint.");
            }
        }
    }
}
