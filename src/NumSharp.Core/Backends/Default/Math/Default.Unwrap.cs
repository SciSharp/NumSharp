// NumPy's np.unwrap (numpy/lib/_function_base_impl.py) as DefaultEngine's member: the period's Python type picks the
// float or integer path, then UnwrapImpl runs NumPy's statements (np.diff, the fused pass, np.cumsum: the operand's
// engine members).

using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        public override NDArray Unwrap(NDArray p, double? discont = null, int axis = -1, object period = null)
        {
            // NumPy's period is a Python float (2 pi by default) or a Python int; only an int reaches its integer path.
            switch (period)
            {
                case null:
                    return UnwrapImpl(p, discont, axis, 2 * Math.PI, periodIsInteger: false, periodInt: 0);
                case double d:
                    return UnwrapImpl(p, discont, axis, d, periodIsInteger: false, periodInt: 0);
                case long l:
                    return UnwrapImpl(p, discont, axis, l, periodIsInteger: true, periodInt: l);
                case int i:
                    return UnwrapImpl(p, discont, axis, i, periodIsInteger: true, periodInt: i);
                default:
                    throw new ArgumentException(
                        $"period must be a double (a Python float) or an integer (a Python int), not {period.GetType().Name}",
                        nameof(period));
            }
        }

        /// <summary>
        ///     The statements behind both <c>np.unwrap</c> overloads, once <see cref="Unwrap"/> has read the period's
        ///     Python type. Mirrors NumPy's composition exactly; np.unwrap.cs's file header has the algorithm and the
        ///     integer/float path rationale.
        /// </summary>
        /// <param name="p">Input array.</param>
        /// <param name="discont">Discontinuity threshold, or <c>null</c> for <c>period/2</c>.</param>
        /// <param name="axis">Axis to unwrap along (may be negative).</param>
        /// <param name="period">The wrap size as a real number (also the double image of an integer period).</param>
        /// <param name="periodIsInteger">
        ///     True when the caller supplied an integer-typed period. It selects NumPy's
        ///     integer code path ONLY for an integer/bool/char input; a float input always
        ///     takes the float path regardless (NumPy's <c>issubdtype(result_type(dd, period), integer)</c>).
        /// </param>
        /// <param name="periodInt">The integer period value (valid only when <paramref name="periodIsInteger"/>).</param>
        /// <returns>The unwrapped array (see the overloads for dtype/shape).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="p"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="p"/> is 0-D.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="p"/>'s rank.</exception>
        /// <exception cref="TypeError"><paramref name="p"/> is complex.</exception>
        /// <exception cref="OverflowException">An integer interval bound or the period does not fit the input
        ///     dtype (NumPy's OverflowError, see the public overloads).</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>: <c>diff</c>, the fused correction, its cumulative sum, the
        ///     <c>p[1:]</c>/<c>up[1:]</c> views, the tail and its cast are released on return, and <c>up</c> is
        ///     yielded; on a throw <c>up</c> is released with them. The body no longer repeats that with a
        ///     hand-written catch/finally. One direct <c>Dispose</c> stays: the bool path frees its bool
        ///     <c>diff</c> as soon as the int64 copy exists, so that full-size array is not held to the peak.</remarks>
        [NDScoped]
        private static NDArray UnwrapImpl(NDArray p, double? discont, int axis, double period,
                                          bool periodIsInteger, long periodInt)
        {
            if (p is null) throw new ArgumentNullException(nameof(p));

            int nd = p.ndim;
            // NumPy computes diff(p) first, and diff rejects a 0-D input with this exact
            // message BEFORE any axis handling — so a 0-D array never reaches axis
            // normalization (matching the order in which NumPy would raise).
            if (nd == 0)
                throw new ValueError("diff requires input that is at least one dimensional");   // np.diff's ValueError, which unwrap's first statement raises

            // normalize_axis_index(axis, nd): valid range is [-nd, nd-1]. NumPy raises
            // AxisError here (via normalize_axis_index) reporting the ORIGINAL axis.
            int ax = Backends.DefaultEngine.check_and_adjust_axis(nd, axis);

            // diff's subtract runs after its rank and axis checks, and has no loop for text: two StringDType operands are
            // named by class (the dispatcher), two fixed-width ones by descriptor (the legacy resolver).
            if (Backends.Strings.NDStringUfuncs.IsText(p)) throw Backends.Strings.NDStringUfuncs.NoLoop("subtract", p.dtype, p.dtype);

            NPTypeCode tc = p.typecode;

            // Complex has no `mod` loop in NumPy; the composition dies inside `mod` with a
            // TypeError. Reproduce that verbatim rather than letting the fused kernel throw
            // an opaque NotSupportedException later.
            if (tc == NPTypeCode.Complex)
                throw new TypeError(
                    "ufunc 'remainder' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the " +
                    "casting rule ''safe''");

            // The integer code path is taken only when result_type(dd, period) is integral,
            // i.e. an integer/bool/char input AND an integer-typed period. A float input, or
            // any input with a float period, promotes to float and takes the float path.
            bool inputIntegerFamily =
                tc == NPTypeCode.Boolean || tc == NPTypeCode.Byte || tc == NPTypeCode.SByte ||
                tc == NPTypeCode.Int16 || tc == NPTypeCode.UInt16 || tc == NPTypeCode.Int32 ||
                tc == NPTypeCode.UInt32 || tc == NPTypeCode.Int64 || tc == NPTypeCode.UInt64 ||
                tc == NPTypeCode.Char;
            bool integerPath = inputIntegerFamily && periodIsInteger;

            // discont defaults to period/2 — Python `period/2` is TRUE division, a float,
            // even on the integer path, so discont is always a real threshold.
            double disc = discont ?? (period / 2.0);

            // Output dtype = result_type(dd, period). diff preserves the input dtype (bool
            // diffs via not_equal, but bool + an integer period promotes to int64), so:
            //   * float family (half/single/double) -> same width; Decimal -> Decimal;
            //   * integer path -> the input's integer dtype (bool -> int64);
            //   * otherwise (integer/bool with a float period) -> float64.
            DType outdt;
            NPTypeCode outTc;
            if (tc == NPTypeCode.Half) { outdt = np.float16; outTc = NPTypeCode.Half; }
            else if (tc == NPTypeCode.Single) { outdt = np.float32; outTc = NPTypeCode.Single; }
            else if (tc == NPTypeCode.Double) { outdt = np.float64; outTc = NPTypeCode.Double; }
            else if (tc == NPTypeCode.Decimal) { outdt = np.@decimal; outTc = NPTypeCode.Decimal; }
            else if (integerPath && tc != NPTypeCode.Boolean) { outdt = p.dtype; outTc = tc; }
            else if (integerPath) { outdt = np.int64; outTc = NPTypeCode.Int64; }   // bool -> int64
            else { outdt = np.float64; outTc = NPTypeCode.Double; }

            // Interval half-width and the "boundary is ambiguous" flag.
            long intervalLowI = 0, intervalHighI = 0;
            double intervalLowF = 0, intervalHighF = 0;
            bool boundaryAmbiguous;
            if (integerPath)
            {
                // Integer path: interval_high = period // 2 (Python FLOOR division, so a
                // negative period floors toward -inf, not toward zero like C#'s `/`), and
                // the boundary is ambiguous only for an even period (rem == 0).
                intervalHighI = FloorDiv(periodInt, 2);
                long rem = periodInt - 2 * intervalHighI;
                boundaryAmbiguous = rem == 0;
                intervalLowI = -intervalHighI;

                // NumPy evaluates `dd - interval_low` first, then `mod(., period)`. Each casts
                // its weak scalar to result_type(dd, scalar): a bool input is HIGHER-kind than
                // the weak int, so bool promotes to the int's default (int64) and the scalars
                // are range-checked against int64 (they always fit) — bool never overflows;
                // an integer input is SAME-kind, so the weak int adopts the input dtype and an
                // out-of-range scalar raises OverflowError there (every unsigned input, since
                // interval_low is negative; a too-narrow signed input). Reproduce the SAME
                // error, value and order up front so it fires for empty inputs too (NumPy's
                // does) and the fused kernel is only ever built for representable scalars —
                // interval_low is checked before period (the order NumPy hits them).
                NPTypeCode computeTc = tc == NPTypeCode.Boolean ? NPTypeCode.Int64 : tc;
                NDExprTypeRules.CheckIntLiteralFits(intervalLowI, computeTc);   // dd - interval_low
                NDExprTypeRules.CheckIntLiteralFits(periodInt, computeTc);      // mod(., period)
            }
            else
            {
                // Float path: real half-width; the boundary is always ambiguous.
                intervalHighF = period / 2.0;
                boundaryAmbiguous = true;
                intervalLowF = -intervalHighF;
            }

            long len = p.Shape.dimensions[ax];

            NDArray dd = null, ph = null, correction = null, pTail = null, tail = null, castTail = null;
            NDArray upTail = null;
            // copy=true: a fresh, writeable, C-contiguous base. On a throw below the scope releases it
            // (it is yielded only by a successful return), so a half-built result never leaks.
            NDArray up = p.astype(outdt);

            // A single element (or empty) along `axis` has no adjacent difference, so
            // there is no correction to apply — `up` (the dtype-cast copy) is the answer.
            // This also keeps np.evaluate / np.cumsum off zero-length operands.
            if (len < 2 || p.size == 0)
                return up;

            dd = p.TensorEngine.Diff(p, 1, ax, null, null);

            // On the integer path a BOOL input computes in int64 (a weak int is higher-kind
            // than bool, so NEP50 promotes bool -> the int's default int64). Materialize that
            // promotion here: diff(bool) is bool, and NDExpr would otherwise range-check the
            // weak interval scalars against bool (and reject the negative interval_low). Every
            // other integer dtype already equals its own compute dtype, so this only fires for
            // bool. The int64 dd, ph_correct, cumsum and int64 output all follow.
            if (integerPath && tc == NPTypeCode.Boolean)
            {
                var ddPromoted = dd.astype(np.int64);
                // A mid-method release of a full-size intermediate: the bool diff is dead the moment
                // its int64 copy exists, and freeing it now keeps it out of the peak the rest of this
                // method reaches. (The scope would release it only at exit; Dispose is idempotent.)
                dd.Dispose();
                dd = ddPromoted;
            }

            // Build the fused ph_correct expression over dd. Scalars are NEP50-weak
            // NDExpr constants: `Const(long)` on the integer path keeps dd's integer
            // dtype, `Const(double)` on the float path never widens dd's float width.
            NDExpr d = NDExpr.Arr(dd);
            NDExpr ilE = integerPath ? NDExpr.Const(intervalLowI) : NDExpr.Const(intervalLowF);
            NDExpr ihE = integerPath ? NDExpr.Const(intervalHighI) : NDExpr.Const(intervalHighF);
            NDExpr perE = integerPath ? NDExpr.Const(periodInt) : NDExpr.Const(period);

            // ddmod = mod(dd - interval_low, period) + interval_low
            NDExpr ddmod = NDExpr.Add(NDExpr.Mod(NDExpr.Subtract(d, ilE), perE), ilE);

            // Boundary fix: where ddmod landed on -period/2 for a POSITIVE jump, snap it
            // back to +period/2 (only meaningful when the boundary is ambiguous — an even
            // integer period or any float period).
            if (boundaryAmbiguous)
                ddmod = NDExpr.Where(
                    NDExpr.BitwiseAnd(NDExpr.Equal(ddmod, ilE), NDExpr.Greater(d, NDExpr.Const(0L))),
                    ihE, ddmod);

            // ph_correct = ddmod - dd, zeroed where the jump is smaller than discont
            // (unwrapping a sub-discont jump would only enlarge the discontinuity).
            NDExpr phExpr = NDExpr.Subtract(ddmod, d);
            phExpr = NDExpr.Where(
                NDExpr.Less(NDExpr.Abs(d), NDExpr.Const(disc)), NDExpr.Const(0L), phExpr);

            ph = p.TensorEngine.Evaluate(phExpr);     // one fused pass, no intermediates (np.evaluate's default options)

            // up[1:] = p[1:] + cumsum(ph_correct) along the axis. cumsum widens integer
            // ph_correct to int64/uint64 (NEP50); the assignment casts it back to `outdt`.
            correction = ph.TensorEngine.ReduceCumAdd(ph, ax, null, null);
            pTail = NDArray.SliceAlongAxis(p, ax, 1, len); // p[1:] (read-only view)
            tail = pTail + correction;                // result_type(p, correction)

            // Cast the tail to the output dtype only when it differs (the float path
            // already matches, so this is a no-op copy avoided there), then write it
            // through the [1:] view of `up`.
            castTail = tail.GetTypeCode == outTc ? tail : tail.astype(outdt);
            upTail = NDArray.SliceAlongAxis(up, ax, 1, len); // up[1:] (writeable view into up)
            upTail.SetData(castTail);
            // Every intermediate above (dd, ph, correction, the p[1:]/up[1:] views, tail and its
            // cast) is released by this method's scope; `up` is yielded.
            return up;
        }

        /// <summary>
        ///     Python-style floor division for a signed integer period: rounds the quotient
        ///     toward negative infinity (so <c>FloorDiv(-5, 2) == -3</c>), unlike C#'s <c>/</c>
        ///     which truncates toward zero. Matches NumPy's <c>divmod(period, 2)</c> half-width,
        ///     which matters for negative periods.
        /// </summary>
        /// <param name="a">Dividend (the period).</param>
        /// <param name="b">Divisor (always 2 here; assumed non-zero and positive).</param>
        /// <returns>The floor of <c>a / b</c>.</returns>
        private static long FloorDiv(long a, long b)
        {
            long q = a / b;
            // C# truncates toward zero; correct down by one when the signs differ and the
            // division was not exact (that is the case where floor != truncation).
            if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
            return q;
        }
    }
}
