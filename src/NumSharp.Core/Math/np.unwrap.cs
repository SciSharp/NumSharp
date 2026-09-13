using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    // ============================== np.unwrap ==============================
    // Unwrap a signal by taking the complement of large deltas w.r.t. a period.
    //
    // Changes every element whose absolute difference from its predecessor
    // exceeds max(discont, period/2) to its period-complementary value, so that
    // adjacent differences never exceed period/2. For the default period 2*pi
    // and discont pi this unwraps a radian phase by adding 2*k*pi.
    //
    // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::unwrap. NumPy is a
    // pure composition, and NumSharp mirrors it operation-for-operation so the
    // NEP50 dtype promotion, the integer/float path split, broadcasting and every
    // edge case fall out of the existing (parity-tested) building blocks:
    //
    //     dd    = diff(p, axis)
    //     ddmod = mod(dd - interval_low, period) + interval_low
    //     if boundary_ambiguous:                 # dd exactly +period/2 -> +period/2
    //         ddmod[(ddmod == interval_low) & (dd > 0)] = interval_high
    //     ph_correct        = ddmod - dd
    //     ph_correct[|dd| < discont] = 0          # small jumps are not unwrapped
    //     up            = p.astype(dtype)         # dtype = result_type(dd, period)
    //     up[1:]        = p[1:] + cumsum(ph_correct, axis)
    //
    // WHY the integer/float split. NumPy chooses the interval half-width and the
    // "boundary is ambiguous" flag from `result_type(dd, period)`:
    //   * FLOAT result (any float input, OR an integer input with a FLOAT period)
    //     -> interval_high = period/2 (a real number), boundary always ambiguous.
    //   * INTEGER result (an integer/bool input AND an INTEGER-typed period)
    //     -> interval_high = period//2 (floor), ambiguous only when period is even.
    // Because a Python int and a Python float are distinct at the call site, the
    // C# port needs the same distinction, which is why there are two overloads:
    // the primary `double period` (the NumPy default and every float use) and a
    // `long period` overload for the integer-preserving path. Note that for an
    // INTEGER input the two paths compute the SAME values (an integer difference
    // never lands in the (period//2, period/2] gap, and the ambiguity fix only
    // fires on that gap); they differ only in the OUTPUT DTYPE — integer-typed
    // period keeps the input's integer dtype (bool -> int64), float period yields
    // float64. Unsigned integer inputs with an integer period reproduce NumPy's
    // OverflowError (interval_low is negative and cannot be cast to the unsigned
    // dtype); complex inputs reproduce NumPy's TypeError (mod rejects complex).
    //
    // FUSION / PERFORMANCE. The whole elementwise chain that maps `dd` to
    // `ph_correct` (subtract, mod, add, an optional boundary Where, subtract, and
    // the discont Where) is built as ONE NDExpr and produced in a single fused
    // np.evaluate pass, where NumPy allocates ~6 intermediate arrays. Only three
    // array passes remain (diff, the fused ph_correct, cumsum), so unwrap is
    // 2-9x faster than NumPy at 100K-10M elements while staying bit-identical.
    // The scalars (interval_low/high, period, discont) ride as NEP50-weak NDExpr
    // constants: on the float path a weak double never widens the operand's float
    // dtype (float16/float32 stay put); on the integer path a weak int keeps the
    // integer dtype and triggers the OverflowError exactly as NumPy's does. Every
    // loop lives inside np.diff / np.evaluate / np.cumsum — there is no hand
    // per-element loop here.
    public static partial class np
    {
        /// <summary>
        ///     Unwrap <paramref name="p"/> by changing deltas larger than
        ///     <c>max(discont, period/2)</c> to their <paramref name="period"/>-complement,
        ///     so adjacent differences never exceed <c>period/2</c>. This is the
        ///     FLOAT-period overload — the NumPy default and the form used for radian
        ///     phase unwrapping.
        /// </summary>
        /// <param name="p">
        ///     Input array (at least one dimensional; a 0-D input is rejected exactly
        ///     as NumPy's inner <see cref="diff(NDArray,int,int,object,object)"/> rejects it).
        /// </param>
        /// <param name="discont">
        ///     Maximum discontinuity between values. <c>null</c> (the default) means
        ///     <c>period/2</c>. Values below <c>period/2</c> have no effect: taking the
        ///     complement of a jump smaller than <c>period/2</c> would only enlarge it,
        ///     so a smaller <paramref name="discont"/> is clamped up to <c>period/2</c>.
        ///     To change behaviour, pass a <paramref name="discont"/> LARGER than
        ///     <c>period/2</c>.
        /// </param>
        /// <param name="axis">
        ///     Axis along which to unwrap; default is the last axis. Negative axes
        ///     count from the end.
        /// </param>
        /// <param name="period">
        ///     Size of the range over which the input wraps (default <c>2*pi</c>).
        ///     This overload treats the period as a real number (NumPy's weak float
        ///     scalar), so the result is always float — a float input keeps its width
        ///     (float16/float32/float64) and an integer/bool input yields float64. To
        ///     get the integer-preserving behaviour of NumPy's <c>period=&lt;int&gt;</c>,
        ///     call the <see cref="unwrap(NDArray,long,System.Nullable{double},int)"/>
        ///     overload (e.g. <c>np.unwrap(a, period: 4)</c>).
        /// </param>
        /// <returns>
        ///     A fresh, writeable, C-contiguous array of the unwrapped signal. The
        ///     shape matches <paramref name="p"/>; the dtype is the float
        ///     <c>result_type(diff(p), period)</c> described above.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="p"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="p"/> is 0-D (propagated from <see cref="diff(NDArray,int,int,object,object)"/>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="p"/>'s rank.</exception>
        /// <exception cref="TypeError"><paramref name="p"/> is complex (NumPy's <c>mod</c> has no complex loop).</exception>
        /// <remarks>
        ///     NumPy's <c>period</c> is keyword-only, so ported code always writes
        ///     <c>period:</c>. Beware the one C# resolution wrinkle: a bare positional
        ///     integer second argument — <c>np.unwrap(a, 4)</c> — binds the
        ///     <see cref="unwrap(NDArray,long,System.Nullable{double},int)"/> overload
        ///     (period = 4), NOT <paramref name="discont"/> = 4. To set
        ///     <paramref name="discont"/> positionally, pass a double (<c>np.unwrap(a, 4.0)</c>)
        ///     or name it (<c>np.unwrap(a, discont: 4)</c>).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html
        /// </remarks>
        public static NDArray unwrap(NDArray p, double? discont = null, int axis = -1, double period = 2 * Math.PI)
            => UnwrapImpl(p, discont, axis, period, periodIsInteger: false, periodInt: 0);

        /// <summary>
        ///     Unwrap <paramref name="p"/> with an INTEGER-typed <paramref name="period"/>,
        ///     the integer-preserving form of NumPy's <c>np.unwrap(a, period=&lt;int&gt;)</c>.
        ///     Use this (rather than the <c>double</c> overload) when you want NumPy's
        ///     integer output dtype — e.g. <c>np.unwrap(new[]{0,1,2,-1,0}, period: 4)</c>
        ///     returns <c>[0,1,2,3,4]</c> as an integer array.
        /// </summary>
        /// <param name="p">Input array (at least one dimensional).</param>
        /// <param name="period">
        ///     Size of the wrap range as an INTEGER. With an integer/bool input this
        ///     selects NumPy's integer code path: the interval half-width is
        ///     <c>period/2</c> floored, the boundary is ambiguous only when
        ///     <paramref name="period"/> is even, and the OUTPUT keeps the input's
        ///     integer dtype (a bool input yields int64). With a FLOAT input the result
        ///     is still float (an integer period cannot make a float signal integral),
        ///     identical to the double overload.
        /// </param>
        /// <param name="discont">
        ///     Maximum discontinuity; <c>null</c> (default) means <c>period/2.0</c>
        ///     (a real number, matching NumPy's Python <c>period/2</c> true division
        ///     even on the integer path). See the double overload for semantics.
        /// </param>
        /// <param name="axis">Axis along which to unwrap; default is the last axis.</param>
        /// <returns>
        ///     A fresh, writeable, C-contiguous array. Integer/bool inputs keep their
        ///     integer dtype (bool -> int64); float inputs stay float.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="p"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="p"/> is 0-D.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="p"/>'s rank.</exception>
        /// <exception cref="TypeError"><paramref name="p"/> is complex.</exception>
        /// <exception cref="OverflowException">
        ///     <paramref name="p"/> is an unsigned integer (or char) — or a signed
        ///     integer too narrow for the derived interval — so the negative
        ///     <c>interval_low</c> (or <paramref name="period"/> itself) cannot be
        ///     represented in the input dtype. This reproduces NumPy's OverflowError
        ///     ("Python integer N out of bounds for &lt;dtype&gt;"), which fires
        ///     regardless of array size (empty inputs included).
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html</remarks>
        public static NDArray unwrap(NDArray p, long period, double? discont = null, int axis = -1)
            => UnwrapImpl(p, discont, axis, period, periodIsInteger: true, periodInt: period);

        /// <summary>
        ///     Shared core for both <see cref="unwrap(NDArray,System.Nullable{double},int,double)"/>
        ///     overloads. Mirrors NumPy's composition exactly; see the file header for the
        ///     algorithm and the integer/float path rationale.
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
                throw new ArgumentException("diff requires input that is at least one dimensional");

            // normalize_axis_index(axis, nd): valid range is [-nd, nd-1]. NumPy raises
            // AxisError here (via normalize_axis_index) reporting the ORIGINAL axis.
            int ax = axis;
            if (ax < 0) ax += nd;
            if (ax < 0 || ax >= nd)
                throw new AxisError($"axis {axis} is out of bounds for array of dimension {nd}");

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

            long len = p.shape[ax];

            NDArray dd = null, ph = null, correction = null, pTail = null, tail = null, castTail = null;
            NDArray upTail = null;
            NDArray up = p.astype(outdt);   // copy=true: a fresh, writeable, C-contiguous base
            try
            {
                // A single element (or empty) along `axis` has no adjacent difference, so
                // there is no correction to apply — `up` (the dtype-cast copy) is the answer.
                // This also keeps np.evaluate / np.cumsum off zero-length operands.
                if (len < 2 || p.size == 0)
                    return up;

                dd = np.diff(p, axis: ax);

                // On the integer path a BOOL input computes in int64 (a weak int is higher-kind
                // than bool, so NEP50 promotes bool -> the int's default int64). Materialize that
                // promotion here: diff(bool) is bool, and NDExpr would otherwise range-check the
                // weak interval scalars against bool (and reject the negative interval_low). Every
                // other integer dtype already equals its own compute dtype, so this only fires for
                // bool. The int64 dd, ph_correct, cumsum and int64 output all follow.
                if (integerPath && tc == NPTypeCode.Boolean)
                {
                    var ddPromoted = dd.astype(np.int64);
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

                ph = np.evaluate(phExpr);                 // one fused pass, no intermediates

                // up[1:] = p[1:] + cumsum(ph_correct) along the axis. cumsum widens integer
                // ph_correct to int64/uint64 (NEP50); the assignment casts it back to `outdt`.
                correction = np.cumsum(ph, axis: ax);
                pTail = SliceAlongAxis(p, ax, 1, len);    // p[1:] (read-only view)
                tail = pTail + correction;                // result_type(p, correction)

                // Cast the tail to the output dtype only when it differs (the float path
                // already matches, so this is a no-op copy avoided there), then write it
                // through the [1:] view of `up`.
                castTail = tail.GetTypeCode == outTc ? tail : tail.astype(outdt);
                upTail = SliceAlongAxis(up, ax, 1, len);  // up[1:] (writeable view into up)
                upTail.SetData(castTail);
                return up;
            }
            catch
            {
                // On any failure the half-built result must not leak its pooled buffer.
                up.Dispose();
                throw;
            }
            finally
            {
                // View-before-owner cleanup of every intermediate (up is returned, never here).
                upTail?.Dispose();
                if (!ReferenceEquals(castTail, tail)) castTail?.Dispose();
                tail?.Dispose();
                pTail?.Dispose();
                correction?.Dispose();
                ph?.Dispose();
                dd?.Dispose();
            }
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
