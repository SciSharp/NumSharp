using System;
using System.Numerics;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise round-half-to-even to <c>decimals==0</c> places — NumPy's <c>np.round</c>/<c>np.around</c>
        /// with no <c>decimals</c> argument, i.e. the <c>rint</c> ufunc but WITH np.round's dtype rules
        /// (probed 2.4.2): integer/char inputs are an IDENTITY copy that PRESERVES the integer dtype (unlike
        /// <c>np.rint</c>, whose float-tier would widen an int to float), floats/complex/decimal PRESERVE their
        /// dtype, and a <b>bool</b> input takes the float16 tier (<c>np.round(bool)</c> → float16, matching the
        /// <c>rint</c> ufunc's bool loop — NOT the <see cref="ResolveUnaryReturnType"/> default of float64).
        /// <c>np.round</c> is a FUNCTION, not a ufunc, so it accepts <c>out=</c> only — there is no
        /// <c>where=</c>/<c>dtype=</c> ufunc kwarg (the <paramref name="where"/>/<paramref name="dtype"/> here are
        /// NumSharp conveniences carried for the engine's uniform unary signature).
        /// </summary>
        /// <param name="nd">Input array; any layout (contiguous / strided / broadcast) is read correctly.</param>
        /// <param name="dtype">Optional NumSharp dtype-target convenience (NumPy's np.round has none); must be a
        /// float/complex loop when supplied, else the "No loop matching …" error fires (via
        /// <see cref="ResolveUnaryReturnType"/>).</param>
        /// <param name="out">Optional destination; returned as-is (same instance), joined by broadcast but never
        /// stretched, and written under a same_kind cast from the loop dtype.</param>
        /// <param name="where">Optional bool mask (NumSharp convenience); validated for dtype only.</param>
        /// <returns>The rounded array — <paramref name="out"/> when supplied, otherwise a fresh array whose dtype
        /// follows the rules above.</returns>
        /// <exception cref="IncorrectTypeException">A <paramref name="dtype"/> below <see cref="NPTypeCode.Single"/>
        /// selects no rint loop.</exception>
        public override NDArray Round(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            ValidateWhereMask(where);

            var inputType = nd.GetTypeCode;
            if (typeCode.HasValue)
                ValidateUnaryInputCast(inputType, typeCode.Value, "rint");

            // Char rides the integer identity path: it is uint16-semantics (NumPy proxy:
            // round(uint16) -> identity uint16), but NPTypeCode.Char fails IsInteger().
            bool isInt = inputType.IsInteger() || inputType == NPTypeCode.Char;
            if (!typeCode.HasValue && isInt && @out is null && where is null)
                return Cast(nd, inputType, copy: true); // np.round int path = identity copy

            // loopType: the integer identity above is already handled; every OTHER input rounds in a FLOAT/complex
            // loop, so bool must resolve to float16 (the rint tier), NOT ResolveUnaryReturnType's float64 default
            // (probed 2.4.2: np.round(bool).dtype == float16). Half/Single/Double/Decimal/Complex are preserved by
            // both resolvers, so the bool remap is the only correction the decimals==0 path needs.
            NPTypeCode loopType = typeCode
                ?? (isInt ? inputType
                          : (inputType == NPTypeCode.Boolean ? NPTypeCode.Half : ResolveUnaryReturnType(nd, null)));
            return ExecuteUnaryOp(nd, UnaryOp.Round, loopType, @out, where);
        }

        /// <summary>
        /// Element-wise round to <paramref name="decimals"/> places — a faithful port of NumPy 2.4.2's
        /// <c>PyArray_Round</c> (<c>numpy/_core/src/multiarray/calculation.c</c>). For <c>decimals != 0</c> NumPy is
        /// NOT a scaled <c>Math.Round(x, digits)</c> (which cannot express negative digits and rounds by a different
        /// algorithm): it composes <c>op2(rint(op1(x, f)), f)</c> with <c>f = 10^|decimals|</c> — <c>op1=multiply,
        /// op2=true_divide</c> for positive decimals, <c>op1=true_divide, op2=multiply</c> for negative — evaluated
        /// AT the input's own float precision (so a float32 stays float32, a float16 stays float16, and a complex
        /// rounds its real/imag parts). This is why the values differ from a naive per-place rounder and why they are
        /// bit-identical to NumPy (verified against 2.4.2 across float16/32/64, complex128 and integer negative
        /// decimals). The previous implementation THREW on negative decimals (<c>Math.Round</c>'s 0..15 digit limit)
        /// and silently left Complex unrounded — both fixed here.
        /// </summary>
        /// <param name="nd">Input array; any layout is read correctly (densified by <see cref="Cast"/>).</param>
        /// <param name="decimals">Places to round to. <c>0</c> routes to the <c>rint</c> path above. Positive rounds
        /// after the decimal point, negative to the left of it (tens, hundreds, …). <see cref="int.MinValue"/> is
        /// NumPy's sentinel meaning <see cref="int.MaxValue"/> (its own magnitude cannot be negated), which makes
        /// <c>f</c> overflow to +∞ and the result NaN, matching NumPy.</param>
        /// <param name="dtype">Optional NumSharp dtype-target (must be float/complex when supplied).</param>
        /// <param name="out">Optional destination. NumPy runs the composition op-by-op, so the CAST error names the
        /// FIRST op — <c>multiply</c> for positive decimals, <c>divide</c> for negative (probed 2.4.2:
        /// <c>round(f8, 1, out=i4)</c> → "Cannot cast ufunc 'multiply' …"; <c>round(i4, -1, out=i4)</c> → "… 'divide'
        /// …", since an integer negative-decimals result is a float64 that cannot cast <c>same_kind</c> into an
        /// integer out even though the no-out form succeeds). One documented micro-divergence: for a WIDENING
        /// cross-dtype out (e.g. <c>round(float32, 1, out=float64)</c>) NumSharp computes wholly at the INPUT
        /// precision then widens the final value, whereas NumPy chains the ufuncs into the wide out buffer
        /// (op1 at input precision, rint+op2 at out precision) — so NumPy's <c>1.6</c> vs NumSharp's
        /// <c>(float64)(1.6f)</c>. This exotic combination is not exercised by the round oracle (out=null); the
        /// no-out and same-dtype-out results are bit-identical to NumPy.</param>
        /// <returns>The rounded array — <paramref name="out"/> when supplied, otherwise fresh.</returns>
        /// <exception cref="NotSupportedException">A <b>bool</b> input with <c>decimals != 0</c> and no
        /// <paramref name="dtype"/> override: NumPy raises because the float64 composition result cannot cast
        /// <c>same_kind</c> back to bool (message reproduced verbatim, naming multiply/divide by sign).</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> whose dtype the loop result cannot reach by
        /// a <c>same_kind</c> cast (via <see cref="ValidateOutCast"/>).</exception>
        public override NDArray Round(NDArray nd, int decimals, DType dtype = null, NDArray @out = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();

            // decimals==0 is the rint ufunc (round-half-to-even), a wholly separate dtype contract handled above.
            if (decimals == 0)
                return Round(nd, dtype, @out, null);

            var inputType = nd.GetTypeCode;
            bool isInt = inputType.IsInteger() || inputType == NPTypeCode.Char;
            // op1 (the first ufunc NumPy applies) drives the out-cast error text and the bool rejection message.
            string op1Name = decimals > 0 ? "multiply" : "divide";

            // bool + decimals!=0: NumPy has no bool round loop for a fractional round — the multiply/divide produces
            // a float64 that fails the same_kind cast back to bool. Reproduced verbatim (probed 2.4.2). A dtype
            // override escapes this (the caller asked for a float/complex loop), so it is gated on !typeCode.
            if (inputType == NPTypeCode.Boolean && !typeCode.HasValue)
                throw new NotSupportedException(
                    $"Cannot cast ufunc '{op1Name}' output from dtype('float64') to dtype('bool') with casting rule 'same_kind'");

            // Integer/char with POSITIVE decimals is an IDENTITY (no fractional part to remove, and — unlike the
            // negative branch — no float round-trip, so a huge int64 is returned untouched). Preserves the integer
            // dtype; an out= receives the identity through the same same_kind write-back the other ufuncs use.
            if (isInt && !typeCode.HasValue && decimals > 0)
            {
                if (@out is null)
                    return Cast(nd, inputType, copy: true);
                return ExecuteUnaryOp(nd, UnaryOp.Positive, inputType, @out, null);
            }

            // Working (loop) dtype:
            //   - explicit dtype override  -> that dtype (ResolveUnaryReturnType rejects an integer target as
            //     "No loop matching … rint", matching np.round having no such loop);
            //   - integer/char, negative decimals -> FLOAT64 (NumPy allocates a double temp and casts back), which
            //     is why round(int, -1) is legal (and wraps on the cast back) where round(int, -1, out=int) raises;
            //   - float/complex/decimal -> the input's own dtype (rounding stays at that precision).
            NPTypeCode work = typeCode.HasValue
                ? ResolveUnaryReturnType(nd, typeCode)
                : (isInt ? NPTypeCode.Double : inputType);

            // The scaled/rinted/unscaled result at the working precision. `core` is a pooled buffer: it is
            // RETURNED as the result on the float/complex/decimal no-out path, but is only an INTERMEDIATE on the
            // integer-cast-back and out= paths — where it MUST be disposed or its pool buffer leaks (a surplus
            // take reclaimable only by a later GC+finalizer, which the ScopeAudit gate fails on).
            var core = RoundScaleCore(nd, decimals, work);

            if (@out is null)
            {
                // Integer/char negative-decimals (no override): cast the float64 result back to the integer dtype.
                // NumPy uses an UNSAFE (C truncation) cast, so an out-of-range magnitude WRAPS — round(int8 127,-1)
                // -> 130 -> -126. The IL float->int cast kernel reproduces NumPy's (MSVC) modular result exactly.
                if (isInt && !typeCode.HasValue)
                {
                    var result = Cast(core, inputType, copy: false);
                    if (!ReferenceEquals(result, core))
                        core.Dispose();          // reclaim the float64 intermediate (result is a fresh int buffer)
                    return result;
                }
                return core;                     // float/complex/decimal: core IS the result the caller owns
            }

            // out= : validate the same_kind cast from the WORKING dtype (the ufunc output NumPy would produce),
            // naming op1 so the message matches NumPy (multiply/divide by sign). The subsequent Positive copy re-runs
            // the identical same_kind check internally (harmless) and performs the write-back into out. `core` is an
            // intermediate here (the result is out), so it is disposed even when ValidateOutCast throws.
            try
            {
                ValidateOutCast(work, @out.typecode, op1Name);
                return ExecuteUnaryOp(core, UnaryOp.Positive, work, @out, null);
            }
            finally
            {
                core.Dispose();
            }
        }

        /// <summary>
        /// NumPy's <c>power_of_ten</c> (<c>calculation.c</c>): an EXACT small table for <c>n &lt; 9</c> (each entry a
        /// power of ten that is exactly representable in double), else <c>1e9</c> multiplied by ten <c>(n-9)</c> more
        /// times. Reproduced so <c>f</c> is bit-identical to NumPy's — <c>Math.Pow(10, n)</c> can be 1 ULP off and
        /// <c>f</c> is applied then un-applied, so the error would not fully cancel. Capped at <c>+∞</c> for
        /// <c>n ≥ 309</c> (10^309 already overflows double, and this both matches NumPy's eventual infinity AND
        /// avoids a multi-billion-iteration loop when the magnitude is <see cref="int.MaxValue"/>).
        /// </summary>
        /// <param name="n">A NON-negative magnitude (<c>|decimals|</c>).</param>
        /// <returns><c>10^n</c> as a double, or <c>+∞</c> once it overflows.</returns>
        private static readonly double[] s_powerOfTen = { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8 };
        private static double PowerOfTen(int n)
        {
            if (n < 9)
                return s_powerOfTen[n];
            if (n >= 309)
                return double.PositiveInfinity;
            double ret = 1e9;
            for (int k = 9; k < n; k++)
                ret *= 10.0;
            return ret;
        }

        /// <summary>
        /// Core of the <c>decimals != 0</c> round: materialise <paramref name="nd"/> as a fresh contiguous copy at
        /// the working dtype <paramref name="work"/> (which also densifies any layout, widens an integer to double,
        /// and yields an empty array for an empty input), then apply <c>op2(rint(op1(x, f)), f)</c> in place at that
        /// precision. The per-element bodies are the SAME primitives the engine's own multiply/rint/divide kernels
        /// use, so the result is bit-identical to composing those ufuncs (and thus to NumPy): <see cref="Math.Round(double,MidpointRounding)"/>
        /// / <see cref="MathF.Round(float,MidpointRounding)"/> / <see cref="Half.Round(Half,MidpointRounding)"/> are
        /// all round-half-to-even, Complex rounds real and imag as independent doubles (NumPy's <c>arr.real =
        /// a.real.round()</c> recursion), and Decimal — which has no NumPy analog — rounds in the exact base-10
        /// domain. A scalar managed loop is used rather than an NDIter pass because this is a precision-control
        /// composition (NumPy itself runs it as three separate ufunc passes), and every layout has already been
        /// collapsed to a contiguous buffer by the <see cref="Cast"/>.
        /// </summary>
        /// <param name="nd">Input array (any dtype/layout).</param>
        /// <param name="decimals">The non-zero decimals (sign selects mul→div vs div→mul).</param>
        /// <param name="work">The working/loop dtype: a float (Half/Single/Double), Complex, Decimal, or Double for
        /// an integer input.</param>
        /// <returns>A fresh, contiguous, writeable array of dtype <paramref name="work"/> holding the rounded values.</returns>
        /// <exception cref="NotSupportedException"><paramref name="work"/> is not a float/complex/decimal dtype.</exception>
        /// <exception cref="OverflowException">A Decimal magnitude beyond decimal's exact range whose rescaled result
        /// no longer fits (an edge with no NumPy analog).</exception>
        private unsafe NDArray RoundScaleCore(NDArray nd, int decimals, NPTypeCode work)
        {
            // copy:true -> a fresh, owned, C-contiguous buffer at `work` we may overwrite in place. This is also the
            // int->double widen and the layout-densification, so the loops below never see strides.
            var res = Cast(nd, work, copy: true);
            long len = res.size;
            if (len == 0)
                return res; // empty input: no elements to round (Cast already produced the right dtype/shape)

            bool up = decimals > 0;                                    // up => multiply first, divide last
            int mag = up ? decimals
                         : (decimals == int.MinValue ? int.MaxValue : -decimals); // |decimals|, INT_MIN sentinel
            double f = PowerOfTen(mag);

            switch (work)
            {
                case NPTypeCode.Double:
                {
                    var p = (double*)res.Address;
                    for (long i = 0; i < len; i++)
                    {
                        double x = p[i];
                        p[i] = up ? Math.Round(x * f, MidpointRounding.ToEven) / f
                                  : Math.Round(x / f, MidpointRounding.ToEven) * f;
                    }
                    break;
                }
                case NPTypeCode.Single:
                {
                    // f is rounded to float32 first (NumPy's weak scalar adopts the array's float32 dtype), then all
                    // three ops run in float32 — the reason a float32 round is NOT a float64 round narrowed.
                    float ff = (float)f;
                    var p = (float*)res.Address;
                    for (long i = 0; i < len; i++)
                    {
                        float x = p[i];
                        p[i] = up ? MathF.Round(x * ff, MidpointRounding.ToEven) / ff
                                  : MathF.Round(x / ff, MidpointRounding.ToEven) * ff;
                    }
                    break;
                }
                case NPTypeCode.Half:
                {
                    // Half arithmetic widens to float32, computes, narrows (there is no hardware f16 math); Half.Round
                    // is round-half-to-even. Verified bit-identical to NumPy's float16 round.
                    Half hf = (Half)f;
                    var p = (Half*)res.Address;
                    for (long i = 0; i < len; i++)
                    {
                        Half x = p[i];
                        Half scaled = up ? x * hf : x / hf;
                        Half r = Half.Round(scaled, MidpointRounding.ToEven);
                        p[i] = up ? r / hf : r * hf;
                    }
                    break;
                }
                case NPTypeCode.Complex:
                {
                    // NumPy rounds a complex by recursing on its real and imag parts as independent float64 arrays;
                    // the a*0/b*0 cross terms of a true complex multiply are exactly zero, so rounding each component
                    // with the double body is identical AND avoids the complex-division double-rounding.
                    var p = (Complex*)res.Address;
                    for (long i = 0; i < len; i++)
                    {
                        Complex z = p[i];
                        double re = up ? Math.Round(z.Real * f, MidpointRounding.ToEven) / f
                                       : Math.Round(z.Real / f, MidpointRounding.ToEven) * f;
                        double im = up ? Math.Round(z.Imaginary * f, MidpointRounding.ToEven) / f
                                       : Math.Round(z.Imaginary / f, MidpointRounding.ToEven) * f;
                        p[i] = new Complex(re, im);
                    }
                    break;
                }
                case NPTypeCode.Decimal:
                {
                    var p = (decimal*)res.Address;
                    for (long i = 0; i < len; i++)
                        p[i] = RoundDecimalScalar(p[i], decimals);
                    break;
                }
                default:
                    throw new NotSupportedException($"np.round with decimals != 0 is not supported for dtype {work}.");
            }

            return res;
        }

        /// <summary>
        /// Round a single <see cref="decimal"/> to <paramref name="decimals"/> places, round-half-to-even, in the
        /// EXACT base-10 domain where possible (Decimal has no NumPy analog, so this is a NumSharp-defined behaviour
        /// chosen for exactness and totality). Positive decimals in <c>[1, 28]</c> use <see cref="decimal.Round(decimal,int,MidpointRounding)"/>
        /// directly (exact, no overflow); positive decimals past decimal's 28-digit fractional precision are an
        /// identity; negative decimals rescale by an exact power of ten (<c>Math.Round(x / 10^|d|) * 10^|d|</c>) in
        /// the decimal domain, collapsing to zero once the divisor exceeds decimal's range.
        /// </summary>
        /// <param name="x">The value to round.</param>
        /// <param name="decimals">Non-zero decimals.</param>
        /// <returns>The rounded decimal.</returns>
        /// <exception cref="OverflowException">A negative-decimals rescale whose product exceeds
        /// <see cref="decimal.MaxValue"/> (an extreme with no NumPy analog).</exception>
        private static decimal RoundDecimalScalar(decimal x, int decimals)
        {
            if (decimals > 0)
            {
                if (decimals >= 29)
                    return x;                                  // more places than decimal can hold -> no-op
                return decimal.Round(x, decimals, MidpointRounding.ToEven);
            }

            int mag = decimals == int.MinValue ? int.MaxValue : -decimals;
            if (mag >= 29)
                return 0m;                                     // |x| < 10^29 always rounds to 0 at this scale

            decimal fdec = 1m;
            for (int k = 0; k < mag; k++)
                fdec *= 10m;                                   // exact for mag <= 28
            return decimal.Round(x / fdec, MidpointRounding.ToEven) * fdec;
        }
    }
}
