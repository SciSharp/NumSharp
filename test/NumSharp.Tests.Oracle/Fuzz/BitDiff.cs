using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Bit-exact comparison of two result buffers, element by element.
    ///
    ///     Everything is compared by raw bytes EXCEPT NaN, which is tokenized so that differing
    ///     NaN payloads (non-contractual) don't false-fail. Note -0.0 vs +0.0 and ±inf ARE
    ///     bit-compared — NumPy preserves signed zero and emits canonical infinity bits, so a
    ///     divergence there is a real bug worth catching.
    /// </summary>
    public static class BitDiff
    {
        public readonly record struct Diff(int Index, string Expected, string Actual);

        public static List<Diff> Compare(byte[] expected, byte[] actual, NPTypeCode tc)
            => Compare(expected, actual, tc, nanBitExact: false);

        /// <summary>
        ///     As <see cref="Compare(byte[],byte[],NPTypeCode)"/>, but when <paramref name="nanBitExact"/>
        ///     is true NaN is compared by its RAW bytes (sign + payload) instead of being tokenized to
        ///     "NaN". Used by the complex-unary NaN-sign contract: NumPy 2.4.2 emits a DETERMINISTIC NaN
        ///     sign per op/path (MSVC UCRT: sqrt/log/... +NaN, csin/ctan the transform's negate, ctanh
        ///     propagates), which NumSharp now reproduces bit-for-bit — so for those ops the sign is
        ///     contractual and must be gated, not tokenized away. Off everywhere else, where a differing
        ///     NaN payload is non-contractual (e.g. order-dependent float32 sum) and would false-fail.
        /// </summary>
        public static List<Diff> Compare(byte[] expected, byte[] actual, NPTypeCode tc, bool nanBitExact)
        {
            var diffs = new List<Diff>();
            if (expected.Length != actual.Length)
            {
                diffs.Add(new Diff(-1, $"len={expected.Length}", $"len={actual.Length}"));
                return diffs;
            }

            int isz = tc.SizeOf();
            int count = isz == 0 ? 0 : expected.Length / isz;
            for (int i = 0; i < count; i++)
            {
                string e = Token(expected, i * isz, tc, nanBitExact);
                string a = Token(actual, i * isz, tc, nanBitExact);
                if (e != a)
                    diffs.Add(new Diff(i, e, a));
            }
            return diffs;
        }

        private static string Token(byte[] b, int off, NPTypeCode tc, bool nanBitExact)
        {
            switch (tc)
            {
                case NPTypeCode.Single:
                {
                    float v = BitConverter.ToSingle(b, off);
                    return float.IsNaN(v) && !nanBitExact ? "NaN" : Hex(b, off, 4);
                }
                case NPTypeCode.Double:
                {
                    double v = BitConverter.ToDouble(b, off);
                    return double.IsNaN(v) && !nanBitExact ? "NaN" : Hex(b, off, 8);
                }
                case NPTypeCode.Half:
                {
                    Half v = BitConverter.ToHalf(b, off);
                    return Half.IsNaN(v) && !nanBitExact ? "NaN" : Hex(b, off, 2);
                }
                case NPTypeCode.Complex:
                {
                    double re = BitConverter.ToDouble(b, off);
                    double im = BitConverter.ToDouble(b, off + 8);
                    string r = double.IsNaN(re) && !nanBitExact ? "NaN" : Hex(b, off, 8);
                    string m = double.IsNaN(im) && !nanBitExact ? "NaN" : Hex(b, off + 8, 8);
                    return r + ":" + m;
                }
                case NPTypeCode.Decimal:
                {
                    // Decimal has no NumPy analog, so SCALE is not part of the contract — 1.0m and
                    // 1.00m are the same VALUE with different bytes. Tokenize by canonical value
                    // (trailing zeros stripped) so the gate flags wrong VALUES, not benign scale.
                    decimal d = MemoryMarshal.Read<decimal>(b.AsSpan(off, 16));
                    string s = d.ToString(CultureInfo.InvariantCulture);
                    if (s.IndexOf('.') >= 0) s = s.TrimEnd('0').TrimEnd('.');
                    return s == "-0" ? "0" : s;
                }
                default:
                    return Hex(b, off, tc.SizeOf());
            }
        }

        private static string Hex(byte[] b, int off, int n)
        {
            var sb = new StringBuilder(n * 2);
            for (int i = 0; i < n; i++)
                sb.Append(b[off + i].ToString("x2"));
            return sb.ToString();
        }

        /// <summary>
        ///     ULP distance between the expected and actual values at <paramref name="index"/>.
        ///     Used only to classify DOCUMENTED near-misses (e.g. complex division) as intended
        ///     divergences — never to relax the default bit-exact gate.
        /// </summary>
        public static bool WithinUlp(byte[] exp, byte[] act, int index, NPTypeCode tc, int maxUlp)
        {
            switch (tc)
            {
                case NPTypeCode.Double:
                    return UlpDouble(BitConverter.ToDouble(exp, index * 8), BitConverter.ToDouble(act, index * 8)) <= maxUlp;
                case NPTypeCode.Single:
                    return UlpSingle(BitConverter.ToSingle(exp, index * 4), BitConverter.ToSingle(act, index * 4)) <= maxUlp;
                case NPTypeCode.Half:
                    return UlpHalf(BitConverter.ToHalf(exp, index * 2), BitConverter.ToHalf(act, index * 2)) <= maxUlp;
                case NPTypeCode.Complex:
                {
                    int o = index * 16;
                    return UlpDouble(BitConverter.ToDouble(exp, o), BitConverter.ToDouble(act, o)) <= maxUlp
                        && UlpDouble(BitConverter.ToDouble(exp, o + 8), BitConverter.ToDouble(act, o + 8)) <= maxUlp;
                }
                default:
                    return false;
            }
        }

        /// <summary>
        ///     Raw ULP distance between the values at element <paramref name="index"/> of two
        ///     same-dtype buffers — the truthful-vs-precise adjudication primitive: the harness
        ///     compares UlpDistance(actual, truth) against UlpDistance(expected, truth) to decide
        ///     which side of a NumPy divergence lost precision. Semantics inherited from the
        ///     private Ulp* helpers: NaN-vs-NaN is 0, equal values (incl. +0 vs -0) are 0,
        ///     opposite signs are long.MaxValue (never "close"); Complex takes the max over its
        ///     two components; non-float dtypes report 0 for byte-equal else long.MaxValue
        ///     (integer math is exact — any divergence is total).
        /// </summary>
        public static long UlpDistance(byte[] a, byte[] b, int index, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Double:
                    return UlpDouble(BitConverter.ToDouble(a, index * 8), BitConverter.ToDouble(b, index * 8));
                case NPTypeCode.Single:
                    return UlpSingle(BitConverter.ToSingle(a, index * 4), BitConverter.ToSingle(b, index * 4));
                case NPTypeCode.Half:
                    return UlpHalf(BitConverter.ToHalf(a, index * 2), BitConverter.ToHalf(b, index * 2));
                case NPTypeCode.Complex:
                {
                    int o = index * 16;
                    return Math.Max(
                        UlpDouble(BitConverter.ToDouble(a, o), BitConverter.ToDouble(b, o)),
                        UlpDouble(BitConverter.ToDouble(a, o + 8), BitConverter.ToDouble(b, o + 8)));
                }
                default:
                {
                    int isz = tc.SizeOf();
                    for (int i = 0; i < isz; i++)
                        if (a[index * isz + i] != b[index * isz + i])
                            return long.MaxValue;
                    return 0;
                }
            }
        }

        /// <summary>
        ///     Absolute distance |expected - actual| at <paramref name="index"/> is within
        ///     <paramref name="absTol"/>. Used only to classify the DOCUMENTED expm1/log1p
        ///     small-|x| accuracy loss (NumSharp computes Exp(x)-1 / Log(1+x), whose absolute
        ///     error near zero is bounded by ~ulp(1) — a subnormal result flushes to 0, a moderate
        ///     one drifts a ULP) as an intended divergence, never to relax the bit-exact gate. Two
        ///     NaNs count as within tolerance (BitDiff already tokenizes NaN, so a NaN reaches here
        ///     only opposite a number — abs is NaN — and is correctly NOT excused).
        /// </summary>
        public static bool WithinAbs(byte[] exp, byte[] act, int index, NPTypeCode tc, double absTol)
        {
            switch (tc)
            {
                case NPTypeCode.Double:
                {
                    double e = BitConverter.ToDouble(exp, index * 8), a = BitConverter.ToDouble(act, index * 8);
                    if (double.IsNaN(e) && double.IsNaN(a)) return true;
                    return Math.Abs(e - a) <= absTol;
                }
                case NPTypeCode.Single:
                {
                    float e = BitConverter.ToSingle(exp, index * 4), a = BitConverter.ToSingle(act, index * 4);
                    if (float.IsNaN(e) && float.IsNaN(a)) return true;
                    return Math.Abs(e - a) <= absTol;
                }
                case NPTypeCode.Half:
                {
                    Half e = BitConverter.ToHalf(exp, index * 2), a = BitConverter.ToHalf(act, index * 2);
                    if (Half.IsNaN(e) && Half.IsNaN(a)) return true;
                    return Math.Abs((double)e - (double)a) <= absTol;
                }
                default:
                    return false;
            }
        }

        /// <summary>
        ///     True iff the difference at <paramref name="index"/> is (at least partly) a pure SIGN /
        ///     payload flip of a NaN or of a signed zero — a NaN with a different sign/payload, or +0.0
        ///     vs -0.0 — rather than a change of numeric value. This is exactly the class of difference
        ///     that <see cref="WithinUlp"/> / <see cref="UlpDistance"/> report as "0 ULP" (NaN-vs-NaN and
        ///     +0-vs-(-0) are treated as equal there), so a ULP-envelope excuse would silently swallow it.
        ///     The complex-unary NaN-sign gate uses this to HARD-FAIL such a flip before the excuses run,
        ///     while a genuine finite rounding difference (the excused interior residual) returns false.
        /// </summary>
        public static bool DiffHasSignFlip(byte[] exp, byte[] act, int index, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Double: return SignFlipD(exp, act, index * 8);
                case NPTypeCode.Single: return SignFlipS(exp, act, index * 4);
                case NPTypeCode.Half:   return SignFlipH(exp, act, index * 2);
                case NPTypeCode.Complex:
                {
                    int o = index * 16;
                    return SignFlipD(exp, act, o) || SignFlipD(exp, act, o + 8);
                }
                default: return false;
            }
        }

        private static bool SignFlipD(byte[] e, byte[] a, int off)
        {
            double x = BitConverter.ToDouble(e, off), y = BitConverter.ToDouble(a, off);
            long lx = BitConverter.DoubleToInt64Bits(x), ly = BitConverter.DoubleToInt64Bits(y);
            if (lx == ly) return false;                          // identical bits: not a diff at all
            if (double.IsNaN(x) && double.IsNaN(y)) return true; // NaN vs NaN with different bits (sign/payload)
            return x == 0.0 && y == 0.0;                         // +0.0 vs -0.0
        }

        private static bool SignFlipS(byte[] e, byte[] a, int off)
        {
            float x = BitConverter.ToSingle(e, off), y = BitConverter.ToSingle(a, off);
            int lx = BitConverter.SingleToInt32Bits(x), ly = BitConverter.SingleToInt32Bits(y);
            if (lx == ly) return false;
            if (float.IsNaN(x) && float.IsNaN(y)) return true;
            return x == 0.0f && y == 0.0f;
        }

        private static bool SignFlipH(byte[] e, byte[] a, int off)
        {
            Half x = BitConverter.ToHalf(e, off), y = BitConverter.ToHalf(a, off);
            short lx = BitConverter.HalfToInt16Bits(x), ly = BitConverter.HalfToInt16Bits(y);
            if (lx == ly) return false;
            if (Half.IsNaN(x) && Half.IsNaN(y)) return true;
            return x == (Half)0.0 && y == (Half)0.0;
        }

        private static long UlpDouble(double a, double b)
        {
            if (double.IsNaN(a) && double.IsNaN(b)) return 0;
            if (a == b) return 0;
            long la = BitConverter.DoubleToInt64Bits(a), lb = BitConverter.DoubleToInt64Bits(b);
            if ((la < 0) != (lb < 0)) return long.MaxValue; // opposite signs: not "close" for our purpose
            return Math.Abs(la - lb);
        }

        private static long UlpSingle(float a, float b)
        {
            if (float.IsNaN(a) && float.IsNaN(b)) return 0;
            if (a == b) return 0;
            int la = BitConverter.SingleToInt32Bits(a), lb = BitConverter.SingleToInt32Bits(b);
            if ((la < 0) != (lb < 0)) return long.MaxValue;
            return Math.Abs((long)la - lb);
        }

        private static long UlpHalf(Half a, Half b)
        {
            if (Half.IsNaN(a) && Half.IsNaN(b)) return 0;
            if (a == b) return 0;
            short la = BitConverter.HalfToInt16Bits(a), lb = BitConverter.HalfToInt16Bits(b);
            if ((la < 0) != (lb < 0)) return long.MaxValue;
            return Math.Abs((long)la - lb);
        }
    }
}
