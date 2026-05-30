using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
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
                string e = Token(expected, i * isz, tc);
                string a = Token(actual, i * isz, tc);
                if (e != a)
                    diffs.Add(new Diff(i, e, a));
            }
            return diffs;
        }

        private static string Token(byte[] b, int off, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Single:
                {
                    float v = BitConverter.ToSingle(b, off);
                    return float.IsNaN(v) ? "NaN" : Hex(b, off, 4);
                }
                case NPTypeCode.Double:
                {
                    double v = BitConverter.ToDouble(b, off);
                    return double.IsNaN(v) ? "NaN" : Hex(b, off, 8);
                }
                case NPTypeCode.Half:
                {
                    Half v = BitConverter.ToHalf(b, off);
                    return Half.IsNaN(v) ? "NaN" : Hex(b, off, 2);
                }
                case NPTypeCode.Complex:
                {
                    double re = BitConverter.ToDouble(b, off);
                    double im = BitConverter.ToDouble(b, off + 8);
                    string r = double.IsNaN(re) ? "NaN" : Hex(b, off, 8);
                    string m = double.IsNaN(im) ? "NaN" : Hex(b, off + 8, 8);
                    return r + ":" + m;
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
    }
}
