using System;
using System.Globalization;
using System.Linq;
using System.Text;
using NumSharp.Utilities;

namespace NumSharp.Tests.Utilities
{
    /// <summary>
    ///     <see cref="NDDoubleParse"/>, the text readers' double parser (Clinger's fast path, the Eisel-Lemire algorithm,
    ///     .NET's parser for the undecided long mantissas): every result must be the correctly rounded double — the bits
    ///     CPython's <c>_Py_dg_strtod</c> (and .NET's own parser) give, which is what NumPy reads.
    /// </summary>
    [TestClass]
    public class NDDoubleParseTests
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        [TestMethod]
        public void EdgeCases_AreCorrectlyRounded()
        {
            string[] texts =
            {
                "0", "-0", "0.0", "-0.0", "1", "-1", "0.1", "0.2", "0.3", "1.5", "123.456", "1e0", "1E5", "1e-5", "1e+5",
                // the exact-power boundary of Clinger's fast path (2^53, 10^22) and just past it
                "9007199254740992", "9007199254740993", "9007199254740995", "9007199254740994e22", "1e22", "1e23", "8.98846567431158e307",
                // the largest finite double, the rounding boundary above it, and overflow
                "1.7976931348623157e308", "1.7976931348623158e308", "1.7976931348623159e308", "1e309", "1e400", "-1e400",
                // subnormals, the smallest one, the halfway point below it (round to even: 0), just above it, underflow
                "2.2250738585072014e-308", "2.2250738585072011e-308", "4.9406564584124654e-324", "5e-324",
                "2.4703282292062327e-324", "2.4703282292062328e-324", "1e-324", "1e-400", "-1e-400",
                // mantissas of 19, 20 and many digits (the >19-digit truncation and its w / w+1 test)
                "1234567890123456789", "12345678901234567890", "123456789012345678901234567890",
                "0.1000000000000000055511151231257827021181583404541015625", "0.10000000000000000555111512312578270211815834045410156250001",
                "17976931348623157081452742373170435679807056752584499659891747680315726078002853876058955863276687817154045895351438246423432132688946418276846754670353751698604991057655128207624549009038932894407586850845513394230458323690322294816580855933212334827479",
                "0." + new string('0', 340) + "1", "1" + new string('0', 300), "0.000000000000000000000000000001",
                // halfway cases between two doubles at several magnitudes
                "9007199254740993.0", "1.00000000000000011102230246251565404236316680908203125",
                "1.000000000000000111022302462515654042363166809082031250000001", "7.2057594037927933e16",
                // leading zeros and a long fraction
                "000000000000000000000000001.5", "0.0000000000000000000000000000000000000000000000000000001234",
            };
            foreach (string t in texts)
                AssertSame(t);
        }

        [TestMethod]
        public void RandomDecimals_MatchDotNetsCorrectlyRoundedParser()
        {
            // A seeded sweep of the digit counts and exponents the readers meet: 1 to 25 significant digits, a point
            // anywhere, exponents across the whole range (overflow and underflow included), both signs.
            var rng = new System.Random(20261015);
            var sb = new StringBuilder();
            for (int n = 0; n < 200_000; n++)
            {
                sb.Clear();
                if (rng.Next(3) == 0)
                    sb.Append(rng.Next(2) == 0 ? '-' : '+');
                int digits = rng.Next(1, 26);
                int point = rng.Next(-1, digits + 1);
                for (int d = 0; d < digits; d++)
                {
                    if (d == point)
                        sb.Append('.');
                    sb.Append((char)('0' + rng.Next(10)));
                }
                if (rng.Next(2) == 0)
                    sb.Append('e').Append(rng.Next(-345, 311).ToString(CI));
                AssertSame(sb.ToString());
            }
        }

        [TestMethod]
        public void BytesAndCharsAgree()
        {
            foreach (string t in new[] { "0.1", "-12345.6789e-10", "1.7976931348623157e308", "4.9e-324", "123456789012345678901234" })
            {
                double c = NDDoubleParse.Parse(t.AsSpan());
                double b = NDDoubleParse.Parse<byte>(Encoding.ASCII.GetBytes(t));
                Assert.AreEqual(BitConverter.DoubleToInt64Bits(c), BitConverter.DoubleToInt64Bits(b), t);
            }
        }

        [TestMethod]
        public void MatchDecimal_IsTheStrtodGrammar()
        {
            // At least one digit in all; an exponent only when a digit follows its sign.
            Assert.AreEqual(2, NDDoubleParse.MatchDecimal("5.".AsSpan(), 0));
            Assert.AreEqual(2, NDDoubleParse.MatchDecimal(".5".AsSpan(), 0));
            Assert.AreEqual(0, NDDoubleParse.MatchDecimal(".".AsSpan(), 0));
            Assert.AreEqual(1, NDDoubleParse.MatchDecimal("1e+".AsSpan(), 0));
            Assert.AreEqual(1, NDDoubleParse.MatchDecimal("1e".AsSpan(), 0));
            Assert.AreEqual(4, NDDoubleParse.MatchDecimal("1e+5x".AsSpan(), 0));
            Assert.AreEqual(0, NDDoubleParse.MatchDecimal("+".AsSpan(), 0));
            Assert.AreEqual(0, NDDoubleParse.MatchDecimal("inf".AsSpan(), 0));
            Assert.IsTrue(NDDoubleParse.TryParseWhole("-2.5e3".AsSpan(), out double v));
            Assert.AreEqual(-2500.0, v);
            Assert.IsFalse(NDDoubleParse.TryParseWhole("2.5e3 ".AsSpan(), out _));
        }

        private static void AssertSame(string text)
        {
            double expected = double.Parse(text, NumberStyles.Float, CI);
            double got = NDDoubleParse.Parse(text.AsSpan());
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(expected), BitConverter.DoubleToInt64Bits(got), text);
        }
    }
}
