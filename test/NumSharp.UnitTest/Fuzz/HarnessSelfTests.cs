using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Fuzz;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     Proves the differential harness has teeth: BitDiff actually detects value/NaN/signed-zero
    ///     divergence, and the corpus pipeline genuinely exercises the float->int wrap semantics whose
    ///     violation (saturation) was the motivating bug. A vacuous (zero-element) pass would fail here.
    /// </summary>
    [TestClass]
    public class HarnessSelfTests
    {
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void BitDiff_DetectsValueDifference()
        {
            var a = BitConverter.GetBytes(1.0);
            var b = BitConverter.GetBytes(2.0);
            var diffs = BitDiff.Compare(a, b, NPTypeCode.Double);
            Assert.AreEqual(1, diffs.Count);
            Assert.AreEqual(0, diffs[0].Index);
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void BitDiff_TreatsDifferentNaNPayloadsAsEqual()
        {
            // Two NaNs with different payloads must NOT be reported as a divergence.
            var nan1 = BitConverter.GetBytes(double.NaN);
            var nan2 = BitConverter.GetBytes(BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000123UL)));
            Assert.IsTrue(double.IsNaN(BitConverter.ToDouble(nan2)));
            var diffs = BitDiff.Compare(nan1, nan2, NPTypeCode.Double);
            Assert.AreEqual(0, diffs.Count, "different NaN payloads should compare equal");
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void BitDiff_DetectsNaNVersusNumber()
        {
            var nan = BitConverter.GetBytes(double.NaN);
            var num = BitConverter.GetBytes(0.0);
            Assert.AreEqual(1, BitDiff.Compare(nan, num, NPTypeCode.Double).Count);
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void BitDiff_DetectsSignedZero()
        {
            // -0.0 and +0.0 differ in bits; NumPy preserves the sign, so this must be caught.
            var negZero = BitConverter.GetBytes(-0.0);
            var posZero = BitConverter.GetBytes(0.0);
            Assert.AreEqual(1, BitDiff.Compare(negZero, posZero, NPTypeCode.Double).Count);
        }

        /// <summary>
        ///     The exact bug class this whole workstream exists to prevent: NumPy float->int on
        ///     overflow/NaN WRAPS to the int-min sentinel (via x86 cvtt), it does NOT saturate.
        ///     Reconstruct a real corpus operand and assert NumSharp matches that on every edge value.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public unsafe void Corpus_FloatToInt_WrapsNotSaturates()
        {
            var cases = FuzzCorpus.Load("astype_smoke.jsonl");
            var c = cases.First(x => x.Layout == "c_contiguous_1d"
                                  && x.Operands[0].Dtype == "float64"
                                  && x.Expected.Dtype == "int32");

            var operand = FuzzCorpus.Reconstruct(c.Operands[0]);
            var inputs = operand.astype(NPTypeCode.Double).flatten();   // logical source values
            var result = operand.astype(NPTypeCode.Int32).flatten();    // NumSharp output

            int edgeChecks = 0;
            for (long i = 0; i < operand.size; i++)
            {
                double sv = Convert.ToDouble(inputs.GetAtIndex((int)i));
                int got = Convert.ToInt32(result.GetAtIndex((int)i));

                if (double.IsNaN(sv) || sv >= 2147483648.0 || sv <= -2147483649.0)
                {
                    // saturation would give int.MaxValue (2147483647) for +overflow — must NOT happen.
                    Assert.AreEqual(int.MinValue, got,
                        $"float->int overflow/NaN at value {sv} must wrap to int.MinValue, got {got}");
                    edgeChecks++;
                }
            }
            Assert.IsTrue(edgeChecks > 0, "expected the corpus to contain overflow/NaN edge values");
        }
    }
}
