using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Open bugs surfaced by widening the differential-fuzz dtype grids toward full coverage
    ///     (gen_oracle per-mode dtype lists grown to ALL_DTYPES). Each is a NumPy-parity gap for a
    ///     dtype the corpus previously did not exercise on that op. NumPy 2.4.2 is the source of
    ///     truth (verified 2026-06-30). The offending (op, dtype) combo is CARVED OUT of the green
    ///     corpus and reproduced here under [OpenBugs] — remove the carve + this test when fixed.
    /// </summary>
    [TestClass]
    public class OpenBugsDtypeCoverageTests : TestClass
    {
        // ============================================================================
        //  BUG: np.clip on a Boolean array throws on the NON-CONTIGUOUS path.
        //
        //  np.clip(bool_array, True, True) -> all-True bool array (NumPy 2.4.2).
        //  NumSharp handles the CONTIGUOUS bool clip fine, but the general strided /
        //  transposed / F-contiguous kernel throws:
        //      NotSupportedException: clip not supported for Boolean
        //  So the bug is layout-dependent — the SIMD/contiguous path supports Boolean
        //  while the coordinate/strided clip kernel omits it.
        // ============================================================================
        private static readonly NDArray ClipTrue = NDArray.Scalar(true);

        [TestMethod, OpenBugs]
        public void Clip_Bool_Transposed_Throws()
        {
            var a = np.array(new bool[] { true, false, false, true, true, false }).reshape(2, 3).T; // non-contiguous
            NDArray r = null;
            Action act = () => r = np.clip(a, ClipTrue, ClipTrue);
            act.Should().NotThrow("NumPy clips a transposed bool array (lo=hi=True -> all True); NumSharp's " +
                                  "strided clip kernel throws NotSupportedException for Boolean.");
            r.typecode.Should().Be(NPTypeCode.Boolean);
        }

        [TestMethod, OpenBugs]
        public void Clip_Bool_Strided_Throws()
        {
            var a = np.array(new bool[] { true, false, false, true, true, false, false, true })["::2"]; // strided view
            NDArray r = null;
            Action act = () => r = np.clip(a, ClipTrue, ClipTrue);
            act.Should().NotThrow("NumPy clips a strided bool view; NumSharp throws NotSupportedException " +
                                  "for Boolean on the non-contiguous clip path (contiguous bool clip works).");
            r.typecode.Should().Be(NPTypeCode.Boolean);
        }

        // ============================================================================
        //  BUG: np.trace of an UNSIGNED dtype returns Int64 instead of uint64.
        //
        //  np.trace(np.arange(16, dtype=uint8).reshape(4,4)) == 30, dtype uint64 (NumPy 2.4.2).
        //  NumSharp returns the right value but dtype Int64 — the trace accumulator upcasts
        //  unsigned to the signed default int instead of the unsigned uint64 (cf. sum(uint8)->uint64,
        //  which IS correct in NumSharp). Carved from the matmul/trace tier (TRACE_DTYPES drops uint8).
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Trace_Unsigned_WrongResultDtype()
        {
            var a = np.arange(16).astype(NPTypeCode.Byte).reshape(4, 4); // uint8
            var r = np.trace(a);
            ((long)r.GetValue()).Should().Be(30L, "value is correct (0+5+10+15)");
            r.typecode.Should().Be(NPTypeCode.UInt64,
                "NumPy trace(uint8) -> uint64 (unsigned sum); NumSharp upcasts to the signed Int64.");
        }

        // ============================================================================
        //  BUG: np.round_/around with NEGATIVE decimals is broken.
        //
        //  np.round([127,153,248], -1) == [130,150,250] (round to tens); float likewise.
        //  NumSharp routes through System.Math.Round(value, digits), which only accepts digits in
        //  [0,15] and THROWS ArgumentOutOfRangeException for the integer loop (and mis-rounds floats).
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Round_NegativeDecimals_Broken()
        {
            var a = np.array(new int[] { 127, 153, 248 });
            NDArray r = null;
            Action act = () => r = np.round_(a, -1);
            act.Should().NotThrow("NumPy round([127,153,248], -1) = [130,150,250]; NumSharp throws " +
                                  "ArgumentOutOfRangeException (Math.Round rejects negative digits).");
            ((int)r.GetValue(0)).Should().Be(130);
            ((int)r.GetValue(1)).Should().Be(150);
            ((int)r.GetValue(2)).Should().Be(250);
        }

        // ============================================================================
        //  BUG: np.round_ on float16 with decimals>=1 diverges from NumPy.
        //
        //  np.round(float16([2.75]), 1) == 2.80078 (banker's rounding of the TRUE float16 value).
        //  NumSharp's float16 fractional rounding diverges (observed off by whole units on some pool
        //  values). Carved from the rounding tier (float16 only kept at decimals=0).
        // ============================================================================
        [TestMethod, OpenBugs]
        public void Round_Float16_Fractional_Diverges()
        {
            var a = np.array(new float[] { 2.75f, 0.05f }).astype(NPTypeCode.Half);
            var r = np.round_(a, 1);
            ((double)(Half)r.GetValue(0)).Should().BeApproximately(2.80078, 0.01,
                "NumPy rounds float16 2.75 -> 2.80078 (banker's); NumSharp diverges.");
            ((double)(Half)r.GetValue(1)).Should().BeApproximately(0.0, 0.01,
                "NumPy rounds float16 0.05 -> 0.0; NumSharp diverges.");
        }

        // ============================================================================
        //  BUG: np.iscomplex / np.isreal IGNORE the imaginary part.
        //
        //  np.iscomplex([1+2j, 3+0j, 0+1j, 5+0j]) == [True, False, True, False] (imag != 0).
        //  np.isreal(...) == [False, True, False, True].
        //  NumSharp's iscomplex returns ALL False and isreal returns ALL True for complex input —
        //  it never inspects the imaginary component. (It also emits garbage bytes on strided real
        //  input — same op, separate manifestation.) Carved from the logic tier.
        // ============================================================================
        [TestMethod, OpenBugs]
        public void IsComplex_IgnoresImaginaryPart()
        {
            var c = np.array(new System.Numerics.Complex[]
                { new(1, 2), new(3, 0), new(0, 1), new(5, 0) });
            var r = np.iscomplex(c);
            ((bool)r.GetValue(0)).Should().BeTrue("1+2j has nonzero imaginary part -> iscomplex True");
            ((bool)r.GetValue(2)).Should().BeTrue("0+1j has nonzero imaginary part -> iscomplex True");
            ((bool)r.GetValue(1)).Should().BeFalse("3+0j is real -> iscomplex False");
        }

        [TestMethod, OpenBugs]
        public void IsReal_IgnoresImaginaryPart()
        {
            var c = np.array(new System.Numerics.Complex[]
                { new(1, 2), new(3, 0), new(0, 1), new(5, 0) });
            var r = np.isreal(c);
            ((bool)r.GetValue(0)).Should().BeFalse("1+2j has nonzero imaginary part -> isreal False");
            ((bool)r.GetValue(1)).Should().BeTrue("3+0j is real -> isreal True");
        }
    }
}
