using System;

namespace NumSharp.Tests
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

        // np.iscomplex / np.isreal previously IGNORED the imaginary part (iscomplex -> all False,
        // isreal -> all True for complex input) AND emitted garbage bytes on strided real input.
        // FIXED: ported to NumPy's own structure (isreal(x) = imag(x) == 0; iscomplex = imag != 0 for
        // complex, else all-False), building the non-complex result from DIMENSIONS only. Now GREEN at
        // full coverage in the logic differential-fuzz tier (ISCOMPLEX_* in gen_oracle.py — all dtypes
        // incl. complex128 × every layout), so the two [OpenBugs] pins here were removed.
    }
}
