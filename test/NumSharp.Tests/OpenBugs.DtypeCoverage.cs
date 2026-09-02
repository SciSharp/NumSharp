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
        // np.clip on a Boolean array previously threw on the NON-CONTIGUOUS path
        // ("NotSupportedException: clip not supported for Boolean") while the contiguous path
        // worked — the strided/transposed/F-order clip kernel (ClipStrided) omitted the Boolean
        // case. FIXED: Boolean now rides the Byte kernel there (bool storage is 0/1, so unsigned
        // Byte Min/Max reproduces the false<true clamp exactly, bit-identical to the contiguous
        // 0/1 scalar select). Now GREEN at full coverage in the stat differential-fuzz tier
        // (CLIP_DTYPES in gen_oracle.py — bool × every layout), so the two [OpenBugs] pins here
        // (Clip_Bool_Transposed/Strided) were removed. Non-contiguous ~/invert on bool was
        // already correct; the NonContiguousTests.NegateBoolean_* pins were likewise retired
        // (np.negative(bool) is a NumPy-2.4.2 TypeError, ~ is the boolean flip).

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
