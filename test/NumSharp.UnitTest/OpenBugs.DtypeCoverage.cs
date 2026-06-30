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
    }
}
