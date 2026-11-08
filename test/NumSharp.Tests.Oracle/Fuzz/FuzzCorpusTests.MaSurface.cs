using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The numpy.ma surface tier (<c>ma_surface.jsonl</c>, test/oracle/gen_ma_surface_oracle.py): NumPy 2.4.2 contracts
    ///     for the numpy.ma functions the coverage inventory listed as available but unproven — the creation wrappers, the
    ///     joins, the fill-value helpers, the predicates and queries, ids, the hard-mask switches, masked_object, putmask,
    ///     convolve / correlate / cov / corrcoef, the edge finders, the compressors, ndenumerate and the apply_* wrappers —
    ///     over masked operands in every data kind and mask pattern.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The tier is held to ZERO excuses: MisalignedRegistry returns null for valueclass <c>masurface</c> and the ma
    ///     harness's own excuses skip it (<c>strict</c> in <see cref="CompareMasked"/>). Error cases compare NumPy's type
    ///     and text. Result kinds: masked (filled(0), mask, form, and the raw data where NumPy defines it), masked_tuple,
    ///     array, scalar, tuple and text (a normalized Python repr).
    ///     </para>
    ///     <para>
    ///     The cells NumSharp gets wrong are carved by the generator (<c>_gap_of</c>) into
    ///     <c>corpus/gaps/ma_surface_gaps.jsonl</c> and replayed by <see cref="MaSurface_KnownGaps"/>, which stays red
    ///     until they are fixed and prints, per gap id, how many of its cases still diverge.
    ///     </para>
    /// </remarks>
    public partial class FuzzCorpusTests
    {
        /// <summary>The gate: every kept ma_surface case matches NumPy with no excuse.</summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MaSurface() => RunMaCorpus("ma_surface.jsonl");

        /// <summary>The carved numpy.ma surface gaps, one verdict per case, grouped by gap id (see the class remarks).</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void MaSurface_KnownGaps() => RunMaGapCorpus("ma_surface_gaps.jsonl");
    }
}
