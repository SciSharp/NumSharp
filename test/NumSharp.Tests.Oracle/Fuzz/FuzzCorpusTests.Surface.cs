using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The surface tier (<c>test/oracle/gen_surface_oracle.py</c>): NumPy 2.4.2 contracts for the APIs the coverage
    ///     inventory listed as available but unproven — the np.linalg Array-API forms (cross, diagonal, outer, trace),
    ///     the histogram family, array_equiv, shares_memory / may_share_memory, broadcast_shapes, apply_along_axis /
    ///     apply_over_axes, bmat, array2string, format_float_positional / format_float_scientific, base_repr,
    ///     binary_repr, typename, datetime_data, the ndarray members tolist / tofile / setfield / base / dtype / shape /
    ///     device / to_device / data (the memoryview), the attributes of finfo, iinfo, dtype, ndarray.flags and poly1d, the
    ///     np.broadcast / flatiter / np.nditer members after a number of steps, the print options, the buffer size,
    ///     dtype.type, np.frompyfunc, and the inventory's aliases numpy.cumulative_sum / cumulative_prod and numpy.shape.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Each array operand also meets the ndarray kinds the earlier tiers proved risky (C, F and permuted views at an
    ///     offset, reversed and stepped views, read-only owners, broadcast views, 0-d views at an offset, empty views
    ///     inside a buffer), and each API meets its refusals, recorded with NumPy's exception type and text. The op keys
    ///     are the API names themselves, so the coverage join (<c>coverage/oracle_evidence.py</c>) credits the
    ///     contracts to the API rows that listed them as unproven.
    ///     </para>
    ///     <para>
    ///     The tier is held to zero registry excuses (valueclass <c>surface</c>, see <see cref="MisalignedRegistry"/>).
    ///     Known divergences are carved by the generator into <c>gaps/surface_gaps.jsonl</c> under named gap ids and
    ///     replayed by <see cref="Surface_KnownGaps"/>, which fails while any of them still differs from NumPy — so a
    ///     fix shows up as a gap to drop from the generator's carve.
    ///     </para>
    /// </remarks>
    public partial class FuzzCorpusTests
    {
        /// <summary>The surface tier: every case bit-exact with NumPy (values, text, dtype names, error type and text).</summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Surface() => RunCorpus("surface.jsonl");

        /// <summary>The carved gaps of <see cref="Surface"/>: fails while any of them still diverges from NumPy.</summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Surface_KnownGaps() => RunGapCorpus("surface_gaps.jsonl");
    }
}
