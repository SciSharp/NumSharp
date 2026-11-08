using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — the NumPy parity QA journey of 2026-11-02 / 03
    // =====================================================================
    //
    // A manual QA pass over branch strings compared NumSharp against NumPy 2.4.2 with a probe harness (a C# probe and a
    // NumPy twin printing the same result lines, then a comparer). Every divergence it confirmed that no existing
    // [OpenBugs] / [Misaligned] entry already pins has an id QA-NN, an explanation in docs/bugs/parity-qa-2026-11-02.md
    // (the journal: symptom, NumPy's answer, root cause where known, severity, fix size) and a reproduction below through
    // public calls, with the smallest input that shows it. Nothing here is fixed yet: the owner picks which ones to fix.
    //
    // Every expected value is NumPy 2.4.2's answer for the same call (probed on win-amd64). EXPECTED (when fixed): each
    // test passes; remove its [OpenBugs], move it next to its API's tests, and update the journal entry.
    // =====================================================================
    public partial class OpenBugs
    {
        // ------------------------------------------------------------------
        // QA-28 — np.asarray (and the rest of the as* family) refuses a Python list (object[])
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.asarray([1, 2.5, 3])</c> is NumPy's canonical array_like conversion (float64 <c>[1.0, 2.5, 3.0]</c>), and
        ///     NumSharp's <c>np.array(object[])</c> / <c>np.asanyarray(object[])</c> run NumPy's Python-list coercion for it.
        ///     <c>np.asarray</c> has no <c>object[]</c> overload, so C# binds <c>asarray(NDArray, …)</c> through the implicit
        ///     <c>Array</c> → <c>NDArray</c> conversion, which treats the <c>object[]</c> as an array of element type
        ///     <c>Object</c> and throws <c>NotSupportedException: Cannot interpret 'System.Object' as a data type</c>. The same
        ///     happens for <c>np.ascontiguousarray</c>, <c>np.asfortranarray</c>, <c>np.asarray_chkfinite</c>, <c>np.copy</c>
        ///     and <c>np.require</c>, and for every other array_like parameter typed <c>NDArray</c> (np.sum, np.sort,
        ///     np.unique, np.add, … in <c>p62</c>): the CLAUDE.md trap ("an object[] (a Python list) needs its own overload")
        ///     applied to np.array / np.asanyarray only.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA28")]
        public void ParityQa_Asarray_PythonList_ShouldCoerceLikeNpArray()
        {
            var list = new object[] { 1, 2.5, 3 };
            var nested = new object[] { new object[] { 1, 2 }, new object[] { 3, 4 } };

            using var flat = np.asarray(list);
            using var grid = np.asarray(nested);
            using var contiguous = np.ascontiguousarray(list);

            flat.dtype.Should().Be(np.float64);
            flat.GetDouble(1).Should().Be(2.5);
            grid.shape.Should().Equal(new long[] { 2, 2 });
            grid.dtype.Should().Be(np.int64);
            contiguous.GetDouble(2).Should().Be(3.0);
        }
    }
}
