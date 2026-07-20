using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.MathSuite
{
    /// <summary>
    /// Guards the `using` on per-iter `slice = arr[slices]` inside
    /// ArgReductionAxisFallback. The fallback rarely fires because IL kernels
    /// cover all common (dtype, op, contig/strided) combinations, but the
    /// refactor must still be correctness-safe for the cases it does cover.
    /// </summary>
    /// <remarks>
    /// <c>ArgMax_Axis_TightLoop_DoesNotLeakWorkingSet</c> used to close this class. Its own comment
    /// conceded it was "belt-and-suspenders" against a path that "rarely fires"; what it actually
    /// measured was process RSS. Removed — see <see cref="LeakGuards"/>.
    /// </remarks>
    [TestClass]
    public class Default_Reduction_Arg_UsingTests : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void ArgMax_Axis_2D_StillCorrect()
        {
            // np.argmax(arange(12).reshape(3,4), axis=1) -> [3, 3, 3]
            var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var r = np.argmax(a, axis: 1);
            ((long)r[0]).Should().Be(3L);
            ((long)r[1]).Should().Be(3L);
            ((long)r[2]).Should().Be(3L);
        }

        [TestMethod]
        public void ArgMin_Axis_2D_StillCorrect()
        {
            // np.argmin(arange(12).reshape(3,4), axis=1) -> [0, 0, 0]
            var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var r = np.argmin(a, axis: 1);
            ((long)r[0]).Should().Be(0L);
            ((long)r[1]).Should().Be(0L);
            ((long)r[2]).Should().Be(0L);
        }

        [TestMethod]
        public void ArgMax_Axis_Transposed_StillCorrect()
        {
            // Transpose forces a non-contig source — the strided IL kernel
            // path. Numerical answers still match the contiguous version.
            var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var aT = a.T;  // (4, 3), non-contig
            var r = np.argmax(aT, axis: 1);  // along the original-rows axis

            r.size.Should().Be(4L);
            // aT[0] = [0, 4, 8]   → argmax = 2
            // aT[1] = [1, 5, 9]   → argmax = 2
            // aT[2] = [2, 6, 10]  → argmax = 2
            // aT[3] = [3, 7, 11]  → argmax = 2
            ((long)r[0]).Should().Be(2L);
            ((long)r[3]).Should().Be(2L);
        }

        // --------------------------- lifetime ---------------------------

        /// <summary>
        /// The per-iteration <c>slice</c> is a VIEW into the caller's array; releasing it must
        /// leave the source — and every later slice of it — intact.
        /// </summary>
        /// <remarks>
        /// The fallback disposes one such view per output element, so a source that survives a
        /// single call is not evidence: the buffer is only at risk once the LAST view releases it.
        /// Reducing twice over the same array, and reading the source afterwards, is what makes
        /// this test able to fail.
        /// </remarks>
        [TestMethod]
        public void ArgMax_Axis_DoesNotDisposeTheSourceItSlices()
        {
            var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);

            var first = np.argmax(a, axis: 1);
            LeakGuards.StillUsable(a, " — the per-iteration slices must not free the source");

            // Second reduction over the same buffer: still correct, so the views released by the
            // first call left the data alone.
            var second = np.argmax(a, axis: 1);
            LeakGuards.StillUsable(a);
            LeakGuards.StillUsable(first, " — an earlier result must not be freed by a later call");

            for (int i = 0; i < 3; i++)
            {
                ((long)first[i]).Should().Be(3L);
                ((long)second[i]).Should().Be(3L);
            }

            // And the source itself still reads back its original values.
            ((int)a[0, 0]).Should().Be(0);
            ((int)a[2, 3]).Should().Be(11);
        }
    }
}
