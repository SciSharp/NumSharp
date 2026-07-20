using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Guards the `using` on sliceI/sliceJ/temp inside SwapSlicesAxis0.
    /// Multi-dim shuffle routes through this helper N times (Fisher-Yates).
    /// </summary>
    /// <remarks>
    /// <c>Shuffle_TightLoop_DoesNotLeakWorkingSet</c> used to close this class. By its own comment
    /// the churn it hoped to see was ~16 MiB against a 30 MiB threshold. Removed — see
    /// <see cref="LeakGuards"/>.
    /// </remarks>
    [TestClass]
    public class np_random_shuffle_using_test : TestClass
    {
        // --------------------------- correctness ---------------------------

        /// <summary>
        /// Multi-dim shuffle still preserves all values and shape. Targets
        /// the SwapSlicesAxis0 code path (1D contig goes through the
        /// stackalloc fast path, not the slice swap).
        /// </summary>
        [TestMethod]
        public void Shuffle_2DArray_PreservesValuesAndShape()
        {
            var rnd = np.random.RandomState(42);
            var nd = np.arange(20).reshape(5, 4).astype(NPTypeCode.Int32);
            var originalSum = (int)np.sum(nd);

            rnd.shuffle(nd);

            ((int)np.sum(nd)).Should().Be(originalSum);
            nd.shape.Should().ContainInOrder(5L, 4L);
        }

        [TestMethod]
        public void Shuffle_3DArray_PreservesValuesAndShape()
        {
            var rnd = np.random.RandomState(7);
            var nd = np.arange(60).reshape(5, 4, 3).astype(NPTypeCode.Int32);
            var originalSum = (int)np.sum(nd);

            rnd.shuffle(nd);

            ((int)np.sum(nd)).Should().Be(originalSum);
            nd.shape.Should().ContainInOrder(5L, 4L, 3L);
        }

        /// <summary>
        /// Identical seeded shuffles of identical inputs produce identical
        /// outputs. Confirms `using` on the swap temps doesn't disturb the
        /// RNG state or the data ordering.
        /// </summary>
        [TestMethod]
        public void Shuffle_2D_DeterministicWithSeed()
        {
            var rnd1 = np.random.RandomState(123);
            var nd1 = np.arange(20).reshape(5, 4);
            rnd1.shuffle(nd1);

            var rnd2 = np.random.RandomState(123);
            var nd2 = np.arange(20).reshape(5, 4);
            rnd2.shuffle(nd2);

            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 4; j++)
                    ((int)nd1[i, j]).Should().Be((int)nd2[i, j]);
        }

        // --------------------------- leak guard ---------------------------

        /// <summary>
        /// Tight loop of multi-dim shuffles. The SwapSlicesAxis0 path
        /// allocates 2 view wrappers + 1 copy buffer per swap, and a 5x4
        /// shuffle invokes it up to 4 times (i = n-1 → 1). The using-bound
        /// temps must release atomically.
        /// </summary>
        /// <remarks>
        /// sliceI and sliceJ are views straight into the array being shuffled, and a 5-row shuffle
        /// disposes eight of them. Shuffling IN PLACE is what makes this able to fail: there is no
        /// fresh output buffer to hide behind, so if a swap's `using` released the array's storage
        /// the very next swap would be writing into freed pages. Running two shuffles back to back
        /// over the same array, then reading every element, is the check.
        /// </remarks>
        [TestMethod]
        public void Shuffle_SwapTemps_DoNotReleaseTheArrayBeingShuffled()
        {
            var rnd = np.random.RandomState(0);
            var nd = np.arange(20).reshape(5, 4).astype(NPTypeCode.Int32);

            rnd.shuffle(nd);
            LeakGuards.StillUsable(nd, " — sliceI/sliceJ are views into this very array");

            // A second in-place pass over storage the first pass's temps have already released
            // would be a write-after-free.
            rnd.shuffle(nd);
            LeakGuards.StillUsable(nd);

            // All 20 values still present exactly once — the swaps moved rows, lost nothing.
            var seen = new bool[20];
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 4; j++)
                {
                    int v = (int)nd[i, j];
                    v.Should().BeInRange(0, 19);
                    seen[v].Should().BeFalse("value {0} must appear exactly once", v);
                    seen[v] = true;
                }
        }
    }
}
