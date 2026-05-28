using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Guards the `using` on sliceI/sliceJ/temp inside SwapSlicesAxis0.
    /// Multi-dim shuffle routes through this helper N times (Fisher-Yates).
    /// </summary>
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
        [TestMethod]
        public void Shuffle_TightLoop_DoesNotLeakWorkingSet()
        {
            // Warm-up
            var rnd = np.random.RandomState(0);
            for (int i = 0; i < 20; i++)
            {
                using var nd = np.arange(200 * 50).reshape(200, 50).astype(NPTypeCode.Double);
                rnd.shuffle(nd);
            }
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 200; i++)
            {
                using var nd = np.arange(200 * 50).reshape(200, 50).astype(NPTypeCode.Double);
                rnd.shuffle(nd);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // 200 iterations × ~199 swaps × (2 view wrappers + 1 row copy
            // of 50 doubles = 400 bytes) — the temp buffer accumulation is
            // the dominant term. Without using, that's ~16 MiB queued.
            deltaMB.Should().BeLessThan(30);
        }
    }
}
