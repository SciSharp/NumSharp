using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Guards the `using` on promoted/broadcasted/contiguous inside np.tile's
    /// general case (the broadcast-then-reshape path).
    /// </summary>
    [TestClass]
    public class np_tile_UsingTests : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void Tile_1DToRepeat_StillCorrect()
        {
            // np.tile([1,2,3], 3) -> [1,2,3,1,2,3,1,2,3]
            var a = np.array(new[] { 1, 2, 3 });
            var r = np.tile(a, 3);

            r.shape.Should().ContainInOrder(9L);
            for (int i = 0; i < 9; i++)
                ((int)r[i]).Should().Be((i % 3) + 1);
        }

        [TestMethod]
        public void Tile_2D_StillCorrect()
        {
            // np.tile([[1,2],[3,4]], (2,2)) -> 4x4 block tiled
            var a = np.array(new[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.tile(a, new long[] { 2, 2 });

            r.shape.Should().ContainInOrder(4L, 4L);
            // Top-left 2x2 == bottom-left 2x2 == top-right 2x2 == bottom-right 2x2
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    ((int)r[i, j]).Should().Be((int)a[i % 2, j % 2]);
        }

        [TestMethod]
        public void Tile_2D_AsymmetricReps_StillCorrect()
        {
            // np.tile([1,2,3], (3,2)) -> 3x6: each row is the input repeated twice.
            var a = np.array(new[] { 1, 2, 3 });
            var r = np.tile(a, new long[] { 3, 2 });

            r.shape.Should().ContainInOrder(3L, 6L);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 6; j++)
                    ((int)r[i, j]).Should().Be((j % 3) + 1);
        }

        // --------------------------- leak guard ---------------------------

        /// <summary>
        /// Tight loop of 2D tile. Each call previously left three NDArray
        /// wrappers (promoted, broadcasted, contiguous) on the finalizer
        /// queue — `contiguous` carries the full output-sized buffer.
        /// </summary>
        [TestMethod]
        public void Tile_TightLoop_DoesNotLeakWorkingSet()
        {
            using var a = np.arange(100).reshape(10, 10).astype(NPTypeCode.Int32);

            for (int i = 0; i < 20; i++)
                _ = np.tile(a, new long[] { 5, 5 });
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 500; i++)
            {
                using var r = np.tile(a, new long[] { 5, 5 });
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // Each call produces a 50×50 Int32 output (~10 KiB) plus its
            // contiguous intermediate of the same size. Without using:
            // 500 × 2 × 10 KiB = ~10 MiB queued.
            deltaMB.Should().BeLessThan(30);
        }
    }
}
