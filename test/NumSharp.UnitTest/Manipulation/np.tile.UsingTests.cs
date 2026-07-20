using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Guards the `using` on promoted/broadcasted/contiguous inside np.tile's
    /// general case (the broadcast-then-reshape path).
    /// </summary>
    /// <remarks>
    /// <c>Tile_TightLoop_DoesNotLeakWorkingSet</c> used to close this class. By its own arithmetic
    /// the intermediates totalled ~10 MiB against a 30 MiB threshold, so reintroducing the leak
    /// could not have tripped it. Removed — see <see cref="LeakGuards"/>.
    /// </remarks>
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

        // --------------------------- lifetime ---------------------------

        /// <summary>
        /// <c>promoted</c> and <c>broadcasted</c> are views of the caller's array, and
        /// <c>contiguous</c> carries the output buffer. Releasing all three must free neither the
        /// input nor the result.
        /// </summary>
        /// <remarks>
        /// The <c>reps=(1,1)</c> case is the one worth pinning: nothing needs promoting or
        /// broadcasting, so the intermediates are at their most likely to BE the input rather than
        /// a copy of it — exactly when a `using` on them does damage. Disposing the source last
        /// checks the other end of the same chain: whether the result holds its own reference to
        /// whatever buffer it ended up with. (It does not prove the result is a copy — a correctly
        /// refcounted alias survives this too — it proves nothing was released early.)
        /// </remarks>
        [TestMethod]
        public void Tile_ReleasesIntermediates_ButNotInputOrOutput()
        {
            var a = np.arange(100).reshape(10, 10).astype(NPTypeCode.Int32);

            var tiled = np.tile(a, new long[] { 5, 5 });
            LeakGuards.StillUsable(a, " — promoted/broadcasted are views of the input");
            LeakGuards.StillUsable(tiled, " — `contiguous` carries the output buffer");

            // The no-op reps: intermediates most likely to alias the input outright.
            var same = np.tile(a, new long[] { 1, 1 });
            LeakGuards.StillUsable(a, " — reps=(1,1) must not hand the input to a `using`");
            LeakGuards.StillUsable(same);
            ((int)a[9, 9]).Should().Be(99);

            // Both results must still stand once the source is gone.
            a.Dispose();
            LeakGuards.StillUsable(tiled, " — the result must hold its own reference to its buffer");
            LeakGuards.StillUsable(same);
            ((int)tiled[49, 49]).Should().Be(99);
            ((int)same[9, 9]).Should().Be(99);
        }
    }
}
