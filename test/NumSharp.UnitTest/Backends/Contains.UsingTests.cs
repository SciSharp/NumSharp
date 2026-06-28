using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    /// Guards the `using` on `comparison = this == scalar` inside NDArray.Contains.
    /// </summary>
    [TestClass]
    public class Contains_UsingTests : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void Contains_PresentValue_True()
        {
            var arr = np.array(new[] { 1, 2, 3, 4, 5 });
            arr.Contains(3).Should().BeTrue();
        }

        [TestMethod]
        public void Contains_AbsentValue_False()
        {
            var arr = np.array(new[] { 1, 2, 3, 4, 5 });
            arr.Contains(10).Should().BeFalse();
        }

        [TestMethod]
        public void Contains_2D_PresentValue_True()
        {
            var arr = np.arange(20).reshape(4, 5);
            arr.Contains(13).Should().BeTrue();
        }

        [TestMethod]
        public void Contains_PreservesCallerInput()
        {
            // When `value` is itself an NDArray, np.asanyarray returns it as-is.
            // The `using` on `comparison` (the equality result) must NOT dispose
            // either the caller's `arr` or `value`.
            var arr = np.array(new[] { 1, 2, 3, 4, 5 });
            var query = np.array(new[] { 3 });
            arr.Contains((object)query).Should().BeTrue();

            // Both the array and the query must remain usable after Contains.
            arr.IsDisposed.Should().BeFalse();
            query.IsDisposed.Should().BeFalse();
            ((int)arr[2]).Should().Be(3);
            ((int)query[0]).Should().Be(3);
        }

        // --------------------------- leak guard ---------------------------

        /// <summary>
        /// Tight loop. Each Contains call allocated a bool array sized to
        /// broadcast(this, scalar) — using on `comparison` should keep
        /// working set near constant.
        /// </summary>
        [TestMethod]
        public void Contains_TightLoop_DoesNotLeakWorkingSet()
        {
            using var arr = np.arange(50_000).astype(NPTypeCode.Int32);

            for (int i = 0; i < 20; i++)
                _ = arr.Contains(42);
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 2000; i++)
                _ = arr.Contains(i);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            // 2000 × 50K-bool ≈ 100 MiB of comparison buffers churned through
            // the finalizer queue without the using. 20 MiB headroom.
            deltaMB.Should().BeLessThan(20);
        }
    }
}
