using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #575, "Performance: Replace heap allocations with stackalloc in 74+ hot paths"
    ///     (https://github.com/SciSharp/NumSharp/issues/575) as superseded: the per-element allocations it listed in
    ///     <c>np.repeat</c> and <c>nonzero</c> are gone, and its benchmark (<c>sum(axis=0)</c> of a float32 cube) computes the
    ///     right result.
    /// </summary>
    /// <remarks>
    ///     The allocation tests read <c>GC.GetAllocatedBytesForCurrentThread</c>, a per-thread counter, so they need no
    ///     <c>[DoNotParallelize]</c> (TestParallelization's rule); the unmanaged result buffers are not counted, only the managed
    ///     objects a call creates. The answer's timings (34.7 ms against NumPy's 43.8 ms) are the benchmark harness's job, not
    ///     a unit test's: a wall-clock assertion would be flaky here.
    /// </remarks>
    [TestClass]
    public class Issue575
    {
        /// <summary>
        ///     Managed bytes one call allocates on this thread, measured after a warm-up call of the same shape (the warm-up
        ///     compiles the kernels and fills the buffer pool's bucket, which are one-time costs, not per-element ones).
        /// </summary>
        /// <param name="call">The NumSharp call to measure; its result is disposed.</param>
        /// <returns>The managed bytes the measured call allocated.</returns>
        private static long ManagedBytesOf(Func<object> call)
        {
            Dispose(call());
            long before = GC.GetAllocatedBytesForCurrentThread();
            var result = call();
            long after = GC.GetAllocatedBytesForCurrentThread();
            Dispose(result);
            return after - before;
        }

        /// <summary>Disposes an array or every array of an array tuple a measured call returned.</summary>
        /// <param name="result">The call's result.</param>
        private static void Dispose(object result)
        {
            if (result is NDArray nd)
                nd.Dispose();
            else if (result is NDArray[] many)
                foreach (var m in many)
                    m.Dispose();
        }

        /// <summary>
        ///     <c>np.repeat</c> allocated a <c>new int[1]</c> per output element (the issue's Tier 1 item 1): 300,000 output
        ///     elements cost about 9.6 MB of managed garbage. Now the managed cost does not grow with the element count: a
        ///     100,000-element repeat allocates no more managed memory than a 1,000-element one, give or take 16 KB.
        /// </summary>
        [TestMethod]
        public void Repeat_AllocatesNoManagedMemoryPerElement()
        {
            var small = np.arange(1_000);
            var large = np.arange(100_000);

            long smallBytes = ManagedBytesOf(() => np.repeat(small, 3));
            long largeBytes = ManagedBytesOf(() => np.repeat(large, 3));

            (largeBytes - smallBytes).Should().BeLessThan(16 * 1024,
                $"repeat of 1,000 elements allocated {smallBytes} managed bytes and of 100,000 elements {largeBytes}");
        }

        /// <summary>
        ///     <c>nonzero</c> cloned a coordinate array per nonzero element (the issue's Tier 1 item 2): an <c>int[]</c> object
        ///     plus its list slot, about 40 managed bytes each. No managed object is allocated per nonzero element any more:
        ///     going from 500 to 50,000 nonzeros adds less than 16 managed bytes per extra element, below the 24 bytes any
        ///     object costs on a 64-bit runtime.
        /// </summary>
        /// <remarks>
        ///     What remains is one bulk buffer: the N-D route stages the flat indices in a single managed <c>long[count]</c>
        ///     (<c>Default.NonZero</c>'s <c>flatBuffer</c>, 8 bytes per nonzero element) before expanding them into the
        ///     unmanaged per-axis results. That is one allocation, not one per element, so the answer's claim holds; pooling it
        ///     would be a separate optimization.
        /// </remarks>
        [TestMethod]
        public void Nonzero_AllocatesNoManagedObjectPerNonzeroElement()
        {
            var sparse = (np.arange(1_000).reshape(10, 100) % 2) == 0;        // 500 nonzeros
            var dense = (np.arange(100_000).reshape(100, 1_000) % 2) == 0;    // 50,000 nonzeros

            long sparseBytes = ManagedBytesOf(() => np.nonzero(sparse));
            long denseBytes = ManagedBytesOf(() => np.nonzero(dense));

            double bytesPerExtraNonzero = (denseBytes - sparseBytes) / (double)(50_000 - 500);
            bytesPerExtraNonzero.Should().BeLessThan(16,
                $"nonzero with 500 nonzeros allocated {sparseBytes} managed bytes and with 50,000 nonzeros {denseBytes}; " +
                "an object per element would cost at least 24 bytes each");
        }

        /// <summary>
        ///     The issue's benchmark operation, at a size a unit test can afford: <c>sum(axis=0)</c> of a float32 ones cube is
        ///     a float32 plane holding the axis length everywhere (NumPy 2.4.2's result for every shape).
        /// </summary>
        [TestMethod]
        public void SumAxis0_OfAFloat32Cube_IsTheAxisLengthEverywhere()
        {
            var r = np.sum(np.ones(new Shape(50, 60, 70), np.float32), axis: 0);

            r.typecode.Should().Be(NPTypeCode.Single);
            r.Should().BeShaped(60, 70);
            np.all(r == 50f).Should().BeTrue();
        }
    }
}
