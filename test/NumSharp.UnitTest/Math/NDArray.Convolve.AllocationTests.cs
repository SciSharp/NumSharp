using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.IO;

namespace NumSharp.UnitTest
{
    /// <summary>
    /// Guards the allocation behaviour of <see cref="NDArray.convolve"/>: the inner loop
    /// must stay O(1) in managed allocations, no matter how many element-pairs it walks.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Correctness and buffer-lifetime for the same modes live in
    ///     <see cref="NdArray_Convolve_UsingTests"/>; this file only asserts the allocation
    ///     shape, which is the one property a runtime probe can actually measure here.
    ///     </para>
    ///     <para>
    ///     <b>Do not add a working-set assertion to this file.</b> The convolve probe was the
    ///     first member of that family to be convicted — it failed CI on Ubuntu (25 MiB) with
    ///     the invariant it named intact, while the buffer it guarded (~8.5 KB, 1.7 MiB over
    ///     the whole loop) could never have reached its own 20 MiB threshold. What actually
    ///     moved that number was per-element boxing churn, which is what the test below
    ///     measures directly. <c>Process.WorkingSet64</c> is process-wide OS RSS that Linux
    ///     does not trim on <c>GC.Collect</c>, so it reports allocator churn as permanent
    ///     growth on Ubuntu and not on Windows.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class NdArray_Convolve_AllocationTests
    {
        /// <summary>
        /// The convolution inner loop must not allocate per element-pair.
        /// </summary>
        /// <remarks>
        ///     A single generic <c>ConvolveFullTyped&lt;T&gt;</c> once read both operands as
        ///     <c>Converts.ToDouble((object)aPtr[j])</c>. The callee takes <see cref="object"/>,
        ///     so the JIT could not elide the boxes: two boxed doubles on EVERY inner-loop
        ///     iteration made managed allocation scale with the work — ~2.9 MB for the 64-tap
        ///     kernel here and ~23.4 MB for the 512-tap one.
        ///
        ///     Convolve has since been rewritten into per-dtype kernels that read their operands
        ///     through typed pointers (<c>ConvolveFullDouble</c> and friends), so no read on any
        ///     path boxes. This test pins that: it is a regression guard against a future generic
        ///     collapse back onto an <see cref="object"/>-taking conversion.
        ///
        ///     Comparing the two sizes is what makes it threshold-free: the bug is defined by
        ///     allocation TRACKING the element count, so the ratio is the assertion and the
        ///     absolute cap is only a backstop. Both are measured with
        ///     <see cref="AllocationTests.MinAllocated"/> (best-of-N floor) because
        ///     <c>GC.GetTotalAllocatedBytes</c> is process-wide and noise can only ever add.
        /// </remarks>
        [TestMethod]
        public void ConvolveSame_DoesNotAllocatePerElementPair()
        {
            var a = np.arange(1_000).astype(NPTypeCode.Double);
            var small = np.arange(64).astype(NPTypeCode.Double);   //  64,000 element-pairs
            var large = np.arange(512).astype(NPTypeCode.Double);  // 512,000 element-pairs — 8x

            long smallAlloc = AllocationTests.MinAllocated(() => { using var r = a.convolve(small, "same"); });
            long largeAlloc = AllocationTests.MinAllocated(() => { using var r = a.convolve(large, "same"); });

            largeAlloc.Should().BeLessThan(smallAlloc * 2,
                "8x the element-pairs must not cost meaningfully more managed memory — per-element "
                + "boxing would make this ratio ~8x, not ~1x");

            smallAlloc.Should().BeLessThan(64 * 1024,
                "a single convolve is a few KB of wrapper churn; per-element boxing cost ~2.9 MB");
        }
    }
}
