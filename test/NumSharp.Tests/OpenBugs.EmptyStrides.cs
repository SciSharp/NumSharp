using System;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — the strides of empty (zero-size) arrays
    // =====================================================================
    //
    // Found 2026-10-05 while writing the empty-array cases of the random-ownership tests (0f904216): NumSharp's
    // `strides` of a zero-size array follow neither of NumPy 2.4.2's two rules, both in numpy/_core/src/multiarray:
    //   - an array NumPy ALLOCATES (np.zeros / np.empty / np.ones / np.full / np.arange(0) …) gets ALL-ZERO strides:
    //     PyArray_NewFromDescr_int allocates one byte for a zero-size array and "Make[s] sure all the strides are 0"
    //     (ctors.c); its transpose permutes those zeros;
    //   - a VIEW keeps the strides its slicing computed, even along an axis of extent 0: arange(12).reshape(4, 3)[:0] has
    //     strides (24, 8), [:, :0] has (24, 8), [1:1, ::2] has (24, 16).
    // NumSharp computes C strides with the zero extents multiplied in (np.zeros((2, 0)) → (0, 8), np.zeros((0, 3)) →
    // (24, 8), np.zeros(0) → (8,)) and zeroes the stride of a sliced axis of extent 0 (a[:0] → (0, 8), a[:, :0] → (24, 0)).
    // A wrap constructor over a caller's buffer is NOT part of this: NumPy's np.ndarray((0, 3), buffer=…) keeps the C
    // strides (24, 8) (the buffer exists, so nothing is zeroed), which Creation/WrapConstructor.ViewShape.Tests.cs pins.
    // np.ndarray(shape) WITHOUT a buffer already follows the allocation rule (all-zero strides, plan
    // docs/plans/np-ndarray-parity.md; Creation/np.ndarray.Test.cs pins it); the allocators below do not yet.
    //
    // Only `strides` is observable here (an empty array has no element to read); the flags already agree.
    // EXPECTED (when fixed): each test passes; remove its [OpenBugs] and move it next to the Shape tests.
    //
    // FIXED 2026-11-05 (the ufunc result-layout port): every fresh zero-size allocation gets all-zero strides
    // (Shape.ForFreshAllocation, the funnel of every allocator), so the allocation rule's pin moved to
    // View/Shape.ForFreshAllocation.Tests.cs (EmptyAllocation_HasAllZeroStrides). The slice rule below is still open.
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>
        ///     A slice keeps the strides its slicing computed, even along an axis of extent 0. With
        ///     <c>a = np.arange(12).reshape(4, 3)</c>: <c>a[:0]</c> → <c>(24, 8)</c>, <c>a[:, :0]</c> → <c>(24, 8)</c>,
        ///     <c>a[1:1, ::2]</c> → <c>(24, 16)</c>, <c>a[:0].T</c> → <c>(8, 24)</c>. NumSharp zeroes the stride of the empty
        ///     axis: <c>(0, 8)</c>, <c>(24, 0)</c>, <c>(0, 16)</c>, <c>(8, 0)</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "empty-strides")]
        public void EmptySlice_ShouldKeepTheParentStrides()
        {
            using var a = np.arange(12).reshape(4, 3);
            foreach (var (name, view, strides) in new (string, NDArray, long[])[]
                     {
                         ("a[:0]", a[":0"], new long[] { 24, 8 }),
                         ("a[:, :0]", a[":, :0"], new long[] { 24, 8 }),
                         ("a[1:1, ::2]", a["1:1, ::2"], new long[] { 24, 16 }),
                         ("a[:0].T", a[":0"].T, new long[] { 8, 24 }),
                     })
            {
                using (view)
                {
                    view.size.Should().Be(0, name);
                    view.strides.Should().Equal(strides, $"{name}: NumPy's slice keeps the parent's stride on the empty axis");
                }
            }
        }
    }
}
