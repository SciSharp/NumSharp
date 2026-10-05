using System;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — a caller's Shape reaching an array's storage
    // =====================================================================
    //
    // Left over from the view-Shape allocation overrun fix (de35a0f2, 2026-10-03). That fix routes every FRESH allocation
    // (new NDArray(dtype, shape), np.empty, np.zeros, np.ones, np.full) through Shape.ForFreshAllocation and every np.random
    // size through Shape.AsClean. Plan: docs/plans/view-shape-open-bugs.md.
    //
    // The WRAP constructors new NDArray(Array | IArraySlice, Shape, order) were the first bug of this file; they now read
    // the shape's dimensions only (NumPy's np.ndarray(shape, buffer=...)), and their pins moved to
    // Creation/WrapConstructor.ViewShape.Tests.cs.
    //
    // Still open: Shape's public indexer setter (this[int dim] { set; }) writes into the dimension ARRAY, and an
    // allocation keeps the caller's Shape (sharing that array) whenever the layout is already fresh. Changing the
    // caller's Shape after the call changes the array: its dimensions grow while its cached size and buffer do not, so
    // indexing the new rows reads past the buffer.
    //   NumSharp: var s = new Shape(3, 4); var a = np.zeros(s); s[0] = 100;  ->  a.shape == (100, 4), a.size == 12
    //   NumPy 2.4.2: s = [3, 4]; a = np.zeros(s); s[0] = 100  ->  a.shape == (3, 4)
    //
    // Every test asserts the SHAPE first, so while the bug is open it fails before any element past the buffer is read.
    // EXPECTED (when fixed): each test passes; remove its [OpenBugs].
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>
        ///     <c>np.zeros(s)</c> must not change when the caller later writes <c>s[0]</c>: NumPy's array keeps its own
        ///     dimensions (<c>s = [3, 4]; a = np.zeros(s); s[0] = 100; a.shape == (3, 4)</c>).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "shape-indexer-shares-dims")]
        public void ShapeIndexerSetter_AfterNpZeros_ShouldNotChangeTheArray()
        {
            var s = new Shape(3, 4);
            using var a = np.zeros(s);
            s[0] = 100;

            a.shape.Should().Equal(new long[] { 3, 4 },
                "the array shares the caller's dimension array, so it becomes (100, 4) over 12 elements");
            a.size.Should().Be(12);
        }

        /// <summary>
        ///     <c>new NDArray(dtype, s)</c>: the same sharing through the allocating constructor every <c>np.*</c> creation
        ///     function reaches.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "shape-indexer-shares-dims")]
        public void ShapeIndexerSetter_AfterNDArrayConstructor_ShouldNotChangeTheArray()
        {
            var s = new Shape(3, 4);
            using var a = new NDArray(np.float64, s);
            s[1] = 100;

            a.shape.Should().Equal(new long[] { 3, 4 }, "the array shares the caller's dimension array");
        }

        /// <summary>
        ///     <c>rng.random(s)</c>: a random sampler keeps a clean size as given (<see cref="Shape.AsClean"/>), so its result
        ///     shares the caller's dimension array too.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "shape-indexer-shares-dims")]
        public void ShapeIndexerSetter_AfterRandomSize_ShouldNotChangeTheArray()
        {
            var s = new Shape(3, 4);
            using var r = np.random.default_rng(42).random(s);
            s[0] = 100;

            r.shape.Should().Equal(new long[] { 3, 4 }, "the draws share the caller's dimension array");
        }
    }
}
