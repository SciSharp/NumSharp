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
    // Fixed so far: the WRAP constructors new NDArray(Array | IArraySlice, Shape, order) read the shape's dimensions only
    // (bug 1; their pins moved to Creation/WrapConstructor.ViewShape.Tests.cs), and Shape's indexer has no setter any more
    // (bug 2a; Shape is immutable, see View/Shape.Immutability.Tests.cs).
    //
    // Still open (bug 2d): the public Shape constructors keep the caller's long[] as the shape's dimension array, and an
    // allocation keeps that shape (sharing the array) whenever its layout is already fresh. NumPy copies the shape
    // sequence into the array's own dimensions, so changing the caller's list afterwards changes nothing:
    //   NumPy 2.4.2: s = [3, 4]; a = np.zeros(s); s[0] = 100  ->  a.shape == (3, 4)
    //   NumSharp:    long[] s = {3, 4}; var a = np.zeros(s); s[0] = 100;  ->  a.shape == (100, 4), a.size == 12
    // Changed dimensions do not change the cached size or the buffer, so indexing the new rows reads past the buffer.
    //
    // Every test asserts the SHAPE first, so while the bug is open it fails before any element past the buffer is read.
    // EXPECTED (when fixed): each test passes; remove its [OpenBugs] and move it next to the Shape immutability tests.
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>
        ///     <c>np.zeros(dims)</c> must not change when the caller later writes <c>dims[0]</c>: NumPy's array keeps its own
        ///     dimensions (<c>s = [3, 4]; a = np.zeros(s); s[0] = 100; a.shape == (3, 4)</c>). The C# analog of NumPy's list
        ///     is the caller's own <c>long[]</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "caller-dims-shared")]
        public void CallerDims_AfterNpZeros_ShouldNotChangeTheArray()
        {
            long[] dims = { 3, 4 };
            using var a = np.zeros(dims);
            dims[0] = 100;

            a.shape.Should().Equal(new long[] { 3, 4 },
                "the array shares the caller's dimension array, so it becomes (100, 4) over 12 elements");
            a.size.Should().Be(12);
        }

        /// <summary>
        ///     <c>new NDArray(dtype, new Shape(dims))</c>: the same sharing through the allocating constructor every
        ///     <c>np.*</c> creation function reaches (NumPy: <c>np.ndarray(s)</c> copies <c>s</c>).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "caller-dims-shared")]
        public void CallerDims_AfterNDArrayConstructor_ShouldNotChangeTheArray()
        {
            long[] dims = { 3, 4 };
            using var a = new NDArray(np.float64, new Shape(dims));
            dims[1] = 100;

            a.shape.Should().Equal(new long[] { 3, 4 }, "the array shares the caller's dimension array");
        }

        /// <summary>
        ///     <c>rng.random(new Shape(dims))</c>: a random sampler keeps a clean size as given (<see cref="Shape.AsClean"/>),
        ///     so its result shares the caller's dimension array too (NumPy: <c>rng.random(s)</c> copies <c>s</c>).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "caller-dims-shared")]
        public void CallerDims_AfterRandomSize_ShouldNotChangeTheArray()
        {
            long[] dims = { 3, 4 };
            using var r = np.random.default_rng(42).random(new Shape(dims));
            dims[0] = 100;

            r.shape.Should().Equal(new long[] { 3, 4 }, "the draws share the caller's dimension array");
        }
    }
}
