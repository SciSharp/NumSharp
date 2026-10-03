using System;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — a caller's Shape reaching an array's storage
    // =====================================================================
    //
    // Left over from the view-Shape allocation overrun fix (de35a0f2, 2026-10-03). That fix routes every FRESH allocation
    // (new NDArray(dtype, shape), np.empty, np.zeros, np.ones, np.full) through Shape.ForFreshAllocation and every np.random
    // size through Shape.AsClean. Two neighbouring paths still let the caller's Shape object decide what an array reads:
    //
    // 1. The WRAP constructors new NDArray(Array, Shape, order) and new NDArray(IArraySlice, Shape, order) install the
    //    given Shape over the caller's buffer as given, strides and offset included. A NumPy port writes
    //    new NDArray(values, x.Shape) for NumPy's np.ndarray(x.shape, buffer=values); with x = np.arange(20)["::4"] the
    //    array reads element i at i * 4 of a 5-element buffer:
    //      NumSharp: new NDArray(new double[] {1, 2, 3, 4, 5}, x.Shape)  ->  [1, 5, 0, 0, 1.16e-311]  (heap memory)
    //      NumPy 2.4.2: np.ndarray(x.shape, dtype=float, buffer=np.array([1., 2, 3, 4, 5]))  ->  [1. 2. 3. 4. 5.]
    //      NumPy 2.4.2: np.ndarray(x.shape, dtype=float, buffer=..., strides=x.strides)  ->  ValueError: strides is
    //                   incompatible with shape of requested array and size of buffer
    //    order='F' rebuilds the shape from its dimensions, but only when NDim > 1, so a 1-D view's strides survive it too.
    //    The fresh-allocation rule cannot simply move into new UnmanagedStorage(IArraySlice, Shape): internal code builds
    //    views through that constructor on purpose (UnmanagedStorage.Getters.cs).
    //
    // 2. Shape's public indexer setter (this[int dim] { set; }) writes into the dimension ARRAY, and an allocation keeps
    //    the caller's Shape (sharing that array) whenever the layout is already fresh. Changing the caller's Shape after
    //    the call changes the array: its dimensions grow while its cached size and buffer do not, so indexing the new rows
    //    reads past the buffer.
    //      NumSharp: var s = new Shape(3, 4); var a = np.zeros(s); s[0] = 100;  ->  a.shape == (100, 4), a.size == 12
    //      NumPy 2.4.2: s = [3, 4]; a = np.zeros(s); s[0] = 100  ->  a.shape == (3, 4)
    //
    // Every test asserts the LAYOUT or SHAPE first, so while a bug is open it fails before any element past the buffer is
    // read. EXPECTED (when fixed): each test passes; remove its [OpenBugs].
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>
        ///     <c>new NDArray(Array, x.Shape)</c> for a stepped view <c>x</c>: the values must be laid out from the shape's
        ///     dimensions in C order (NumPy's <c>np.ndarray(x.shape, buffer=values)</c>), not under the view's stride 4.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "wrap-ctor-view-shape")]
        public void WrapCtor_ArrayValues_SteppedViewShape_ShouldLayOutInCOrder()
        {
            using var x = np.arange(20)["::4"];
            var values = new double[] { 1, 2, 3, 4, 5 };
            using var a = new NDArray(values, x.Shape);

            a.shape.Should().Equal(new long[] { 5 });
            a.Shape.strides.Should().Equal(new long[] { 1 },
                "the 5 values fill a 5-element buffer; the view's stride 4 reads [1, 5, heap, heap, heap]");
            a.Shape.offset.Should().Be(0);
            a.ToArray<double>().Should().Equal(values);
        }

        /// <summary>
        ///     <c>new NDArray(IArraySlice, x.Shape)</c> for a stepped view <c>x</c>: the same contract as the
        ///     <see cref="Array"/> constructor, through the slice overload <c>np.ones</c> also uses.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "wrap-ctor-view-shape")]
        public void WrapCtor_ArraySliceValues_SteppedViewShape_ShouldLayOutInCOrder()
        {
            using var x = np.arange(20)["::4"];
            using var a = new NDArray(ArraySlice.FromArray(new double[] { 1, 2, 3, 4, 5 }), x.Shape);

            a.Shape.strides.Should().Equal(new long[] { 1 }, "the view's stride 4 reads past the 5-element slice");
            a.Shape.offset.Should().Be(0);
            a.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0);
        }

        /// <summary>
        ///     A 2-D column-stepped view (<c>arange(24).reshape(4, 6)[:, ::2]</c>, strides <c>(6, 2)</c>): NumPy's
        ///     <c>np.ndarray((4, 3), buffer=arange(12))</c> is <c>[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "wrap-ctor-view-shape")]
        public void WrapCtor_ArrayValues_ColumnSteppedView2DShape_ShouldLayOutInCOrder()
        {
            using var v = np.arange(24).reshape(4, 6)[":, ::2"];
            var values = new long[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            using var a = new NDArray(values, v.Shape);

            a.shape.Should().Equal(new long[] { 4, 3 });
            a.Shape.strides.Should().Equal(new long[] { 3, 1 },
                "the view's strides (6, 2) reach element 22 of a 12-element buffer");
            a.Shape.offset.Should().Be(0);
            a.ToArray<long>().Should().Equal(values, "values fill the array in C order");
        }

        /// <summary>
        ///     <c>order: 'F'</c> rebuilds the shape from its dimensions only when <c>NDim &gt; 1</c>, so a 1-D stepped view keeps
        ///     its stride 4 under <c>'F'</c> as well. A 1-D array is both C- and F-ordered: the values must read back in order.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "wrap-ctor-view-shape")]
        public void WrapCtor_ArrayValues_SteppedViewShape_FOrder1D_ShouldLayOutDensely()
        {
            using var x = np.arange(20)["::4"];
            using var a = new NDArray(new double[] { 1, 2, 3, 4, 5 }, x.Shape, 'F');

            a.Shape.strides.Should().Equal(new long[] { 1 }, "order 'F' of a 1-D shape is the dense stride 1");
            a.Shape.offset.Should().Be(0);
            a.ToArray<double>().Should().Equal(1.0, 2.0, 3.0, 4.0, 5.0);
        }

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
