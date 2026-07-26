using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    ///     Regression tests for crashes/corruption discovered while battle-testing np.pad
    ///     against NumPy 2.4.2. Each of these previously segfaulted, divided by zero, or
    ///     threw on arrays with a zero-size dimension. They are reduction/cast/binary-op
    ///     root causes that np.pad's stat and linear_ramp modes surfaced.
    ///
    ///     NumPy reference behaviour (2.4.2) is noted inline.
    /// </summary>
    [TestClass]
    public class EmptyArrayCrashRegressionTests
    {
        // ---------------- np.median / quantile / percentile on empty ----------------

        [TestMethod]
        public void Median_EmptyVector_ReturnsNaN()
        {
            // NumPy: np.median([]) -> nan (previously segfaulted in the quantile IL kernel).
            var r = np.median(np.array(new double[0]));
            double.IsNaN(Convert.ToDouble(r.GetAtIndex(0))).Should().BeTrue();
        }

        [TestMethod]
        public void Median_EmptyAxis_ReturnsNaNFilled()
        {
            // NumPy: np.median(np.zeros((2,0)), axis=1) -> [nan, nan]
            var r = np.median(np.zeros(new Shape(2, 0), typeof(double)), axis: 1);
            r.shape.Should().BeEquivalentTo(new int[] { 2 });
            double.IsNaN(r.GetDouble(0)).Should().BeTrue();
            double.IsNaN(r.GetDouble(1)).Should().BeTrue();
        }

        [TestMethod]
        public void Median_EmptyIntVector_ReturnsNaN()
        {
            // Integer median promotes to float64; empty -> nan (no crash).
            var r = np.median(np.array(new int[0]));
            double.IsNaN(Convert.ToDouble(r.GetAtIndex(0))).Should().BeTrue();
        }

        [TestMethod]
        public void Quantile_EmptyVector_Throws()
        {
            // NumPy diverges from median here: np.quantile([], 0.5) raises IndexError.
            // NumSharp must throw (not segfault).
            Action act = () => np.quantile(np.array(new double[0]), 0.5);
            act.Should().Throw<Exception>();
        }

        [TestMethod]
        public void Percentile_EmptyVector_Throws()
        {
            Action act = () => np.percentile(np.array(new double[0]), 50);
            act.Should().Throw<Exception>();
        }

        // ---------------- astype / Cast on a zero-size-dimension array ----------------

        [TestMethod]
        public void Astype_ZeroSizeDimension_PreservesShapeAndRetypes()
        {
            // (1,0) int32 -> double. Previously threw ArgumentOutOfRangeException in CastTo
            // because Shape.IsEmpty only catches the uninitialized sentinel, not a real
            // shape with a zero-size dimension.
            var a = np.zeros(new Shape(1, 0), typeof(int));
            var r = a.astype(typeof(double));
            r.shape.Should().BeEquivalentTo(new int[] { 1, 0 });
            r.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void Astype_ZeroSizeDimension_SlicedView()
        {
            var a = np.zeros(new Shape(4, 3, 0), typeof(int))[new Slice(1, 2), Slice.All, Slice.All];
            var r = a.astype(typeof(double));
            r.shape.Should().BeEquivalentTo(new int[] { 1, 3, 0 });
            r.dtype.Should().Be(typeof(double));
        }

        // ---------------- binary op with a zero-element broadcast result ----------------

        [TestMethod]
        public void BinaryOp_EmptyBroadcast_ReturnsEmptyResult()
        {
            // (3,1,1) * (1,0,2) broadcasts to (3,0,2) — zero elements. The IL/NDIter
            // element-wise path previously corrupted the heap; it must short-circuit.
            var a = np.ones(new Shape(3, 1, 1), typeof(double));
            var b = np.zeros(new Shape(1, 0, 2), typeof(double));
            var r = a * b;
            r.shape.Should().BeEquivalentTo(new int[] { 3, 0, 2 });
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void BinaryOp_EmptyBroadcast_Subtract2D()
        {
            var a = np.ones(new Shape(3, 1), typeof(double));
            var b = np.zeros(new Shape(1, 0), typeof(double));
            var r = a - b;
            r.shape.Should().BeEquivalentTo(new int[] { 3, 0 });
            r.size.Should().Be(0);
        }

        // ---------------- arr[slice] = emptyArray (SetData) ----------------

        [TestMethod]
        public void SetData_AssignEmptyIntoEmptyRegion_NoCrash()
        {
            // np.pad's _PadSimple does `padded[originalSlice] = array` where array.size == 0
            // when padding the non-empty axis of a zero-dim array. Previously divide-by-zero
            // in UnmanagedStorage.SetData (subShape.size % valueshape.size).
            var padded = np.zeros(new Shape(8, 0), typeof(double));
            Action act = () => { padded[new Slice(3, 5), new Slice(0, 0)] = np.ones(new Shape(2, 0), typeof(double)); };
            act.Should().NotThrow();
        }

        // ---------------- basic indexing of an array with a zero-sized axis ----------------
        //
        // A valid integer index into an array whose SIZE is 0 used to throw
        // `IndexOutOfRangeException: The offset 0 is out of range in Shape 0`. The offset
        // bounds-check in Shape.GetSubshape compares against `bufferSize > 0 ? bufferSize : size`,
        // which is 0 for an empty array — so every offset, the correct one included, was "out of
        // range". Per-axis validation (NumPy's real rule) already happened earlier in
        // Shape.InferNegativeCoordinates, so the offset test is only a buffer backstop, and there
        // is no buffer to overrun when the sub-view addresses no element.

        [TestMethod]
        public void Index_TrailingZeroAxis_ReturnsEmptyView()
        {
            // NumPy: a[0] -> (3,0); a[0,0] -> (0,); a[1,2] -> (0,)
            var a = np.zeros(new Shape(2, 3, 0), typeof(double));

            a[0].shape.Should().BeEquivalentTo(new long[] { 3, 0 });
            a[1].shape.Should().BeEquivalentTo(new long[] { 3, 0 });
            a[-1].shape.Should().BeEquivalentTo(new long[] { 3, 0 });
            a[0, 0].shape.Should().BeEquivalentTo(new long[] { 0 });
            a[1, 2].shape.Should().BeEquivalentTo(new long[] { 0 });
            a.GetData(new long[] { 0 }).shape.Should().BeEquivalentTo(new long[] { 3, 0 });

            np.zeros(new Shape(3, 0), typeof(double))[2].shape.Should().BeEquivalentTo(new long[] { 0 });
        }

        [TestMethod]
        public void Index_TrailingZeroAxis_StillRejectsOutOfRange()
        {
            // The backstop was relaxed, not removed — NumPy's per-axis IndexError must survive.
            var a = np.zeros(new Shape(2, 3, 0), typeof(double));

            new Action(() => { var _ = a[2]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = a[-3]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = a[0, 3]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = a[0, 0, 0]; }).Should().Throw<IndexError>("axis 2 has size 0");
            // a zero in the LEADING axis makes even index 0 invalid, as in NumPy
            new Action(() => { var _ = np.zeros(new Shape(0, 3, 4), typeof(double))[0]; }).Should().Throw<IndexError>();
        }

        [TestMethod]
        public void Index_EmptyWindowOverALiveBuffer_ReturnsEmptyView()
        {
            // (5,0) carved out of a live 10-element buffer: the strides still step across the
            // original rows, so [i] computes an offset of 2i into an InternalArray already
            // narrowed to Count 0. NumPy returns the empty view for every valid i.
            var big = np.arange(10).astype(NPTypeCode.Double).reshape(5, 2);
            var window = big[":, 1:1"];
            window.shape.Should().BeEquivalentTo(new long[] { 5, 0 });

            for (int i = 0; i < 5; i++)
                window[i].shape.Should().BeEquivalentTo(new long[] { 0 }, $"window[{i}]");

            new Action(() => { var _ = window[5]; }).Should().Throw<IndexError>();
        }

        [TestMethod]
        public void Index_NonEmptyNeighbours_AreUnaffected()
        {
            // The relaxed backstop must not let a real overrun through on ordinary arrays.
            var big = np.arange(10).astype(NPTypeCode.Double).reshape(5, 2);

            Convert.ToDouble(big[0].GetAtIndex(0)).Should().Be(0d);
            Convert.ToDouble(big[3].GetAtIndex(1)).Should().Be(7d);
            big[4].shape.Should().BeEquivalentTo(new long[] { 2 });
            Convert.ToDouble(big[4, 1].GetAtIndex(0)).Should().Be(9d);

            new Action(() => { var _ = big[5]; }).Should().Throw<IndexError>();
            new Action(() => { var _ = big[0, 2]; }).Should().Throw<IndexError>();
        }

        [TestMethod]
        public void Index_EmptySubView_IsAUsableArray()
        {
            // Not just a shape: the result has to behave like NumPy's empty view.
            var sub = np.zeros(new Shape(2, 3, 0), typeof(double))[0];

            sub.size.Should().Be(0);
            Convert.ToDouble(np.sum(sub).GetAtIndex(0)).Should().Be(0d, "NumPy: sum of empty is 0");
            (sub + sub).shape.Should().BeEquivalentTo(new long[] { 3, 0 });
            sub.T.shape.Should().BeEquivalentTo(new long[] { 0, 3 });
        }
    }
}
