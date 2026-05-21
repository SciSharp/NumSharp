using System;
using System.Linq;
using AwesomeAssertions;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Tests for the <c>axis</c> parameter of <see cref="np.repeat(NDArray, long, int?)"/>.
    /// Mirrors NumPy 2.x — see <c>numpy/_core/src/multiarray/item_selection.c</c> (PyArray_Repeat).
    /// </summary>
    [TestClass]
    public class np_repeat_axis_tests
    {
        #region 2D scalar repeats

        [TestMethod]
        public void Axis0_2D_Shape()
        {
            // np.repeat([[1,2],[3,4]], 2, axis=0).shape == (4, 2)
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, 2, axis: 0);
            r.shape.Should().Equal(new long[] { 4, 2 });
            r.ravel().ToArray<int>().Should().ContainInOrder(1, 2, 1, 2, 3, 4, 3, 4);
        }

        [TestMethod]
        public void Axis1_2D_Shape()
        {
            // np.repeat([[1,2],[3,4]], 3, axis=1).shape == (2, 6)
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, 3, axis: 1);
            r.shape.Should().Equal(new long[] { 2, 6 });
            r.ravel().ToArray<int>().Should().ContainInOrder(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4);
        }

        [TestMethod]
        public void NegativeAxis_EquivalentToPositive()
        {
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r1 = np.repeat(x, 2, axis: -1);
            var r2 = np.repeat(x, 2, axis: 1);
            r1.shape.Should().Equal(r2.shape);
            r1.ravel().ToArray<int>().Should().Equal(r2.ravel().ToArray<int>());
        }

        #endregion

        #region 3D scalar repeats — every axis position

        [TestMethod]
        public void Axis0_3D()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var r = np.repeat(arr, 2, axis: 0);
            r.shape.Should().Equal(new long[] { 4, 3, 4 });
            r.ravel().ToArray<long>().Take(12).Should().ContainInOrder(Enumerable.Range(0, 12).Select(i => (long)i));
            r.ravel().ToArray<long>().Skip(12).Take(12).Should().ContainInOrder(Enumerable.Range(0, 12).Select(i => (long)i));
        }

        [TestMethod]
        public void Axis1_3D()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var r = np.repeat(arr, 2, axis: 1);
            r.shape.Should().Equal(new long[] { 2, 6, 4 });
        }

        [TestMethod]
        public void Axis2_3D_Innermost()
        {
            var arr = np.arange(24).reshape(2, 3, 4);
            var r = np.repeat(arr, 2, axis: 2);
            r.shape.Should().Equal(new long[] { 2, 3, 8 });
            // First row should be 0,0,1,1,2,2,3,3
            r.ravel().ToArray<long>().Take(8).Should().ContainInOrder(0L, 0L, 1L, 1L, 2L, 2L, 3L, 3L);
        }

        #endregion

        #region Per-element repeats along axis

        [TestMethod]
        public void Axis0_PerElement()
        {
            // np.repeat([[1,2],[3,4]], [1,2], axis=0) -> [[1,2],[3,4],[3,4]]
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, np.array(new int[] { 1, 2 }), axis: 0);
            r.shape.Should().Equal(new long[] { 3, 2 });
            r.ravel().ToArray<int>().Should().ContainInOrder(1, 2, 3, 4, 3, 4);
        }

        [TestMethod]
        public void Axis1_PerElement()
        {
            // np.repeat([[1,2],[3,4]], [1,2], axis=1) -> [[1,2,2],[3,4,4]]
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, np.array(new int[] { 1, 2 }), axis: 1);
            r.shape.Should().Equal(new long[] { 2, 3 });
            r.ravel().ToArray<int>().Should().ContainInOrder(1, 2, 2, 3, 4, 4);
        }

        [TestMethod]
        public void Axis_SizeOneBroadcast()
        {
            // NumPy: size-1 repeats array broadcasts along the axis.
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, np.array(new int[] { 2 }), axis: 0);
            r.shape.Should().Equal(new long[] { 4, 2 });
        }

        [TestMethod]
        public void Axis_PerElement_MixedZero()
        {
            // np.repeat([[1,2],[3,4]], [0,2], axis=0) -> [[3,4],[3,4]]
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, np.array(new int[] { 0, 2 }), axis: 0);
            r.shape.Should().Equal(new long[] { 2, 2 });
            r.ravel().ToArray<int>().Should().ContainInOrder(3, 4, 3, 4);
        }

        #endregion

        #region 0-D input

        [TestMethod]
        public void ZeroDim_Axis0()
        {
            // np.repeat(np.array(5), 4, axis=0) -> [5,5,5,5]
            var r = np.repeat(np.array(5), 4, axis: 0);
            r.shape.Should().Equal(new long[] { 4 });
            r.ravel().ToArray<int>().Should().ContainInOrder(5, 5, 5, 5);
        }

        [TestMethod]
        public void ZeroDim_AxisNeg1()
        {
            var r = np.repeat(np.array(5), 4, axis: -1);
            r.shape.Should().Equal(new long[] { 4 });
        }

        [TestMethod]
        public void ZeroDim_InvalidAxis_Throws()
        {
            Action act = () => np.repeat(np.array(5), 4, axis: 1);
            act.Should().Throw<AxisError>();
        }

        #endregion

        #region Edge cases

        [TestMethod]
        public void Empty_2D_Axis0()
        {
            var arr = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
            var r = np.repeat(arr, 2, axis: 0);
            r.shape.Should().Equal(new long[] { 0, 3 });
            r.size.Should().Be(0);
        }

        [TestMethod]
        public void Empty_2D_Axis1()
        {
            var arr = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
            var r = np.repeat(arr, 2, axis: 1);
            r.shape.Should().Equal(new long[] { 0, 6 });
        }

        [TestMethod]
        public void Repeats0_ProducesEmptyAxis()
        {
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, 0, axis: 0);
            r.shape.Should().Equal(new long[] { 0, 2 });
        }

        [TestMethod]
        public void OutOfBoundsAxis_Throws()
        {
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            Action act = () => np.repeat(x, 2, axis: 5);
            act.Should().Throw<AxisError>();
        }

        [TestMethod]
        public void OutOfBoundsNegativeAxis_Throws()
        {
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            Action act = () => np.repeat(x, 2, axis: -5);
            act.Should().Throw<AxisError>();
        }

        [TestMethod]
        public void PerElement_SizeMismatch_Throws()
        {
            // NumPy: ValueError when len(repeats) != shape[axis] and not size-1.
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            Action act = () => np.repeat(x, np.array(new int[] { 1, 2, 3 }), axis: 0);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void PerElement_NegativeCount_Throws()
        {
            var x = np.array(new int[] { 1, 2, 3, 4 }).reshape(2, 2);
            Action act = () => np.repeat(x, np.array(new int[] { 1, -1 }), axis: 0);
            act.Should().Throw<ArgumentException>();
        }

        #endregion

        #region Non-contiguous input

        [TestMethod]
        public void Transposed_Axis0()
        {
            // Transpose is non-contig; the IL kernel reads from the materialized contig copy.
            var x = np.array(new int[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            var t = x.T; // shape (3, 2): [[1,4],[2,5],[3,6]]
            var r = np.repeat(t, 2, axis: 0);
            r.shape.Should().Equal(new long[] { 6, 2 });
            r.ravel().ToArray<int>().Should().ContainInOrder(1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6);
        }

        [TestMethod]
        public void FortranContig_Axis1()
        {
            var arr = np.array(new int[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            var fc = np.asfortranarray(arr);
            var r = np.repeat(fc, 2, axis: 1);
            r.shape.Should().Equal(new long[] { 2, 6 });
            r.ravel().ToArray<int>().Should().ContainInOrder(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6);
        }

        [TestMethod]
        public void SlicedView_Axis0()
        {
            // x[::2] takes rows 0,2 from a (3,2) base — non-contig view.
            var x = np.arange(6).reshape(3, 2);
            var view = x["::2"];
            view.shape.Should().Equal(new long[] { 2, 2 });
            var r = np.repeat(view, 2, axis: 0);
            r.shape.Should().Equal(new long[] { 4, 2 });
            r.ravel().ToArray<long>().Should().ContainInOrder(0L, 1L, 0L, 1L, 4L, 5L, 4L, 5L);
        }

        #endregion

        #region Dtype coverage along axis

        [TestMethod]
        public void Dtype_Byte_Axis0()
        {
            var x = np.array(new byte[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, 2, axis: 0);
            r.shape.Should().Equal(new long[] { 4, 2 });
            r.ravel().ToArray<byte>().Should().ContainInOrder((byte)1, (byte)2, (byte)1, (byte)2, (byte)3, (byte)4, (byte)3, (byte)4);
        }

        [TestMethod]
        public void Dtype_Int16_Axis1()
        {
            var x = np.array(new short[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, 2, axis: 1);
            r.ravel().ToArray<short>().Should().ContainInOrder((short)1, (short)1, (short)2, (short)2, (short)3, (short)3, (short)4, (short)4);
        }

        [TestMethod]
        public void Dtype_Int64_Axis0()
        {
            var x = np.array(new long[] { 1, 2, 3, 4 }).reshape(2, 2);
            var r = np.repeat(x, 2, axis: 0);
            r.ravel().ToArray<long>().Should().ContainInOrder(1L, 2L, 1L, 2L, 3L, 4L, 3L, 4L);
        }

        [TestMethod]
        public void Dtype_Single_Axis0()
        {
            var x = np.array(new float[] { 1.5f, 2.5f, 3.5f, 4.5f }).reshape(2, 2);
            var r = np.repeat(x, 2, axis: 0);
            r.ravel().ToArray<float>().Should().ContainInOrder(1.5f, 2.5f, 1.5f, 2.5f, 3.5f, 4.5f, 3.5f, 4.5f);
        }

        [TestMethod]
        public void Dtype_Double_Axis1()
        {
            var x = np.array(new double[] { 1.5, 2.5, 3.5, 4.5 }).reshape(2, 2);
            var r = np.repeat(x, 2, axis: 1);
            r.ravel().ToArray<double>().Should().ContainInOrder(1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5, 4.5);
        }

        [TestMethod]
        public void Dtype_Decimal_Axis1()
        {
            // 16-byte chunk -> exercises the Vector128 IL path.
            var x = np.array(new decimal[] { 1.5m, 2.5m, 3.5m, 4.5m }).reshape(2, 2);
            var r = np.repeat(x, 2, axis: 1);
            r.ravel().ToArray<decimal>().Should().ContainInOrder(1.5m, 1.5m, 2.5m, 2.5m, 3.5m, 3.5m, 4.5m, 4.5m);
        }

        [TestMethod]
        public void Dtype_Boolean_Axis0()
        {
            var x = np.array(new bool[] { true, false, false, true }).reshape(2, 2);
            var r = np.repeat(x, 2, axis: 0);
            r.ravel().ToArray<bool>().Should().ContainInOrder(true, false, true, false, false, true, false, true);
        }

        #endregion

        #region Multi-element chunk

        [TestMethod]
        public void Int32_Axis0_NonTrivialChunk()
        {
            // axis=0 on (2,3) int32 produces chunk = 3*4 = 12 bytes per slab —
            // exercises the cpblk path of the IL kernel.
            var x = np.array(new int[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            var r = np.repeat(x, 2, axis: 0);
            r.shape.Should().Equal(new long[] { 4, 3 });
            r.ravel().ToArray<int>().Should().ContainInOrder(1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6);
        }

        [TestMethod]
        public void Double_Axis0_NonTrivialChunk()
        {
            // chunk = 3*8 = 24 bytes
            var x = np.array(new double[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            var r = np.repeat(x, 2, axis: 0);
            r.shape.Should().Equal(new long[] { 4, 3 });
            r.ravel().ToArray<double>().Should().ContainInOrder(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0);
        }

        #endregion

        #region axis=None size-1 broadcast (NumPy-compat fix)

        [TestMethod]
        public void AxisNone_PerElement_SizeOneBroadcast()
        {
            // NumPy: np.repeat([1,2,3], np.array([2])) -> [1,1,2,2,3,3]
            // Existing impl rejected this; the unified IL kernel honors broadcast.
            var r = np.repeat(np.array(new int[] { 1, 2, 3 }), np.array(new int[] { 2 }));
            r.ravel().ToArray<int>().Should().ContainInOrder(1, 1, 2, 2, 3, 3);
        }

        #endregion
    }
}
