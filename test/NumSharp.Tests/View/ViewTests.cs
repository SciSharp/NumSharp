using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.View
{
    /// <summary>
    ///     Battle tests for NDArray.view() - NumPy parity verification.
    ///     Tests byte reinterpretation, not value conversion.
    /// </summary>
    [TestClass]
    public class ViewTests
    {
        #region Same Type Views

        [TestMethod]
        public void View_SameType_SharesMemory()
        {
            var arr = np.array(new int[] { 1, 2, 3, 4, 5 });
            var view = arr.view<int>();

            view[0] = 100;

            arr.GetInt32(0).Should().Be(100);
            view.GetInt32(0).Should().Be(100);
        }

        [TestMethod]
        public void View_SameType_PreservesShape()
        {
            var arr = np.arange(12).reshape(3, 4);
            var view = arr.view<long>();

            view.shape.Should().BeEquivalentTo(new long[] { 3, 4 });
        }

        #endregion

        #region Byte Reinterpretation (Same Size Types)

        [TestMethod]
        public void View_Float64AsInt64_ReinterpretsIEEE754Bits()
        {
            // NumPy: np.array([1.0, 2.0]).view(np.int64)
            // Returns IEEE 754 bit patterns, not converted values
            var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
            var view = arr.view<long>();

            // IEEE 754 double precision:
            // 1.0 = 0x3FF0000000000000
            // 2.0 = 0x4000000000000000
            // 3.0 = 0x4008000000000000
            view.GetInt64(0).Should().Be(0x3FF0000000000000L);
            view.GetInt64(1).Should().Be(0x4000000000000000L);
            view.GetInt64(2).Should().Be(0x4008000000000000L);
        }

        [TestMethod]
        public void View_Float32AsInt32_ReinterpretsIEEE754Bits()
        {
            // NumPy: np.array([1.0, 2.0], dtype=np.float32).view(np.int32)
            var arr = np.array(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var view = arr.view<int>();

            // IEEE 754 single precision:
            // 1.0f = 0x3F800000
            // 2.0f = 0x40000000
            // 3.0f = 0x40400000
            // 4.0f = 0x40800000
            view.GetInt32(0).Should().Be(0x3F800000);
            view.GetInt32(1).Should().Be(0x40000000);
            view.GetInt32(2).Should().Be(0x40400000);
            view.GetInt32(3).Should().Be(0x40800000);
        }

        [TestMethod]
        public void View_Int64AsFloat64_ReinterpretsToDouble()
        {
            // Put IEEE 754 bit pattern for 2.0, should read as 2.0
            var arr = np.array(new long[] { 0x4000000000000000L });
            var view = arr.view<double>();

            view.GetDouble(0).Should().Be(2.0);
        }

        #endregion

        #region Different Size Types (Shape Changes)

        [TestMethod]
        public void View_Float64AsFloat32_DoublesLastDimension()
        {
            // NumPy: np.array([1.0, 2.0], dtype=np.float64).view(np.float32)
            // float64[2] (16 bytes) -> float32[4] (16 bytes)
            var arr = np.array(new double[] { 1.0, 2.0 });
            var view = arr.view<float>();

            view.shape.Should().BeEquivalentTo(new long[] { 4 });
            view.size.Should().Be(4);
        }

        [TestMethod]
        public void View_Float64AsInt8_ExpandsLastDimension()
        {
            // NumPy: np.array([1.0, 2.0], dtype=np.float64).view(np.int8)
            // float64[2] (16 bytes) -> int8[16]
            var arr = np.array(new double[] { 1.0, 2.0 });
            var view = arr.view<byte>();

            view.shape.Should().BeEquivalentTo(new long[] { 16 });
            view.size.Should().Be(16);
        }

        [TestMethod]
        public void View_Int8AsFloat64_ContractsLastDimension()
        {
            // int8[16] (16 bytes) -> float64[2]
            var arr = np.zeros(new Shape(16), NPTypeCode.Byte);
            var view = arr.view<double>();

            view.shape.Should().BeEquivalentTo(new long[] { 2 });
            view.size.Should().Be(2);
        }

        [TestMethod]
        public void View_2D_AdjustsLastDimension()
        {
            // NumPy: np.zeros((3, 4), dtype=np.float64).view(np.float32)
            // float64[3, 4] (96 bytes) -> float32[3, 8] (96 bytes)
            var arr = np.zeros(new Shape(3, 4), NPTypeCode.Double);
            var view = arr.view<float>();

            view.shape.Should().BeEquivalentTo(new long[] { 3, 8 });
        }

        #endregion

        #region Memory Sharing

        [TestMethod]
        public void View_ModifyThroughView_AffectsOriginal()
        {
            var original = np.array(new double[] { 1.0, 2.0 });
            var intView = original.view<long>();

            // Set IEEE 754 bits for 3.0
            intView.SetInt64(0x4008000000000000L, 0);

            // Original should now read as 3.0
            original.GetDouble(0).Should().Be(3.0);
        }

        [TestMethod]
        public void View_ModifyOriginal_AffectsView()
        {
            var original = np.array(new double[] { 1.0, 2.0 });
            var intView = original.view<long>();

            original.SetDouble(5.0, 0);

            // View bits should change (5.0 = 0x4014000000000000)
            intView.GetInt64(0).Should().Be(0x4014000000000000L);
        }

        #endregion

        #region Edge Cases

        [TestMethod]
        public void View_WithNull_ReturnsSameTypeCopy()
        {
            var arr = np.array(new int[] { 1, 2, 3 });
            var view = arr.view();

            view.dtype.Should().Be(arr.dtype);
            view.shape.Should().BeEquivalentTo(arr.shape);
        }

        [TestMethod]
        public void View_OuterStridedDifferentSize_Works()
        {
            // NumPy 2.x: view(dtype) with a different itemsize only requires the LAST axis to be
            // contiguous, NOT the whole array. An outer-strided slice (contiguous last axis) is allowed.
            //   np.arange(12).reshape(3,4)[::2].view(np.int32)
            //     -> shape (2, 8), strides (64, 4) bytes, shares memory.
            var arr = np.arange(12).reshape(3, 4); // int64
            var sliced = arr["::2"];               // (2,4), outer-strided, last axis contiguous

            // Same size type works (any layout).
            sliced.view<long>().Should().NotBeNull();

            // Different (smaller) size: last axis is rescaled, outer stride preserved.
            var view = sliced.view<int>();
            view.shape.Should().BeEquivalentTo(new long[] { 2, 8 });
            view.Shape.Strides.Should().BeEquivalentTo(new long[] { 16, 1 }); // element strides (bytes/4)
            // int64 rows [0,1,2,3] and [8,9,10,11] reinterpreted little-endian as int32.
            view.flatten().ToArray<int>().Should().BeEquivalentTo(
                new[] { 0, 0, 1, 0, 2, 0, 3, 0, 8, 0, 9, 0, 10, 0, 11, 0 });

            // Writes through to the shared base.
            view.SetInt32(999, 0, 0);
            sliced.GetInt64(0, 0).Should().Be(999);
        }

        [TestMethod]
        public void View_LastAxisNonContiguousDifferentSize_Throws()
        {
            // The last axis is NOT contiguous (reversed / transposed), so a different-size view is
            // rejected — matching NumPy's ValueError "the last axis must be contiguous".
            Action reversed = () => np.arange(5).astype(np.int32)["::-1"].view<short>();
            reversed.Should().Throw<InvalidOperationException>();

            Action transposed = () => np.arange(6).astype(np.int32).reshape(2, 3).T.view<short>();
            transposed.Should().Throw<InvalidOperationException>();
        }

        [TestMethod]
        public void View_BroadcastOuterDifferentSize_StaysReadOnly()
        {
            // NumPy allows a different-size view of a broadcast (stride-0 outer, contiguous last axis)
            // and the result is READ-ONLY, like the source.
            //   np.broadcast_to(np.arange(4, dtype=int32), (3,4)).view(np.int16)
            //     -> shape (3, 8), strides (0, 2) bytes, WRITEABLE=False.
            var b = np.broadcast_to(np.arange(4).astype(np.int32), new Shape(3, 4));
            var view = b.view<short>();
            view.shape.Should().BeEquivalentTo(new long[] { 3, 8 });
            view.Shape.Strides.Should().BeEquivalentTo(new long[] { 0, 1 });
            view.Shape.IsWriteable.Should().BeFalse();
            view.flatten().ToArray<short>().Should().BeEquivalentTo(
                new short[] { 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 1, 0, 2, 0, 3, 0 });
        }

        [TestMethod]
        public void View_0dDifferentSize_Throws()
        {
            // NumPy: "Changing the dtype of a 0d array is only supported if the itemsize is unchanged".
            var scalar = np.array(5).astype(np.int32).reshape(new int[0]);
            Action act = () => scalar.view<short>();
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void View_IncompatibleByteSize_Throws()
        {
            // 3 * 8 = 24 bytes, not divisible by sizeof(decimal) = 16
            var arr = np.array(new double[] { 1.0, 2.0, 3.0 });

            Action act = () => arr.view<decimal>();
            act.Should().Throw<ArgumentException>();
        }

        #endregion
    }
}
