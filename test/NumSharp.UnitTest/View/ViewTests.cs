using System;
using AwesomeAssertions;

namespace NumSharp.UnitTest.View
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
        public void View_NonContiguous_ThrowsForDifferentSize()
        {
            // Non-contiguous arrays can't be viewed as different-sized types
            var arr = np.arange(12).reshape(3, 4);
            var sliced = arr["::2"]; // Non-contiguous slice

            // Same size type should work
            var sameSizeView = sliced.view<long>();
            sameSizeView.Should().NotBeNull();

            // Different size type should throw
            Action act = () => sliced.view<int>();
            act.Should().Throw<InvalidOperationException>();
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
