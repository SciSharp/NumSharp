using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Comprehensive tests for np.reshape and NDArray.reshape against NumPy 2.4.2 ground truth.
    /// All tests based on verified NumPy behavior.
    /// </summary>
    [TestClass]
    public class np_reshape_Test : TestClass
    {
        #region Basic Reshapes

        [TestMethod]
        public void Reshape_1D_to_2D()
        {
            // NumPy: np.arange(6).reshape(2,3) = [[0,1,2],[3,4,5]]
            var a = np.arange(6);
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_1D_to_3D()
        {
            // NumPy: np.arange(6).reshape(2,1,3) = [[[0,1,2]],[[3,4,5]]]
            var a = np.arange(6);
            var r = np.reshape(a, 2, 1, 3);

            r.Should().BeShaped(2, 1, 3);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_2D_to_1D()
        {
            // NumPy: np.arange(6).reshape(2,3).reshape(6) = [0,1,2,3,4,5]
            var a = np.arange(6).reshape(2, 3);
            var r = np.reshape(a, 6);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_2D_to_3D()
        {
            // NumPy: np.arange(6).reshape(2,3).reshape(3,1,2) = [[[0,1]],[[2,3]],[[4,5]]]
            var a = np.arange(6).reshape(2, 3);
            var r = np.reshape(a, 3, 1, 2);

            r.Should().BeShaped(3, 1, 2);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_3D_to_2D()
        {
            // NumPy: np.arange(24).reshape(2,3,4).reshape(6,4)
            // First row: [0,1,2,3], Last row: [20,21,22,23]
            var a = np.arange(24).reshape(2, 3, 4);
            var r = np.reshape(a, 6, 4);

            r.Should().BeShaped(6, 4);
            r.Should().BeOfSize(24);
            r.GetInt32(0, 0).Should().Be(0);
            r.GetInt32(0, 1).Should().Be(1);
            r.GetInt32(0, 2).Should().Be(2);
            r.GetInt32(0, 3).Should().Be(3);
            r.GetInt32(5, 0).Should().Be(20);
            r.GetInt32(5, 1).Should().Be(21);
            r.GetInt32(5, 2).Should().Be(22);
            r.GetInt32(5, 3).Should().Be(23);
        }

        [TestMethod]
        public void Reshape_3D_to_1D()
        {
            // NumPy: np.arange(24).reshape(2,3,4).reshape(24) = [0..23]
            var a = np.arange(24).reshape(2, 3, 4);
            var r = np.reshape(a, 24);

            r.Should().BeShaped(24);
            r.Should().BeOfValues(
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23);
        }

        [TestMethod]
        public void Reshape_4D_to_2D()
        {
            // NumPy: np.arange(24).reshape(2,3,2,2).reshape(6,4)
            // Same linear order as 3D→2D test
            var a = np.arange(24).reshape(2, 3, 2, 2);
            var r = np.reshape(a, 6, 4);

            r.Should().BeShaped(6, 4);
            r.GetInt32(0, 0).Should().Be(0);
            r.GetInt32(0, 3).Should().Be(3);
            r.GetInt32(5, 0).Should().Be(20);
            r.GetInt32(5, 3).Should().Be(23);
        }

        [TestMethod]
        public void Reshape_SameShape()
        {
            // NumPy: np.arange(6).reshape(2,3).reshape(2,3) = [[0,1,2],[3,4,5]] (no-op)
            var a = np.arange(6).reshape(2, 3);
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_SingleElement()
        {
            // NumPy: np.array([42]).reshape(1,1,1) = value 42, shape (1,1,1)
            var a = np.array(new[] { 42 });
            var r = np.reshape(a, 1, 1, 1);

            r.Should().BeShaped(1, 1, 1);
            r.GetInt32(0, 0, 0).Should().Be(42);
        }

        [TestMethod]
        public void Reshape_1x6_to_6x1()
        {
            // NumPy: np.arange(6).reshape(1,6).reshape(6,1) = [[0],[1],[2],[3],[4],[5]]
            var a = np.arange(6).reshape(1, 6);
            var r = np.reshape(a, 6, 1);

            r.Should().BeShaped(6, 1);
            r.GetInt32(0, 0).Should().Be(0);
            r.GetInt32(1, 0).Should().Be(1);
            r.GetInt32(2, 0).Should().Be(2);
            r.GetInt32(3, 0).Should().Be(3);
            r.GetInt32(4, 0).Should().Be(4);
            r.GetInt32(5, 0).Should().Be(5);
        }

        #endregion

        #region Minus-One Inference

        [TestMethod]
        public void Reshape_Neg1_First()
        {
            // NumPy: np.arange(12).reshape(-1,3) → shape (4,3)
            var a = np.arange(12);
            var r = np.reshape(a, -1, 3);

            r.Should().BeShaped(4, 3);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Reshape_Neg1_Last()
        {
            // NumPy: np.arange(12).reshape(3,-1) → shape (3,4)
            var a = np.arange(12);
            var r = np.reshape(a, 3, -1);

            r.Should().BeShaped(3, 4);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Reshape_Neg1_Middle()
        {
            // NumPy: np.arange(12).reshape(2,-1,3) → shape (2,2,3)
            var a = np.arange(12);
            var r = np.reshape(a, 2, -1, 3);

            r.Should().BeShaped(2, 2, 3);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Reshape_Neg1_Flatten()
        {
            // NumPy: np.arange(12).reshape(-1) → shape (12,)
            var a = np.arange(12).reshape(3, 4);
            var r = np.reshape(a, -1);

            r.Should().BeShaped(12);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Reshape_Neg1_With1()
        {
            // NumPy: np.arange(5).reshape(-1,1) → shape (5,1)
            var a = np.arange(5);
            var r = np.reshape(a, -1, 1);

            r.Should().BeShaped(5, 1);
            r.GetInt32(0, 0).Should().Be(0);
            r.GetInt32(1, 0).Should().Be(1);
            r.GetInt32(2, 0).Should().Be(2);
            r.GetInt32(3, 0).Should().Be(3);
            r.GetInt32(4, 0).Should().Be(4);
        }

        #endregion

        #region View Semantics

        [TestMethod]
        public void Reshape_Contiguous_ReturnsView()
        {
            // NumPy: reshape of contiguous array returns view, modifications visible
            var a = np.arange(6);
            var r = a.reshape(2, 3);

            // Modify reshaped array
            r.SetInt32(99, 0, 1);

            // Original array should see the change at index 1
            a.GetInt32(1).Should().Be(99);
        }

        [TestMethod]
        public void Reshape_DoubleReshape_SharesMemory()
        {
            // NumPy: arange(24)→(4,6)→(2,3,4), modify r2[0,1,2]=888 → original a[6]=888
            var a = np.arange(24);
            var r1 = a.reshape(4, 6);
            var r2 = r1.reshape(2, 3, 4);

            r2.SetInt32(888, 0, 1, 2);

            a.GetInt32(6).Should().Be(888);
        }

        [TestMethod]
        public void Reshape_Back_SharesMemory()
        {
            // NumPy: arange(12)→(3,4)→(12) shares memory with original
            var a = np.arange(12);
            var r1 = a.reshape(3, 4);
            var r2 = r1.reshape(12);

            r2.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

            r2.SetInt32(777, 5);
            a.GetInt32(5).Should().Be(777);
        }

        #endregion

        #region Scalar / 0-dim

        [TestMethod]
        public void Reshape_ScalarTo1D()
        {
            // NumPy: np.array(42).reshape(1) → shape (1,), val [42]
            // Note: NumSharp doesn't have true 0-dim scalars; np.array(42) creates shape (1,)
            // This test passes because NumSharp scalar is already (1,)
            var a = np.array(42);
            var r = np.reshape(a, 1);

            r.Should().BeShaped(1);
            r.GetInt32(0).Should().Be(42);
        }

        [TestMethod]
        public void Reshape_ScalarTo2D()
        {
            // NumPy: np.array(42).reshape(1,1) → shape (1,1), val [[42]]
            // Note: NumSharp doesn't have true 0-dim scalars; np.array(42) creates shape (1,)
            var a = np.array(42);
            var r = np.reshape(a, 1, 1);

            r.Should().BeShaped(1, 1);
            r.GetInt32(0, 0).Should().Be(42);
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Reshape_1DToScalar()
        {
            // NumPy: np.array([42]).reshape(()) → scalar 42
            // Bug: Known NRE bug when reshaping to empty shape
            var a = np.array(new[] { 42 });
            var r = np.reshape(a, new Shape());

            r.Should().BeScalar(42);
        }

        #endregion

        #region Empty Arrays

        [TestMethod]
        public void Reshape_Empty_To0x3()
        {
            // NumPy: np.array([]).reshape(0,3) → shape (0,3), size 0
            var a = np.array(new int[0]);
            var r = np.reshape(a, 0, 3);

            r.Should().BeShaped(0, 3);
            r.Should().BeOfSize(0);
        }

        [TestMethod]
        public void Reshape_Empty_To3x0()
        {
            // NumPy: np.array([]).reshape(3,0) → shape (3,0), size 0
            var a = np.array(new int[0]);
            var r = np.reshape(a, 3, 0);

            r.Should().BeShaped(3, 0);
            r.Should().BeOfSize(0);
        }

        [TestMethod]
        public void Reshape_Empty_To0x0()
        {
            // NumPy: np.array([]).reshape(0,0) → shape (0,0), size 0
            var a = np.array(new int[0]);
            var r = np.reshape(a, 0, 0);

            r.Should().BeShaped(0, 0);
            r.Should().BeOfSize(0);
        }

        [TestMethod]
        public void Reshape_0x3_To0()
        {
            // NumPy: np.empty((0,3)).reshape(0) → shape (0,)
            var a = np.empty(new Shape(0, 3), np.int32);
            var r = np.reshape(a, 0);

            r.Should().BeShaped(0);
            r.Should().BeOfSize(0);
        }

        [TestMethod]
        public void Reshape_Empty_Neg1()
        {
            // NumPy: np.array([]).reshape(-1,3) → shape (0,3)
            var a = np.array(new int[0]);
            var r = np.reshape(a, -1, 3);

            r.Should().BeShaped(0, 3);
            r.Should().BeOfSize(0);
        }

        #endregion

        #region Sliced + Reshape

        [TestMethod]
        public void Reshape_ContiguousSlice_Values()
        {
            // NumPy: np.arange(10)[2:8].reshape(2,3) = [[2,3,4],[5,6,7]]
            var a = np.arange(10);
            var s = a["2:8"];
            var r = np.reshape(s, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfValues(2, 3, 4, 5, 6, 7);
        }

        [TestMethod]
        public void Reshape_StepSlice_Values()
        {
            // NumPy: np.arange(10)[::2].reshape(1,5) = [[0,2,4,6,8]]
            var a = np.arange(10);
            var s = a["::2"];
            var r = np.reshape(s, 1, 5);

            r.Should().BeShaped(1, 5);
            r.Should().BeOfValues(0, 2, 4, 6, 8);
        }

        [TestMethod]
        public void Reshape_2D_ColSlice_Values()
        {
            // NumPy: np.arange(12).reshape(3,4)[:,1:3].reshape(6) = [1,2,5,6,9,10]
            var a = np.arange(12).reshape(3, 4);
            var s = a[":,1:3"];
            var r = np.reshape(s, 6);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(1, 2, 5, 6, 9, 10);
        }

        [TestMethod]
        public void Reshape_2D_RowSlice_Values()
        {
            // NumPy: np.arange(12).reshape(3,4)[1:3].reshape(8) = [4,5,6,7,8,9,10,11]
            var a = np.arange(12).reshape(3, 4);
            var s = a["1:3"];
            var r = np.reshape(s, 8);

            r.Should().BeShaped(8);
            r.Should().BeOfValues(4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Reshape_Reversed_Values()
        {
            // NumPy: np.arange(6)[::-1].reshape(2,3) = [[5,4,3],[2,1,0]]
            var a = np.arange(6);
            var s = a["::-1"];
            var r = np.reshape(s, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfValues(5, 4, 3, 2, 1, 0);
        }

        [TestMethod]
        public void Reshape_Slice_WriteThrough()
        {
            // NumPy: slice a[2:8], reshape(2,3), set [0,0]=99 → original a[2] becomes 99
            var a = np.arange(10);
            var s = a["2:8"];
            var r = s.reshape(2, 3);

            r.SetInt32(99, 0, 0);

            a.GetInt32(2).Should().Be(99);
        }

        #endregion

        #region Broadcast + Reshape

        [TestMethod]
        public void Reshape_RowBroadcast_CopyReshape()
        {
            // NumPy: broadcast_to([1,2,3], (3,3)), copy then reshape(9) = [1,2,3,1,2,3,1,2,3]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.broadcast_to(a, new Shape(3, 3));
            var c = b.copy();
            var r = np.reshape(c, 9);

            r.Should().BeShaped(9);
            r.Should().BeOfValues(1, 2, 3, 1, 2, 3, 1, 2, 3);
        }

        [TestMethod]
        public void Reshape_ColBroadcast_CopyReshape()
        {
            // NumPy: broadcast_to([[10],[20],[30]], (3,3)), copy then reshape(9)
            // = [10,10,10,20,20,20,30,30,30]
            var a = np.array(new int[,] { { 10 }, { 20 }, { 30 } });
            var b = np.broadcast_to(a, new Shape(3, 3));
            var c = b.copy();
            var r = np.reshape(c, 9);

            r.Should().BeShaped(9);
            r.Should().BeOfValues(10, 10, 10, 20, 20, 20, 30, 30, 30);
        }

        [TestMethod]
        public void Reshape_Broadcast_DirectReshape()
        {
            // NumPy: np.reshape(broadcast_to([1,2,3], (3,3)), 9) = [1,2,3,1,2,3,1,2,3]
            // This should work since reshape handles broadcast through _reshapeBroadcast
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.broadcast_to(a, new Shape(3, 3));
            var r = np.reshape(b, 9);

            r.Should().BeShaped(9);
            r.Should().BeOfValues(1, 2, 3, 1, 2, 3, 1, 2, 3);
        }

        [TestMethod]
        public void Reshape_ColBroadcast_DirectReshape()
        {
            // NumPy: broadcast_to column [[10],[20],[30]], reshape(9)
            // = [10,10,10,20,20,20,30,30,30]
            var a = np.array(new int[,] { { 10 }, { 20 }, { 30 } });
            var b = np.broadcast_to(a, new Shape(3, 3));
            var r = np.reshape(b, 9);

            r.Should().BeShaped(9);
            r.Should().BeOfValues(10, 10, 10, 20, 20, 20, 30, 30, 30);
        }

        #endregion

        #region Dtypes

        [TestMethod]
        public void Reshape_Boolean()
        {
            var a = np.array(new[] { true, false, true, false, true, false });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<bool>();
            r.Should().BeOfValues(true, false, true, false, true, false);
        }

        [TestMethod]
        public void Reshape_Byte()
        {
            var a = np.array(new byte[] { 0, 1, 2, 3, 4, 5 });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<byte>();
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_Int16()
        {
            var a = np.array(new short[] { 0, 1, 2, 3, 4, 5 });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<short>();
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_UInt16()
        {
            var a = np.array(new ushort[] { 0, 1, 2, 3, 4, 5 });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<ushort>();
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_Int32()
        {
            var a = np.array(new[] { 0, 1, 2, 3, 4, 5 });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<int>();
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_UInt32()
        {
            var a = np.array(new uint[] { 0, 1, 2, 3, 4, 5 });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<uint>();
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_Int64()
        {
            var a = np.array(new long[] { 0, 1, 2, 3, 4, 5 });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<long>();
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_UInt64()
        {
            var a = np.array(new ulong[] { 0, 1, 2, 3, 4, 5 });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<ulong>();
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_Char()
        {
            var a = np.array(new[] { 'a', 'b', 'c', 'd', 'e', 'f' });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<char>();
            r.Should().BeOfValues('a', 'b', 'c', 'd', 'e', 'f');
        }

        [TestMethod]
        public void Reshape_Single()
        {
            var a = np.array(new[] { 0f, 1f, 2f, 3f, 4f, 5f });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<float>();
            r.Should().BeOfValues(0f, 1f, 2f, 3f, 4f, 5f);
        }

        [TestMethod]
        public void Reshape_Double()
        {
            var a = np.array(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<double>();
            r.Should().BeOfValues(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        }

        [TestMethod]
        public void Reshape_Decimal()
        {
            var a = np.array(new[] { 0m, 1m, 2m, 3m, 4m, 5m });
            var r = np.reshape(a, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfType<decimal>();
            r.Should().BeOfValues(0m, 1m, 2m, 3m, 4m, 5m);
        }

        #endregion

        #region Large Arrays

        [TestMethod]
        public void Reshape_Large_100x100_to_10000()
        {
            // Test large array reshape
            var a = np.arange(10000).reshape(100, 100);
            var r = np.reshape(a, 10000);

            r.Should().BeShaped(10000);
            r.Should().BeOfSize(10000);

            // Check first and last 5 elements
            r.GetInt32(0).Should().Be(0);
            r.GetInt32(1).Should().Be(1);
            r.GetInt32(2).Should().Be(2);
            r.GetInt32(3).Should().Be(3);
            r.GetInt32(4).Should().Be(4);
            r.GetInt32(9995).Should().Be(9995);
            r.GetInt32(9996).Should().Be(9996);
            r.GetInt32(9997).Should().Be(9997);
            r.GetInt32(9998).Should().Be(9998);
            r.GetInt32(9999).Should().Be(9999);
        }

        [TestMethod]
        public void Reshape_Large_100x100_to_50x200()
        {
            // Test large array different shape
            var a = np.arange(10000).reshape(100, 100);
            var r = np.reshape(a, 50, 200);

            r.Should().BeShaped(50, 200);
            r.Should().BeOfSize(10000);

            // Check corner elements
            r.GetInt32(0, 0).Should().Be(0);
            r.GetInt32(0, 199).Should().Be(199);
            r.GetInt32(49, 0).Should().Be(9800);
            r.GetInt32(49, 199).Should().Be(9999);
        }

        #endregion

        #region Error Cases

        [TestMethod]
        public void Reshape_IncompatibleShape_Throws()
        {
            // NumPy: np.arange(6).reshape(2,4) → raises ValueError
            var a = np.arange(6);

            Action act = () => np.reshape(a, 2, 4);

            act.Should().Throw<Exception>(); // Could be IncorrectShapeException or similar
        }

        [TestMethod]
        public void Reshape_TwoNeg1_Throws()
        {
            // NumPy: np.arange(6).reshape(-1,-1) → raises ValueError
            var a = np.arange(6);

            Action act = () => np.reshape(a, -1, -1);

            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Reshape_Neg1_NonDivisible_Throws()
        {
            // NumPy: np.arange(7).reshape(-1,3) → raises ValueError (7 not divisible by 3)
            var a = np.arange(7);

            Action act = () => np.reshape(a, -1, 3);

            act.Should().Throw<Exception>();
        }

        #endregion

        #region Static vs Instance

        [TestMethod]
        public void Reshape_Static_Equals_Instance()
        {
            // NumPy: np.reshape(a, (3,4)) == a.reshape(3,4)
            var a = np.arange(12);
            var r1 = np.reshape(a, 3, 4);
            var r2 = a.reshape(3, 4);

            r1.Should().BeShaped(3, 4);
            r2.Should().BeShaped(3, 4);
            np.array_equal(r1, r2).Should().BeTrue();
        }

        #endregion

        #region Transposed + Reshape

        [TestMethod]
        public void Reshape_Transposed_Values()
        {
            // NumPy: np.arange(6).reshape(2,3).T.reshape(6) = [0,3,1,4,2,5]
            // (NumPy copies because not C-contiguous; NumSharp's transpose creates copy anyway)
            var a = np.arange(6).reshape(2, 3);
            var t = a.T;
            var r = np.reshape(t, 6);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(0, 3, 1, 4, 2, 5);
        }

        [TestMethod]
        public void Reshape_Transposed_NoWriteThrough()
        {
            // After transposed reshape, writing to result does NOT modify original
            // (since transpose creates a copy in NumSharp)
            var a = np.arange(6).reshape(2, 3);
            var t = a.T;
            var r = t.reshape(6);

            r.SetInt32(999, 0);

            // Original should be unchanged (transpose made a copy)
            a.GetInt32(0, 0).Should().Be(0);
        }

        #endregion

        #region Fancy Combinations

        [TestMethod]
        public void Reshape_Slice_Reshape_Values()
        {
            // NumPy: np.arange(24).reshape(4,6)[1:3,2:5].reshape(6) = [8,9,10,14,15,16]
            var a = np.arange(24).reshape(4, 6);
            var s = a["1:3,2:5"];
            var r = np.reshape(s, 6);

            r.Should().BeShaped(6);
            r.Should().BeOfValues(8, 9, 10, 14, 15, 16);
        }

        [TestMethod]
        public void Reshape_NewAxis_Values()
        {
            // NumPy: np.arange(6)[np.newaxis,:].reshape(2,3) = [[0,1,2],[3,4,5]]
            // NumSharp equivalent: np.expand_dims(a, 0)
            var a = np.arange(6);
            var expanded = np.expand_dims(a, 0);
            var r = np.reshape(expanded, 2, 3);

            r.Should().BeShaped(2, 3);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Reshape_Chain_4Steps()
        {
            // NumPy: arange(24)→(4,6)→(2,2,6)→(2,2,2,3)→(24), verify equal to original
            var a = np.arange(24);
            var r1 = np.reshape(a, 4, 6);
            var r2 = np.reshape(r1, 2, 2, 6);
            var r3 = np.reshape(r2, 2, 2, 2, 3);
            var r4 = np.reshape(r3, 24);

            r4.Should().BeShaped(24);
            np.array_equal(r4, a).Should().BeTrue();
        }

        [TestMethod]
        public void Reshape_Unsafe_ParamsInt()
        {
            // Test reshape_unsafe with int[] params works correctly
            var a = np.arange(12);
            var r = a.reshape_unsafe(3, 4);

            r.Should().BeShaped(3, 4);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Reshape_Unsafe_Shape()
        {
            // Test reshape_unsafe with Shape overload works correctly
            var a = np.arange(12);
            var newShape = new Shape(3, 4);
            var r = a.reshape_unsafe(newShape);

            r.Should().BeShaped(3, 4);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        #endregion
    }
}
