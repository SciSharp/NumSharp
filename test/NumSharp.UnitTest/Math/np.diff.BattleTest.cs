using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Maths
{
    /// <summary>
    /// NumPy-parity battle tests for np.diff and np.ediff1d.
    /// Every expected value was produced by running NumPy 2.4.2.
    /// Note: NumSharp's default integer dtype is Int32 (NumPy uses int64); diff
    /// PRESERVES dtype, so int32 input → int32 output (values match NumPy exactly).
    /// </summary>
    [TestClass]
    public class DiffBattleTests : TestClass
    {
        // ---------------------------------------------------------------- diff: basic

        [TestMethod]
        public void Diff_1D_Default()
        {
            // np.diff([1,2,4,7,0]) -> [1, 2, 3, -7]
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.diff(x).Should().Be(new int[] { 1, 2, 3, -7 });
        }

        [TestMethod]
        public void Diff_1D_HigherOrder()
        {
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.diff(x, 2).Should().Be(new int[] { 1, 1, -10 });   // np.diff(x, n=2)
            np.diff(x, 3).Should().Be(new int[] { 0, -11 });      // np.diff(x, n=3)
            np.diff(x, 4).Should().Be(new int[] { -11 });
        }

        [TestMethod]
        public void Diff_N_GreaterEqual_Length_ReturnsEmpty()
        {
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.diff(x, 5).Should().BeShaped(0);   // n == len -> empty
            np.diff(x, 6).Should().BeShaped(0);   // n  > len -> empty
        }

        [TestMethod]
        public void Diff_N0_ReturnsInputUnchanged()
        {
            // np.diff(x, 0) returns the input unchanged (same object in NumPy).
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            var r = np.diff(x, 0);
            ReferenceEquals(r, x).Should().BeTrue();
            r.Should().Be(new int[] { 1, 2, 4, 7, 0 });
        }

        [TestMethod]
        public void Diff_NegativeN_Throws()
        {
            // np.diff(x, -1) -> ValueError "order must be non-negative but got -1"
            NDArray x = new int[] { 1, 2, 3 };
            Assert.ThrowsException<ArgumentException>(() => np.diff(x, -1));
        }

        [TestMethod]
        public void Diff_0D_Throws()
        {
            // np.diff(np.array(5)) -> ValueError "diff requires input that is at least one dimensional"
            NDArray scalar = np.asarray(5);
            Assert.ThrowsException<ArgumentException>(() => np.diff(scalar));
        }

        // ---------------------------------------------------------------- diff: axes

        [TestMethod]
        public void Diff_2D_DefaultAxis()
        {
            // np.diff([[1,3,6,10],[0,5,6,8]]) -> [[2,3,4],[5,1,2]]
            NDArray m = new int[,] { { 1, 3, 6, 10 }, { 0, 5, 6, 8 } };
            np.diff(m).Should().Be(new int[,] { { 2, 3, 4 }, { 5, 1, 2 } });
        }

        [TestMethod]
        public void Diff_2D_Axis0()
        {
            NDArray m = new int[,] { { 1, 3, 6, 10 }, { 0, 5, 6, 8 } };
            np.diff(m, axis: 0).Should().Be(new int[,] { { -1, 2, 0, -2 } });
        }

        [TestMethod]
        public void Diff_2D_Axis1_HigherOrder()
        {
            NDArray m = new int[,] { { 1, 3, 6, 10 }, { 0, 5, 6, 8 } };
            np.diff(m, 2, 1).Should().Be(new int[,] { { 1, 1 }, { -4, 1 } });
        }

        [TestMethod]
        public void Diff_NegativeAxis()
        {
            // axis=-2 on a 2-D array == axis 0
            NDArray m = new int[,] { { 1, 3, 6, 10 }, { 0, 5, 6, 8 } };
            np.diff(m, axis: -2).Should().Be(new int[,] { { -1, 2, 0, -2 } });
        }

        [TestMethod]
        public void Diff_AxisOutOfBounds_Throws()
        {
            // np.diff(m, axis=2) on 2-D -> AxisError
            NDArray m = new int[,] { { 1, 2 }, { 3, 4 } };
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.diff(m, axis: 2));
        }

        // ---------------------------------------------------------------- diff: dtypes

        [TestMethod]
        public void Diff_Bool_UsesNotEqual()
        {
            // bool diff uses not_equal: np.diff([T,F,F,T]) -> [T,F,T]
            NDArray b = new bool[] { true, false, false, true };
            np.diff(b).Should().Be(new bool[] { true, false, true });
        }

        [TestMethod]
        public void Diff_UInt8_Wraps()
        {
            // np.diff(uint8[1,0]) -> [255] (0 - 1 wraps in uint8)
            NDArray u8 = new byte[] { 1, 0 };
            np.diff(u8).Should().Be(new byte[] { 255 });
        }

        [TestMethod]
        public void Diff_Int8_Wraps()
        {
            NDArray i8 = new sbyte[] { -128, 127 };
            np.diff(i8).Should().Be(new sbyte[] { -1 }); // 127 - (-128) = 255 -> -1
        }

        [TestMethod]
        public void Diff_Float32_PreservesDtype()
        {
            NDArray f = new float[] { 1.5f, 2.5f, 10f };
            np.diff(f).Should().Be(new float[] { 1.0f, 7.5f });
        }

        [TestMethod]
        public void Diff_Complex()
        {
            // np.diff([1+2j, 3+1j, 0+0j]) -> [(2-1j), (-3-1j)]
            NDArray c = new Complex[] { new(1, 2), new(3, 1), new(0, 0) };
            var r = np.diff(c);
            r.Should().BeShaped(2);
            ((Complex)r[0]).Should().Be(new Complex(2, -1));
            ((Complex)r[1]).Should().Be(new Complex(-3, -1));
        }

        // ---------------------------------------------------------------- diff: prepend / append

        [TestMethod]
        public void Diff_PrependScalar()
        {
            // np.diff([1,2,4,7,0], prepend=0) -> [1,1,2,3,-7]
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.diff(x, prepend: 0).Should().Be(new int[] { 1, 1, 2, 3, -7 });
        }

        [TestMethod]
        public void Diff_AppendScalar()
        {
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.diff(x, append: 0).Should().Be(new int[] { 1, 2, 3, -7, 0 });
        }

        [TestMethod]
        public void Diff_PrependAndAppendScalar()
        {
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.diff(x, prepend: 0, append: 10).Should().Be(new int[] { 1, 1, 2, 3, -7, 10 });
        }

        [TestMethod]
        public void Diff_PrependArray()
        {
            // np.diff([1,2,4,7,0], prepend=[0,0]) -> [0,1,1,2,3,-7]
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.diff(x, prepend: (NDArray)new int[] { 0, 0 }).Should().Be(new int[] { 0, 1, 1, 2, 3, -7 });
        }

        [TestMethod]
        public void Diff_PrependFloat_PromotesDtype()
        {
            // np.diff([1,2,3], prepend=0.5) -> float64 [0.5, 1.0, 1.0]
            NDArray x = new int[] { 1, 2, 3 };
            np.diff(x, prepend: 0.5).Should().Be(new double[] { 0.5, 1.0, 1.0 });
        }

        [TestMethod]
        public void Diff_2D_PrependScalar_Axis1()
        {
            NDArray m = new int[,] { { 1, 3, 6, 10 }, { 0, 5, 6, 8 } };
            np.diff(m, axis: 1, prepend: 0).Should().Be(new int[,] { { 1, 2, 3, 4 }, { 0, 5, 1, 2 } });
        }

        [TestMethod]
        public void Diff_2D_PrependScalar_Axis0()
        {
            NDArray m = new int[,] { { 1, 3, 6, 10 }, { 0, 5, 6, 8 } };
            np.diff(m, axis: 0, prepend: 0).Should().Be(new int[,] { { 1, 3, 6, 10 }, { -1, 2, 0, -2 } });
        }

        [TestMethod]
        public void Diff_2D_AppendArray_Axis1()
        {
            NDArray m = new int[,] { { 1, 3, 6, 10 }, { 0, 5, 6, 8 } };
            NDArray app = new int[,] { { 99 }, { 99 } };
            np.diff(m, axis: 1, append: app).Should().Be(new int[,] { { 2, 3, 4, 89 }, { 5, 1, 2, 91 } });
        }

        [TestMethod]
        public void Diff_Bool_WithIntPrepend_UsesSubtract()
        {
            // post-concat dtype is integer -> op becomes subtract, not not_equal
            // np.diff([T,F,T], prepend=0) -> [1, -1, 1]
            NDArray b = new bool[] { true, false, true };
            np.diff(b, prepend: 0).Should().Be(new int[] { 1, -1, 1 });
        }

        [TestMethod]
        public void Diff_Bool_WithBoolPrepend_StaysNotEqual()
        {
            // post-concat dtype is bool -> not_equal: np.diff([T,F,T], prepend=True) -> [F,T,T]
            NDArray b = new bool[] { true, false, true };
            np.diff(b, prepend: true).Should().Be(new bool[] { false, true, true });
        }

        [TestMethod]
        public void Diff_HigherOrder_WithPrepend()
        {
            // prepend applied once, then 2 diffs: np.diff(x, n=2, prepend=0) -> [0,1,1,-10]
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.diff(x, 2, prepend: 0).Should().Be(new int[] { 0, 1, 1, -10 });
        }

        // ---------------------------------------------------------------- diff: empty

        [TestMethod]
        public void Diff_Empty1D()
        {
            NDArray e = new double[] { };
            np.diff(e).Should().BeShaped(0);
        }

        [TestMethod]
        public void Diff_SingleElement_ReturnsEmpty()
        {
            NDArray s = new int[] { 5 };
            np.diff(s).Should().BeShaped(0);
        }

        // ---------------------------------------------------------------- diff: layouts (NDIter fast path)

        [TestMethod]
        public void Diff_3D_AllAxes()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            np.diff(a, axis: 0).Should().BeShaped(1, 3, 4);
            np.diff(a, axis: 1).Should().BeShaped(2, 2, 4);
            np.diff(a, axis: 2).Should().BeShaped(2, 3, 3);
            // axis=2 of arange(24) reshaped: every consecutive diff along last axis is 1
            np.diff(a, axis: 2).flatten().Should().Be(np.ones(new Shape(2, 3, 3), NPTypeCode.Int32).flatten());
        }

        [TestMethod]
        public void Diff_Transposed()
        {
            // a = arange(12).reshape(3,4).T  -> shape (4,3); diff along last axis -> all 4s
            var t = np.arange(12).reshape(3, 4).T;
            var r = np.diff(t);
            r.Should().BeShaped(4, 2);
            r.flatten().Should().Be((NDArray)new int[] { 4, 4, 4, 4, 4, 4, 4, 4 });
        }

        [TestMethod]
        public void Diff_FortranOrder()
        {
            var f = np.arange(12).reshape(3, 4).copy('F');
            var r = np.diff(f, axis: 1);
            r.Should().BeShaped(3, 3);
            r.flatten().Should().Be((NDArray)new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 });
        }

        [TestMethod]
        public void Diff_NegativeStride()
        {
            // arange(10)[::-1] -> [9,8,...,0]; diff -> all -1
            var rev = np.arange(10)["::-1"];
            np.diff(rev).Should().Be(new int[] { -1, -1, -1, -1, -1, -1, -1, -1, -1 });
        }

        [TestMethod]
        public void Diff_BroadcastView_ReadOnly()
        {
            // broadcast_to(arange(4), (3,4)); diff axis=1 -> rows of [1,1,1]; diff axis=0 -> [0,...]
            var b = np.broadcast_to(np.arange(4), new Shape(3, 4));
            np.diff(b, axis: 1).flatten().Should().Be((NDArray)new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 });
            np.diff(b, axis: 0).flatten().Should().Be((NDArray)new int[] { 0, 0, 0, 0, 0, 0, 0, 0 });
        }

        // ---------------------------------------------------------------- ediff1d

        [TestMethod]
        public void Ediff1d_Basic()
        {
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.ediff1d(x).Should().Be(new int[] { 1, 2, 3, -7 });
        }

        [TestMethod]
        public void Ediff1d_ToBeginScalar()
        {
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.ediff1d(x, to_begin: -99).Should().Be(new int[] { -99, 1, 2, 3, -7 });
        }

        [TestMethod]
        public void Ediff1d_ToEndArray()
        {
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.ediff1d(x, to_end: (NDArray)new int[] { 88, 99 }).Should().Be(new int[] { 1, 2, 3, -7, 88, 99 });
        }

        [TestMethod]
        public void Ediff1d_Both()
        {
            NDArray x = new int[] { 1, 2, 4, 7, 0 };
            np.ediff1d(x, to_begin: -99, to_end: (NDArray)new int[] { 88, 99 })
              .Should().Be(new int[] { -99, 1, 2, 3, -7, 88, 99 });
        }

        [TestMethod]
        public void Ediff1d_2D_Ravels()
        {
            // np.ediff1d([[1,2,4],[1,6,24]]) -> [1,2,-3,5,18]
            NDArray y = new int[,] { { 1, 2, 4 }, { 1, 6, 24 } };
            np.ediff1d(y).Should().Be(new int[] { 1, 2, -3, 5, 18 });
        }

        [TestMethod]
        public void Ediff1d_Empty()
        {
            NDArray e = new double[] { };
            np.ediff1d(e).Should().BeShaped(0);
        }

        [TestMethod]
        public void Ediff1d_SingleElement_ReturnsEmpty()
        {
            NDArray s = new int[] { 5 };
            np.ediff1d(s).Should().BeShaped(0);
        }

        [TestMethod]
        public void Ediff1d_Empty_WithToBegin()
        {
            // empty input + to_begin=7 -> [7]
            NDArray e = new long[] { };
            np.ediff1d(e, to_begin: 7L).Should().Be(new long[] { 7 });
        }

        [TestMethod]
        public void Ediff1d_Single_WithBeginAndEnd()
        {
            // single element [5] -> empty middle; [1,2] + [] + [3] -> [1,2,3]
            NDArray s = new int[] { 5 };
            np.ediff1d(s, to_begin: (NDArray)new int[] { 1, 2 }, to_end: 3)
              .Should().Be(new int[] { 1, 2, 3 });
        }

        [TestMethod]
        public void Ediff1d_UInt8_Wraps()
        {
            NDArray u8 = new byte[] { 1, 0 };
            np.ediff1d(u8).Should().Be(new byte[] { 255 });
        }

        [TestMethod]
        public void Ediff1d_Bool_Throws()
        {
            // ediff1d uses subtract (no not_equal special case); NumPy rejects bool subtract.
            NDArray b = new bool[] { true, false, true };
            Assert.ThrowsException<NotSupportedException>(() => np.ediff1d(b));
        }

        [TestMethod]
        public void Ediff1d_IncompatibleToBegin_Throws()
        {
            // float to_begin into int array: can_cast(float, int, same_kind) == False -> TypeError
            NDArray x = new int[] { 1, 2, 3 };
            Assert.ThrowsException<ArgumentException>(() => np.ediff1d(x, to_begin: 0.5));
        }

        [TestMethod]
        public void Ediff1d_StridedInput_Ravels()
        {
            // arange-like strided view ravels in C order before differencing
            var strided = ((NDArray)new int[] { 1, 2, 4, 7, 0 })["::2"]; // [1,4,0]
            np.ediff1d(strided).Should().Be(new int[] { 3, -4 });
        }
    }
}
