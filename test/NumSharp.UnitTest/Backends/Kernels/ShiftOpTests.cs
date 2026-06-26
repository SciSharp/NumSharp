using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests for bit shift operations (left_shift, right_shift).
/// All expected values are verified against NumPy 2.x output.
/// </summary>
[TestClass]
public class ShiftOpTests
{
    #region Left Shift Tests

    [TestMethod]
    public void LeftShift_Int32_Scalar()
    {
        // NumPy: np.left_shift([5, 10, 15], 2) = [20, 40, 60]
        var arr = np.array(new int[] { 5, 10, 15 });
        var result = np.left_shift(arr, 2);

        result.Should().BeOfValues(20, 40, 60).And.BeOfType(NPTypeCode.Int32);
    }

    [TestMethod]
    public void LeftShift_Int32_Array()
    {
        // NumPy: np.left_shift([1, 2, 4], [1, 2, 3]) = [2, 8, 32]
        var arr1 = np.array(new int[] { 1, 2, 4 });
        var arr2 = np.array(new int[] { 1, 2, 3 });
        var result = np.left_shift(arr1, arr2);

        result.Should().BeOfValues(2, 8, 32).And.BeOfType(NPTypeCode.Int32);
    }

    [TestMethod]
    public void LeftShift_Byte()
    {
        // NumPy: np.left_shift(np.array([1, 2, 4], dtype=np.uint8), 2) = [4, 8, 16]
        var arr = np.array(new byte[] { 1, 2, 4 });
        var result = np.left_shift(arr, 2);

        result.Should().BeOfValues(4, 8, 16).And.BeOfType(NPTypeCode.Byte);
    }

    [TestMethod]
    public void LeftShift_Int64()
    {
        // NumPy: np.left_shift(np.array([1, 2, 4], dtype=np.int64), 32) = [4294967296, 8589934592, 17179869184]
        var arr = np.array(new long[] { 1L, 2L, 4L });
        var result = np.left_shift(arr, 32);

        result.Should().BeOfValues(4294967296L, 8589934592L, 17179869184L).And.BeOfType(NPTypeCode.Int64);
    }

    [TestMethod]
    public void LeftShift_Overflow()
    {
        // NumPy: np.left_shift(np.array([128], dtype=np.uint8), 1) = [0] (overflow wraps)
        var arr = np.array(new byte[] { 128 });
        var result = np.left_shift(arr, 1);

        result.Should().BeOfValues(0).And.BeOfType(NPTypeCode.Byte);
    }

    #endregion

    #region Right Shift Tests

    [TestMethod]
    public void RightShift_Int32_Scalar()
    {
        // NumPy: np.right_shift([20, 40, 60], 2) = [5, 10, 15]
        var arr = np.array(new int[] { 20, 40, 60 });
        var result = np.right_shift(arr, 2);

        result.Should().BeOfValues(5, 10, 15).And.BeOfType(NPTypeCode.Int32);
    }

    [TestMethod]
    public void RightShift_Int32_Array()
    {
        // NumPy: np.right_shift([16, 32, 64], [1, 2, 3]) = [8, 8, 8]
        var arr1 = np.array(new int[] { 16, 32, 64 });
        var arr2 = np.array(new int[] { 1, 2, 3 });
        var result = np.right_shift(arr1, arr2);

        result.Should().BeOfValues(8, 8, 8).And.BeOfType(NPTypeCode.Int32);
    }

    [TestMethod]
    public void RightShift_Byte()
    {
        // NumPy: np.right_shift(np.array([16, 32, 64], dtype=np.uint8), 2) = [4, 8, 16]
        var arr = np.array(new byte[] { 16, 32, 64 });
        var result = np.right_shift(arr, 2);

        result.Should().BeOfValues(4, 8, 16).And.BeOfType(NPTypeCode.Byte);
    }

    [TestMethod]
    public void RightShift_SignedArithmetic()
    {
        // NumPy: np.right_shift(np.array([-8], dtype=np.int32), 1) = [-4]
        // Arithmetic right shift preserves sign
        var arr = np.array(new int[] { -8 });
        var result = np.right_shift(arr, 1);

        result.Should().BeOfValues(-4).And.BeOfType(NPTypeCode.Int32);
    }

    [TestMethod]
    public void RightShift_UnsignedLogical()
    {
        // NumPy: np.right_shift(np.array([255], dtype=np.uint8), 1) = [127]
        // Logical right shift fills with zeros
        var arr = np.array(new byte[] { 255 });
        var result = np.right_shift(arr, 1);

        result.Should().BeOfValues(127).And.BeOfType(NPTypeCode.Byte);
    }

    [TestMethod]
    public void RightShift_Int64()
    {
        // NumPy: np.right_shift(np.array([4294967296], dtype=np.int64), 32) = [1]
        var arr = np.array(new long[] { 4294967296L });
        var result = np.right_shift(arr, 32);

        result.Should().BeOfValues(1L).And.BeOfType(NPTypeCode.Int64);
    }

    #endregion

    #region Broadcasting Tests

    [TestMethod]
    public void LeftShift_Broadcasting()
    {
        // NumPy: np.left_shift([[1], [2], [4]], [1, 2, 3]) = [[2, 4, 8], [4, 8, 16], [8, 16, 32]]
        var arr1 = np.array(new int[,] { { 1 }, { 2 }, { 4 } });
        var arr2 = np.array(new int[] { 1, 2, 3 });
        var result = np.left_shift(arr1, arr2);

        result.Should().BeShaped(3, 3);
        result.Should().BeOfValues(2, 4, 8, 4, 8, 16, 8, 16, 32);
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public void LeftShift_Float_ThrowsNotSupported()
    {
        var arr = np.array(new float[] { 1.0f, 2.0f });
        Assert.ThrowsException<TypeError>(() => np.left_shift(arr, 1));
    }

    [TestMethod]
    public void RightShift_Double_ThrowsNotSupported()
    {
        var arr = np.array(new double[] { 1.0, 2.0 });
        Assert.ThrowsException<TypeError>(() => np.right_shift(arr, 1));
    }

    #endregion

    #region UInt Types Tests

    [TestMethod]
    public void LeftShift_UInt16()
    {
        var arr = np.array(new ushort[] { 1, 2, 4 });
        var result = np.left_shift(arr, 4);

        result.Should().BeOfValues(16, 32, 64).And.BeOfType(NPTypeCode.UInt16);
    }

    [TestMethod]
    public void LeftShift_UInt32()
    {
        var arr = np.array(new uint[] { 1, 2, 4 });
        var result = np.left_shift(arr, 8);

        result.Should().BeOfValues(256u, 512u, 1024u).And.BeOfType(NPTypeCode.UInt32);
    }

    [TestMethod]
    public void LeftShift_UInt64()
    {
        var arr = np.array(new ulong[] { 1, 2, 4 });
        var result = np.left_shift(arr, 40);

        result.Should().BeOfValues(1099511627776UL, 2199023255552UL, 4398046511104UL).And.BeOfType(NPTypeCode.UInt64);
    }

    #endregion

    #region Type Promotion (NEP50) Tests

    [TestMethod]
    public void LeftShift_Int8_Int32_PromotesToInt32_Array()
    {
        // NumPy: np.left_shift(np.array([1,2,4],np.int8), np.array([1,2,3],np.int32)) -> int32 [2,8,32]
        // (regression: NumSharp previously used lhs.typecode and returned int8)
        var a = np.array(new sbyte[] { 1, 2, 4 });
        var b = np.array(new int[] { 1, 2, 3 });
        np.left_shift(a, b).Should().BeOfValues(2, 8, 32).And.BeOfType(NPTypeCode.Int32);
    }

    [TestMethod]
    public void LeftShift_Int8_Int32_PromotesToInt32_Scalar()
    {
        // The promoted width matters: shifting an int8 by 20 at int32 width yields 1048576,
        // whereas the old int8-width path overflowed to 0.
        var a = np.array(new sbyte[] { 1, 2, 4 });
        var b = np.array(new int[] { 20 });
        np.left_shift(a, b).Should().BeOfValues(1048576, 2097152, 4194304).And.BeOfType(NPTypeCode.Int32);
    }

    [TestMethod]
    public void LeftShift_Int32_Int64_PromotesToInt64()
    {
        // NumPy: int32 << int64 -> int64
        var a = np.array(new int[] { 1, 2, 4 });
        var b = np.array(new long[] { 2 });
        np.left_shift(a, b).Should().BeOfValues(4L, 8L, 16L).And.BeOfType(NPTypeCode.Int64);
    }

    [TestMethod]
    public void LeftShift_UInt8_UInt16_PromotesToUInt16()
    {
        var a = np.array(new byte[] { 1, 2, 4 });
        var b = np.array(new ushort[] { 1, 2, 3 });
        np.left_shift(a, b).Should().BeOfValues(2, 8, 32).And.BeOfType(NPTypeCode.UInt16);
    }

    [TestMethod]
    public void LeftShift_Bool_Bool_PromotesToInt8()
    {
        // NumPy has no bool shift loop; bool promotes to int8: True<<True == 1<<1 == 2.
        var a = np.array(new bool[] { true, false });
        var b = np.array(new bool[] { true, true });
        np.left_shift(a, b).Should().BeOfValues(2, 0).And.BeOfType(NPTypeCode.SByte);
    }

    [TestMethod]
    public void LeftShift_SByte_ScalarSimd_PreservesDtype()
    {
        // int8 has a vector shift loop; the SIMD scalar fast path must keep int8 and wrap.
        var a = np.array(new sbyte[] { 5, 10, 15, 64 });
        np.left_shift(a, 1).Should().BeOfValues(10, 20, 30, -128).And.BeOfType(NPTypeCode.SByte);
    }

    #endregion

    #region Overflow & Negative Count Tests (NumPy shift semantics)

    [TestMethod]
    public void LeftShift_CountGEWidth_IsZero()
    {
        // count >= bit width: left shift is always 0 (probed NumPy 2.4.2).
        np.left_shift(np.array(new sbyte[] { 1, 127, -1 }), 8)
            .Should().BeOfValues(0, 0, 0).And.BeOfType(NPTypeCode.SByte);
        np.left_shift(np.array(new int[] { 1 }), 40).Should().BeOfValues(0);
        np.left_shift(np.array(new long[] { 1 }), 100).Should().BeOfValues(0L);
    }

    [TestMethod]
    public void RightShift_CountGEWidth_SignedIsSignFill()
    {
        // signed right shift overflow: -1 for negative values, 0 otherwise.
        np.right_shift(np.array(new sbyte[] { 1, 127, -1 }), 8)
            .Should().BeOfValues(0, 0, -1).And.BeOfType(NPTypeCode.SByte);
        np.right_shift(np.array(new int[] { -8, 8 }), 40).Should().BeOfValues(-1, 0);
    }

    [TestMethod]
    public void RightShift_CountGEWidth_UnsignedIsZero()
    {
        np.right_shift(np.array(new byte[] { 255, 128 }), 8).Should().BeOfValues(0, 0).And.BeOfType(NPTypeCode.Byte);
    }

    [TestMethod]
    public void Shift_NegativeCount_BehavesLikeOverflow()
    {
        // count < 0 behaves like count >= width (probed NumPy 2.4.2).
        np.left_shift(np.array(new int[] { 5 }), -1).Should().BeOfValues(0);
        np.right_shift(np.array(new int[] { 5, -5 }), -1).Should().BeOfValues(0, -1);
    }

    #endregion

    #region Layout Tests (strided / broadcast)

    [TestMethod]
    public void LeftShift_StridedView_MatchesNumPy()
    {
        // np.left_shift(np.arange(8,dtype=int32)[::2], 1) -> [0,4,8,12]
        var a = np.arange(8).astype(NPTypeCode.Int32)["::2"];
        np.left_shift(a, 1).Should().BeOfValues(0, 4, 8, 12).And.BeOfType(NPTypeCode.Int32);
    }

    [TestMethod]
    public void LeftShift_ColumnBroadcast_MatchesNumPy()
    {
        // (3,4) << (3,1): each row shifted by its own count.
        var a = np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4);
        var col = np.array(new int[] { 0, 1, 2 }).reshape(3, 1);
        var r = np.left_shift(a, col);
        r.Should().BeShaped(3, 4);
        r.Should().BeOfValues(0, 1, 2, 3, 8, 10, 12, 14, 32, 36, 40, 44);
    }

    #endregion
}
