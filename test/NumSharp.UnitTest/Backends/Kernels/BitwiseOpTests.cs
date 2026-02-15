using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for bitwise operations (And, Or, Xor).
/// All expected values are verified against NumPy 2.x output.
/// </summary>
public class BitwiseOpTests
{
    #region Boolean AND Tests

    [Test]
    public void BitwiseAnd_Bool_TruthTable()
    {
        // NumPy: [True, True, False, False] & [True, False, True, False]
        //      = [True, False, False, False]
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a & b;

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    [Test]
    public void BitwiseAnd_Bool_WithScalar_True()
    {
        var a = np.array(new[] { true, true, false, false });
        var result = a & true;

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    [Test]
    public void BitwiseAnd_Bool_WithScalar_False()
    {
        var a = np.array(new[] { true, true, false, false });
        var result = a & false;

        Assert.IsFalse(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    #endregion

    #region Boolean OR Tests

    [Test]
    public void BitwiseOr_Bool_TruthTable()
    {
        // NumPy: [True, True, False, False] | [True, False, True, False]
        //      = [True, True, True, False]
        var a = np.array(new[] { true, true, false, false });
        var b = np.array(new[] { true, false, true, false });
        var result = a | b;

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    [Test]
    public void BitwiseOr_Bool_WithScalar_True()
    {
        var a = np.array(new[] { true, true, false, false });
        var result = a | true;

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
        Assert.IsTrue(result.GetBoolean(3));
    }

    [Test]
    public void BitwiseOr_Bool_WithScalar_False()
    {
        var a = np.array(new[] { true, true, false, false });
        var result = a | false;

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    #endregion

    #region Boolean XOR Tests

    [Test]
    public void BitwiseXor_Bool_TruthTable()
    {
        // NumPy: [True, True, False, False] ^ [True, False, True, False]
        //      = [False, True, True, False]
        // Use NDArray<bool> to get the typed ^ operator
        NDArray<bool> a = np.array(new[] { true, true, false, false }).MakeGeneric<bool>();
        NDArray<bool> b = np.array(new[] { true, false, true, false }).MakeGeneric<bool>();
        var result = a ^ b;

        Assert.IsFalse(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    #endregion

    #region Boolean NOT Tests

    [Test]
    public void BitwiseNot_Bool()
    {
        // NumPy: ~[True, True, False, False] = [False, False, True, True]
        var a = np.array(new[] { true, true, false, false });
        var result = !a;

        Assert.IsFalse(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
        Assert.IsTrue(result.GetBoolean(3));
    }

    #endregion

    #region Integer Bitwise AND Tests

    [Test]
    public void BitwiseAnd_Byte()
    {
        // NumPy: [0b11110000, 0b10101010, 0b00001111] & [0b11001100, 0b01010101, 0b11110000]
        //      = [0b11000000, 0b00000000, 0b00000000] = [192, 0, 0]
        var a = np.array(new byte[] { 0b11110000, 0b10101010, 0b00001111 });
        var b = np.array(new byte[] { 0b11001100, 0b01010101, 0b11110000 });
        var result = a & b;

        result.Should().BeOfValues(192, 0, 0).And.BeOfType(NPTypeCode.Byte);
    }

    [Test]
    public void BitwiseAnd_Int32()
    {
        var a = np.array(new[] { 0b11110000, 0b10101010, 0b00001111 });
        var b = np.array(new[] { 0b11001100, 0b01010101, 0b11110000 });
        var result = a & b;

        result.Should().BeOfValues(192, 0, 0).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void BitwiseAnd_Int64()
    {
        var a = np.array(new long[] { 0b11110000, 0b10101010, 0b00001111 });
        var b = np.array(new long[] { 0b11001100, 0b01010101, 0b11110000 });
        var result = a & b;

        result.Should().BeOfValues(192L, 0L, 0L).And.BeOfType(NPTypeCode.Int64);
    }

    #endregion

    #region Integer Bitwise OR Tests

    [Test]
    public void BitwiseOr_Byte()
    {
        // NumPy: [0b11110000, 0b10101010, 0b00001111] | [0b11001100, 0b01010101, 0b11110000]
        //      = [0b11111100, 0b11111111, 0b11111111] = [252, 255, 255]
        var a = np.array(new byte[] { 0b11110000, 0b10101010, 0b00001111 });
        var b = np.array(new byte[] { 0b11001100, 0b01010101, 0b11110000 });
        var result = a | b;

        result.Should().BeOfValues(252, 255, 255).And.BeOfType(NPTypeCode.Byte);
    }

    [Test]
    public void BitwiseOr_Int32()
    {
        var a = np.array(new[] { 0b11110000, 0b10101010, 0b00001111 });
        var b = np.array(new[] { 0b11001100, 0b01010101, 0b11110000 });
        var result = a | b;

        result.Should().BeOfValues(252, 255, 255).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region Integer Bitwise XOR Tests

    [Test]
    public void BitwiseXor_Byte()
    {
        // NumPy: [0b11110000, 0b10101010, 0b00001111] ^ [0b11001100, 0b01010101, 0b11110000]
        //      = [0b00111100, 0b11111111, 0b11111111] = [60, 255, 255]
        NDArray<byte> a = np.array(new byte[] { 0b11110000, 0b10101010, 0b00001111 }).MakeGeneric<byte>();
        NDArray<byte> b = np.array(new byte[] { 0b11001100, 0b01010101, 0b11110000 }).MakeGeneric<byte>();
        var result = a ^ b;

        result.Should().BeOfValues(60, 255, 255).And.BeOfType(NPTypeCode.Byte);
    }

    [Test]
    public void BitwiseXor_Int32()
    {
        NDArray<int> a = np.array(new[] { 0b11110000, 0b10101010, 0b00001111 }).MakeGeneric<int>();
        NDArray<int> b = np.array(new[] { 0b11001100, 0b01010101, 0b11110000 }).MakeGeneric<int>();
        var result = a ^ b;

        result.Should().BeOfValues(60, 255, 255).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region Broadcasting Tests

    [Test]
    [OpenBugs] // Bitwise broadcasting may have issues
    public void BitwiseAnd_2D_With_1D_Broadcasting()
    {
        // NumPy: [[True, False], [True, False]] & [True, False]
        //      = [[True, False], [True, False]]
        var a = np.array(new[,] { { true, false }, { true, false } });
        var b = np.array(new[] { true, false });
        var result = a & b;

        result.Should().BeShaped(2, 2);
        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    [Test]
    [OpenBugs] // Bitwise broadcasting may have issues
    public void BitwiseOr_2D_With_1D_Broadcasting()
    {
        var a = np.array(new[,] { { true, false }, { false, false } });
        var b = np.array(new[] { false, true });
        var result = a | b;

        result.Should().BeShaped(2, 2);
        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
        Assert.IsTrue(result.GetBoolean(3));
    }

    [Test]
    public void BitwiseAnd_Int32_Scalar_Broadcasting()
    {
        // Array AND with scalar
        var a = np.array(new[] { 0xFF, 0xF0, 0x0F, 0x00 });
        var result = a & 0x0F;

        result.Should().BeOfValues(0x0F, 0x00, 0x0F, 0x00).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region Boolean Mask Scenario Tests

    [Test]
    public void BitwiseAnd_BooleanMasks()
    {
        // Common pattern: combining comparison results
        // data > 2 AND data < 5
        var data = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask1 = data > 2;  // [False, False, True, True, True]
        var mask2 = data < 5;  // [True, True, True, True, False]
        var combined = mask1 & mask2;  // [False, False, True, True, False]

        Assert.IsFalse(combined.GetBoolean(0));
        Assert.IsFalse(combined.GetBoolean(1));
        Assert.IsTrue(combined.GetBoolean(2));
        Assert.IsTrue(combined.GetBoolean(3));
        Assert.IsFalse(combined.GetBoolean(4));
    }

    [Test]
    public void BitwiseOr_BooleanMasks()
    {
        // data > 2 OR data < 5 = all True in this case
        var data = np.array(new[] { 1, 2, 3, 4, 5 });
        var mask1 = data > 2;  // [False, False, True, True, True]
        var mask2 = data < 5;  // [True, True, True, True, False]
        var combined = mask1 | mask2;  // [True, True, True, True, True]

        Assert.IsTrue(combined.GetBoolean(0));
        Assert.IsTrue(combined.GetBoolean(1));
        Assert.IsTrue(combined.GetBoolean(2));
        Assert.IsTrue(combined.GetBoolean(3));
        Assert.IsTrue(combined.GetBoolean(4));
    }

    #endregion

    #region Typed NDArray Tests

    [Test]
    public void BitwiseOr_TypedBoolArray()
    {
        // This tests the NDArray<bool> | NDArray<bool> operator
        var a = np.array(new[] { true, false, true, false }).MakeGeneric<bool>();
        var b = np.array(new[] { true, true, false, false }).MakeGeneric<bool>();
        var result = a | b;

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    [Test]
    public void BitwiseAnd_TypedBoolArray()
    {
        var a = np.array(new[] { true, false, true, false }).MakeGeneric<bool>();
        var b = np.array(new[] { true, true, false, false }).MakeGeneric<bool>();
        var result = a & b;

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    [Test]
    public void BitwiseXor_TypedBoolArray()
    {
        var a = np.array(new[] { true, false, true, false }).MakeGeneric<bool>();
        var b = np.array(new[] { true, true, false, false }).MakeGeneric<bool>();
        var result = a ^ b;

        Assert.IsFalse(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
        Assert.IsFalse(result.GetBoolean(3));
    }

    #endregion

    #region Sliced Array Tests

    [Test]
    public void BitwiseAnd_SlicedArrays()
    {
        var a = np.array(new[] { true, false, true, false, true, false });
        var b = np.array(new[] { true, true, true, false, false, false });

        // Slice: every other element
        var a_sliced = a["::2"];  // [True, True, True]
        var b_sliced = b["::2"];  // [True, True, False]
        var result = a_sliced & b_sliced;

        Assert.AreEqual(3, result.size);
        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
    }

    #endregion

    #region Empty Array Tests

    [Test]
    public void BitwiseAnd_EmptyArrays()
    {
        var a = np.array(Array.Empty<bool>());
        var b = np.array(Array.Empty<bool>());
        var result = a & b;

        Assert.AreEqual(0, result.size);
        result.Should().BeShaped(0);
    }

    #endregion
}
