using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Comprehensive tests for operations on sliced (non-contiguous) arrays.
/// All expected values are verified against NumPy 2.x output.
/// Sliced arrays test the strided path of IL kernels rather than the SIMD contiguous path.
/// </summary>
public class SlicedArrayOpTests
{
    #region Binary Operations on Sliced Arrays

    [Test]
    public void Add_SlicedArrays_Int32()
    {
        // NumPy: a[::2] + b[::2] where a=[1,2,3,4,5,6,7,8], b=[10,20,30,40,50,60,70,80]
        // a[::2] = [1, 3, 5, 7], b[::2] = [10, 30, 50, 70]
        // Result: [11, 33, 55, 77]
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var b = np.array(new[] { 10, 20, 30, 40, 50, 60, 70, 80 });

        var result = a["::2"] + b["::2"];

        result.Should().BeOfValues(11, 33, 55, 77).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Add_SlicedArrays_SameSource()
    {
        // NumPy: a[::2] + a[1::2] where a=[1,2,3,4,5,6,7,8]
        // a[::2] = [1, 3, 5, 7], a[1::2] = [2, 4, 6, 8]
        // Result: [3, 7, 11, 15]
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8 });

        var result = a["::2"] + a["1::2"];

        result.Should().BeOfValues(3, 7, 11, 15).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Subtract_SlicedArrays_Float64()
    {
        // a[::2] - a[1::2]
        var a = np.array(new[] { 10.0, 2.0, 30.0, 4.0, 50.0, 6.0 });
        // a[::2] = [10, 30, 50], a[1::2] = [2, 4, 6]
        // Result: [8, 26, 44]

        var result = a["::2"] - a["1::2"];

        Assert.AreEqual(8.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(26.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(44.0, result.GetDouble(2), 1e-10);
    }

    [Test]
    public void Multiply_SlicedArrays_Int32()
    {
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
        // a[::2] = [1, 3, 5], a[1::2] = [2, 4, 6]
        // Result: [2, 12, 30]

        var result = a["::2"] * a["1::2"];

        result.Should().BeOfValues(2, 12, 30).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Divide_SlicedArrays_Float64()
    {
        var a = np.array(new[] { 10.0, 2.0, 30.0, 5.0, 60.0, 6.0 });
        // a[::2] = [10, 30, 60], a[1::2] = [2, 5, 6]
        // Result: [5, 6, 10]

        var result = a["::2"] / a["1::2"];

        Assert.AreEqual(5.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(10.0, result.GetDouble(2), 1e-10);
    }

    [Test]
    [OpenBugs]  // Sliced array + scalar fails
    public void Add_SlicedWithScalar()
    {
        // NumPy: j[::2] + 5 where j=[10, 20, 30, 40, 50]
        // j[::2] = [10, 30, 50]
        // Result: [15, 35, 55]
        var j = np.array(new[] { 10, 20, 30, 40, 50 });

        var result = j["::2"] + 5;

        result.Should().BeOfValues(15, 35, 55).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    [OpenBugs]  // Sliced array * scalar fails
    public void Multiply_SlicedWithScalar()
    {
        // NumPy: j[1::2] * 2 where j=[10, 20, 30, 40, 50]
        // j[1::2] = [20, 40]
        // Result: [40, 80]
        var j = np.array(new[] { 10, 20, 30, 40, 50 });

        var result = j["1::2"] * 2;

        result.Should().BeOfValues(40, 80).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region 2D Sliced Operations

    [Test]
    public void Add_2DSliced_Rows()
    {
        // c[::2, :] + c[1::1, :] for 3x3 array
        var c = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        // c[0, :] = [1, 2, 3], c[2, :] = [7, 8, 9]
        // c[::2, :] = [[1,2,3], [7,8,9]]

        var sliced = c["::2, :"];

        Assert.AreEqual(2, sliced.shape[0]);
        Assert.AreEqual(3, sliced.shape[1]);
        Assert.AreEqual(1.0, sliced.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(7.0, sliced.GetDouble(1, 0), 1e-10);
    }

    [Test]
    public void Add_2DSliced_Cols()
    {
        var c = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        // c[:, ::2] = [[1, 3], [4, 6], [7, 9]]

        var sliced = c[":, ::2"];

        Assert.AreEqual(3, sliced.shape[0]);
        Assert.AreEqual(2, sliced.shape[1]);
        Assert.AreEqual(1.0, sliced.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(3.0, sliced.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(6.0, sliced.GetDouble(1, 1), 1e-10);
    }

    [Test]
    public void Add_2DSliced_Corners()
    {
        var c = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        // c[::2, ::2] = [[1, 3], [7, 9]]

        var sliced = c["::2, ::2"];

        Assert.AreEqual(2, sliced.shape[0]);
        Assert.AreEqual(2, sliced.shape[1]);
        Assert.AreEqual(1.0, sliced.GetDouble(0, 0), 1e-10);
        Assert.AreEqual(3.0, sliced.GetDouble(0, 1), 1e-10);
        Assert.AreEqual(7.0, sliced.GetDouble(1, 0), 1e-10);
        Assert.AreEqual(9.0, sliced.GetDouble(1, 1), 1e-10);
    }

    #endregion

    #region Unary Operations on Sliced Arrays

    [Test]
    public void Sin_SlicedArray()
    {
        // NumPy: np.sin(d[::2]) where d=[0, 1, 2, 3, 4, 5]
        // d[::2] = [0, 2, 4]
        // Result: [0.0, 0.90929743, -0.7568025]
        var d = np.array(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });

        var result = np.sin(d["::2"]);

        Assert.AreEqual(0.0, result.GetDouble(0), 1e-7);
        Assert.AreEqual(0.90929743, result.GetDouble(1), 1e-7);
        Assert.AreEqual(-0.7568025, result.GetDouble(2), 1e-7);
    }

    [Test]
    public void Sqrt_SlicedArray()
    {
        // NumPy: np.sqrt(d[1::2]) where d=[0, 1, 2, 3, 4, 5]
        // d[1::2] = [1, 3, 5]
        // Result: [1.0, 1.73205081, 2.23606798]
        var d = np.array(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });

        var result = np.sqrt(d["1::2"]);

        Assert.AreEqual(1.0, result.GetDouble(0), 1e-7);
        Assert.AreEqual(1.73205081, result.GetDouble(1), 1e-7);
        Assert.AreEqual(2.23606798, result.GetDouble(2), 1e-7);
    }

    [Test]
    public void Cos_SlicedArray()
    {
        var d = np.array(new[] { 0.0, Math.PI / 2, Math.PI, Math.PI * 1.5, Math.PI * 2 });
        // d[::2] = [0, π, 2π]
        // cos: [1, -1, 1]

        var result = np.cos(d["::2"]);

        Assert.AreEqual(1.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(-1.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(1.0, result.GetDouble(2), 1e-10);
    }

    [Test]
    public void Exp_SlicedArray()
    {
        var d = np.array(new[] { 0.0, 1.0, 2.0, 3.0 });
        // d[::2] = [0, 2]
        // exp: [1, e^2=7.389...]

        var result = np.exp(d["::2"]);

        Assert.AreEqual(1.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(Math.Exp(2.0), result.GetDouble(1), 1e-10);
    }

    [Test]
    public void Log_SlicedArray()
    {
        var d = np.array(new[] { 1.0, 2.0, Math.E, 4.0, Math.E * Math.E });
        // d[::2] = [1, e, e^2]
        // log: [0, 1, 2]

        var result = np.log(d["::2"]);

        Assert.AreEqual(0.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(1.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(2.0, result.GetDouble(2), 1e-10);
    }

    [Test]
    public void Abs_SlicedArray()
    {
        var d = np.array(new[] { -5.0, 3.0, -2.0, 7.0, -1.0, 9.0 });
        // d[::2] = [-5, -2, -1]
        // abs: [5, 2, 1]

        var result = np.abs(d["::2"]);

        Assert.AreEqual(5.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(2.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(1.0, result.GetDouble(2), 1e-10);
    }

    #endregion

    #region Comparison Operations on Sliced Arrays

    [Test]
    public void LessThan_SlicedArrays()
    {
        // NumPy: e[::2] < e[1::2] where e=[1, 5, 2, 6, 3, 7]
        // e[::2] = [1, 2, 3], e[1::2] = [5, 6, 7]
        // Result: [True, True, True]
        var e = np.array(new[] { 1, 5, 2, 6, 3, 7 });

        var result = e["::2"] < e["1::2"];

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
    }

    [Test]
    [OpenBugs]  // Sliced comparison with scalar fails
    public void GreaterThan_SlicedWithScalar()
    {
        // NumPy: e[::2] > 2 where e=[1, 5, 2, 6, 3, 7]
        // e[::2] = [1, 2, 3]
        // Result: [False, False, True]
        var e = np.array(new[] { 1, 5, 2, 6, 3, 7 });

        var result = e["::2"] > 2;

        Assert.IsFalse(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
    }

    [Test]
    public void Equal_SlicedArrays()
    {
        var a = np.array(new[] { 1, 2, 3, 2, 5, 2 });
        // a[::2] = [1, 3, 5], a[1::2] = [2, 2, 2]
        // Result: [False, False, False]

        var result = a["::2"] == a["1::2"];

        Assert.IsFalse(result.GetBoolean(0));
        Assert.IsFalse(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
    }

    #endregion

    #region Bitwise Operations on Sliced Arrays

    [Test]
    public void BitwiseAnd_SlicedBoolArrays()
    {
        // NumPy: f[::2] & g[::2]
        // f = [True, False, True, False, True, False], g = [False, True, True, False, False, True]
        // f[::2] = [True, True, True], g[::2] = [False, True, False]
        // Result: [False, True, False]
        var f = np.array(new[] { true, false, true, false, true, false });
        var g = np.array(new[] { false, true, true, false, false, true });

        var result = f["::2"] & g["::2"];

        Assert.IsFalse(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsFalse(result.GetBoolean(2));
    }

    [Test]
    public void BitwiseOr_SlicedBoolArrays()
    {
        // NumPy: f[::2] | g[::2]
        // f[::2] = [True, True, True], g[::2] = [False, True, False]
        // Result: [True, True, True]
        var f = np.array(new[] { true, false, true, false, true, false });
        var g = np.array(new[] { false, true, true, false, false, true });

        var result = f["::2"] | g["::2"];

        Assert.IsTrue(result.GetBoolean(0));
        Assert.IsTrue(result.GetBoolean(1));
        Assert.IsTrue(result.GetBoolean(2));
    }

    [Test]
    public void BitwiseXor_SlicedIntArrays()
    {
        var a = np.array(new[] { 0b1010, 0b0000, 0b1111, 0b0000, 0b0101 });
        var b = np.array(new[] { 0b1100, 0b0000, 0b1010, 0b0000, 0b0011 });
        // a[::2] = [0b1010, 0b1111, 0b0101]
        // b[::2] = [0b1100, 0b1010, 0b0011]
        // XOR:   = [0b0110, 0b0101, 0b0110] = [6, 5, 6]

        var aTyped = a.MakeGeneric<int>();
        var bTyped = b.MakeGeneric<int>();

        var result = aTyped["::2"] ^ bTyped["::2"];

        Assert.AreEqual(0b0110, result.GetInt32(0));
        Assert.AreEqual(0b0101, result.GetInt32(1));
        Assert.AreEqual(0b0110, result.GetInt32(2));
    }

    #endregion

    #region Reduction Operations on Sliced Arrays

    [Test]
    public void Sum_SlicedArray()
    {
        // NumPy: np.sum(h[::2]) where h=[1, 2, 3, 4, 5, 6, 7, 8]
        // h[::2] = [1, 3, 5, 7]
        // Result: 16
        var h = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var result = np.sum(h["::2"]);

        Assert.AreEqual(16.0, result.GetDouble(0), 1e-10);
    }

    [Test]
    public void Mean_SlicedArray()
    {
        // NumPy: np.mean(h[1::2]) where h=[1, 2, 3, 4, 5, 6, 7, 8]
        // h[1::2] = [2, 4, 6, 8]
        // Result: 5.0
        var h = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var result = np.mean(h["1::2"]);

        Assert.AreEqual(5.0, result.GetDouble(0), 1e-10);
    }

    [Test]
    public void Max_SlicedArray()
    {
        // NumPy: np.max(h[::3]) where h=[1, 2, 3, 4, 5, 6, 7, 8]
        // h[::3] = [1, 4, 7]
        // Result: 7
        var h = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });

        var result = np.amax(h["::3"]);

        Assert.AreEqual(7.0, result.GetDouble(0), 1e-10);
    }

    [Test]
    public void Min_SlicedArray()
    {
        var h = np.array(new[] { 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 });
        // h[::2] = [8, 6, 4, 2]
        // min: 2

        var result = np.amin(h["::2"]);

        Assert.AreEqual(2.0, result.GetDouble(0), 1e-10);
    }

    [Test]
    public void Sum_2DSliced()
    {
        // NumPy: np.sum(c[::2, :]) where c=[[1,2,3],[4,5,6],[7,8,9]]
        // c[::2, :] = [[1,2,3], [7,8,9]]
        // Result: 30
        var c = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

        var result = np.sum(c["::2, :"]);

        Assert.AreEqual(30.0, result.GetDouble(0), 1e-10);
    }

    [Test]
    public void Sum_2DSlicedCols()
    {
        // NumPy: np.sum(c[:, ::2]) where c=[[1,2,3],[4,5,6],[7,8,9]]
        // c[:, ::2] = [[1,3], [4,6], [7,9]]
        // Result: 1+3+4+6+7+9 = 30
        var c = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });

        var result = np.sum(c[":, ::2"]);

        Assert.AreEqual(30.0, result.GetDouble(0), 1e-10);
    }

    #endregion

    #region Reversed Slice Operations

    [Test]
    public void Add_ReversedSlice()
    {
        // NumPy: i[::-1] + i where i=[1, 2, 3, 4, 5]
        // i[::-1] = [5, 4, 3, 2, 1]
        // Result: [6, 6, 6, 6, 6]
        var i = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var result = i["::-1"] + i;

        Assert.AreEqual(6.0, result.GetDouble(0), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(1), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(2), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(3), 1e-10);
        Assert.AreEqual(6.0, result.GetDouble(4), 1e-10);
    }

    [Test]
    public void Sum_ReversedSlice()
    {
        // NumPy: np.sum(i[::-1]) = 15 (same as forward)
        var i = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var result = np.sum(i["::-1"]);

        Assert.AreEqual(15.0, result.GetDouble(0), 1e-10);
    }

    [Test]
    public void Sin_ReversedStridedSlice()
    {
        // NumPy: np.sin(i[::-2]) where i=[1, 2, 3, 4, 5]
        // i[::-2] = [5, 3, 1]
        // Result: sin([5, 3, 1]) = [-0.95892427, 0.14112001, 0.84147098]
        var i = np.array(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var result = np.sin(i["::-2"]);

        Assert.AreEqual(-0.95892427, result.GetDouble(0), 1e-7);
        Assert.AreEqual(0.14112001, result.GetDouble(1), 1e-7);
        Assert.AreEqual(0.84147098, result.GetDouble(2), 1e-7);
    }

    [Test]
    public void Multiply_ReversedSlice()
    {
        var a = np.array(new[] { 1, 2, 3, 4 });
        // a[::-1] = [4, 3, 2, 1]
        // a * a[::-1] = [4, 6, 6, 4]

        var result = a * a["::-1"];

        result.Should().BeOfValues(4, 6, 6, 4).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region Mixed Contiguous and Sliced Operations

    [Test]
    public void Add_ContiguousWithSliced()
    {
        // Contiguous + Sliced
        var a = np.array(new[] { 1, 2, 3, 4 });
        var b = np.array(new[] { 10, 20, 30, 40, 50, 60, 70, 80 });

        var result = a + b["::2"];  // a + [10, 30, 50, 70]

        result.Should().BeOfValues(11, 32, 53, 74).And.BeOfType(NPTypeCode.Int32);
    }

    [Test]
    public void Multiply_SlicedWithContiguous()
    {
        // Sliced * Contiguous
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var b = np.array(new[] { 2, 2, 2, 2 });

        var result = a["::2"] * b;  // [1, 3, 5, 7] * [2, 2, 2, 2]

        result.Should().BeOfValues(2, 6, 10, 14).And.BeOfType(NPTypeCode.Int32);
    }

    #endregion

    #region Type Preservation on Sliced Arrays

    [Test]
    public void Add_SlicedByte()
    {
        var a = np.array(new byte[] { 1, 2, 3, 4, 5, 6 });
        var b = np.array(new byte[] { 10, 20, 30, 40, 50, 60 });

        var result = a["::2"] + b["::2"];  // [1, 3, 5] + [10, 30, 50] = [11, 33, 55]

        result.Should().BeOfValues(11, 33, 55).And.BeOfType(NPTypeCode.Byte);
    }

    [Test]
    public void Add_SlicedFloat32()
    {
        var a = np.array(new float[] { 1.5f, 2.5f, 3.5f, 4.5f });
        var b = np.array(new float[] { 0.5f, 1.0f, 1.5f, 2.0f });

        var result = a["::2"] + b["::2"];  // [1.5, 3.5] + [0.5, 1.5] = [2.0, 5.0]

        Assert.AreEqual(NPTypeCode.Single, result.typecode);
        Assert.AreEqual(2.0f, result.GetSingle(0), 1e-5f);
        Assert.AreEqual(5.0f, result.GetSingle(1), 1e-5f);
    }

    [Test]
    public void Add_SlicedInt64()
    {
        var a = np.array(new long[] { 1000000000000L, 2, 3000000000000L, 4 });
        var b = np.array(new long[] { 1, 2, 3, 4 });

        var result = a["::2"] + b["::2"];  // [1e12, 3e12] + [1, 3]

        Assert.AreEqual(NPTypeCode.Int64, result.typecode);
        Assert.AreEqual(1000000000001L, result.GetInt64(0));
        Assert.AreEqual(3000000000003L, result.GetInt64(1));
    }

    #endregion

    #region Edge Cases

    [Test]
    public void SingleElement_Slice()
    {
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        // a[2:3] = [3] (single element slice, still 1D)

        var sliced = a["2:3"];

        Assert.AreEqual(1, sliced.size);
        Assert.AreEqual(1, sliced.ndim);
        Assert.AreEqual(3, sliced.GetInt32(0));
    }

    [Test]
    public void Step_GreaterThanSize()
    {
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        // a[::10] = [1] (step larger than array)

        var sliced = a["::10"];

        Assert.AreEqual(1, sliced.size);
        Assert.AreEqual(1, sliced.GetInt32(0));
    }

    [Test]
    public void NegativeStep_FullReverse()
    {
        var a = np.array(new[] { 1, 2, 3, 4, 5 });
        // a[::-1] = [5, 4, 3, 2, 1]

        var sliced = a["::-1"];

        Assert.AreEqual(5, sliced.GetInt32(0));
        Assert.AreEqual(4, sliced.GetInt32(1));
        Assert.AreEqual(3, sliced.GetInt32(2));
        Assert.AreEqual(2, sliced.GetInt32(3));
        Assert.AreEqual(1, sliced.GetInt32(4));
    }

    [Test]
    public void SliceOfSlice()
    {
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        // a[::2] = [1, 3, 5, 7, 9]
        // a[::2][::2] = [1, 5, 9]

        var sliced = a["::2"]["::2"];

        Assert.AreEqual(3, sliced.size);
        Assert.AreEqual(1, sliced.GetInt32(0));
        Assert.AreEqual(5, sliced.GetInt32(1));
        Assert.AreEqual(9, sliced.GetInt32(2));
    }

    [Test]
    public void SliceOfSlice_Operations()
    {
        var a = np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        // a[::2][::2] = [1, 5], a[1::2][::2] = [2, 6]
        // Sum: [3, 11]

        var result = a["::2"]["::2"] + a["1::2"]["::2"];

        Assert.AreEqual(2, result.size);
        Assert.AreEqual(3, result.GetInt32(0));
        Assert.AreEqual(11, result.GetInt32(1));
    }

    #endregion
}
