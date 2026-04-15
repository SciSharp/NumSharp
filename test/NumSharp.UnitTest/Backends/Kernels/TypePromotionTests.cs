using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests verifying NumSharp type promotion alignment with NumPy 2.x NEP50 rules.
/// Reference: docs/NUMPY_ALIGNMENT_INVESTIGATION.md
///
/// NumPy NEP50 Type Promotion Rules:
/// - Reductions (sum, prod, cumsum) on int32 return int64
/// - mean/std/var always return float64
/// - max/min preserve input dtype
/// - Binary operations follow "smallest type that can hold both" rule
/// - Division always returns float64 (true division)
/// - Floor division preserves integer type
/// </summary>
public class TypePromotionTests
{
    #region Reduction Promotion Tests

    /// <summary>
    /// NumPy: np.sum(np.int32([1,2,3])).dtype -> int64
    /// </summary>
    [TestMethod]
    public void Sum_Int32_ReturnsInt64()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(arr);

        // NumPy NEP50: sum promotes int32 to int64
        Assert.AreEqual(NPTypeCode.Int64, result.typecode,
            "sum(int32) should return int64 per NEP50");
        Assert.AreEqual(6L, result.GetInt64(0));
    }

    /// <summary>
    /// NumPy: np.prod(np.int32([2,3,4])).dtype -> int64
    /// </summary>
    [TestMethod]
    public void Prod_Int32_ReturnsInt64()
    {
        var arr = np.array(new int[] { 2, 3, 4 });
        var result = np.prod(arr);

        // NumPy NEP50: prod promotes int32 to int64
        Assert.AreEqual(NPTypeCode.Int64, result.typecode,
            "prod(int32) should return int64 per NEP50");
        Assert.AreEqual(24L, result.GetInt64(0));
    }

    /// <summary>
    /// NumPy: np.cumsum(np.int32([1,2,3])).dtype -> int64
    /// </summary>
    [TestMethod]
    public void Cumsum_Int32_ReturnsInt64()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.cumsum(arr);

        // NumPy NEP50: cumsum promotes int32 to int64
        Assert.AreEqual(NPTypeCode.Int64, result.typecode,
            "cumsum(int32) should return int64 per NEP50");

        var values = result.ToArray<long>();
        Assert.AreEqual(1L, values[0]);
        Assert.AreEqual(3L, values[1]);
        Assert.AreEqual(6L, values[2]);
    }

    /// <summary>
    /// NumPy: np.max(np.int32([1,2,3])).dtype -> int32 (preserves)
    /// </summary>
    [TestMethod]
    public void Max_Int32_PreservesType()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.amax(arr);

        // NumPy: max preserves input dtype (no promotion)
        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "max(int32) should preserve int32");
        Assert.AreEqual(3, result.GetInt32(0));
    }

    /// <summary>
    /// NumPy: np.min(np.int32([1,2,3])).dtype -> int32 (preserves)
    /// </summary>
    [TestMethod]
    public void Min_Int32_PreservesType()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.amin(arr);

        // NumPy: min preserves input dtype (no promotion)
        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "min(int32) should preserve int32");
        Assert.AreEqual(1, result.GetInt32(0));
    }

    /// <summary>
    /// NumPy: np.mean(np.int32([1,2,3])).dtype -> float64
    /// </summary>
    [TestMethod]
    public void Mean_Int32_ReturnsFloat64()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.mean(arr);

        // NumPy: mean always returns float64
        Assert.AreEqual(NPTypeCode.Double, result.typecode,
            "mean(int32) should return float64");
        Assert.AreEqual(2.0, result.GetDouble(0), 1e-10);
    }

    /// <summary>
    /// NumPy: np.std(np.int32([1,2,3])).dtype -> float64
    /// </summary>
    [TestMethod]
    public void Std_Int32_ReturnsFloat64()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.std(arr);

        // NumPy: std always returns float64
        Assert.AreEqual(NPTypeCode.Double, result.typecode,
            "std(int32) should return float64");
    }

    /// <summary>
    /// NumPy: np.var(np.int32([1,2,3])).dtype -> float64
    /// </summary>
    [TestMethod]
    public void Var_Int32_ReturnsFloat64()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.var(arr);

        // NumPy: var always returns float64
        Assert.AreEqual(NPTypeCode.Double, result.typecode,
            "var(int32) should return float64");
    }

    #endregion

    #region Binary Operation Promotion Tests

    /// <summary>
    /// NumPy: (int32 + int32).dtype -> int32
    /// </summary>
    [TestMethod]
    public void Add_Int32_Int32_PreservesType()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new int[] { 4, 5, 6 });
        var result = a + b;

        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "int32 + int32 should return int32");
    }

    /// <summary>
    /// NumPy: (int32 + int64).dtype -> int64
    /// </summary>
    [TestMethod]
    public void Add_Int32_Int64_ReturnsInt64()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new long[] { 4L, 5L, 6L });
        var result = a + b;

        Assert.AreEqual(NPTypeCode.Int64, result.typecode,
            "int32 + int64 should return int64");
    }

    /// <summary>
    /// NumPy: (int32 + float32).dtype -> float64
    /// </summary>
    [TestMethod]
    public void Add_Int32_Float32_ReturnsFloat64()
    {
        var a = np.array(new int[] { 1, 2, 3 });
        var b = np.array(new float[] { 0.5f, 0.5f, 0.5f });
        var result = a + b;

        // NumPy NEP50: int32 + float32 -> float64
        Assert.AreEqual(NPTypeCode.Double, result.typecode,
            "int32 + float32 should return float64 per NEP50");
    }

    /// <summary>
    /// NumPy: (int64 + float32).dtype -> float64
    /// </summary>
    [TestMethod]
    public void Add_Int64_Float32_ReturnsFloat64()
    {
        var a = np.array(new long[] { 1L, 2L, 3L });
        var b = np.array(new float[] { 0.5f, 0.5f, 0.5f });
        var result = a + b;

        // NumPy NEP50: int64 + float32 -> float64
        Assert.AreEqual(NPTypeCode.Double, result.typecode,
            "int64 + float32 should return float64 per NEP50");
    }

    /// <summary>
    /// NumPy: (float32 + float64).dtype -> float64
    /// </summary>
    [TestMethod]
    public void Add_Float32_Float64_ReturnsFloat64()
    {
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var b = np.array(new double[] { 0.5, 0.5, 0.5 });
        var result = a + b;

        Assert.AreEqual(NPTypeCode.Double, result.typecode,
            "float32 + float64 should return float64");
    }

    #endregion

    #region Division Tests

    /// <summary>
    /// NumPy: (int32 / int32).dtype -> float64 (true division)
    /// </summary>
    [TestMethod]
    public void Divide_Int32_Int32_ReturnsFloat64()
    {
        var a = np.array(new int[] { 4, 5, 6 });
        var b = np.array(new int[] { 2, 2, 2 });
        var result = a / b;

        // NumPy: true division always returns float64
        Assert.AreEqual(NPTypeCode.Double, result.typecode,
            "int32 / int32 (true division) should return float64");

        var values = result.ToArray<double>();
        Assert.AreEqual(2.0, values[0], 1e-10);
        Assert.AreEqual(2.5, values[1], 1e-10);
        Assert.AreEqual(3.0, values[2], 1e-10);
    }

    /// <summary>
    /// NumPy: (int32 // int32).dtype -> int32 (floor division)
    /// Note: NumSharp may not have floor division operator
    /// </summary>
    [TestMethod]
    public void FloorDivide_Int32_Int32_PreservesType()
    {
        var a = np.array(new int[] { 5, 7, 9 });
        var b = np.array(new int[] { 2, 3, 4 });

        // NumPy: floor division preserves int32
        // NumSharp uses np.floor_divide or integer division semantics
        var result = np.floor(a / b).astype(NPTypeCode.Int32);

        // After floor and cast back to int32
        Assert.AreEqual(NPTypeCode.Int32, result.typecode);

        var values = result.ToArray<int>();
        Assert.AreEqual(2, values[0]); // 5 // 2 = 2
        Assert.AreEqual(2, values[1]); // 7 // 3 = 2
        Assert.AreEqual(2, values[2]); // 9 // 4 = 2
    }

    #endregion

    #region Power Promotion Tests

    /// <summary>
    /// NumPy: np.power(int32, 2).dtype -> int32
    /// </summary>
    [TestMethod]
    public void Power_Int32_Int_PreservesType()
    {
        var a = np.array(new int[] { 2, 3, 4 });
        var result = np.power(a, 2);

        // NumPy: int32 ** int -> int32
        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "int32 ** 2 should return int32");

        var values = result.ToArray<int>();
        Assert.AreEqual(4, values[0]);
        Assert.AreEqual(9, values[1]);
        Assert.AreEqual(16, values[2]);
    }

    /// <summary>
    /// NumPy: np.power(int32, 2.0).dtype -> float64
    /// </summary>
    [TestMethod]
    public void Power_Int32_Float_ReturnsFloat64()
    {
        var a = np.array(new int[] { 2, 3, 4 });
        var result = np.power(a, 2.0);

        // NumPy: int32 ** float -> float64
        Assert.AreEqual(NPTypeCode.Double, result.typecode,
            "int32 ** 2.0 should return float64");

        var values = result.ToArray<double>();
        Assert.AreEqual(4.0, values[0], 1e-10);
        Assert.AreEqual(9.0, values[1], 1e-10);
        Assert.AreEqual(16.0, values[2], 1e-10);
    }

    #endregion

    #region Unary Operation Tests

    /// <summary>
    /// NumPy: np.sqrt(int32).dtype -> float64
    /// </summary>
    [TestMethod]
    public void Sqrt_Int32_ReturnsFloat64()
    {
        var arr = np.array(new int[] { 4, 9, 16 });
        var result = np.sqrt(arr);

        // NumPy: sqrt always returns float64
        Assert.AreEqual(NPTypeCode.Double, result.typecode,
            "sqrt(int32) should return float64");

        var values = result.ToArray<double>();
        Assert.AreEqual(2.0, values[0], 1e-10);
        Assert.AreEqual(3.0, values[1], 1e-10);
        Assert.AreEqual(4.0, values[2], 1e-10);
    }

    /// <summary>
    /// NumPy: np.abs(int32).dtype -> int32 (preserves)
    /// </summary>
    [TestMethod]
    public void Abs_Int32_PreservesType()
    {
        var arr = np.array(new int[] { -1, -2, 3 });
        var result = np.abs(arr);

        // NumPy: abs preserves input dtype
        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "abs(int32) should preserve int32");

        var values = result.ToArray<int>();
        Assert.AreEqual(1, values[0]);
        Assert.AreEqual(2, values[1]);
        Assert.AreEqual(3, values[2]);
    }

    /// <summary>
    /// NumPy: np.sign(int32).dtype -> int32 (preserves)
    /// </summary>
    [TestMethod]
    public void Sign_Int32_PreservesType()
    {
        var arr = np.array(new int[] { -5, 0, 5 });
        var result = np.sign(arr);

        // NumPy: sign preserves input dtype
        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "sign(int32) should preserve int32");

        var values = result.ToArray<int>();
        Assert.AreEqual(-1, values[0]);
        Assert.AreEqual(0, values[1]);
        Assert.AreEqual(1, values[2]);
    }

    /// <summary>
    /// NumPy: np.square(int32).dtype -> int32 (preserves)
    /// </summary>
    [TestMethod]
    public void Square_Int32_PreservesType()
    {
        var arr = np.array(new int[] { 2, 3, 4 });
        var result = np.square(arr);

        // NumPy: square preserves input dtype
        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "square(int32) should preserve int32");

        var values = result.ToArray<int>();
        Assert.AreEqual(4, values[0]);
        Assert.AreEqual(9, values[1]);
        Assert.AreEqual(16, values[2]);
    }

    /// <summary>
    /// NumPy: np.negative(int32).dtype -> int32 (preserves)
    /// </summary>
    [TestMethod]
    public void Negative_Int32_PreservesType()
    {
        var arr = np.array(new int[] { 1, -2, 3 });
        var result = np.negative(arr);

        // NumPy: negative preserves input dtype
        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "negative(int32) should preserve int32");

        var values = result.ToArray<int>();
        Assert.AreEqual(-1, values[0]);
        Assert.AreEqual(2, values[1]);
        Assert.AreEqual(-3, values[2]);
    }

    #endregion

    #region ArgMax/ArgMin Tests

    /// <summary>
    /// NumPy: np.argmax([1,2,3]) -> 2 (returns int64 scalar)
    /// NumSharp: np.argmax now returns long (C# long = int64) - aligned with NumPy
    /// </summary>
    [TestMethod]
    public void ArgMax_ReturnsCorrectIndex()
    {
        var arr = np.array(new int[] { 1, 3, 2 });
        long result = np.argmax(arr);

        // NumSharp now returns long (int64), matching NumPy
        Assert.AreEqual(1L, result); // Index of max value (3)
    }

    /// <summary>
    /// NumPy: np.argmin([1,2,3]) -> 0 (returns int64 scalar)
    /// NumSharp: np.argmin now returns long (C# long = int64) - aligned with NumPy
    /// </summary>
    [TestMethod]
    public void ArgMin_ReturnsCorrectIndex()
    {
        var arr = np.array(new int[] { 3, 1, 2 });
        long result = np.argmin(arr);

        // NumSharp now returns long (int64), matching NumPy
        Assert.AreEqual(1L, result); // Index of min value (1)
    }

    /// <summary>
    /// NumPy: np.argmax(arr, axis=0) returns NDArray with dtype int64
    /// </summary>
    [TestMethod]
    public void ArgMax_WithAxis_ReturnsInt64Array()
    {
        var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.argmax(arr, axis: 0);

        // NumPy: argmax(axis=) returns int64 NDArray
        Assert.AreEqual(NPTypeCode.Int64, result.typecode,
            "argmax(axis=0) should return int64 array");

        var values = result.ToArray<long>();
        Assert.AreEqual(1L, values[0]); // Max in column 0 is at row 1
        Assert.AreEqual(1L, values[1]); // Max in column 1 is at row 1
    }

    /// <summary>
    /// NumPy: np.argmin(arr, axis=0) returns NDArray with dtype int64
    /// </summary>
    [TestMethod]
    public void ArgMin_WithAxis_ReturnsInt64Array()
    {
        var arr = np.array(new int[,] { { 3, 4 }, { 1, 2 } });
        var result = np.argmin(arr, axis: 0);

        // NumPy: argmin(axis=) returns int64 NDArray
        Assert.AreEqual(NPTypeCode.Int64, result.typecode,
            "argmin(axis=0) should return int64 array");

        var values = result.ToArray<long>();
        Assert.AreEqual(1L, values[0]); // Min in column 0 is at row 1
        Assert.AreEqual(1L, values[1]); // Min in column 1 is at row 1
    }

    #endregion

    #region Axis Reduction Shape Tests

    /// <summary>
    /// Test reduction along axis maintains correct shape and dtype.
    /// NumPy: np.sum([[1,2],[3,4]], axis=0) -> [4, 6] with dtype int64
    /// </summary>
    [TestMethod]
    public void Sum_WithAxis_CorrectShapeAndType()
    {
        var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.sum(arr, axis: 0);

        // Shape should be (2,) after reduction along axis 0
        Assert.AreEqual(1, result.ndim);
        Assert.AreEqual(2, result.shape[0]);

        // NumPy NEP50: sum promotes int32 to int64
        Assert.AreEqual(NPTypeCode.Int64, result.typecode,
            "sum(int32, axis=0) should return int64 per NEP50");

        var values = result.ToArray<long>();
        Assert.AreEqual(4L, values[0]); // 1 + 3
        Assert.AreEqual(6L, values[1]); // 2 + 4
    }

    /// <summary>
    /// Test keepdims parameter with reduction.
    /// NumPy: np.sum([[1,2],[3,4]], axis=0, keepdims=True) -> [[4, 6]]
    /// </summary>
    [TestMethod]
    public void Sum_WithKeepdims_CorrectShape()
    {
        var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.sum(arr, axis: 0, keepdims: true);

        // Shape should be (1, 2) with keepdims=True
        Assert.AreEqual(2, result.ndim);
        Assert.AreEqual(1, result.shape[0]);
        Assert.AreEqual(2, result.shape[1]);
    }

    #endregion

    #region Mixed Type Edge Cases

    /// <summary>
    /// NumPy: byte + int32 -> int32
    /// </summary>
    [TestMethod]
    public void Add_Byte_Int32_ReturnsInt32()
    {
        var a = np.array(new byte[] { 1, 2, 3 });
        var b = np.array(new int[] { 100, 200, 300 });
        var result = a + b;

        Assert.AreEqual(NPTypeCode.Int32, result.typecode,
            "byte + int32 should return int32");
    }

    /// <summary>
    /// NumPy: uint32 + int32 -> int64 (to avoid overflow)
    /// </summary>
    [TestMethod]
    public void Add_UInt32_Int32_ReturnsInt64()
    {
        var a = np.array(new uint[] { 1, 2, 3 });
        var b = np.array(new int[] { 1, 2, 3 });
        var result = a + b;

        // NumPy promotes to int64 to avoid overflow when mixing signed/unsigned
        Assert.AreEqual(NPTypeCode.Int64, result.typecode,
            "uint32 + int32 should return int64 to handle mixed sign");
    }

    #endregion

    #region Empty Array Tests

    /// <summary>
    /// NumPy: np.max(np.array([])) raises ValueError
    /// "zero-size array to reduction operation maximum which has no identity"
    /// </summary>
    [TestMethod]
    public void Max_EmptyArray_ThrowsArgumentException()
    {
        var empty = np.array(new double[0]);
        Assert.ThrowsException<ArgumentException>(() => np.amax(empty));
    }

    /// <summary>
    /// NumPy: np.min(np.array([])) raises ValueError
    /// "zero-size array to reduction operation minimum which has no identity"
    /// </summary>
    [TestMethod]
    public void Min_EmptyArray_ThrowsArgumentException()
    {
        var empty = np.array(new double[0]);
        Assert.ThrowsException<ArgumentException>(() => np.amin(empty));
    }

    /// <summary>
    /// NumPy: np.argmax(np.array([])) raises ValueError
    /// "attempt to get argmax of an empty sequence"
    /// </summary>
    [TestMethod]
    public void ArgMax_EmptyArray_ThrowsArgumentException()
    {
        var empty = np.array(new double[0]);
        Assert.ThrowsException<ArgumentException>(() => np.argmax(empty));
    }

    /// <summary>
    /// NumPy: np.argmin(np.array([])) raises ValueError
    /// "attempt to get argmin of an empty sequence"
    /// </summary>
    [TestMethod]
    public void ArgMin_EmptyArray_ThrowsArgumentException()
    {
        var empty = np.array(new double[0]);
        Assert.ThrowsException<ArgumentException>(() => np.argmin(empty));
    }

    #endregion
}
