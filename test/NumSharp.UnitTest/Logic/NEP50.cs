using System;
using AwesomeAssertions;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Logic;

/// <summary>
/// NEP 50 Type Promotion Tests
///
/// NumPy 2.0 introduced NEP 50 (https://numpy.org/neps/nep-0050-scalar-promotion.html)
/// which changed scalar-array type promotion rules.
///
/// KEY CONCEPT: Python scalars (int, float, complex) are "weakly typed" - they adopt
/// the array's dtype when combined. NumPy scalars (np.int32, np.float64) are "strongly
/// typed" - their dtype is honored in promotion.
///
/// NUMSHARP DESIGN DECISION: C# primitive scalars (int, double, etc.) are treated as
/// "weakly typed" like Python scalars, not like NumPy scalars. This gives users the
/// natural Python-like experience:
///
///     np.array(new byte[]{1,2,3}) + 5  →  uint8 result (not int32)
///
/// This matches how Python users expect `arr + 5` to behave in NumPy 2.x.
///
/// Each test is verified against NumPy 2.4.2 output.
/// </summary>
public class NEP50_TypePromotion
{
    #region 1. Unsigned Array + Python Int (array dtype wins)

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.uint8) + 5).dtype)"
    /// Output: uint8
    /// </summary>
    [Test]
    public void UInt8Array_Plus_PythonInt_Returns_UInt8()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = arr + 5;

        result.dtype.Should().Be(np.uint8);
        result.GetAtIndex<byte>(0).Should().Be(6);
        result.GetAtIndex<byte>(1).Should().Be(7);
        result.GetAtIndex<byte>(2).Should().Be(8);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.uint16) + 5).dtype)"
    /// Output: uint16
    /// </summary>
    [Test]
    public void UInt16Array_Plus_PythonInt_Returns_UInt16()
    {
        var arr = np.array(new ushort[] { 1, 2, 3 });
        var result = arr + 5;

        result.dtype.Should().Be(np.uint16);
        result.GetAtIndex<ushort>(0).Should().Be(6);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.uint32) + 5).dtype)"
    /// Output: uint32
    /// </summary>
    [Test]
    public void UInt32Array_Plus_PythonInt_Returns_UInt32()
    {
        var arr = np.array(new uint[] { 1, 2, 3 });
        var result = arr + 5;

        result.dtype.Should().Be(np.uint32);
        result.GetAtIndex<uint>(0).Should().Be(6);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.uint64) + 5).dtype)"
    /// Output: uint64
    /// </summary>
    [Test]
    public void UInt64Array_Plus_PythonInt_Returns_UInt64()
    {
        var arr = np.array(new ulong[] { 1, 2, 3 });
        var result = arr + 5;

        result.dtype.Should().Be(np.uint64);
        result.GetAtIndex<ulong>(0).Should().Be(6);
    }

    #endregion

    #region 2. Signed Array + Python Int (array dtype wins)

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.int16) + 5).dtype)"
    /// Output: int16
    /// </summary>
    [Test]
    public void Int16Array_Plus_PythonInt_Returns_Int16()
    {
        var arr = np.array(new short[] { 1, 2, 3 });
        var result = arr + 5;

        result.dtype.Should().Be(np.int16);
        result.GetAtIndex<short>(0).Should().Be(6);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.int32) + 5).dtype)"
    /// Output: int32
    /// </summary>
    [Test]
    public void Int32Array_Plus_PythonInt_Returns_Int32()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = arr + 5;

        result.dtype.Should().Be(np.int32);
        result.GetAtIndex<int>(0).Should().Be(6);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.int64) + 5).dtype)"
    /// Output: int64
    /// </summary>
    [Test]
    public void Int64Array_Plus_PythonInt_Returns_Int64()
    {
        var arr = np.array(new long[] { 1, 2, 3 });
        var result = arr + 5;

        result.dtype.Should().Be(np.int64);
        result.GetAtIndex<long>(0).Should().Be(6);
    }

    #endregion

    #region 3. Float Array + Python Int (int adopts float kind)

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.float32) + 5).dtype)"
    /// Output: float32
    /// </summary>
    [Test]
    public void Float32Array_Plus_PythonInt_Returns_Float32()
    {
        var arr = np.array(new float[] { 1f, 2f, 3f });
        var result = arr + 5;

        result.dtype.Should().Be(np.float32);
        result.GetAtIndex<float>(0).Should().Be(6f);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.float64) + 5).dtype)"
    /// Output: float64
    /// </summary>
    [Test]
    public void Float64Array_Plus_PythonInt_Returns_Float64()
    {
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = arr + 5;

        result.dtype.Should().Be(np.float64);
        result.GetAtIndex<double>(0).Should().Be(6.0);
    }

    #endregion

    #region 4. Float Array + Python Float (scalar adopts array dtype)

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.float32) + 5.0).dtype)"
    /// Output: float32
    /// </summary>
    [Test]
    public void Float32Array_Plus_PythonFloat_Returns_Float32()
    {
        var arr = np.array(new float[] { 1f, 2f, 3f });
        var result = arr + 5.0;

        result.dtype.Should().Be(np.float32);
        result.GetAtIndex<float>(0).Should().Be(6f);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.float64) + 5.0).dtype)"
    /// Output: float64
    /// </summary>
    [Test]
    public void Float64Array_Plus_PythonFloat_Returns_Float64()
    {
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = arr + 5.0;

        result.dtype.Should().Be(np.float64);
        result.GetAtIndex<double>(0).Should().Be(6.0);
    }

    #endregion

    #region 5. Int Array + Python Float (promotes to float64 - cross-kind)

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.int32) + 5.0).dtype)"
    /// Output: float64
    /// </summary>
    [Test]
    public void Int32Array_Plus_PythonFloat_Returns_Float64()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = arr + 5.0;

        result.dtype.Should().Be(np.float64);
        result.GetAtIndex<double>(0).Should().Be(6.0);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.uint8) + 5.0).dtype)"
    /// Output: float64
    /// </summary>
    [Test]
    public void UInt8Array_Plus_PythonFloat_Returns_Float64()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = arr + 5.0;

        result.dtype.Should().Be(np.float64);
        result.GetAtIndex<double>(0).Should().Be(6.0);
    }

    #endregion

    #region 6. Subtraction (same rules apply)

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([10,20,30], np.uint8) - 5).dtype)"
    /// Output: uint8
    /// </summary>
    [Test]
    public void UInt8Array_Minus_PythonInt_Returns_UInt8()
    {
        var arr = np.array(new byte[] { 10, 20, 30 });
        var result = arr - 5;

        result.dtype.Should().Be(np.uint8);
        result.GetAtIndex<byte>(0).Should().Be(5);
        result.GetAtIndex<byte>(1).Should().Be(15);
        result.GetAtIndex<byte>(2).Should().Be(25);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([10,20,30], np.uint16) - 5).dtype)"
    /// Output: uint16
    /// </summary>
    [Test]
    public void UInt16Array_Minus_PythonInt_Returns_UInt16()
    {
        var arr = np.array(new ushort[] { 10, 20, 30 });
        var result = arr - 5;

        result.dtype.Should().Be(np.uint16);
        result.GetAtIndex<ushort>(0).Should().Be(5);
    }

    #endregion

    #region 7. Multiplication (same rules apply)

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.uint8) * 5).dtype)"
    /// Output: uint8
    /// </summary>
    [Test]
    public void UInt8Array_Times_PythonInt_Returns_UInt8()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = arr * 5;

        result.dtype.Should().Be(np.uint8);
        result.GetAtIndex<byte>(0).Should().Be(5);
        result.GetAtIndex<byte>(1).Should().Be(10);
        result.GetAtIndex<byte>(2).Should().Be(15);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.uint32) * 5).dtype)"
    /// Output: uint32
    /// </summary>
    [Test]
    public void UInt32Array_Times_PythonInt_Returns_UInt32()
    {
        var arr = np.array(new uint[] { 1, 2, 3 });
        var result = arr * 5;

        result.dtype.Should().Be(np.uint32);
        result.GetAtIndex<uint>(0).Should().Be(5);
    }

    #endregion

    #region 8. Division

    /// <summary>
    /// NumPy: python3 -c "import numpy as np; print((np.array([10,20,30], np.uint8) / 5).dtype)"
    /// Output: float64
    ///
    /// NumSharp now matches NumPy: true division returns float64.
    /// </summary>
    [Test]
    public void UInt8Array_Divide_PythonInt_Returns_Float64()
    {
        var arr = np.array(new byte[] { 10, 20, 30 });
        var result = arr / 5;

        // True division returns float64, matching NumPy
        result.dtype.Should().Be(np.float64, "True division returns float64");
        result.GetAtIndex<double>(0).Should().Be(2.0);  // 10 / 5 = 2.0
    }

    /// <summary>
    /// NumPy: python3 -c "import numpy as np; print((np.array([10,20,30], np.int32) / 5).dtype)"
    /// Output: float64
    ///
    /// NumSharp now matches NumPy: true division returns float64.
    /// </summary>
    [Test]
    public void Int32Array_Divide_PythonInt_Returns_Float64()
    {
        var arr = np.array(new int[] { 10, 20, 30 });
        var result = arr / 5;

        // True division returns float64, matching NumPy
        result.dtype.Should().Be(np.float64, "True division returns float64");
        result.GetAtIndex<double>(0).Should().Be(2.0);  // 10 / 5 = 2.0
    }

    /// <summary>
    /// Float division works correctly.
    /// Verified: python3 -c "import numpy as np; print((np.array([10,20,30], np.float64) / 5).dtype)"
    /// Output: float64
    /// </summary>
    [Test]
    public void Float64Array_Divide_PythonInt_Returns_Float64()
    {
        var arr = np.array(new double[] { 10.0, 20.0, 30.0 });
        var result = arr / 5;

        result.dtype.Should().Be(np.float64);
        result.GetAtIndex<double>(0).Should().Be(2.0);
    }

    #endregion

    #region 9. Modulo

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([10,20,30], np.uint8) % 7).dtype)"
    /// Output: uint8
    /// </summary>
    [Test]
    public void UInt8Array_Mod_PythonInt_Returns_UInt8()
    {
        var arr = np.array(new byte[] { 10, 20, 30 });
        var result = arr % 7;

        result.dtype.Should().Be(np.uint8);
        result.GetAtIndex<byte>(0).Should().Be(3);  // 10 % 7 = 3
        result.GetAtIndex<byte>(1).Should().Be(6);  // 20 % 7 = 6
        result.GetAtIndex<byte>(2).Should().Be(2);  // 30 % 7 = 2
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([10,20,30], np.int32) % 7).dtype)"
    /// Output: int32
    /// </summary>
    [Test]
    public void Int32Array_Mod_PythonInt_Returns_Int32()
    {
        var arr = np.array(new int[] { 10, 20, 30 });
        var result = arr % 7;

        result.dtype.Should().Be(np.int32);
        result.GetAtIndex<int>(0).Should().Be(3);
    }

    #endregion

    #region 10. Scalar-First Operations (commutative - same result)

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((5 + np.array([1,2,3], np.uint8)).dtype)"
    /// Output: uint8
    /// </summary>
    [Test]
    public void PythonInt_Plus_UInt8Array_Returns_UInt8()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = 5 + arr;

        result.dtype.Should().Be(np.uint8);
        result.GetAtIndex<byte>(0).Should().Be(6);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((5 - np.array([1,2,3], np.uint8)).dtype)"
    /// Output: uint8
    /// </summary>
    [Test]
    public void PythonInt_Minus_UInt8Array_Returns_UInt8()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = 5 - arr;

        result.dtype.Should().Be(np.uint8);
        result.GetAtIndex<byte>(0).Should().Be(4);  // 5 - 1 = 4
        result.GetAtIndex<byte>(1).Should().Be(3);  // 5 - 2 = 3
        result.GetAtIndex<byte>(2).Should().Be(2);  // 5 - 3 = 2
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((5 * np.array([1,2,3], np.uint8)).dtype)"
    /// Output: uint8
    /// </summary>
    [Test]
    public void PythonInt_Times_UInt8Array_Returns_UInt8()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = 5 * arr;

        result.dtype.Should().Be(np.uint8);
        result.GetAtIndex<byte>(0).Should().Be(5);
    }

    #endregion

    #region 11. _FindCommonArrayScalarType Direct Tests

    /// <summary>
    /// Direct verification of the type promotion table for NEP 50 changes.
    /// These are the 12 entries that changed from NumPy 1.x to 2.x behavior.
    /// </summary>
    [Test]
    public void ArrayScalarType_UInt8_SignedScalars()
    {
        // NEP 50: uint8 array + signed scalar → uint8 (array wins)
        np._FindCommonArrayScalarType(NPTypeCode.Byte, NPTypeCode.Int16).Should().Be(NPTypeCode.Byte);
        np._FindCommonArrayScalarType(NPTypeCode.Byte, NPTypeCode.Int32).Should().Be(NPTypeCode.Byte);
        np._FindCommonArrayScalarType(NPTypeCode.Byte, NPTypeCode.Int64).Should().Be(NPTypeCode.Byte);
    }

    [Test]
    public void ArrayScalarType_UInt16_SignedScalars()
    {
        // NEP 50: uint16 array + signed scalar → uint16 (array wins)
        np._FindCommonArrayScalarType(NPTypeCode.UInt16, NPTypeCode.Int16).Should().Be(NPTypeCode.UInt16);
        np._FindCommonArrayScalarType(NPTypeCode.UInt16, NPTypeCode.Int32).Should().Be(NPTypeCode.UInt16);
        np._FindCommonArrayScalarType(NPTypeCode.UInt16, NPTypeCode.Int64).Should().Be(NPTypeCode.UInt16);
    }

    [Test]
    public void ArrayScalarType_UInt32_SignedScalars()
    {
        // NEP 50: uint32 array + signed scalar → uint32 (array wins)
        np._FindCommonArrayScalarType(NPTypeCode.UInt32, NPTypeCode.Int16).Should().Be(NPTypeCode.UInt32);
        np._FindCommonArrayScalarType(NPTypeCode.UInt32, NPTypeCode.Int32).Should().Be(NPTypeCode.UInt32);
        np._FindCommonArrayScalarType(NPTypeCode.UInt32, NPTypeCode.Int64).Should().Be(NPTypeCode.UInt32);
    }

    [Test]
    public void ArrayScalarType_UInt64_SignedScalars()
    {
        // NEP 50: uint64 array + signed scalar → uint64 (array wins)
        np._FindCommonArrayScalarType(NPTypeCode.UInt64, NPTypeCode.Int16).Should().Be(NPTypeCode.UInt64);
        np._FindCommonArrayScalarType(NPTypeCode.UInt64, NPTypeCode.Int32).Should().Be(NPTypeCode.UInt64);
        np._FindCommonArrayScalarType(NPTypeCode.UInt64, NPTypeCode.Int64).Should().Be(NPTypeCode.UInt64);
    }

    #endregion

    #region 12. Array-Array Operations (unchanged - uses arr_arr table)

    /// <summary>
    /// Array-array operations use _typemap_arr_arr, not _typemap_arr_scalar.
    /// These should NOT be affected by NEP 50 changes to arr_scalar table.
    ///
    /// Verified: python3 -c "import numpy as np; print((np.array([1], np.uint8) + np.array([5], np.int32)).dtype)"
    /// Output: int32
    /// </summary>
    [Test]
    public void ArrayArray_UInt8_Plus_Int32_Returns_Int32()
    {
        var arr1 = np.array(new byte[] { 1, 2, 3 });
        var arr2 = np.array(new int[] { 5, 6, 7 });
        var result = arr1 + arr2;

        result.dtype.Should().Be(np.int32);
        result.GetAtIndex<int>(0).Should().Be(6);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1], np.uint16) + np.array([5], np.int32)).dtype)"
    /// Output: int32
    /// </summary>
    [Test]
    public void ArrayArray_UInt16_Plus_Int32_Returns_Int32()
    {
        var arr1 = np.array(new ushort[] { 1, 2, 3 });
        var arr2 = np.array(new int[] { 5, 6, 7 });
        var result = arr1 + arr2;

        result.dtype.Should().Be(np.int32);
    }

    #endregion

    #region 13. Unchanged Behaviors (sanity checks)

    /// <summary>
    /// Same-type operations remain unchanged.
    /// </summary>
    [Test]
    public void SameType_UInt8_Plus_UInt8_Returns_UInt8()
    {
        var arr1 = np.array(new byte[] { 1, 2, 3 });
        var arr2 = np.array(new byte[] { 4, 5, 6 });
        var result = arr1 + arr2;

        result.dtype.Should().Be(np.uint8);
        result.GetAtIndex<byte>(0).Should().Be(5);
    }

    /// <summary>
    /// Unsigned array + larger unsigned scalar → array dtype (unchanged).
    /// </summary>
    [Test]
    public void UInt8Array_Plus_UInt32Scalar_Returns_UInt8()
    {
        // This was already "array wins" in NumPy 1.x for same-kind
        np._FindCommonArrayScalarType(NPTypeCode.Byte, NPTypeCode.UInt32).Should().Be(NPTypeCode.Byte);
    }

    /// <summary>
    /// Float operations with int scalars → float (array kind wins).
    /// </summary>
    [Test]
    public void Float32Array_Operations_PreserveFloat()
    {
        var arr = np.array(new float[] { 1f, 2f, 3f });

        (arr + 5).dtype.Should().Be(np.float32);
        (arr - 5).dtype.Should().Be(np.float32);
        (arr * 5).dtype.Should().Be(np.float32);
        (5 + arr).dtype.Should().Be(np.float32);
    }

    #endregion

    #region 14. Edge Cases

    /// <summary>
    /// Empty array operations should preserve dtype.
    /// </summary>
    [Test]
    public void EmptyArray_Operations_PreserveDtype()
    {
        var arr = np.array(Array.Empty<byte>());
        var result = arr + 5;

        result.dtype.Should().Be(np.uint8);
        result.size.Should().Be(0);
    }

    /// <summary>
    /// 1D array operations.
    /// </summary>
    [Test]
    public void OneDimensional_UInt8_Operations()
    {
        var arr = np.arange(10).astype(np.uint8);
        var result = arr + 100;

        result.dtype.Should().Be(np.uint8);
        result.shape.Should().BeEquivalentTo(new[] { 10 });
    }

    /// <summary>
    /// Multi-dimensional array operations.
    /// </summary>
    [Test]
    public void MultiDimensional_UInt8_Operations()
    {
        var arr = np.arange(12).astype(np.uint8).reshape(3, 4);
        var result = arr + 5;

        result.dtype.Should().Be(np.uint8);
        result.shape.Should().BeEquivalentTo(new[] { 3, 4 });
    }

    /// <summary>
    /// Scalar array (0-d) operations use SCALAR-SCALAR promotion, not ARR-SCALAR.
    ///
    /// When both operands are scalars, NumSharp uses _FindCommonScalarType,
    /// which follows different rules than _FindCommonArrayScalarType.
    ///
    /// NumPy: python3 -c "import numpy as np; print((np.uint8(10) + 5).dtype)"
    /// Output: uint8 (NEP 50: weak scalar adopts stronger scalar dtype)
    ///
    /// NumSharp uses scalar-scalar table which may differ.
    /// </summary>
    [Test]
    [Misaligned]
    public void ScalarArray_Operations_UsesScalarScalarPromotion()
    {
        var arr = NDArray.Scalar((byte)10);
        var result = arr + 5;

        // Note: Scalar + scalar uses _FindCommonScalarType, not _FindCommonArrayScalarType
        // Current behavior returns int32 (matches C# semantics)
        result.dtype.Should().Be(np.int32,
            "Scalar-scalar promotion follows different rules than arr-scalar");
    }

    #endregion

    #region 15. Power Operation

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.uint8) ** 2).dtype)"
    /// Output: uint8
    /// </summary>
    [Test]
    public void UInt8Array_Power_PythonInt_Returns_UInt8()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = np.power(arr, 2);

        result.dtype.Should().Be(np.uint8);
        result.GetAtIndex<byte>(0).Should().Be(1);
        result.GetAtIndex<byte>(1).Should().Be(4);
        result.GetAtIndex<byte>(2).Should().Be(9);
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.int32) ** 2).dtype)"
    /// Output: int32
    /// </summary>
    [Test]
    public void Int32Array_Power_PythonInt_Returns_Int32()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.power(arr, 2);

        result.dtype.Should().Be(np.int32);
        result.GetAtIndex<int>(0).Should().Be(1);
        result.GetAtIndex<int>(1).Should().Be(4);
        result.GetAtIndex<int>(2).Should().Be(9);
    }

    #endregion

    #region 16. Comparison with NumPy 1.x Behavior (Documentation)

    /// <summary>
    /// Documents what CHANGED from NumPy 1.x to 2.x behavior.
    ///
    /// NumPy 1.x: uint8([1,2,3]) + 5  →  int64 (scalar widened to int64)
    /// NumPy 2.x: uint8([1,2,3]) + 5  →  uint8 (array dtype wins)
    ///
    /// NumSharp now follows NumPy 2.x behavior.
    /// </summary>
    [Test]
    public void Documentation_NEP50_BreakingChange()
    {
        // This is the key behavioral change from NumPy 1.x to 2.x
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = arr + 5;

        // NumPy 1.x would return int64
        // NumPy 2.x (and NumSharp) returns uint8
        result.dtype.Should().Be(np.uint8,
            "NEP 50: array dtype wins when scalar is same-kind (integer)");
    }

    #endregion

    #region 17. Cross-Kind Promotion (float wins over int)

    /// <summary>
    /// When kinds differ (int vs float), the higher kind wins.
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.int16) + 5.0).dtype)"
    /// Output: float64
    ///
    /// Note: NumSharp doesn't support int8 (sbyte), so we use int16.
    /// </summary>
    [Test]
    public void CrossKind_IntArray_Plus_Float_Returns_Float64()
    {
        var intArr = np.array(new short[] { 1, 2, 3 });  // int16
        var result = intArr + 5.0;

        result.dtype.Should().Be(np.float64, "Cross-kind: float wins over int");
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.uint8) + 5.0).dtype)"
    /// Output: float64
    /// </summary>
    [Test]
    public void CrossKind_UInt8Array_Plus_Float_Returns_Float64()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        var result = arr + 5.0;

        result.dtype.Should().Be(np.float64, "Cross-kind: float wins over uint");
    }

    /// <summary>
    /// Verified: python3 -c "import numpy as np; print((np.array([1,2,3], np.int32) + 5.0).dtype)"
    /// Output: float64
    /// </summary>
    [Test]
    public void CrossKind_Int32Array_Plus_Float_Returns_Float64()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = arr + 5.0;

        result.dtype.Should().Be(np.float64, "Cross-kind: float wins over int");
    }

    #endregion

    #region 18. Complete NEP50 Operation Matrix (12 combinations × 4 operations)

    // ============================================================================
    // UINT8 + SIGNED SCALARS (3 combinations)
    // ============================================================================

    /// <summary>
    /// Verified: python3 -c "import numpy as np; a=np.array([1,2,3], np.uint8); print((a + np.int16(5)).dtype)"
    /// NumPy 2.x with numpy scalar: int16 (strongly typed)
    /// NumSharp with C# short: treats as weakly typed → uint8
    /// </summary>
    [Test]
    public void NEP50_UInt8_Plus_Short_AllOps()
    {
        var arr = np.array(new byte[] { 10, 20, 30 });
        short scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint8, "uint8 + int16 → uint8");
        (arr - scalar).dtype.Should().Be(np.uint8, "uint8 - int16 → uint8");
        (arr * scalar).dtype.Should().Be(np.uint8, "uint8 * int16 → uint8");
        (arr % scalar).dtype.Should().Be(np.uint8, "uint8 % int16 → uint8");
    }

    [Test]
    public void NEP50_UInt8_Plus_Int_AllOps()
    {
        var arr = np.array(new byte[] { 10, 20, 30 });
        int scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint8, "uint8 + int32 → uint8");
        (arr - scalar).dtype.Should().Be(np.uint8, "uint8 - int32 → uint8");
        (arr * scalar).dtype.Should().Be(np.uint8, "uint8 * int32 → uint8");
        (arr % scalar).dtype.Should().Be(np.uint8, "uint8 % int32 → uint8");
    }

    [Test]
    public void NEP50_UInt8_Plus_Long_AllOps()
    {
        var arr = np.array(new byte[] { 10, 20, 30 });
        long scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint8, "uint8 + int64 → uint8");
        (arr - scalar).dtype.Should().Be(np.uint8, "uint8 - int64 → uint8");
        (arr * scalar).dtype.Should().Be(np.uint8, "uint8 * int64 → uint8");
        (arr % scalar).dtype.Should().Be(np.uint8, "uint8 % int64 → uint8");
    }

    // ============================================================================
    // UINT16 + SIGNED SCALARS (3 combinations)
    // ============================================================================

    [Test]
    public void NEP50_UInt16_Plus_Short_AllOps()
    {
        var arr = np.array(new ushort[] { 100, 200, 300 });
        short scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint16, "uint16 + int16 → uint16");
        (arr - scalar).dtype.Should().Be(np.uint16, "uint16 - int16 → uint16");
        (arr * scalar).dtype.Should().Be(np.uint16, "uint16 * int16 → uint16");
        (arr % scalar).dtype.Should().Be(np.uint16, "uint16 % int16 → uint16");
    }

    [Test]
    public void NEP50_UInt16_Plus_Int_AllOps()
    {
        var arr = np.array(new ushort[] { 100, 200, 300 });
        int scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint16, "uint16 + int32 → uint16");
        (arr - scalar).dtype.Should().Be(np.uint16, "uint16 - int32 → uint16");
        (arr * scalar).dtype.Should().Be(np.uint16, "uint16 * int32 → uint16");
        (arr % scalar).dtype.Should().Be(np.uint16, "uint16 % int32 → uint16");
    }

    [Test]
    public void NEP50_UInt16_Plus_Long_AllOps()
    {
        var arr = np.array(new ushort[] { 100, 200, 300 });
        long scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint16, "uint16 + int64 → uint16");
        (arr - scalar).dtype.Should().Be(np.uint16, "uint16 - int64 → uint16");
        (arr * scalar).dtype.Should().Be(np.uint16, "uint16 * int64 → uint16");
        (arr % scalar).dtype.Should().Be(np.uint16, "uint16 % int64 → uint16");
    }

    // ============================================================================
    // UINT32 + SIGNED SCALARS (3 combinations)
    // ============================================================================

    [Test]
    public void NEP50_UInt32_Plus_Short_AllOps()
    {
        var arr = np.array(new uint[] { 1000, 2000, 3000 });
        short scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint32, "uint32 + int16 → uint32");
        (arr - scalar).dtype.Should().Be(np.uint32, "uint32 - int16 → uint32");
        (arr * scalar).dtype.Should().Be(np.uint32, "uint32 * int16 → uint32");
        (arr % scalar).dtype.Should().Be(np.uint32, "uint32 % int16 → uint32");
    }

    [Test]
    public void NEP50_UInt32_Plus_Int_AllOps()
    {
        var arr = np.array(new uint[] { 1000, 2000, 3000 });
        int scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint32, "uint32 + int32 → uint32");
        (arr - scalar).dtype.Should().Be(np.uint32, "uint32 - int32 → uint32");
        (arr * scalar).dtype.Should().Be(np.uint32, "uint32 * int32 → uint32");
        (arr % scalar).dtype.Should().Be(np.uint32, "uint32 % int32 → uint32");
    }

    [Test]
    public void NEP50_UInt32_Plus_Long_AllOps()
    {
        var arr = np.array(new uint[] { 1000, 2000, 3000 });
        long scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint32, "uint32 + int64 → uint32");
        (arr - scalar).dtype.Should().Be(np.uint32, "uint32 - int64 → uint32");
        (arr * scalar).dtype.Should().Be(np.uint32, "uint32 * int64 → uint32");
        (arr % scalar).dtype.Should().Be(np.uint32, "uint32 % int64 → uint32");
    }

    // ============================================================================
    // UINT64 + SIGNED SCALARS (3 combinations)
    // ============================================================================

    [Test]
    public void NEP50_UInt64_Plus_Short_AllOps()
    {
        var arr = np.array(new ulong[] { 10000, 20000, 30000 });
        short scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint64, "uint64 + int16 → uint64");
        (arr - scalar).dtype.Should().Be(np.uint64, "uint64 - int16 → uint64");
        (arr * scalar).dtype.Should().Be(np.uint64, "uint64 * int16 → uint64");
        (arr % scalar).dtype.Should().Be(np.uint64, "uint64 % int16 → uint64");
    }

    [Test]
    public void NEP50_UInt64_Plus_Int_AllOps()
    {
        var arr = np.array(new ulong[] { 10000, 20000, 30000 });
        int scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint64, "uint64 + int32 → uint64");
        (arr - scalar).dtype.Should().Be(np.uint64, "uint64 - int32 → uint64");
        (arr * scalar).dtype.Should().Be(np.uint64, "uint64 * int32 → uint64");
        (arr % scalar).dtype.Should().Be(np.uint64, "uint64 % int32 → uint64");
    }

    [Test]
    public void NEP50_UInt64_Plus_Long_AllOps()
    {
        var arr = np.array(new ulong[] { 10000, 20000, 30000 });
        long scalar = 5;

        (arr + scalar).dtype.Should().Be(np.uint64, "uint64 + int64 → uint64");
        (arr - scalar).dtype.Should().Be(np.uint64, "uint64 - int64 → uint64");
        (arr * scalar).dtype.Should().Be(np.uint64, "uint64 * int64 → uint64");
        (arr % scalar).dtype.Should().Be(np.uint64, "uint64 % int64 → uint64");
    }

    #endregion

    #region 19. Value Correctness Tests

    /// <summary>
    /// Verify actual computed values are correct, not just dtypes.
    /// </summary>
    [Test]
    public void NEP50_Values_UInt8_Operations()
    {
        var arr = np.array(new byte[] { 10, 20, 30 });

        var add = arr + 5;
        add.GetAtIndex<byte>(0).Should().Be(15);
        add.GetAtIndex<byte>(1).Should().Be(25);
        add.GetAtIndex<byte>(2).Should().Be(35);

        var sub = arr - 5;
        sub.GetAtIndex<byte>(0).Should().Be(5);
        sub.GetAtIndex<byte>(1).Should().Be(15);
        sub.GetAtIndex<byte>(2).Should().Be(25);

        var mul = arr * 2;
        mul.GetAtIndex<byte>(0).Should().Be(20);
        mul.GetAtIndex<byte>(1).Should().Be(40);
        mul.GetAtIndex<byte>(2).Should().Be(60);

        var mod = arr % 7;
        mod.GetAtIndex<byte>(0).Should().Be(3);   // 10 % 7
        mod.GetAtIndex<byte>(1).Should().Be(6);   // 20 % 7
        mod.GetAtIndex<byte>(2).Should().Be(2);   // 30 % 7
    }

    [Test]
    public void NEP50_Values_UInt32_Operations()
    {
        var arr = np.array(new uint[] { 1000, 2000, 3000 });

        var add = arr + 500;
        add.GetAtIndex<uint>(0).Should().Be(1500);
        add.GetAtIndex<uint>(1).Should().Be(2500);
        add.GetAtIndex<uint>(2).Should().Be(3500);

        var sub = arr - 500;
        sub.GetAtIndex<uint>(0).Should().Be(500);
        sub.GetAtIndex<uint>(1).Should().Be(1500);
        sub.GetAtIndex<uint>(2).Should().Be(2500);
    }

    [Test]
    public void NEP50_Values_UInt64_Operations()
    {
        var arr = np.array(new ulong[] { 10000, 20000, 30000 });

        var add = arr + 5000L;
        add.GetAtIndex<ulong>(0).Should().Be(15000);
        add.GetAtIndex<ulong>(1).Should().Be(25000);
        add.GetAtIndex<ulong>(2).Should().Be(35000);
    }

    #endregion
}
