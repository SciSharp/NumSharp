using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Logic;

/// <summary>
///     NEP 50's weak Python scalar in the comparison ufuncs and the binary ufuncs: a C# literal (NumPy's Python int,
///     float, complex or bool) adopts the array's precision and is packed into it as NumPy's <c>PyArray_Pack</c> does.
///     Every expected value was probed against NumPy 2.4.2; the oracle tier <c>weak_scalar.jsonl</c> covers the matrix,
///     these tests pin the cases a regression would hit first.
/// </summary>
[TestClass]
public class ComparisonWeakScalarTests
{
    private static bool[] B(NDArray a) => a.ToArray<bool>();

    /// <summary>
    ///     The reported bug: a float32 array compared with a Python float runs at float32. 0.1 packs into float32(0.1),
    ///     which equals the first element; comparing at float64 gave [True, True] for <c>&gt;</c>.
    /// </summary>
    [TestMethod]
    public void Float32Array_PythonFloat_ComparesAtFloat32()
    {
        // NumPy: np.greater(np.array([0.1, 0.5], np.float32), 0.1) -> [False, True], and so on.
        var x = np.array(new[] { 0.1f, 0.5f });
        B(np.greater(x, 0.1)).Should().Equal(false, true);
        B(np.greater_equal(x, 0.1)).Should().Equal(true, true);
        B(np.less(x, 0.1)).Should().Equal(false, false);
        B(np.less_equal(x, 0.1)).Should().Equal(true, false);
        B(np.equal(x, 0.1)).Should().Equal(true, false);
        B(np.not_equal(x, 0.1)).Should().Equal(false, true);
        // Scalar first: np.greater(0.1, x) -> [False, False], np.less(0.1, x) -> [False, True].
        B(np.greater(0.1, x)).Should().Equal(false, false);
        B(np.less(0.1, x)).Should().Equal(false, true);
    }

    /// <summary>The C# operators bind the same comparison, through both their NDArray and object overloads.</summary>
    [TestMethod]
    public void Operators_UseTheWeakScalar()
    {
        var x = np.array(new[] { 0.1f, 0.5f });
        B(x > 0.1).Should().Equal(false, true);
        B(x <= 0.1).Should().Equal(true, false);
        B(x == 0.1).Should().Equal(true, false);
        B(x != 0.1).Should().Equal(false, true);
        B(0.1 < x).Should().Equal(false, true);
        B(x > (object)0.1).Should().Equal(false, true);
        B((object)0.1 == x).Should().Equal(true, false);
    }

    /// <summary>The out= and where= routes run the same weak comparison (the masked-off slot keeps its prior value).</summary>
    [TestMethod]
    public void OutAndWhere_UseTheWeakScalar()
    {
        var x = np.array(new[] { 0.1f, 0.5f, 0.1f });
        var o = np.zeros(new Shape(3), np.@bool);
        np.greater(x, 0.1, @out: o);
        B(o).Should().Equal(false, true, false);

        var w = np.array(new[] { true, false, true });
        var o2 = np.array(new[] { false, true, false });
        np.equal(x, 0.1, @out: o2, where: w);
        B(o2).Should().Equal(true, true, true);
    }

    /// <summary>float16: the Python float packs with npy_double_to_half, an int past 2048 rounds, an int past 65504
    ///     becomes infinity.</summary>
    [TestMethod]
    public void Float16Array_PacksThePythonScalar()
    {
        // NumPy: np.array([0.1, 0.5], np.float16) == 0.1 -> [True, False]; [2048] == 2049 -> [True];
        // [inf] == 70000 -> [True] (70000 packs as inf).
        var h = np.array(new[] { (Half)0.1, (Half)0.5 });
        B(h == 0.1).Should().Equal(true, false);
        B(np.array(new[] { (Half)2048 }) == 2049).Should().Equal(true);
        B(np.array(new[] { Half.PositiveInfinity }) == 70000).Should().Equal(true);
        // A C# float literal is a Python float of the same value: float32(0.1) packs into float16(0.1) too.
        B(h == 0.1f).Should().Equal(true, false);
    }

    /// <summary>A Python int packs into float32 through the correctly rounded double: an int past 2^53 rounds twice.</summary>
    [TestMethod]
    public void Float32Array_PythonInt_PacksThroughTheDouble()
    {
        // NumPy: np.array([2.0**24], np.float32) == 16777217 -> [True];
        // np.array([2.0**60], np.float32) == 2**60 + 2**36 + 1 -> [True] (double rounding),
        // np.array([2.0**60 + 2.0**37], np.float32) == 2**60 + 2**36 + 1 -> [False].
        B(np.array(new[] { 16777216f }) == 16777217).Should().Equal(true);
        long v = (1L << 60) + (1L << 36) + 1;
        B(np.array(new[] { (float)(1L << 60) }) == v).Should().Equal(true);
        B(np.array(new[] { (float)((1L << 60) + (1L << 37)) }) == v).Should().Equal(false);
    }

    /// <summary>A Python complex against a float32 array compares as complex64: each part rounds to float32.</summary>
    [TestMethod]
    public void Float32Array_PythonComplex_ComparesAsComplex64()
    {
        // NumPy: np.array([0.1, 0.5], np.float32) == (0.1+0j) -> [True, False]; == (0.1+1e-50j) -> [True, False];
        // < (0.1+1e-9j) -> [True, False].
        var x = np.array(new[] { 0.1f, 0.5f });
        B(x == new Complex(0.1, 0)).Should().Equal(true, false);
        B(x == new Complex(0.1, 1e-50)).Should().Equal(true, false);
        B(x < new Complex(0.1, 1e-9)).Should().Equal(true, false);
    }

    /// <summary>An integer array compares EXACTLY with any Python int, in range or not (NumPy 2.x).</summary>
    [TestMethod]
    public void IntegerArray_PythonInt_ComparesExactly()
    {
        // NumPy: np.array([1, 2], np.int8) > 1000 -> [False, False]; np.array([1, 2], np.uint8) > -1 -> [True, True];
        // np.array([2**63 - 1], np.int64) == 2**63 -> [False].
        B(np.array(new sbyte[] { 1, 2 }) > 1000).Should().Equal(false, false);
        B(np.array(new byte[] { 1, 2 }) > -1).Should().Equal(true, true);
        B(np.array(new[] { long.MaxValue }) == 9223372036854775808UL).Should().Equal(false);
    }

    /// <summary>
    ///     A bool array has no exact Python-int comparison: NumPy compares in the int64 loop, whose pack fails past int64.
    ///     The text is the LP64 (Linux / macOS) one NumSharp pins; Windows NumPy says "int too big to convert".
    /// </summary>
    [TestMethod]
    public void BoolArray_PythonIntPastInt64_RaisesTheLp64OverflowError()
    {
        var b = np.array(new[] { true, false });
        Action eq = () => np.equal(b, 9223372036854775808UL);
        eq.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
        B(np.equal(b, long.MaxValue)).Should().Equal(false, false);
    }

    /// <summary>The out= route compares an int64 / uint64 pair exactly too (NumPy's qQ loops), like the bare route.</summary>
    [TestMethod]
    public void OutRoute_Int64UInt64Pair_IsExact()
    {
        // NumPy: np.equal(np.array([2**53 + 1], np.int64), np.array([2**53], np.uint64), out=o) -> [False].
        var o = np.zeros(new Shape(1), np.@bool);
        np.equal(np.array(new[] { (1L << 53) + 1 }), np.array(new[] { 1UL << 53 }), @out: o);
        B(o).Should().Equal(false);
    }

    /// <summary>
    ///     NumSharp cannot tell a C# literal from an explicit 0-d array, so a 0-d operand against an array is read as the
    ///     Python scalar its type stands for — the rule NumSharp's arithmetic uses. NumPy treats a 0-d ARRAY as strong:
    ///     <c>np.greater(float32 [0.1, 0.5], np.array(0.1))</c> is [True, True] there.
    /// </summary>
    [TestMethod]
    [Misaligned]
    public void ZeroDimArrayOperand_IsReadAsThePythonScalar()
    {
        var x = np.array(new[] { 0.1f, 0.5f });
        B(np.greater(x, np.array(0.1))).Should().Equal(false, true);
    }

    /// <summary>
    ///     Two 0-d operands keep the strong promotion, as in NumSharp's arithmetic. NumPy's answer depends on which one is
    ///     a Python scalar (<c>np.float32(0.1) &gt; 0.1</c> is False there, <c>np.float32(0.1) &gt; np.float64(0.1)</c> True).
    /// </summary>
    [TestMethod]
    [Misaligned]
    public void TwoZeroDimOperands_KeepTheStrongPromotion()
    {
        B(np.greater(np.array(0.1f), np.array(0.1))).Should().Equal(true);
    }

    /// <summary>np.isclose keeps Python scalars weak in NumPy 2.x, so it now compares at the array's float32.</summary>
    [TestMethod]
    public void IsClose_Float32_PythonFloat_IsWeak()
    {
        // NumPy: np.isclose(np.array([0.1, 0.5], np.float32), 0.1, rtol=0, atol=0) -> [True, False].
        B(np.isclose(np.array(new[] { 0.1f, 0.5f }), 0.1, rtol: 0, atol: 0)).Should().Equal(true, false);
    }

    /// <summary>np.evaluate types a complex literal against a float32 operand as complex64 too.</summary>
    [TestMethod]
    public void Evaluate_ComplexLiteral_ComparesAsComplex64()
    {
        var x = np.array(new[] { 0.1f, 0.5f });
        B(np.evaluate(NDExpr.Equal(x, new Complex(0.1, 0)))).Should().Equal(true, false);
        B(np.evaluate(NDExpr.Equal(x, 0.1))).Should().Equal(true, false);
    }

    /// <summary>
    ///     Empty operands broadcast through the comparisons and the C# operators exactly as NumPy broadcasts them, or raise
    ///     its broadcast error (the operators used to answer the left operand's shape; the iterator route raised a
    ///     reduction error).
    /// </summary>
    [TestMethod]
    public void EmptyOperands_BroadcastOrRaise()
    {
        // NumPy: (np.zeros((2,1,3)) > np.zeros((0,3))).shape == (2,0,3); (np.zeros((0,3)) == np.zeros((2,1,3))).shape the
        // same; np.zeros(0) > np.zeros(3) raises "operands could not be broadcast together with shapes (0,) (3,) ".
        (np.zeros(new Shape(2, 1, 3)) > np.zeros(new Shape(0, 3))).shape.Should().Equal(2L, 0L, 3L);
        (np.zeros(new Shape(0, 3)) == np.zeros(new Shape(2, 1, 3))).shape.Should().Equal(2L, 0L, 3L);
        np.greater(np.zeros(new Shape(4, 0, 1)), np.zeros(new Shape(1, 5))).shape.Should().Equal(4L, 0L, 5L);
        Action mismatch = () => _ = np.zeros(new Shape(0)) > np.zeros(new Shape(3));
        mismatch.Should().Throw<IncorrectShapeException>()
            .WithMessage("operands could not be broadcast together with shapes (0,) (3,) ");
    }
}

/// <summary>
///     NEP 50's pack of a weak Python int into a binary ufunc's loop dtype: NumPy range-checks it
///     (<c>npy_update_operand_for_scalar</c> → setitem) instead of wrapping, and packs it into a float32 loop through the
///     correctly rounded double. Probed against NumPy 2.4.2.
/// </summary>
[TestClass]
public class BinaryUfuncWeakIntPackTests
{
    /// <summary>The reported family's silent wrong value: int8 + 1000 used to wrap to [-23, -22].</summary>
    [TestMethod]
    public void IntegerLoop_OutOfRangePythonInt_RaisesOverflowError()
    {
        var i8 = np.array(new sbyte[] { 1, 2 });
        Action add = () => _ = i8 + 1000;
        add.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
        Action sub = () => _ = np.array(new byte[] { 1, 2 }) + -1;
        sub.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
        Action max = () => np.maximum(i8, 1000);
        max.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
        Action shl = () => np.left_shift(i8, 1000);
        shl.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
        Action outRoute = () => np.add(i8, 1000, @out: np.zeros(new Shape(2), np.int8));
        outRoute.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
        // In range: the array's dtype, no error.
        (i8 + 100).ToArray<sbyte>().Should().Equal((sbyte)101, (sbyte)102);
    }

    /// <summary>dtype= selects the loop the int is packed into.</summary>
    [TestMethod]
    public void DtypeOverride_IsTheLoopThePackChecks()
    {
        var i8 = np.array(new sbyte[] { 1, 2 });
        np.add(i8, 1000, dtype: np.int16).ToArray<short>().Should().Equal((short)1001, (short)1002);
        Action narrow = () => np.add(i8, 1000, dtype: np.int8);
        narrow.Should().Throw<OverflowException>().WithMessage("Python integer 1000 out of bounds for int8");
    }

    /// <summary>
    ///     np.power packs before its integer-power rule: uint8 ** -1 is the pack's OverflowError, int8 ** -1 the
    ///     negative-power error (plan api-parameter-parity X21).
    /// </summary>
    [TestMethod]
    public void Power_PacksBeforeTheNegativePowerRule()
    {
        Action unsigned = () => np.power(np.array(new byte[] { 1, 2 }), -1);
        unsigned.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
        Action signed = () => np.power(np.array(new sbyte[] { 1, 2 }), -1);
        signed.Should().Throw<ArgumentException>().WithMessage("Integers to negative integer powers are not allowed.");
    }

    /// <summary>A bool array meets a Python int in the int64 loop: past int64 the pack fails with the LP64 text.</summary>
    [TestMethod]
    public void BoolArray_PythonIntPastInt64_RaisesTheLp64OverflowError()
    {
        var b = np.array(new[] { true, false });
        Action add = () => _ = b + 9223372036854775808UL;
        add.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
    }

    /// <summary>An int64 / uint64 literal past 2^53 packs into a float32 loop through the double: two roundings.</summary>
    [TestMethod]
    public void Float32Loop_BigInt_RoundsThroughTheDouble()
    {
        // NumPy: np.zeros(1, np.float32) + (2**60 + 2**36 + 1) is 2^60 exactly (the array cast would give 2^60 + 2^37);
        // the same for np.maximum, np.clip and a uint64 literal (2**63 + 2**39 + 1 packs as 2^63).
        long v = (1L << 60) + (1L << 36) + 1;
        var z = np.array(new[] { 0f });
        (z + v).ToArray<float>()[0].Should().Be((float)(1L << 60));
        np.maximum(z, v).ToArray<float>()[0].Should().Be((float)(1L << 60));
        np.clip(z, v, null).ToArray<float>()[0].Should().Be((float)(1L << 60));
        // arctan2's float-tier route packs the same way: the result equals arctan2 against an explicit float32 2^60.
        np.arctan2(np.array(new[] { 1f }), v).ToArray<float>()[0]
            .Should().Be(np.arctan2(np.array(new[] { 1f }), np.array(new[] { (float)(1L << 60) })).ToArray<float>()[0]);
        (z + ((1UL << 63) + (1UL << 39) + 1)).ToArray<float>()[0].Should().Be((float)(1UL << 63));
    }
}
