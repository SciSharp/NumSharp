using System;

namespace NumSharp.Tests.Logic;

/// <summary>
///     NEP 50's weak operand is the Python scalar alone (parity QA-05): in C# the 0-d array a number literal becomes through
///     the implicit conversion (<c>NDArray.WasPythonLiteral</c>). Every other 0-d operand — an element read <c>a[i]</c>, a
///     reduction result, <c>np.array(x)</c>, <c>NDArray.Scalar(x)</c>, a <see cref="Half"/> / <c>char</c> value — is a
///     NumPy scalar or 0-d array, STRONG in NumPy 2, and promotes like an array. NumSharp used to read any 0-d operand as
///     a Python scalar, so <c>uint8 + int64[1]</c> stayed uint8 and wrapped. Every expected value was probed against
///     NumPy 2.4.2.
/// </summary>
[TestClass]
public class Nep50StrongZeroDimOperandTests
{
    /// <summary>An element read is NumPy's <c>np.int64</c> scalar: <c>u8 + i64[1]</c> is int64 <c>[201, 202, 450]</c>, where
    ///     the weak reading kept uint8 and wrapped the last element to 194 (the QA-05 pin).</summary>
    [TestMethod]
    public void ElementRead_IsStrong_NoSilentWrap()
    {
        using var u8 = np.array(new byte[] { 1, 2, 250 });
        using var i64 = np.array(new long[] { 100, 200, 300 });
        using var element = i64[1];

        using var sum = u8 + element;

        sum.dtype.Should().Be(np.int64);
        sum.ToArray<long>().Should().Equal(201L, 202L, 450L);
    }

    /// <summary>A reduction result is a NumPy scalar of the reduction's dtype: <c>f16 * f32.mean()</c> is float32,
    ///     <c>f32 + f64.max()</c> float64, <c>np.add(i8, i64.sum())</c> int64 (the weak reading raised NumPy's
    ///     OverflowError for the last one, as if the sum were a Python int).</summary>
    [TestMethod]
    public void ReductionResults_AreStrong()
    {
        using var f16 = np.array(new[] { 1.0, 2.5, 3.0 }).astype(np.float16);
        using var f32 = np.array(new[] { 1.5f, 2.0f, 0.25f });
        using var mean = f32.mean();
        using var product = f16 * mean;
        product.dtype.Should().Be(np.float32);
        product.ToArray<float>().Should().Equal(1.25f, 3.125f, 3.75f);

        using var small = np.array(new[] { 0.1f, 0.2f });
        using var f64 = np.array(new[] { 1.0, 2.0 });
        using var max = f64.max();
        using var widened = small + max;
        widened.dtype.Should().Be(np.float64);
        widened.ToArray<double>().Should().Equal(2.100000001490116, 2.2000000029802322);

        using var i8 = np.array(new sbyte[] { 1, -2, 3 });
        using var i64 = np.array(new long[] { 100, 200, 300 });
        using var total = i64.sum();
        using var added = np.add(i8, total);
        added.dtype.Should().Be(np.int64);
        added.ToArray<long>().Should().Equal(601L, 598L, 603L);
    }

    /// <summary>An explicit 0-d array is strong in every binary ufunc: shift, multiply, maximum, floor_divide, power and
    ///     true division promote with it (the weak reading gave int8 zeros for the shift, NumPy's OverflowError for the
    ///     product, and float32 for the division).</summary>
    [TestMethod]
    public void ExplicitZeroDimArrays_AreStrong()
    {
        using var i8 = np.array(new sbyte[] { 1, -2, 3 });
        using var nine = np.array(9L);
        using var shifted = np.left_shift(i8, nine);
        shifted.dtype.Should().Be(np.int64);
        shifted.ToArray<long>().Should().Equal(512L, -1024L, 1536L);

        using var thousand = np.array(1000L);
        using var product = i8 * thousand;
        product.dtype.Should().Be(np.int64);
        product.ToArray<long>().Should().Equal(1000L, -2000L, 3000L);

        using var u8 = np.array(new byte[] { 1, 2 });
        using var threeHundred = np.array(300L);
        using var maximum = np.maximum(u8, threeHundred);
        maximum.dtype.Should().Be(np.int64);
        maximum.ToArray<long>().Should().Equal(300L, 300L);

        using var odd = np.array(new sbyte[] { 7, 9 });
        using var two = np.array(2L);
        using var quotient = np.floor_divide(odd, two);
        quotient.dtype.Should().Be(np.int64);
        quotient.ToArray<long>().Should().Equal(3L, 4L);

        using var bases = np.array(new sbyte[] { 2, 3 });
        using var squared = np.power(bases, two);
        squared.dtype.Should().Be(np.int64);
        squared.ToArray<long>().Should().Equal(4L, 9L);

        using var f32 = np.array(new[] { 1f, 2f });
        using var three = np.array(3L);
        using var divided = f32 / three;
        divided.dtype.Should().Be(np.float64);
        divided.ToArray<double>().Should().Equal(1.0 / 3.0, 2.0 / 3.0);
    }

    /// <summary>NumPy scalars of the boundary map and <c>NDArray.Scalar</c> values are strong: <c>i8 + np.float16(0.5)</c> is
    ///     float16, <c>i8 - np.int16(1)</c> int16, <c>bool + np.int32(2)</c> int32.</summary>
    [TestMethod]
    public void NumPyScalarValues_AreStrong()
    {
        using var i8 = np.array(new sbyte[] { 1, 2 });
        using var half = i8 + (Half)0.5;
        half.dtype.Should().Be(np.float16);
        half.ToArray<Half>().Should().Equal((Half)1.5, (Half)2.5);

        using var one = NDArray.Scalar((short)1);
        using var diff = i8 - one;
        diff.dtype.Should().Be(np.int16);

        using var flags = np.array(new[] { true, false });
        using var two = NDArray.Scalar(2);
        using var counted = flags + two;
        counted.dtype.Should().Be(np.int32);
    }

    /// <summary>np.clip: a strong bound is neither dropped nor packed and promotes (<c>np.clip(i8, i64[0], i64[1])</c> is
    ///     int64 <c>[100, 100, 100]</c>); literal bounds keep NumPy's Python-int rules (<c>np.clip(i8, -1000, 1000)</c> is
    ///     the unchanged int8).</summary>
    [TestMethod]
    public void Clip_StrongBoundsPromote_LiteralBoundsStayWeak()
    {
        using var i8 = np.array(new sbyte[] { 1, -2, 3 });
        using var i64 = np.array(new long[] { 100, 200, 300 });
        using var lo = i64[0];
        using var hi = i64[1];

        using var strong = np.clip(i8, lo, hi);
        strong.dtype.Should().Be(np.int64);
        strong.ToArray<long>().Should().Equal(100L, 100L, 100L);

        using var weak = np.clip(i8, -1000, 1000);
        weak.dtype.Should().Be(np.int8);
        weak.ToArray<sbyte>().Should().Equal((sbyte)1, (sbyte)-2, (sbyte)3);
    }

    /// <summary>A C# literal stays the weak Python scalar: <c>u8 + 100</c> is uint8 (and wraps, as NumPy's does),
    ///     <c>i8 + 1000</c> raises NumPy's OverflowError, <c>f32 * 2.5</c> stays float32.</summary>
    [TestMethod]
    public void Literals_StayWeak()
    {
        using var u8 = np.array(new byte[] { 1, 2, 250 });
        using var wrapped = u8 + 100;
        wrapped.dtype.Should().Be(np.uint8);
        wrapped.ToArray<byte>().Should().Equal((byte)101, (byte)102, (byte)94);

        using var i8 = np.array(new sbyte[] { 1, -2, 3 });
        Action overflow = () => (i8 + 1000).Dispose();
        overflow.Should().Throw<OverflowError>().WithMessage("Python integer 1000 out of bounds for int8");

        using var f32 = np.array(new[] { 1f, 2f });
        using var scaled = f32 * 2.5;
        scaled.dtype.Should().Be(np.float32);
    }

    /// <summary>A literal against a strong 0-d operand is weak too: <c>np.uint8(10) + 5</c> is uint8 15,
    ///     <c>np.int8(1) + 1000</c> raises NumPy's OverflowError, <c>np.float32(2.5) * 1.5</c> is float32 3.75.</summary>
    [TestMethod]
    public void LiteralAgainstAStrongScalar_IsWeak()
    {
        using var u8 = NDArray.Scalar((byte)10);
        using var sum = u8 + 5;
        sum.dtype.Should().Be(np.uint8);
        sum.GetByte().Should().Be((byte)15);

        using var i8 = NDArray.Scalar((sbyte)1);
        Action overflow = () => (i8 + 1000).Dispose();
        overflow.Should().Throw<OverflowError>().WithMessage("Python integer 1000 out of bounds for int8");

        using var f32 = NDArray.Scalar(2.5f);
        using var product = f32 * 1.5;
        product.dtype.Should().Be(np.float32);
        product.GetSingle().Should().Be(3.75f);
    }

    /// <summary>The <c>object</c> operator overloads read a boxed C# number as the same weak literal the NDArray overloads
    ///     receive: <c>f32 + (object)2.5</c> is float32, <c>x &gt; (object)0.1</c> compares in float32, <c>u8 + (object)100</c>
    ///     stays uint8.</summary>
    [TestMethod]
    public void ObjectOperators_ReadABoxedNumberAsTheLiteral()
    {
        using var f32 = np.array(new[] { 1f, 2f });
        using var sum = f32 + (object)2.5;
        sum.dtype.Should().Be(np.float32);

        using var x = np.array(new[] { 0.1f, 0.5f });
        using var greater = x > (object)0.1;
        greater.ToArray<bool>().Should().Equal(false, true);

        using var u8 = np.array(new byte[] { 1, 250 });
        using var wrapped = u8 + (object)100;
        wrapped.dtype.Should().Be(np.uint8);
        wrapped.ToArray<byte>().Should().Equal((byte)101, (byte)94);
    }

    /// <summary>np.isclose keeps a Python scalar weak and promotes a NumPy one: against float32 <c>[0.1, 0.5]</c> the
    ///     literal 0.1 gives <c>[True, False]</c>, <c>np.array(0.1)</c> <c>[False, False]</c> (float64); an int literal is
    ///     <c>float(y)</c>, weak.</summary>
    [TestMethod]
    public void IsClose_LiteralWeak_ZeroDimArrayStrong()
    {
        using var x = np.array(new[] { 0.1f, 0.5f });
        np.isclose(x, 0.1, rtol: 0, atol: 0).ToArray<bool>().Should().Equal(true, false);
        using var strong = np.array(0.1);
        np.isclose(x, strong, rtol: 0, atol: 0).ToArray<bool>().Should().Equal(false, false);

        using var near = np.array(new[] { 5.0000001f, 6f });
        np.isclose(near, 5, rtol: 0, atol: 0).ToArray<bool>().Should().Equal(true, false);
    }
}
