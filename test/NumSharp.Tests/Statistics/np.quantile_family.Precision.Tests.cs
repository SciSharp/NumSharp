using System;
using System.Linq;

namespace NumSharp.Tests.Statistics;

/// <summary>
///     NumPy 2.4.2's quantile index arithmetic in the precision of q's dtype, the lerp's dtype, the <c>out=</c> routes and
///     the nan forms' result objects: every expected value, bit pattern and text probed against <c>numpy==2.4.2</c>. The
///     oracle section <c>qprec</c> gates the whole matrix; these pin the rules a reader of <c>QuantileIndexes</c> and
///     <c>QuantileEngine</c> needs first.
/// </summary>
[TestClass]
public class np_quantile_family_Precision_Tests
{
    /// <summary>The elements of an array as doubles in C order (float16 widened exactly).</summary>
    /// <param name="a">The array.</param>
    /// <returns>The values.</returns>
    private static double[] D(NDArray a)
    {
        using var dense = a.copy();
        return Enumerable.Range(0, (int)dense.size)
            .Select(i => dense.Storage.GetAtIndex(i) is Half h ? (double)h : Convert.ToDouble(dense.Storage.GetAtIndex(i)))
            .ToArray();
    }

    /// <summary>The bits of a double.</summary>
    /// <param name="x">The value.</param>
    /// <returns>Its IEEE pattern.</returns>
    private static ulong Bits(double x) => (ulong)BitConverter.DoubleToInt64Bits(x);

    /// <summary>A float16 q array (NumPy's <c>np.array(values, np.float16)</c>: each value rounded once).</summary>
    /// <param name="values">The quantiles.</param>
    /// <returns>The array.</returns>
    private static NDArray Q16(params double[] values)
    {
        using var q = np.array(values);
        return q.astype(np.float16);
    }

    /// <summary>
    ///     A float16 / float32 q computes the virtual index in its own dtype: <c>(n - 1) * q</c> with q = float16(0.999)
    ///     over 2053 elements is 2050 in float16 (the exact 2049.99609375 rounds to the float16 grid of step 2), so the
    ///     result is 2050.0 where the same q as float64 gives 2049.99609375; a float32 q of 0.3 over 1,000,001 elements
    ///     gives 300000.0 (float64: 300000.01192092896).
    /// </summary>
    [TestMethod]
    public void FloatQ_IndexArithmeticRoundsInQsDtype()
    {
        using var a = np.arange(2053.0);
        using var q16 = Q16(0.999);
        using var q16As64 = q16.astype(np.float64);
        using var r16 = np.quantile(a, q16);
        using var r64 = np.quantile(a, q16As64);
        D(r16).Should().Equal(2050.0);
        D(r64).Should().Equal(2049.99609375);

        using var big = np.arange(1_000_001.0);
        using var q32 = np.array(new[] { 0.3f });
        using var q32As64 = q32.astype(np.float64);
        using var s32 = np.quantile(big, q32);
        using var s64 = np.quantile(big, q32As64);
        D(s32).Should().Equal(300000.0);
        D(s64).Should().Equal(300000.01192092896);
    }

    /// <summary>
    ///     The rounding can carry an index past the line: over 2052 elements <c>n - 1 = 2051</c> is 2052 in float16, so
    ///     'lower' partitions at 2052 and raises the partition's <c>kth(=2052) out of bounds (2052)</c>, while 'linear'
    ///     clips the index (<c>vi &gt;= n - 1</c>) and answers 2051.0. From 65520 elements on <c>n - 1</c> is inf in float16:
    ///     <c>inf * 0.5 - floor(inf)</c> is NaN, and the result is x86's default NaN (negative, <c>0xFFF8...</c>).
    /// </summary>
    [TestMethod]
    public void FloatQ_RoundedIndexPastTheEnd()
    {
        using var a = np.arange(2052.0);
        using var one = Q16(1.0);
        Action lower = () => np.quantile(a, one, method: "lower");
        lower.Should().Throw<ValueError>().WithMessage("kth(=2052) out of bounds (2052)");
        using var linear = np.quantile(a, one);
        D(linear).Should().Equal(2051.0);

        using var huge = np.arange(70000.0);
        using var half = Q16(0.5);
        using var nan = np.quantile(huge, half);
        Bits(D(nan)[0]).Should().Be(0xFFF8000000000000UL);
    }

    /// <summary>
    ///     np.percentile divides <c>q / 100</c> in q's dtype first: float16(33.3) / 100 is 0.333 in float16, so the 11-element
    ///     percentile is 3.330078125 (float64 q: 3.33125); and the range check reads the quotient's dtype too, so a float16
    ///     q of 100.04 (stored as 100.0625) raises <c>Percentiles must be in the range [0, 100]</c>.
    /// </summary>
    [TestMethod]
    public void Percentile_DividesInQsDtype()
    {
        using var a = np.arange(11.0);
        using var p16 = Q16(33.3);
        using var p64 = p16.astype(np.float64);
        using var r16 = np.percentile(a, p16);
        using var r64 = np.percentile(a, p64);
        D(r16).Should().Equal(3.330078125);
        D(r64).Should().Equal(3.33125);

        using var over = Q16(100.04);
        Action range = () => np.percentile(a, over);
        range.Should().Throw<ValueError>().WithMessage("Percentiles must be in the range [0, 100]");
    }

    /// <summary>
    ///     The lerp runs in <c>result_type(data, q)</c> for a q array: a float16 q over float32 data gives float32, over
    ///     int32 data float64 (the weight rounded to float16 either way: 1.2001953125), and a float32 q over float16 data
    ///     float32 (1.2000000476837158).
    /// </summary>
    [TestMethod]
    public void LerpDtype_IsResultTypeOfDataAndQ()
    {
        using var f32 = np.arange(5).astype(np.float32);
        using var i32 = np.arange(5).astype(np.int32);
        using var f16 = np.arange(5).astype(np.float16);
        using var q16 = Q16(0.3);
        using var q32 = np.array(new[] { 0.3f });
        using var r1 = np.quantile(f32, q16);
        using var r2 = np.quantile(i32, q16);
        using var r3 = np.quantile(f16, q32);
        r1.dtype.Should().Be(np.float32);
        D(r1).Should().Equal(1.2001953125);
        r2.dtype.Should().Be(np.float64);
        D(r2).Should().Equal(1.2001953125);
        r3.dtype.Should().Be(np.float32);
        D(r3).Should().Equal(1.2000000476837158);
    }

    /// <summary>
    ///     An out of another float dtype receives the lerp computed in ITS dtype (NumPy's
    ///     <c>subtract(b, diff * (1 - t), out=out, dtype=type(out.dtype))</c> for t &gt;= 0.5), which is not the float64
    ///     answer cast: <c>0x40B0F51A</c> against <c>0x40B0F51B</c> here. The take path casts unsafely into an integer out
    ///     (300 into int8 is 44) but refuses a NaN there (NumPy's <c>copyto(..., casting='same_kind')</c> text); a wrong
    ///     shape or a read-only out raises NumPy's texts.
    /// </summary>
    [TestMethod]
    public void Out_ReceivesTheLerpInItsOwnDtype()
    {
        using var data = np.array(new[] { 6.962, 2.927, 0.015, 9.735, 2.984 });
        using var q = np.array(new[] { 0.66 });
        using var o32 = np.zeros(new Shape(1), np.float32);
        using var r = np.quantile(data, q, @out: o32);
        ReferenceEquals(r, o32).Should().BeTrue("the out is returned itself");
        BitConverter.SingleToUInt32Bits(o32.GetSingle(0)).Should().Be(0x40B0F51Au);
        using var f64 = np.quantile(data, q);
        BitConverter.SingleToUInt32Bits((float)f64.GetDouble(0)).Should().Be(0x40B0F51Bu, "the float64 answer cast differs");

        using var big = np.array(new[] { 1.0, 2.0, 300.0 });
        using var top = np.array(new[] { 1.0 });
        using var o8 = np.zeros(new Shape(1), np.int8);
        np.quantile(big, top, method: "lower", @out: o8);
        o8.GetSByte(0).Should().Be(44);

        using var withNaN = np.array(new[] { 1.0, double.NaN, 3.0 });
        using var mid = np.array(new[] { 0.5 });
        using var o64 = np.zeros(new Shape(1), np.int64);
        Action nanInto = () => np.quantile(withNaN, mid, method: "lower", @out: o64);
        nanInto.Should().Throw<TypeError>()
            .WithMessage("Cannot cast scalar from dtype('float64') to dtype('int64') according to the rule 'same_kind'");

        using var five = np.arange(5.0);
        using var three = np.array(new[] { 0.2, 0.5, 0.9 });
        using var two = np.zeros(new Shape(2), np.float32);
        Action shape = () => np.quantile(five, three, @out: two);
        shape.Should().Throw<ValueError>().WithMessage("operands could not be broadcast together with shapes (3,) (3,) (2,) ");
        using var ro = np.zeros(new Shape(1));
        ro.Shape.IsWriteable.Should().BeTrue();
        ro.setflags(write: false);
        Action readOnly = () => np.quantile(five, mid, @out: ro);
        readOnly.Should().Throw<ValueError>().WithMessage("output array is read-only");
    }

    /// <summary>
    ///     An integer q array computes its indexes in its integer dtype: weibull over float data lerps with an integer
    ///     weight in float64 ([0, 9]) and over int32 data in int64; 'linear' takes (int16 data stays int16). The integer
    ///     arithmetic follows NumPy: <c>n = 300</c> does not fit a uint8 q (<c>Python integer 300 out of bounds for
    ///     uint8</c>), weibull's clipped index -1 cannot be stored in one (<c>Python integer -1 out of bounds for uint8</c>),
    ///     a uint64 'linear' index cannot partition (<c>Partition index must be integer</c>), and a bool q computes in int64.
    /// </summary>
    [TestMethod]
    public void IntegerQ_ComputesInItsIntegerDtype()
    {
        using var ten = np.arange(10.0);
        using var qi = np.array(new long[] { 0, 1 });
        using var w = np.quantile(ten, qi, method: "weibull");
        w.dtype.Should().Be(np.float64);
        D(w).Should().Equal(0.0, 9.0);
        using var ten32 = np.arange(10).astype(np.int32);
        using var wi = np.quantile(ten32, qi, method: "weibull");
        wi.dtype.Should().Be(np.int64);
        D(wi).Should().Equal(0.0, 9.0);
        using var ten16 = np.arange(10).astype(np.int16);
        using var qi8 = np.array(new sbyte[] { 0, 1 });
        using var lin = np.quantile(ten16, qi8);
        lin.dtype.Should().Be(np.int16);
        D(lin).Should().Equal(0.0, 9.0);

        using var qu8 = np.array(new byte[] { 0, 1 });
        using var n300 = np.arange(300.0);
        Action overflowN = () => np.quantile(n300, qu8, method: "inverted_cdf");
        overflowN.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for uint8");
        using var n255 = np.arange(255.0);
        Action overflowMinusOne = () => np.quantile(n255, qu8, method: "weibull");
        overflowMinusOne.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");

        using var five = np.arange(5.0);
        using var qu64 = np.array(new ulong[] { 0, 1 });
        Action uint64Linear = () => np.quantile(five, qu64);
        uint64Linear.Should().Throw<TypeError>().WithMessage("Partition index must be integer");
        using var lower = np.quantile(five, qu64, method: "lower");
        D(lower).Should().Equal(0.0, 4.0);
        using var qb = np.array(new[] { false, true });
        using var mp = np.quantile(five, qb, method: "midpoint");
        D(mp).Should().Equal(0.0, 4.0);
    }

    /// <summary>
    ///     uint64 data past 2^63 widens to float64 rounded ONCE, as NumPy's C cast does: 2^63 + 1025 is 2^63 + 2048
    ///     (<c>0x43E0000000000001</c>), so the median of two such values is that, and the 0.25 quantile of 2^63 + 1025 and
    ///     2^63 + 5121 is <c>0x43E0000000000002</c>. .NET 8's own conversion rounds twice there (signed, then + 2^64) and
    ///     gave 2^63 for both.
    /// </summary>
    [TestMethod]
    public void UInt64Data_WidensRoundedOnce()
    {
        const ulong x = 9223372036854776833UL;   // 2^63 + 1025
        using var same = np.array(new[] { x, x });
        using var median = np.quantile(same, 0.5);
        Bits(D(median)[0]).Should().Be(0x43E0000000000001UL);
        using var pair = np.array(new[] { x, 9223372036854780929UL });   // 2^63 + 5121
        using var q = np.array(new[] { 0.25 });
        using var quarter = np.quantile(pair, q);
        Bits(D(quarter)[0]).Should().Be(0x43E0000000000002UL);
    }

    /// <summary>
    ///     An empty line raises NumPy's indexing error of its first access: <c>index -1 is out of bounds for axis 0 with
    ///     size 0</c> for float data (the NaN check reads the last element), <c>index 0 ...</c> for integer data.
    /// </summary>
    [TestMethod]
    public void EmptyData_RaisesNumPysIndexError()
    {
        using var empty = np.zeros(new Shape(0));
        Action interp = () => np.quantile(empty, 0.5);
        interp.Should().Throw<IndexOutOfRangeException>().WithMessage("index -1 is out of bounds for axis 0 with size 0");
        Action take = () => np.quantile(empty, 0.5, method: "lower");
        take.Should().Throw<IndexOutOfRangeException>().WithMessage("index -1 is out of bounds for axis 0 with size 0");
        using var emptyInt = np.zeros(new Shape(0), np.int64);
        Action intInterp = () => np.quantile(emptyInt, 0.5);
        intInterp.Should().Throw<IndexOutOfRangeException>().WithMessage("index 0 is out of bounds for axis 0 with size 0");
    }

    /// <summary>
    ///     The nan forms store a 0-d result into a 0-d out as NumPy's <c>out[...] = result</c> stores a NumPy SCALAR
    ///     (<c>PyArray_Pack</c>). A signed integer out converts it with <c>int()</c>: 2.5 into int64 is 2, a NaN (an all-NaN
    ///     slice, or <c>inf - inf * 0</c> from the lerp) raises <c>cannot convert float NaN to integer</c>, 400 into int8
    ///     raises <c>Python integer 400 out of bounds for int8</c>. An unsigned out CASTS it instead: a NaN is 0 in uint8 and
    ///     2^63 in uint64, 400 is 144 in uint8 and -5 is 251; a bool out takes the truth; a float32 out the float64 value
    ///     rounded once; a read-only out refuses before anything converts.
    /// </summary>
    [TestMethod]
    public void NanForms_ZeroDimOut_IsAScalarStore()
    {
        using var a = np.array(new[] { 1.0, double.NaN, 4.0 });
        using var o = np.zeros(new Shape(), np.int64);
        np.nanquantile(a, 0.5, @out: o);
        o.GetInt64().Should().Be(2);

        using var allNaN = np.array(new[] { double.NaN, double.NaN });
        using var o2 = np.zeros(new Shape(), np.int64);
        Action nan = () => np.nanquantile(allNaN, 0.5, @out: o2);
        nan.Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
        using var inf = np.array(new[] { double.PositiveInfinity, 1.0 });
        Action infNaN = () => np.nanquantile(inf, 1.0, @out: o2);
        infNaN.Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");

        using var u8 = np.zeros(new Shape(), np.uint8);
        np.nanquantile(allNaN, 0.5, @out: u8);
        u8.GetByte().Should().Be(0);
        using var u64 = np.zeros(new Shape(), np.uint64);
        np.nanquantile(allNaN, 0.5, @out: u64);
        u64.GetUInt64().Should().Be(9223372036854775808UL);
        using var flag = np.zeros(new Shape(), np.@bool);
        np.nanquantile(allNaN, 0.5, @out: flag);
        flag.GetBoolean().Should().BeTrue();

        using var big = np.array(new[] { 300.0, double.NaN, 500.0 });
        using var i8 = np.zeros(new Shape(), np.int8);
        Action range = () => np.nanquantile(big, 0.5, @out: i8);
        range.Should().Throw<OverflowException>().WithMessage("Python integer 400 out of bounds for int8");
        np.nanquantile(big, 0.5, @out: u8);
        u8.GetByte().Should().Be(144);
        using var negative = np.array(new[] { -7.5, double.NaN, -2.5 });
        np.nanquantile(negative, 0.5, @out: u8);
        u8.GetByte().Should().Be(251);

        using var b = np.array(new[] { 0.1, double.NaN, 0.7 });
        using var o32 = np.zeros(new Shape(), np.float32);
        np.nanquantile(b, 0.3, @out: o32);
        BitConverter.SingleToUInt32Bits(o32.GetSingle()).Should().Be(0x3E8F5C29u);

        using var ro = np.zeros(new Shape(), np.int8);
        ro.setflags(write: false);
        Action readOnly = () => np.nanquantile(allNaN, 0.5, @out: ro);
        readOnly.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
    }

    /// <summary>
    ///     The nan forms along an axis return <c>np.apply_along_axis</c>'s object: a view of its buffer (OWNDATA false),
    ///     F-ordered for an array q (the q axis moved to the front of a C buffer); an all-NaN slice is Python's
    ///     <c>np.nan</c> (positive, <c>0x7FF8...</c>), where the plain form copies the data's own NaN (.NET's
    ///     <see cref="double.NaN"/> is NumPy's <c>-nan</c>, <c>0xFFF8...</c>).
    /// </summary>
    [TestMethod]
    public void NanForms_AlongAnAxis_ReturnApplyAlongAxisLayout()
    {
        using var m = np.arange(12.0).reshape(3, 4);
        m[1, 2] = double.NaN;
        using var q = np.array(new[] { 0.25, 0.75 });
        using var r1 = np.nanquantile(m, q, axis: 1);
        D(r1).Should().Equal(0.75, 4.5, 8.75, 2.25, 6.0, 10.25);
        r1.Shape.IsContiguous.Should().BeFalse();
        r1.Shape.IsFContiguous.Should().BeTrue();
        r1.flags.owndata.Should().BeFalse();
        using var r0 = np.nanquantile(m, q, axis: 0);
        D(r0).Should().Equal(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        r0.Shape.IsFContiguous.Should().BeTrue();
        r0.flags.owndata.Should().BeFalse();
        using var s = np.nanquantile(m, 0.5, axis: 1);
        D(s).Should().Equal(1.5, 5.0, 9.5);
        s.flags.owndata.Should().BeFalse();

        using var slices = np.array(new[,] { { double.NaN, double.NaN }, { 1.0, 2.0 } });
        using var nanForm = np.nanquantile(slices, 0.5, axis: 1);
        Bits(D(nanForm)[0]).Should().Be(0x7FF8000000000000UL);
        using var plainData = np.array(new[,] { { 1.0, double.NaN }, { 1.0, 2.0 } });
        using var plain = np.quantile(plainData, 0.5, axis: 1);
        Bits(D(plain)[0]).Should().Be(0xFFF8000000000000UL);
        D(plain)[1].Should().Be(1.5);
    }

    /// <summary>
    ///     Each slice of a nan form plans with its own NaN-free length (the kernel reuses a plan only while that length
    ///     repeats): rows with 1, 0, 3 and 0 NaNs under a float16 q give NumPy's twelve values, and the row whose length
    ///     makes the float16 index reach the end raises <c>kth(=2052) out of bounds (2052)</c> in either row order.
    /// </summary>
    [TestMethod]
    public void NanForms_SlicesOfDifferentLengths_PlanEach()
    {
        using var r = np.arange(40.0).reshape(4, 10);
        r[0, 3] = double.NaN;
        r[2, 1] = double.NaN;
        r[2, 5] = double.NaN;
        r[2, 7] = double.NaN;
        using var q = Q16(0.1, 0.5, 0.9);
        using var res = np.nanquantile(r, q, axis: 1);
        D(res).Should().Equal(0.7998046875, 10.89990234375, 21.19921875, 30.89990234375,
            5.0, 14.5, 24.0, 34.5,
            8.19921875, 18.1015625, 28.3984375, 38.1015625);

        using var line = np.arange(2052.0);
        using var big = np.stack(new[] { line, line });
        big[0, 5] = double.NaN;
        using var one = Q16(1.0);
        Action secondRow = () => np.nanquantile(big, one, axis: 1, method: "lower");
        secondRow.Should().Throw<ValueError>().WithMessage("kth(=2052) out of bounds (2052)");
        using var flipped = big["::-1"].copy();
        Action firstRow = () => np.nanquantile(flipped, one, axis: 1, method: "lower");
        firstRow.Should().Throw<ValueError>().WithMessage("kth(=2052) out of bounds (2052)");
    }

    /// <summary>
    ///     q = 0 and 1 place the line's ends with NumPy's linear passes (introselect_'s dumb_select and its float maximum
    ///     pass), which keep NumPy's choice among ties: the FIRST of several equal minimums, so a line starting with
    ///     <c>-0.0</c> answers <c>-0.0</c> and one starting with <c>+0.0</c> answers <c>+0.0</c>; an integer q's weibull
    ///     and midpoint (kth 0, 1 and n - 1) answer the minimum and maximum.
    /// </summary>
    [TestMethod]
    public void EndQuantiles_KeepNumPysTieChoice()
    {
        var negFirst = Enumerable.Range(0, 20).Select(i => i % 2 == 0 ? -0.0 : 0.0).Append(1.0).ToArray();
        var posFirst = Enumerable.Range(0, 20).Select(i => i % 2 == 0 ? 0.0 : -0.0).Append(1.0).ToArray();
        using var neg = np.array(negFirst);
        using var pos = np.array(posFirst);
        using var zero = np.array(new[] { 0.0 });
        using var rn = np.quantile(neg, zero, method: "lower");
        using var rp = np.quantile(pos, zero, method: "lower");
        Bits(D(rn)[0]).Should().Be(0x8000000000000000UL);
        Bits(D(rp)[0]).Should().Be(0UL);

        using var a = np.arange(100_000.0)["::-1"].copy();
        using var qi = np.array(new[] { 0, 1 });
        using var mid = np.quantile(a, qi, method: "midpoint");
        D(mid).Should().Equal(0.0, 99999.0);
    }
}
