using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Statistics;

namespace NumSharp.Tests.Statistics;

/// <summary>
///     NumPy's <c>_ureduce</c> prologue of the quantile family — <c>np.median</c>, <c>np.nanmedian</c>,
///     <c>np.quantile</c>, <c>np.percentile</c>, <c>np.nanquantile</c>, <c>np.nanpercentile</c> — runs before the reduction
///     function's checks: the axes, keepdims' <c>out[(Ellipsis,) + index_out]</c>, then the merge of several reduced axes
///     (<c>reshape(kept + (-1,))</c>, which cannot resolve its -1 beside a kept axis of length 0). Each route of the family
///     had its own copy of it until the np-layer dtype plan's P1a step 2 (<see cref="DefaultEngine.UReduce"/>), and the
///     copies disagreed (the plan's Appendix B1). Every expected outcome and text was probed against <c>numpy==2.4.2</c>;
///     the oracle section <c>qprologue</c> gates the whole matrix.
/// </summary>
[TestClass]
public class np_quantile_family_Prologue_Tests
{
    /// <summary>NumPy's merge error for <c>(0, 3, 4)</c> reduced over two axes beside the kept empty one.</summary>
    private const string MergeText = "cannot reshape array of size 0 into shape (0,newaxis)";

    /// <summary>NumPy's IndexError for a keepdims out of one dimension under 2-D data.</summary>
    private const string TooManyIndices = "too many indices for array: array is 1-dimensional, but 2 were indexed";

    /// <summary>The <c>(0, 3, 4)</c> array of Appendix B1 in a dtype (text as its <c>U3</c> / <c>S3</c> / StringDType
    ///     conversion of zeros).</summary>
    /// <param name="dtype">The dtype string.</param>
    /// <returns>The empty array.</returns>
    private static NDArray EmptyB1(string dtype) => np.zeros(new Shape(0, 3, 4), np.dtype(dtype));

    /// <summary>A <c>(2, 3)</c> array of a dtype: <c>arange(6)</c> for numbers, one letter per element for text.</summary>
    /// <param name="dtype">The dtype string.</param>
    /// <returns>The array.</returns>
    private static NDArray Data23(string dtype)
    {
        if (dtype is "U3" or "S3" or "T")
        {
            using var text = np.array(new[,] { { "b", "a", "c" }, { "d", "e", "f" } });
            return text.astype(np.dtype(dtype));
        }
        using var basis = np.arange(6.0).reshape(2, 3);
        return basis.astype(np.dtype(dtype));
    }

    /// <summary>
    ///     Appendix B1 itself: <c>axis=(1, 2)</c> over <c>zeros((0, 3, 4))</c> is NumPy's reshape ValueError for numbers and
    ///     text alike, in every function that is not a nan form. NumSharp returned a <c>(0,)</c> float64 array for numeric
    ///     median / quantile and the closing mean's refusal for <c>U3</c> median, before the shared prologue.
    /// </summary>
    [TestMethod]
    public void B1_MergeOfAKeptEmptyAxis_IsNumPysReshapeError()
    {
        int[] axes = { 1, 2 };
        foreach (string dtype in new[] { "float64", "int32", "bool", "U3", "S3", "T" })
        {
            using var a = EmptyB1(dtype);
            Action median = () => np.median(a, axes);
            median.Should().Throw<IncorrectShapeException>().WithMessage(MergeText, dtype);
            Action quantile = () => np.quantile(a, 0.5, axes);
            quantile.Should().Throw<IncorrectShapeException>().WithMessage(MergeText, dtype);
            Action percentile = () => np.percentile(a, 50.0, axes);
            percentile.Should().Throw<IncorrectShapeException>().WithMessage(MergeText, dtype);
        }
    }

    /// <summary>
    ///     The empty tuple merges NOTHING into a trailing axis of length 1, so empty data fails the same reshape
    ///     (<c>(0,3,4,newaxis)</c>), while non-empty data reduces each element alone: <c>np.median(arange(6.).reshape(2, 3),
    ///     axis=())</c> is the data itself, float64 <c>(2, 3)</c>.
    /// </summary>
    [TestMethod]
    public void EmptyTuple_MergesIntoALengthOneAxis()
    {
        var none = Array.Empty<int>();
        using var z = EmptyB1("float64");
        Action median = () => np.median(z, none);
        median.Should().Throw<IncorrectShapeException>().WithMessage("cannot reshape array of size 0 into shape (0,3,4,newaxis)");
        Action quantile = () => np.quantile(z, 0.5, none);
        quantile.Should().Throw<IncorrectShapeException>().WithMessage("cannot reshape array of size 0 into shape (0,3,4,newaxis)");
        using var a = np.arange(6.0).reshape(2, 3);
        using var m = np.median(a, none);
        m.dtype.Should().Be(np.float64);
        m.shape.Should().Equal(2L, 3L);
        m.ToArray<double>().Should().Equal(0, 1, 2, 3, 4, 5);
    }

    /// <summary>
    ///     The merge comes before everything the reduction function checks: an unknown method, a 3-D q, an empty text q
    ///     (which passes NumPy's [0, 1] check), weights. Each of these is NumPy's reshape error over B1's data.
    /// </summary>
    [TestMethod]
    public void Merge_PrecedesTheMethodQsRankTextQAndWeights()
    {
        int[] axes = { 1, 2 };
        using var a = EmptyB1("float64");
        Action bogus = () => np.quantile(a, 0.5, axes, method: "bogus");
        bogus.Should().Throw<IncorrectShapeException>().WithMessage(MergeText);
        using var q3 = np.full(new Shape(1, 1, 1), 0.5);
        Action rank = () => np.quantile(a, q3, axes);
        rank.Should().Throw<IncorrectShapeException>().WithMessage(MergeText);
        using var textQ = np.zeros(new Shape(0), np.dtype("U1"));
        Action text = () => np.quantile(a, textQ, axes);
        text.Should().Throw<IncorrectShapeException>().WithMessage(MergeText);
        using var stringQ = np.zeros(new Shape(0), np.dtype("T"));
        Action vstring = () => np.quantile(a, stringQ, axes);
        vstring.Should().Throw<IncorrectShapeException>().WithMessage(MergeText);
        using var w = np.ones(new Shape(0, 3, 4));
        Action weighted = () => np.quantile(a, 0.5, axes, method: "inverted_cdf", weights: w);
        weighted.Should().Throw<IncorrectShapeException>().WithMessage(MergeText);
    }

    /// <summary>
    ///     The nan forms send EMPTY data to <c>np.nanmean</c> before the prologue: numbers give an empty float64 result
    ///     (shape <c>(0,)</c> over B1's data), fixed-width text the add.reduce refusal — never the merge's error.
    /// </summary>
    [TestMethod]
    public void NanForms_AnswerEmptyDataBeforeThePrologue()
    {
        int[] axes = { 1, 2 };
        using var a = EmptyB1("float64");
        using var nanmedian = np.nanmedian(a, axes);
        nanmedian.dtype.Should().Be(np.float64);
        nanmedian.shape.Should().Equal(0L);
        using var nanquantile = np.nanquantile(a, 0.5, axes);
        nanquantile.shape.Should().Equal(0L);
        using var u = EmptyB1("U3");
        Action text = () => np.nanmedian(u, axes);
        text.Should().Throw<TypeError>().WithMessage(
            "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U3'), dtype('<U3'), dtype('<U6'))");
    }

    /// <summary>
    ///     keepdims' out index is the second step of the prologue, before every check of the reduction function and before
    ///     the text refusals: a 1-D out under 2-D data is NumPy's IndexError for numbers, <c>U</c>, <c>S</c> and
    ///     StringDType, in all four functions with an out. NumSharp reported the text refusals (and numeric median's
    ///     result-then-copy shape error) first, before the shared prologue.
    /// </summary>
    [TestMethod]
    public void KeepdimsOutIndex_PrecedesTheReductionFunctionAndTheTextRefusals()
    {
        foreach (string dtype in new[] { "float64", "int32", "bool", "U3", "S3", "T" })
        {
            using var a = Data23(dtype);
            Action median = () => np.median(a, 0, np.zeros(new Shape(3)), keepdims: true);
            median.Should().Throw<IndexError>().WithMessage(TooManyIndices, dtype);
            Action nanmedian = () => np.nanmedian(a, 0, np.zeros(new Shape(3)), keepdims: true);
            nanmedian.Should().Throw<IndexError>().WithMessage(TooManyIndices, dtype);
            Action quantile = () => np.quantile(a, 0.5, 0, np.zeros(new Shape(3)), keepdims: true, method: "bogus");
            quantile.Should().Throw<IndexError>().WithMessage(TooManyIndices, dtype);
            Action nanquantile = () => np.nanquantile(a, new[] { 0.5 }, 0, np.zeros(new Shape(3)), keepdims: true);
            nanquantile.Should().Throw<IndexError>().WithMessage(TooManyIndices, dtype);
            using var textQ = np.zeros(new Shape(0), np.dtype("U1"));
            Action withTextQ = () => np.nanquantile(a, textQ, 0, np.zeros(new Shape(3)), keepdims: true);
            withTextQ.Should().Throw<IndexError>().WithMessage(TooManyIndices, dtype);
            using var w = np.ones(new Shape(2, 3));
            Action weighted = () => np.quantile(a, 0.5, 0, np.zeros(new Shape(3)), keepdims: true, method: "inverted_cdf",
                weights: w);
            weighted.Should().Throw<IndexError>().WithMessage(TooManyIndices, dtype);
        }
    }

    /// <summary>
    ///     keepdims' index takes every reduced axis at 0, so an out whose reduced axis is EMPTY is NumPy's IndexError
    ///     (<c>index 0 is out of bounds for axis 0 with size 0</c>), and a LARGER out passes it: the quantile then
    ///     broadcasts into the view (<c>np.quantile(arange(6.).reshape(2, 3), 0.5, axis=0, keepdims=True,
    ///     out=zeros((4, 1, 3)))</c> fills every row with <c>[1.5, 2.5, 3.5]</c>).
    /// </summary>
    [TestMethod]
    public void KeepdimsOutIndex_TakesEachReducedAxisAtZero()
    {
        foreach (string dtype in new[] { "float64", "U3", "T" })
        {
            using var a = Data23(dtype);
            Action median = () => np.median(a, 0, np.zeros(new Shape(0, 3)), keepdims: true);
            median.Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0", dtype);
            Action quantile = () => np.quantile(a, 0.5, 0, np.zeros(new Shape(0, 3)), keepdims: true);
            quantile.Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0", dtype);
        }
        using var f = Data23("float64");
        var larger = np.zeros(new Shape(4, 1, 3));
        using var r = np.quantile(f, 0.5, 0, larger, keepdims: true);
        ReferenceEquals(r, larger).Should().BeTrue("NumPy returns the out itself");
        larger.ToArray<double>().Should().Equal(1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 1.5, 2.5, 3.5);
    }

    /// <summary>
    ///     <see cref="DefaultEngine.UReduce"/> hands the reduction what NumPy's hands it: the caller's data for no axis or
    ///     one axis; for several axes the kept ones first and the reduced ones merged into one LAST axis in their original
    ///     order (<c>np.moveaxis(a, keep, range(nkeep)).reshape(kept + (-1,))</c>), the axis given in any order; under
    ///     keepdims a view of out with each reduced axis taken at 0. The quantiles over <c>axis=(2, 0)</c> of
    ///     <c>arange(24.).reshape(2, 3, 4)</c> at 0 and 1 are <c>[[0, 4, 8], [15, 19, 23]]</c> (NumPy).
    /// </summary>
    [TestMethod]
    public void UReduce_HandsTheReductionNumPysData()
    {
        using var b = np.arange(24.0).reshape(2, 3, 4);
        using (var scope = NDScope.Open())
        {
            UReduced none = DefaultEngine.UReduce(b, null, null, false);
            ReferenceEquals(none.Data, b).Should().BeTrue();
            none.Axis.Should().BeNull();
            none.Reduced.Should().BeNull();
            UReduced one = DefaultEngine.UReduce(b, new[] { -2 }, null, false);
            ReferenceEquals(one.Data, b).Should().BeTrue();
            one.Axis.Should().Be(1);
            one.Reduced.Should().Equal(1);
            UReduced two = DefaultEngine.UReduce(b, new[] { 2, 0 }, null, false);
            two.Reduced.Should().Equal(2, 0);
            two.Data.shape.Should().Equal(3L, 8L);
            two.Axis.Should().Be(1);
            // Row k of the merged data is b[:, k, :] in C order: the reduced axes keep their original order.
            two.Data[Slice.Index(1)].ToArray<double>().Should().Equal(4, 5, 6, 7, 16, 17, 18, 19);
            var outArray = np.zeros(new Shape(1, 3, 1));
            UReduced kept = DefaultEngine.UReduce(b, new[] { 2, 0 }, outArray, true);
            kept.Target.shape.Should().Equal(3L);
        }
        using var q = np.quantile(b, new[] { 0.0, 1.0 }, new[] { 2, 0 });
        q.shape.Should().Equal(2L, 3L);
        q.ToArray<double>().Should().Equal(0, 4, 8, 15, 19, 23);
    }

    /// <summary>
    ///     <see cref="UReduced.NumPyAxis"/> is the axis exactly as NumPy's <c>_ureduce</c> hands it on: null without an axis,
    ///     the normalized axis for one, and the UNnormalized <c>-1</c> for merged axes (the empty tuple included), where
    ///     <see cref="UReduced.Axis"/> is the merged axis' index.
    /// </summary>
    [TestMethod]
    public void UReduce_NumPyAxis_IsTheArgumentNumPysFunctionReceives()
    {
        using var b = np.arange(24.0).reshape(2, 3, 4);
        using var scope = NDScope.Open();
        DefaultEngine.UReduce(b, null, null, false).NumPyAxis.Should().BeNull();
        DefaultEngine.UReduce(b, new[] { -1 }, null, false).NumPyAxis.Should().Be(2);
        UReduced merged = DefaultEngine.UReduce(b, new[] { 1, 2 }, null, false);
        merged.NumPyAxis.Should().Be(-1);
        merged.Axis.Should().Be(1);
        DefaultEngine.UReduce(b, Array.Empty<int>(), null, false).NumPyAxis.Should().Be(-1);
    }

    /// <summary>
    ///     The nan forms with a q of two or more dimensions over merged axes: NumPy's <c>_nanquantile_ureduce_func</c> moves
    ///     q's axes with <c>from_ax = [axis + i for i in range(q.ndim)]</c> on <c>_ureduce</c>'s <c>-1</c>, so
    ///     <c>np.nanquantile(arange(24.).reshape(2, 3, 4), q of shape (3, 2), axis=(1, 2))</c> is <c>(2, 2, 3)</c> (q's last
    ///     axis, the kept axis, q's first), where np.quantile gives <c>(3, 2, 2)</c>; keepdims indexes that result, and an out
    ///     must take its shape (an out in q-first order is NumPy's broadcast error). StringDType data follows the same
    ///     statement. A 1-d q and one reduced axis are unaffected. NumSharp returned np.quantile's order until this fix.
    /// </summary>
    [TestMethod]
    public void NanForms_TwoDimensionalQ_OverMergedAxes_KeepNumPysAxisOrder()
    {
        using var a = np.arange(24.0).reshape(2, 3, 4);
        using var q2 = np.array(new[,] { { 0.0, 0.5 }, { 0.25, 1.0 }, { 0.75, 0.1 } });
        int[] axes = { 1, 2 };
        double[] expected = { 0.0, 2.75, 8.25, 12.0, 14.75, 20.25, 5.5, 11.0, 1.1, 17.5, 23.0, 13.1 };
        using (var r = np.nanquantile(a, q2, axes))
        {
            r.shape.Should().Equal(2L, 2L, 3L);
            r.ToArray<double>().Should().Equal(expected);
            r.flags.owndata.Should().BeFalse("NumPy returns a view of np.apply_along_axis's buffer");
        }
        using (var hundred = q2 * 100)
        using (var p = np.nanpercentile(a, hundred, axes))
        {
            p.shape.Should().Equal(2L, 2L, 3L);
            p.ToArray<double>().Should().Equal(expected);
        }
        using (var plain = np.quantile(a, q2, axes))
            plain.shape.Should().Equal(3L, 2L, 2L);
        using (var other = np.nanquantile(a, q2, new[] { 0, 2 }))
        {
            other.shape.Should().Equal(2L, 3L, 3L);
            other.ToArray<double>().Should().Equal(0.0, 1.75, 13.25, 4.0, 5.75, 17.25, 8.0, 9.75, 21.25, 7.5, 15.0,
                0.7000000000000001, 11.5, 19.0, 4.7, 15.5, 23.0, 8.7);
        }
        using (var oneD = np.nanquantile(a, new[] { 0.25, 0.75 }, axes))
            oneD.shape.Should().Equal(2L, 2L);
        using (var single = np.nanquantile(a, q2, 1))
            single.shape.Should().Equal(3L, 2L, 2L, 4L);
        using (var kd = np.nanquantile(a, q2, axes, keepdims: true))
            kd.shape.Should().Equal(2L, 2L, 3L, 1L, 1L);
        var moved = np.zeros(new Shape(2, 2, 3));
        using (var written = np.nanquantile(a, q2, axes, moved))
        {
            ReferenceEquals(written, moved).Should().BeTrue();
            moved.ToArray<double>().Should().Equal(expected);
        }
        using var qFirst = np.zeros(new Shape(3, 2, 2));
        Action refused = () => np.nanquantile(a, q2, axes, qFirst);
        refused.Should().Throw<ValueError>().WithMessage("could not broadcast input array from shape (2,2,3) into shape (3,2,2)");
        using var letters = np.array(Enumerable.Range(0, 24).Select(i => ((char)('a' + i)).ToString()).ToArray());
        using var shaped = letters.reshape(2, 3, 4);
        using var text = shaped.astype(np.dtype("T"));
        using var t = np.nanquantile(text, q2, axes, method: "lower");
        t.shape.Should().Equal(2L, 2L, 3L);
        NDArray.AsStringArray(t).Should().Equal("a", "c", "i", "m", "o", "u", "f", "l", "b", "r", "x", "n");
    }

    /// <summary>
    ///     The raw bits of every element of a float16 / float32 / float64 array, widened to a long: what the NaN tests
    ///     compare, because <c>double.NaN.Equals(double.NaN)</c> hides the sign and payload NumPy's answer has.
    /// </summary>
    /// <param name="r">The array (any layout).</param>
    /// <returns>The element bits in C order.</returns>
    /// <exception cref="NotSupportedException">Another dtype (a test bug: the nan forms answer float data only).</exception>
    private static long[] FloatBits(NDArray r) => r.typecode switch
    {
        NPTypeCode.Double => r.ToArray<double>().Select(BitConverter.DoubleToInt64Bits).ToArray(),
        NPTypeCode.Single => r.ToArray<float>().Select(f => (long)BitConverter.SingleToInt32Bits(f)).ToArray(),
        NPTypeCode.Half => r.ToArray<Half>().Select(h => (long)BitConverter.HalfToInt16Bits(h)).ToArray(),
        _ => throw new NotSupportedException(r.typecode.ToString()),
    };

    /// <summary>
    ///     The nan forms over data that is NaN everywhere: every slice is <c>_nanquantile_1d</c>'s
    ///     <c>np.full(q.shape, np.nan, dtype=arr1d.dtype)[()]</c> — Python's POSITIVE NaN (float64 <c>0x7ff8...</c>, float32
    ///     <c>0x7fc00000</c>, float16 <c>0x7e00</c>; .NET's <c>double.NaN</c> is the negative one), a view for a q array and a
    ///     scalar for a 0-d q over the flattened data, stacked and moved along an axis (merged axes with a 3-D q give
    ///     <c>(3, 2, 2, 1)</c> for <c>axis=(1, 2)</c> of a <c>(2, 3, 4)</c> array) — and NumPy reaches neither the rank check
    ///     of a 3-D q nor the method lookup. An out takes the answer through <c>out[...] = result</c>, which broadcasts: a
    ///     larger out takes it in every row, a q-first out along merged axes is the broadcast ValueError. Probed against
    ///     <c>numpy==2.4.2</c>; NumSharp filled .NET's negative NaN into owning q-first buffers and required an out of the
    ///     exact shape until this fix.
    /// </summary>
    [TestMethod]
    public void NanForms_AllNaNData_AnswerNumPysPositiveNaN_InNumPysLayout()
    {
        double numpyNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000L);
        int[] merged = { 1, 2 };
        foreach (var (dtype, bits) in new[] { ("float64", 0x7FF8000000000000L), ("float32", 0x7FC00000L), ("float16", 0x7E00L) })
        {
            using var basis = np.full(new Shape(2, 3, 4), numpyNaN);
            using var a = basis.astype(np.dtype(dtype));
            using var q1 = np.array(new[] { 0.25, 0.75 });
            using var q3 = np.full(new Shape(2, 1, 3), 0.5);

            // The flattened data: one slice, the q array's np.full(...)[()] view, or the scalar of a 0-d q.
            using (var flat = np.nanquantile(a, q1))
            {
                flat.dtype.Should().Be(np.dtype(dtype));
                flat.shape.Should().Equal(2L);
                flat.flags.owndata.Should().BeFalse("[()] of a 1-d array is a view ({0})", dtype);
                FloatBits(flat).Should().AllBeEquivalentTo(bits, dtype);
            }
            using (var scalar = np.nanquantile(a, 0.5))
            {
                scalar.ndim.Should().Be(0);
                scalar.flags.writeable.Should().BeFalse("a NumPy scalar ({0})", dtype);
                FloatBits(scalar).Should().Equal(new[] { bits }, dtype);
            }

            // Merged axes with a 3-D q: no rank check, NumPy's moved order, a view.
            using (var moved = np.nanquantile(a, q3, merged))
            {
                moved.shape.Should().Equal(new[] { 3L, 2L, 2L, 1L }, dtype);
                moved.flags.owndata.Should().BeFalse(dtype);
                FloatBits(moved).Should().AllBeEquivalentTo(bits, dtype);
            }

            // An unknown method is never looked up; a larger out takes the answer in every row (the same instance back).
            using (var bogus = np.nanquantile(a, q1, merged, method: "bogus"))
            {
                bogus.shape.Should().Equal(new[] { 2L, 2L }, dtype);
                FloatBits(bogus).Should().AllBeEquivalentTo(bits, dtype);
            }
            using var big = np.zeros(new Shape(2, 2, 2), np.dtype(dtype));
            using (var written = np.nanquantile(a, q1, merged, big, method: "bogus"))
            {
                ReferenceEquals(written, big).Should().BeTrue(dtype);
                FloatBits(big).Should().AllBeEquivalentTo(bits, dtype);
            }

            // A q-first out along merged axes: the moved (2, 2, 3) answer does not broadcast into (3, 2, 2).
            using var q2 = np.full(new Shape(3, 2), 0.5);
            using var qFirst = np.zeros(new Shape(3, 2, 2), np.dtype(dtype));
            Action refused = () => np.nanquantile(a, q2, merged, qFirst, method: "bogus");
            refused.Should().Throw<ValueError>()
                .WithMessage("could not broadcast input array from shape (2,2,3) into shape (3,2,2)", dtype);
        }
    }
}
