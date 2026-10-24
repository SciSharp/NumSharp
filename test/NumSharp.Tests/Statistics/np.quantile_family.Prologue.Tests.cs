using System;
using NumSharp.Backends;
using NumSharp.Statistics;

namespace NumSharp.Tests.Statistics;

/// <summary>
///     NumPy's <c>_ureduce</c> prologue of the quantile family — <c>np.median</c>, <c>np.nanmedian</c>,
///     <c>np.quantile</c>, <c>np.percentile</c>, <c>np.nanquantile</c>, <c>np.nanpercentile</c> — runs before the reduction
///     function's checks: the axes, keepdims' <c>out[(Ellipsis,) + index_out]</c>, then the merge of several reduced axes
///     (<c>reshape(kept + (-1,))</c>, which cannot resolve its -1 beside a kept axis of length 0). Each route of the family
///     had its own copy of it until the np-layer dtype plan's P1a step 2 (<see cref="QuantileEngine.UReduce"/>), and the
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
    ///     <see cref="QuantileEngine.UReduce"/> hands the reduction what NumPy's hands it: the caller's data for no axis or
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
            UReduced none = QuantileEngine.UReduce(b, null, null, false);
            ReferenceEquals(none.Data, b).Should().BeTrue();
            none.Axis.Should().BeNull();
            none.Reduced.Should().BeNull();
            UReduced one = QuantileEngine.UReduce(b, new[] { -2 }, null, false);
            ReferenceEquals(one.Data, b).Should().BeTrue();
            one.Axis.Should().Be(1);
            one.Reduced.Should().Equal(1);
            UReduced two = QuantileEngine.UReduce(b, new[] { 2, 0 }, null, false);
            two.Reduced.Should().Equal(2, 0);
            two.Data.shape.Should().Equal(3L, 8L);
            two.Axis.Should().Be(1);
            // Row k of the merged data is b[:, k, :] in C order: the reduced axes keep their original order.
            two.Data[Slice.Index(1)].ToArray<double>().Should().Equal(4, 5, 6, 7, 16, 17, 18, 19);
            var outArray = np.zeros(new Shape(1, 3, 1));
            UReduced kept = QuantileEngine.UReduce(b, new[] { 2, 0 }, outArray, true);
            kept.Target.shape.Should().Equal(3L);
        }
        using var q = np.quantile(b, new[] { 0.0, 1.0 }, new[] { 2, 0 });
        q.shape.Should().Equal(2L, 3L);
        q.ToArray<double>().Should().Equal(0, 4, 8, 15, 19, 23);
    }
}
