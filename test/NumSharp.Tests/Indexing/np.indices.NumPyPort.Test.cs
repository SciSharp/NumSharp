using System;

namespace NumSharp.Tests.Indexing
{
    /// <summary>
    ///     np.indices / np.indices_sparse as NumPy's <c>numeric.py</c> statements: <c>res = empty((N,) + dimensions,
    ///     dtype=dtype)</c> before anything else (dense only), then one <c>arange(dim, dtype=dtype)</c> per axis, reshaped to
    ///     broadcast along the others. Every expected value, text and flag is NumPy 2.4.2's, probed on Windows and on Linux
    ///     (the same answers on both).
    /// </summary>
    [TestClass]
    public class np_indices_numpy_port_tests
    {
        /// <summary>
        ///     The dense grid is allocated first, so a negative dimension is the allocation's ValueError with NumPy's text
        ///     (no position suffix), and it comes before a text dtype's arange refusal.
        /// </summary>
        [TestMethod]
        public void Dense_NegativeDimension_IsTheAllocationsValueError()
        {
            Action negative = () => np.indices(new long[] { 2, -3 });
            negative.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
            Action negativeText = () => np.indices(new long[] { -1 }, np.dtype("U5"));
            negativeText.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");
        }

        /// <summary>
        ///     A bool arange allows at most two elements (NumPy's BOOL_fill), and each axis runs one, so a bool grid with a
        ///     dimension of 3 raises even when another dimension is 0; a (2, 2) grid is NumPy's [False, True] coordinates.
        /// </summary>
        [TestMethod]
        public void Bool_ArangePastTwoElements_RaisesItsTypeError()
        {
            const string text = "arange() is only supported for booleans when the result has at most length 2.";
            Action three = () => np.indices(new long[] { 3 }, np.bool_);
            three.Should().Throw<TypeError>().WithMessage(text);
            Action emptyGrid = () => np.indices(new long[] { 0, 3 }, np.bool_);
            emptyGrid.Should().Throw<TypeError>().WithMessage(text);
            Action sparse = () => np.indices_sparse(new long[] { 3, 2 }, np.bool_);
            sparse.Should().Throw<TypeError>().WithMessage(text);

            using var g = np.indices(new long[] { 2, 2 }, np.bool_);
            g.dtype.Should().Be(np.bool_);
            g.ToArray<bool>().Should().Equal(false, false, true, true, false, true, false, true);
        }

        /// <summary>
        ///     A text dtype has no arange fill: its arange raises whatever its length (a 0 included), dense and sparse alike;
        ///     an empty tuple runs no arange, so the dense result is NumPy's empty (0,) array of that descriptor.
        /// </summary>
        [TestMethod]
        public void TextDtype_ArangeRefuses_TheEmptyTupleIsAnEmptyTextArray()
        {
            Action zeroLength = () => np.indices(new long[] { 0 }, np.dtype("U5"));
            zeroLength.Should().Throw<TypeError>()
                .WithMessage("arange() not supported for inputs with DType <class 'numpy.dtypes.StrDType'>.");
            Action stringDType = () => np.indices(new long[] { 2, 3 }, np.dtypes.StringDType());
            stringDType.Should().Throw<TypeError>()
                .WithMessage("arange() not supported for inputs with DType <class 'numpy.dtypes.StringDType'>.");
            Action sparseNegative = () => np.indices_sparse(new long[] { -1 }, np.dtype("U5"));
            sparseNegative.Should().Throw<TypeError>()
                .WithMessage("arange() not supported for inputs with DType <class 'numpy.dtypes.StrDType'>.");

            using var u = np.indices(new long[0], np.dtype("U5"));
            u.dtype.str.Should().Be("<U5");
            string.Join(",", u.shape).Should().Be("0");
            using var t = np.indices(new long[0], np.dtypes.StringDType());
            t.dtype.ToString().Should().Be("StringDType()");
            string.Join(",", t.shape).Should().Be("0");
            np.indices_sparse(new long[0], np.dtype("U5")).Should().BeEmpty();
        }

        /// <summary>
        ///     The sparse grid has no allocation to refuse a negative dimension: its arange is empty and the reshape reads the
        ///     negative entry as the unknown dimension, so that axis is 0 long - (2, -3) gives a (2, 1) and a (1, 0) array.
        /// </summary>
        [TestMethod]
        public void Sparse_NegativeDimension_IsAnEmptyAxis()
        {
            var pair = np.indices_sparse(new long[] { 2, -3 });
            pair.Length.Should().Be(2);
            string.Join(",", pair[0].shape).Should().Be("2,1");
            pair[0].ToArray<long>().Should().Equal(0L, 1L);
            string.Join(",", pair[1].shape).Should().Be("1,0");
            var single = np.indices_sparse(new long[] { -1 });
            string.Join(",", single[0].shape).Should().Be("0");
            single[0].dtype.Should().Be(np.int64);
        }

        /// <summary>
        ///     Each sparse array is NumPy's reshape of its arange, a view: <c>flags.owndata</c> is False for every dtype (a
        ///     float64 grid was an owning cast of an int64 arange).
        /// </summary>
        [TestMethod]
        public void Sparse_ResultsAreViewsOfTheirAranges()
        {
            var floats = np.indices_sparse(new long[] { 2, 3 }, np.float64);
            floats[0].dtype.Should().Be(np.float64);
            floats[0].flags.owndata.Should().BeFalse();
            floats[1].flags.owndata.Should().BeFalse();
            floats[1].ToArray<double>().Should().Equal(0.0, 1.0, 2.0);
            var ints = np.indices_sparse(new long[] { 2, 3 });
            ints[0].flags.owndata.Should().BeFalse();
            ints[1].flags.owndata.Should().BeFalse();
        }

        /// <summary>
        ///     A grid of another dtype holds its arange's values: an int8 arange wraps past 127 (NumPy's BYTE_fill stores
        ///     <c>start + i*delta</c> truncated), a float16 one is each position rounded once.
        /// </summary>
        [TestMethod]
        public void OtherDtypes_HoldTheirArangesValues()
        {
            using var bytes = np.indices(new long[] { 200 }, np.int8);
            bytes.dtype.Should().Be(np.int8);
            bytes.flags.owndata.Should().BeTrue();
            var values = bytes.ToArray<sbyte>();
            values[127].Should().Be((sbyte)127);
            values[128].Should().Be((sbyte)-128);
            values[199].Should().Be((sbyte)-57);

            using var halves = np.indices(new long[] { 2, 3 }, np.float16);
            halves.ToArray<Half>().Should().Equal((Half)0, (Half)0, (Half)0, (Half)1, (Half)1, (Half)1,
                (Half)0, (Half)1, (Half)2, (Half)0, (Half)1, (Half)2);
        }
    }
}
