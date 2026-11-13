using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     Parity QA-31: an <c>out=</c> of the wrong shape is refused with the text of the NumPy 2.4.2 statement that refuses
    ///     it (scratchpad probes p168_numpy.py / p169_numpy.py against p170.cs: every refusal below equal). np.cumsum /
    ///     np.cumprod are <c>PyUFunc_Accumulate</c>: its outer ITERATOR (an input of more than one dimension, a loop dtype
    ///     other than an operand's, an unaligned operand, an out overlapping the input) raises NpyIter's texts, its direct
    ///     1-D path <c>provided out is the wrong size for the accumulation.</c>. np.ptp is <c>maximum.reduce(a, axis,
    ///     out=out)</c> first; np.median ends with <c>mean(part[indexer], out=out)</c> over the partition's middle slice;
    ///     np.nanmedian assigns <c>out[...] = result</c>; an empty input under the nan forms is <c>np.nanmean</c>.
    /// </summary>
    [TestClass]
    public class OutShapeRefusalsNumPyTests
    {
        /// <summary>The head of NpyIter's broadcast_error text with op_axes set.</summary>
        private const string Remapped = "operands could not be broadcast together with remapped shapes [original->remapped]: ";

        /// <summary>PyUFunc_Accumulate's direct-path text.</summary>
        private const string WrongSize = "provided out is the wrong size for the accumulation.";

        /// <summary>A fresh float64 zeros array of the given dimensions.</summary>
        /// <param name="dims">The dimensions; none gives a 0-d array.</param>
        /// <returns>The new array (the caller owns it).</returns>
        private static NDArray Z(params long[] dims) => np.zeros(new Shape(dims));

        /// <summary>A fresh zeros array of the given dtype and dimensions.</summary>
        /// <param name="dtype">The element type.</param>
        /// <param name="dims">The dimensions; none gives a 0-d array.</param>
        /// <returns>The new array (the caller owns it).</returns>
        private static NDArray ZT(NPTypeCode dtype, params long[] dims) => np.zeros(new Shape(dims), dtype);

        /// <summary>The 3 x 4 float64 operand of NumPy's probe: <c>np.arange(12.0).reshape(3, 4)</c>.</summary>
        /// <returns>A fresh array.</returns>
        private static NDArray A() => np.arange(12.0).reshape(3, 4);

        /// <summary>Along an axis of an N-D input the accumulate iterator refuses with its three texts (cumsum and cumprod
        ///     alike): the op_axes entry an out with fewer dimensions lacks, the remapped shapes, the non-broadcastable out
        ///     whose "remapped to" lists its extents reversed.</summary>
        [TestMethod]
        public void CumsumCumprod_AlongAnAxis_RaiseTheAccumulateIteratorsTexts()
        {
            using var a = A();
            (int Axis, long[] Out, string Message)[] rows =
            {
                (0, new long[] { 4, 3 }, Remapped + "(4,3)->(4,3) (3,4)->(3,4) "),
                (1, new long[] { 3, 5 }, Remapped + "(3,5)->(3,5) (3,4)->(3,4) "),
                (-1, new long[] { 2, 4 }, Remapped + "(2,4)->(2,4) (3,4)->(3,4) "),
                (0, new long[] { 1, 3, 4 }, Remapped + "(1,3,4)->(1,3) (3,4)->(3,4) "),
                (0, new long[] { 3 }, "Iterator input op_axes[0][0] (==1) is not a valid axis of op[0], which has 1 dimensions "),
                (1, new long[] { 12 }, "Iterator input op_axes[0][0] (==1) is not a valid axis of op[0], which has 1 dimensions "),
                (0, new long[0], "Iterator input op_axes[0][1] (==0) is not a valid axis of op[0], which has 0 dimensions "),
                (0, new long[] { 3, 1 }, "non-broadcastable output operand with shape (3,1) [remapped to (1,3)] doesn't match the broadcast shape (3,4)"),
                (1, new long[] { 1, 4 }, "non-broadcastable output operand with shape (1,4) [remapped to (4,1)] doesn't match the broadcast shape (3,4)"),
            };
            foreach (var (axis, outDims, message) in rows)
            {
                using var o = Z(outDims);
                new Action(() => np.cumsum(a, axis, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(message, $"cumsum axis {axis}");
                new Action(() => np.cumprod(a, axis, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(message, $"cumprod axis {axis}");
            }
        }

        /// <summary>The direct 1-D path (the flattened form, a 1-D input, a 0-d input: NumPy's CheckAxis ravels them) keeps
        ///     <see cref="WrongSize"/>; a loop dtype other than an operand's, or an out overlapping the input, takes the
        ///     iterator even in 1-D, and with it NpyIter's texts.</summary>
        [TestMethod]
        public void CumsumCumprod_OneDimension_DirectPathOrIteratorByNumPysTest()
        {
            using var a = A();
            using var v = np.arange(3.0);
            using var i4 = np.arange(3).astype(np.int32);
            foreach (var shape in new[] { new long[] { 5 }, new long[] { 3, 4 }, new long[] { 12, 1 }, new long[] { 13 }, new long[] { 1 } })
            {
                using var o = Z(shape);
                new Action(() => np.cumsum(a, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(WrongSize);
                new Action(() => np.cumprod(a, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(WrongSize);
            }
            foreach (var shape in new[] { new long[] { 4 }, new long[] { 1 }, new long[] { 3, 1 }, new long[] { 1, 3 } })
            {
                using var o = Z(shape);
                new Action(() => np.cumsum(v, 0, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(WrongSize);
            }
            using (var o = ZT(NPTypeCode.Int32, 4))
            {
                new Action(() => np.cumsum(i4, 0, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(WrongSize);
                new Action(() => np.cumsum(i4, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(WrongSize);
            }
            using (var scalar = NDArray.Scalar(5.0))
            using (var o = Z(2))
            {
                new Action(() => np.cumsum(scalar, 0, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(WrongSize);
                new Action(() => np.cumsum(scalar, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(WrongSize);
            }

            // Another out dtype (the loop is promote_types(out, arr)), another input dtype, or a dtype= request: the iterator.
            const string FourThree = Remapped + "(4,)->(4,) (3,)->(3,) ";
            using (var o = ZT(NPTypeCode.Single, 4))
                new Action(() => np.cumsum(v, 0, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(FourThree);
            using (var o = ZT(NPTypeCode.Int64, 4))
            {
                new Action(() => np.cumsum(i4, 0, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(FourThree);
                new Action(() => np.cumprod(i4, @out: o)).Should().Throw<IncorrectShapeException>().WithMessage(FourThree);
            }
            using (var o = Z(4))
                new Action(() => np.cumsum(v, 0, NPTypeCode.Single, o)).Should().Throw<IncorrectShapeException>().WithMessage(FourThree);
            using (var o = ZT(NPTypeCode.Single, 1))
                new Action(() => np.cumsum(v, 0, @out: o)).Should().Throw<IncorrectShapeException>()
                    .WithMessage("non-broadcastable output operand with shape (1,) [remapped to (1,)] doesn't match the broadcast shape (3,)");
            using (var o = ZT(NPTypeCode.Single))
                new Action(() => np.cumsum(v, 0, @out: o)).Should().Throw<IncorrectShapeException>()
                    .WithMessage("Iterator input op_axes[0][0] (==0) is not a valid axis of op[0], which has 0 dimensions ");
            using (var o = ZT(NPTypeCode.Single, 5))
                new Action(() => np.cumsum(a, @out: o)).Should().Throw<IncorrectShapeException>()
                    .WithMessage(Remapped + "(5,)->(5,) (12,)->(12,) ");

            // An out that is a longer view of the input's own buffer overlaps it (NPY_MAY_SHARE_BOUNDS): the iterator. A view
            // past the input's end does not.
            using var buf = np.arange(8.0);
            new Action(() => np.cumsum(buf[":3"], 0, @out: buf[":4"])).Should().Throw<IncorrectShapeException>().WithMessage(FourThree);
            new Action(() => np.cumsum(buf[":3"], @out: buf["4:8"])).Should().Throw<IncorrectShapeException>().WithMessage(WrongSize);

            // A read-only out refuses first.
            using var ro = Z(4, 3);
            ro.setflags(write: false);
            new Action(() => np.cumsum(a, 0, @out: ro)).Should().Throw<ValueError>().WithMessage("output array is read-only");
        }

        /// <summary>np.ptp's out meets maximum.reduce's prologue before anything is computed (NumPy's texts, ufunc
        ///     'maximum'), so a misshapen out is reported before an empty reduction's "no identity".</summary>
        [TestMethod]
        public void Ptp_RaisesMaximumReducesPrologueTexts()
        {
            using var a = A();
            using var a3 = np.arange(24.0).reshape(2, 3, 4);
            using var three = Z(3);
            using var five = Z(5);
            using var row = Z(1, 4);
            using var one = Z(1);
            using var four = Z(4);
            using var full = Z(3, 4);

            new Action(() => np.ptp(a, 0, three)).Should().Throw<IncorrectShapeException>().WithMessage(Remapped + "(3,)->(3) (3,4) ");
            new Action(() => np.ptp(a, 1, five)).Should().Throw<IncorrectShapeException>().WithMessage(Remapped + "(5,)->(5,newaxis) (3,4) ");
            new Action(() => np.ptp(a, 0, row)).Should().Throw<ValueError>()
                .WithMessage("output parameter for reduction operation maximum has the wrong number of dimensions: Found 2 but expected 1");
            new Action(() => np.ptp(a, (int?)null, three)).Should().Throw<ValueError>()
                .WithMessage("output parameter for reduction operation maximum has the wrong number of dimensions: Found 1 but expected 0");
            new Action(() => np.ptp(a, 0, one)).Should().Throw<ValueError>()
                .WithMessage("output operand requires a reduction along dimension 0, but the reduction is not enabled. The dimension size of 1 does not match the expected output shape.");
            new Action(() => np.ptp(a, 0, four, keepdims: true)).Should().Throw<ValueError>()
                .WithMessage("output parameter for reduction operation maximum has the wrong number of dimensions: Found 1 but expected 2 (must match the operand's when keepdims=True)");
            new Action(() => np.ptp(a, 0, full, keepdims: true)).Should().Throw<ValueError>()
                .WithMessage("operand was set up as a reduction along axis 0, but the length of the axis is 3 (it has to be 1)");
            new Action(() => np.ptp(a3, new[] { 0, 1 }, three)).Should().Throw<IncorrectShapeException>()
                .WithMessage(Remapped + "(3,)->(3) (2,3,4) ");
            using (var empty = Z(0, 4))
                new Action(() => np.ptp(empty, 0, three)).Should().Throw<IncorrectShapeException>().WithMessage(Remapped + "(3,)->(3) (0,4) ");
            using (var ro = Z(3))
            {
                ro.setflags(write: false);
                new Action(() => np.ptp(a, 0, ro)).Should().Throw<ValueError>().WithMessage("output array is read-only");
            }

            // A fitting out still receives the range.
            using var ok = Z(3);
            ReferenceEquals(np.ptp(a, 1, ok), ok).Should().BeTrue("the out itself is returned");
            ok.GetDouble(0).Should().Be(3.0);
        }

        /// <summary>np.median's out meets np.mean's add.reduce prologue over the partition's middle slice (one element for an
        ///     odd length, two for an even one, none for an empty axis), after _ureduce's keepdims view of it.</summary>
        [TestMethod]
        public void Median_RaisesMeansPrologueTextsOverTheMiddleSlice()
        {
            using var a = A();
            using var a3 = np.arange(24.0).reshape(2, 3, 4);
            using var three = Z(3);
            using var five = Z(5);
            using var row = Z(1, 4);
            using var four = Z(4);

            new Action(() => np.median(a, 0, three)).Should().Throw<IncorrectShapeException>().WithMessage(Remapped + "(3,)->(3) (1,4) ");
            new Action(() => np.median(a, 1, five)).Should().Throw<IncorrectShapeException>().WithMessage(Remapped + "(5,)->(5,newaxis) (3,2) ");
            new Action(() => np.median(a, 1, four)).Should().Throw<IncorrectShapeException>().WithMessage(Remapped + "(4,)->(4,newaxis) (3,2) ");
            new Action(() => np.median(a, 0, row)).Should().Throw<ValueError>()
                .WithMessage("output parameter for reduction operation add has the wrong number of dimensions: Found 2 but expected 1");
            new Action(() => np.median(a, (int?)null, three)).Should().Throw<ValueError>()
                .WithMessage("output parameter for reduction operation add has the wrong number of dimensions: Found 1 but expected 0");
            new Action(() => np.median(a3, new[] { 0, 1 }, three)).Should().Throw<IncorrectShapeException>()
                .WithMessage(Remapped + "(3,)->(3,newaxis) (4,2) ");
            using (var empty = Z(0, 4))
                new Action(() => np.median(empty, 0, three)).Should().Throw<IncorrectShapeException>().WithMessage(Remapped + "(3,)->(3) (0,4) ");
            using (var ro = Z(3))
            {
                ro.setflags(write: false);
                new Action(() => np.median(a, 1, ro)).Should().Throw<ValueError>().WithMessage("output array is read-only");
            }

            // keepdims: _ureduce indexes out[..., 0, :] first (too few dimensions is its IndexError), then mean checks that
            // view, so an out whose reduced axis is longer than 1 is NumPy's: its row 0 receives the median.
            new Action(() => np.median(a, 0, four, keepdims: true)).Should().Throw<IndexError>()
                .WithMessage("too many indices for array: array is 1-dimensional, but 2 were indexed");
            using (var col = Z(3, 1))
                new Action(() => np.median(a, 0, col, keepdims: true)).Should().Throw<ValueError>()
                    .WithMessage("output operand requires a reduction along dimension 0, but the reduction is not enabled. The dimension size of 1 does not match the expected output shape.");
            using (var tall = Z(3, 4))
            {
                ReferenceEquals(np.median(a, 0, tall, keepdims: true), tall).Should().BeTrue("the out itself is returned");
                tall["0"].ToArray<double>().Should().Equal(4.0, 5.0, 6.0, 7.0);
                tall["1"].ToArray<double>().Should().Equal(0.0, 0.0, 0.0, 0.0);
            }

            // An empty slice's median is NaN, into a fitting out too.
            using (var empty = Z(0, 4))
            {
                ReferenceEquals(np.median(empty, 0, four), four).Should().BeTrue("the out itself is returned");
                double.IsNaN(four.GetDouble(0)).Should().BeTrue();
            }
        }

        /// <summary>np.nanmedian stores with <c>out[...] = result</c>: np.copyto's broadcast (an out the result broadcasts to is
        ///     NumPy's, a scalar median fills any out), its unsafe cast and its texts; an empty input is np.nanmean's
        ///     add.reduce prologue against the input.</summary>
        [TestMethod]
        public void NanMedian_AssignsLikeNumPy()
        {
            using var a = A();
            using var three = Z(3);
            new Action(() => np.nanmedian(a, 0, three)).Should().Throw<ValueError>()
                .WithMessage("could not broadcast input array from shape (4,) into shape (3,)");
            using (var row = Z(1, 4))
                new Action(() => np.nanmedian(a, 1, row)).Should().Throw<ValueError>()
                    .WithMessage("could not broadcast input array from shape (3,) into shape (1,4)");
            using (var col = Z(3, 1))
                new Action(() => np.nanmedian(a, 0, col, keepdims: true)).Should().Throw<ValueError>()
                    .WithMessage("could not broadcast input array from shape (4,) into shape (1,)");
            using (var ro = Z(4))
            {
                ro.setflags(write: false);
                new Action(() => np.nanmedian(a, 0, ro)).Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
            }
            using (var empty = Z(0, 4))
                new Action(() => np.nanmedian(empty, 0, three)).Should().Throw<IncorrectShapeException>()
                    .WithMessage(Remapped + "(3,)->(3) (0,4) ");

            // Broadcast stores NumPy makes: the column medians into every row, the flat median into every cell.
            using (var two = Z(2, 4))
            {
                ReferenceEquals(np.nanmedian(a, 0, two), two).Should().BeTrue("the out itself is returned");
                two["1"].ToArray<double>().Should().Equal(4.0, 5.0, 6.0, 7.0);
            }
            using (var cube = Z(1, 3, 4))
                ReferenceEquals(np.nanmedian(a, 0, cube), cube).Should().BeTrue("the out itself is returned");
            using (var grid = Z(2, 4))
            {
                np.nanmedian(a, (int?)null, grid);
                grid.ToArray<double>().Should().OnlyContain(x => x == 5.5);
            }
            using (var kept = Z(2, 1, 4))
                ReferenceEquals(np.nanmedian(a, 0, kept, keepdims: true), kept).Should().BeTrue("the out itself is returned");

            // The unsafe cast: an integer out truncates (1.5, 5.5, 9.5 -> 1, 5, 9).
            using var ints = ZT(NPTypeCode.Int64, 3);
            np.nanmedian(a, 1, ints);
            ints.ToArray<long>().Should().Equal(1L, 5L, 9L);
        }

        /// <summary>
        ///     np.around / np.round_ follow <c>PyArray_Round</c>'s statements (scratchpad probes p172_numpy.py / p173): an out
        ///     of another SIZE is <c>invalid output shape</c> before anything else; integer data at <c>decimals &gt;= 0</c> is
        ///     <c>PyArray_AssignArray</c> (np.copyto's texts); float data at <c>decimals == 0</c> is the rint ufunc, at any other
        ///     count <c>multiply(a, 10**d, out)</c> (three operands, the Python float as <c>()</c>); complex data assigns its
        ///     parts into the out.
        /// </summary>
        [TestMethod]
        public void Around_RaisesPyArrayRoundsTexts()
        {
            using var a = A();
            using var ai = np.arange(12).astype(np.int64).reshape(3, 4);
            const string InvalidShape = "invalid output shape";

            foreach (int decimals in new[] { 0, 1, -1 })
                foreach (var dims in new[] { new long[] { 3 }, new long[] { 3, 5 }, new long[] { 1, 4 }, new long[0] })
                {
                    using var o = Z(dims);
                    using var oi = ZT(NPTypeCode.Int64, dims);
                    new Action(() => np.around(a, decimals, o)).Should().Throw<ValueError>().WithMessage(InvalidShape);
                    // Before the integer route's divide cast refusal too (decimals -1 into an int64 out).
                    new Action(() => np.around(ai, decimals, oi)).Should().Throw<ValueError>().WithMessage(InvalidShape);
                }

            using var transposed = Z(4, 3);
            using var transposedInt = ZT(NPTypeCode.Int64, 4, 3);
            new Action(() => np.around(a, 0, transposed)).Should().Throw<ArgumentException>()
                .WithMessage("operands could not be broadcast together with shapes (3,4) (4,3) ");
            new Action(() => np.around(a, 1, transposed)).Should().Throw<ArgumentException>()
                .WithMessage("operands could not be broadcast together with shapes (3,4) () (4,3) ");
            new Action(() => np.round(a, -1, transposed)).Should().Throw<ArgumentException>()
                .WithMessage("operands could not be broadcast together with shapes (3,4) () (4,3) ");
            new Action(() => np.around(ai, 0, transposedInt)).Should().Throw<ValueError>()
                .WithMessage("could not broadcast input array from shape (3,4) into shape (4,3)");
            new Action(() => np.around(ai, 1, transposedInt)).Should().Throw<ValueError>()
                .WithMessage("could not broadcast input array from shape (3,4) into shape (4,3)");
            new Action(() => np.around(ai, -1, transposedInt)).Should().Throw<TypeError>()
                .WithMessage("Cannot cast ufunc 'divide' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");

            // The integer copy's read-only text is the assignment's, not a ufunc's.
            using (var ro = ZT(NPTypeCode.Int64, 3, 4))
            {
                ro.setflags(write: false);
                new Action(() => np.around(ai, 0, ro)).Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
            }

            // Complex data: the parts' assignment refuses an out it does not broadcast to, and fills a larger one.
            using var c = (np.arange(12.0) + np.arange(12.0) * new System.Numerics.Complex(0, 1)).reshape(3, 4);
            using (var ct = ZT(NPTypeCode.Complex, 4, 3))
                new Action(() => np.around(c, 2, ct)).Should().Throw<ValueError>()
                    .WithMessage("could not broadcast input array from shape (3,4) into shape (4,3)");
            using (var cubeC = ZT(NPTypeCode.Complex, 1, 3, 4))
                ReferenceEquals(np.around(c, 0, cubeC), cubeC).Should().BeTrue("the out itself is returned");

            // Outs NumPy fills: a larger broadcast out, and an integer copy into a narrower integer out.
            using (var cube = Z(1, 3, 4))
            {
                using var sevenths = np.arange(12.0).reshape(3, 4) / 7;
                np.around(sevenths, 2, cube);
                cube.ToArray<double>()[1].Should().Be(0.14);
            }
            using (var narrow = ZT(NPTypeCode.Int32, 3, 4))
            {
                np.around(ai, 1, narrow);
                narrow.ToArray<int>()[3].Should().Be(3);
            }
        }
    }
}
