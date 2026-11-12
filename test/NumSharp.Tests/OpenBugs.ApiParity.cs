using System;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Bugs the API-parity batches found on the way (docs/plans/API_PARITY_DIFF.md, 2026-11-11 / 12; batches 2, 3 and 4): each is pinned here as
    ///     NumPy 2.4.2's behaviour and left for its own fix, per the owner's rule that a bug found while fixing something
    ///     else gets a reproduction and is deferred.
    /// </summary>
    public partial class OpenBugs
    {
        /// <summary>
        ///     np.concatenate's argument refusals - and np.stack's, which concatenates - raise .NET types with NumPy's texts
        ///     where NumPy raises <c>TypeError</c>: a casting refusal ("Cannot cast array data from dtype('float64') to
        ///     dtype('int32') according to the rule 'safe'") is <see cref="InvalidCastException"/>, and out with dtype
        ///     ("concatenate() only takes `out` or `dtype` as an argument, but both were provided.") is
        ///     <see cref="ArgumentException"/>. The site (<c>NDArray.concatenate</c> in Creation/NDArray.Multiarray.cs) still
        ///     follows the house convention from before NumSharp had Python's exception classes ("NumPy raises TypeError; we
        ///     raise InvalidCastException"), which the Python-exceptions work switched elsewhere. Found by API-parity batch
        ///     2's np.stack(out=, dtype=, casting=) test.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "api-parity-batch-2")]
        public void ApiParity_ConcatenateCastRefusal_IsNumPysTypeError()
        {
            var a = np.arange(6.0).reshape(2, 3);
            const string text = "Cannot cast array data from dtype('float64') to dtype('int32') according to the rule 'safe'";

            Action concatenate = () => np.concatenate(new[] { a, a }, axis: 0, dtype: np.int32, casting: "safe");
            concatenate.Should().Throw<TypeError>().WithMessage(text);

            Action stack = () => np.stack(new[] { a, a }, dtype: np.int32, casting: "safe");
            stack.Should().Throw<TypeError>().WithMessage(text);

            Action both = () => np.concatenate(new[] { a, a }, axis: 0, @out: np.zeros(new Shape(4, 3)), dtype: np.float64);
            both.Should().Throw<TypeError>()
                .WithMessage("concatenate() only takes `out` or `dtype` as an argument, but both were provided.");
        }

        /// <summary>
        ///     np.repeat's count refusals raise <see cref="ArgumentException"/> where NumPy raises ValueError, and the
        ///     scalar-count text differs: a negative scalar count is NumPy's allocation error ("negative dimensions are not
        ///     allowed"), a negative entry of an array count "repeats may not contain negative values." (with the period), and
        ///     counts that do not broadcast "operands could not be broadcast together with shape (1,) (2,)". NumSharp raises
        ///     "repeats may not contain negative values" for both negative forms. Found by API-parity batch 3's repeat
        ///     differential (the one signature left the texts as they were).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "api-parity-batch-3")]
        public void ApiParity_RepeatCountRefusals_AreNumPysValueError()
        {
            var a = np.array(new[] { 1, 2, 3 });

            Action scalar = () => np.repeat(a, -1);
            scalar.Should().Throw<ValueError>().WithMessage("negative dimensions are not allowed");

            Action array = () => np.repeat(a, new[] { 1, -1, 2 });
            array.Should().Throw<ValueError>().WithMessage("repeats may not contain negative values.");

            Action broadcast = () => np.repeat(5, new[] { 1, 2 });
            broadcast.Should().Throw<ValueError>().WithMessage("operands could not be broadcast together with shape (1,) (2,)");
        }

        /// <summary>
        ///     NumPy converts a Python float count (and a list of them) to intp by truncation - <c>np.repeat([1, 2, 3], 2.5)</c>
        ///     is <c>[1, 1, 2, 2, 3, 3]</c>; only an ARRAY count must cast 'safely' (<c>np.array(2.0)</c> raises "Cannot cast
        ///     scalar from dtype('float64') to dtype('int64') according to the rule 'safe'"). A C# double literal is NumSharp's
        ///     Python float (a 0-d array marked as the literal), but np.repeat applies the array rule to it and raises the
        ///     TypeError. Found by API-parity batch 3's repeat probe; it predates the batch.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "api-parity-batch-3")]
        public void ApiParity_RepeatFloatLiteral_TruncatesAsPythonFloat()
        {
            var a = np.array(new[] { 1, 2, 3 });
            np.repeat(a, 2.5).ToArray<int>().Should().Equal(1, 1, 2, 2, 3, 3);
        }

        /// <summary>
        ///     The nan reductions' new keywords on a <c>numpy.matrix</c> operand: NumPy hands them to the matrix method its
        ///     body reaches, which has no such keyword, so <c>np.nansum(m, where=w)</c> raises "matrix.sum() got an
        ///     unexpected keyword argument 'where'" (also <c>initial=</c>; <c>np.nanprod(m, initial=2)</c> names
        ///     matrix.prod; nanmean / nanvar with <c>where=</c> name matrix.sum, the count's method). NumSharp computes a value:
        ///     NumSharp.Build's <c>[ReducesThrough]</c> entry check knows only <c>keepdims</c>
        ///     (<c>NDSubclass.ReducesThroughCheck</c>). Found by API-parity batch 3, which added the keywords; the check needs
        ///     the weaver to load <c>initial</c> / <c>where</c> (and, for an integer matrix, <c>mean</c> / <c>correction</c>,
        ///     which np.var hands to matrix.var) by name.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "api-parity-batch-3")]
        public void ApiParity_NanReductions_MatrixOperand_RefuseWhereAndInitial()
        {
            var m = np.asmatrix(np.array(new double[,] { { 1, double.NaN }, { 3, 4 } }));
            var w = np.array(new[,] { { true, false }, { true, true } });

            Action sumWhere = () => np.nansum(m, where: w);
            sumWhere.Should().Throw<TypeError>().WithMessage("matrix.sum() got an unexpected keyword argument 'where'");
            Action sumInitial = () => np.nansum(m, initial: 1);
            sumInitial.Should().Throw<TypeError>().WithMessage("matrix.sum() got an unexpected keyword argument 'initial'");
            Action prodInitial = () => np.nanprod(m, initial: 2);
            prodInitial.Should().Throw<TypeError>().WithMessage("matrix.prod() got an unexpected keyword argument 'initial'");
            Action meanWhere = () => np.nanmean(m, where: w);
            meanWhere.Should().Throw<TypeError>().WithMessage("matrix.sum() got an unexpected keyword argument 'where'");
            Action varWhere = () => np.nanvar(m, where: w);
            varWhere.Should().Throw<TypeError>().WithMessage("matrix.sum() got an unexpected keyword argument 'where'");
        }

        /// <summary>
        ///     np.all / np.any of a 0-d operand with a TUPLE axis: NumPy's ufunc reduction keeps the 0-d special case (axis 0
        ///     or -1 accepted) for an INTEGER axis only, so <c>np.all(np.array(5.), (0,))</c> is <c>AxisError: axis 0 is out
        ///     of bounds for array of dimension 0</c> (also <c>(-1,)</c> and <c>(0, 0)</c>). NumSharp's tuple route
        ///     (<c>DefaultEngine.ReduceLogicalMultiAxis</c>) lets a one-element <c>(0,)</c> / <c>(-1,)</c> through and answers
        ///     the scalar. Found by API-parity batch 4, whose <see cref="NDAxis"/> tells the two spellings apart; the route
        ///     predates the batch (the <c>int[]</c> overloads reached it).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "api-parity-batch-4")]
        public void ApiParity_AllAny_ZeroDOperand_TupleAxis_IsAxisError()
        {
            var s = np.array(5.0);
            Action all = () => np.all(s, new[] { 0 });
            all.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");
            Action any = () => np.any(s, new[] { -1 });
            any.Should().Throw<AxisError>().WithMessage("axis -1 is out of bounds for array of dimension 0*");
        }

        /// <summary>
        ///     np.count_nonzero of a <c>numpy.matrix</c> with an axis (or keepdims): NumPy's body is
        ///     <c>a.astype(bool).sum(axis=axis, dtype=np.intp, keepdims=keepdims)</c>, and the matrix's <c>sum</c> has no
        ///     keepdims keyword, so every such call is <c>TypeError: matrix.sum() got an unexpected keyword argument
        ///     'keepdims'</c> (NumPy 2.4.2: axis 0, <c>(0, 1)</c>, and None with keepdims). NumSharp counts on the plain
        ///     view and returns the plain counts: TensorEngine.CountNonZero declares no class rule. Found by API-parity batch
        ///     4's matrix evidence; it predates the batch (the int-axis overload answered the same).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "api-parity-batch-4")]
        public void ApiParity_CountNonzero_MatrixOperand_IsMatrixSumsKeywordError()
        {
            var m = np.matrix(np.array(new long[,] { { 0, 0, 0 }, { 0, 7, 0 } }));
            const string text = "matrix.sum() got an unexpected keyword argument 'keepdims'";
            Action axis = () => np.count_nonzero(m, 0);
            axis.Should().Throw<TypeError>().WithMessage(text);
            Action tuple = () => np.count_nonzero(m, (0, 1));
            tuple.Should().Throw<TypeError>().WithMessage(text);
            Action kept = () => np.count_nonzero(m, null, true);
            kept.Should().Throw<TypeError>().WithMessage(text);
            np.count_nonzero(m).Should().Be(1L, "without an axis NumPy's multiarray.count_nonzero answers");
        }

        /// <summary>
        ///     np.percentile / np.quantile of a <c>numpy.matrix</c> along axis 1: NumPy's <c>_quantile</c> moves the axis first
        ///     and takes rows of the matrix, so <c>np.percentile(m, 50, 1)</c> and <c>np.quantile(m, 0.5, (1,))</c> are the
        ///     1 x 2 ROW. NumSharp's <c>[ReducesThrough]</c> collapse puts a 1-d result along axis 1 back as a column, as
        ///     <c>matrix._collapse</c> does for the reductions that call the matrix method (np.median's <c>(1,)</c> IS that
        ///     column). Found by API-parity batch 4 (batch4_matrix_evidence.py), predating it; batch 5 made the collapse read
        ///     a tuple (np.median's and np.percentile's <c>(0, 1)</c> are now NumPy's 1 x 1 matrix:
        ///     <c>NumPyKeywordParityBatch5Tests.MatrixOperand_QuantileFamily_FullTuple_IsTheOneByOneMatrix</c>), leaving this
        ///     row.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "api-parity-batch-4")]
        public void ApiParity_QuantileFamily_MatrixOperand_KeepsNumPysMatrixShape()
        {
            var m = np.matrix(np.array(new double[,] { { 1, 5, 2 }, { 4, 0, 6 } }));
            void Matrix(NDArray r, long rows, long cols, params double[] values)
            {
                (r is NDMatrix).Should().BeTrue();
                r.shape.Should().Equal(rows, cols);
                r.ToArray<double>().Should().Equal(values);
            }

            Matrix(np.percentile(m, 50, 1), 1, 2, 2.0, 4.0);
            Matrix(np.quantile(m, 0.5, new[] { 1 }), 1, 2, 2.0, 4.0);
        }

        /// <summary>
        ///     np.unwrap with a period of several elements: NumPy broadcasts the period into its elementwise statements, so a
        ///     float period array that broadcasts against <c>diff(p)</c> unwraps each lane by its own period, and one that does
        ///     not is the broadcast ValueError. NumSharp's one-signature unwrap (API-parity batch 6, an <see cref="NDArray"/>
        ///     period) reads one number only and raises <see cref="NotSupportedException"/> for more. Found by API-parity
        ///     batch 6 (NumPy 2.4.2 probes: <c>np.unwrap(p2, period=np.array([[4.0], [6.0]]))</c>; an integer period array
        ///     is NumPy's own "truth value ... ambiguous" error from <c>if boundary_ambiguous</c>).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "api-parity-batch-6")]
        public void ApiParity_Unwrap_PeriodArray_BroadcastsAsNumPy()
        {
            var p2 = np.array(new double[,] { { 0, 1, 2, -1, 0 }, { 0, 1, 2, -1, 0 } });
            var lanes = np.unwrap(p2, period: np.array(new double[,] { { 4.0 }, { 6.0 } }));
            lanes.dtype.Should().Be(np.float64);
            lanes.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, -1.0, 0.0);

            Action mismatch = () => np.unwrap(np.array(new[] { 0.0, 1.0, 2.0, -1.0, 0.0 }), period: np.array(new[] { 4.0, 4.0 }));
            mismatch.Should().Throw<ValueError>().WithMessage("operands could not be broadcast together with shapes (4,) (2,) ");
        }
    }
}
