using System;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Bugs the API-parity batches found on the way (docs/plans/API_PARITY_DIFF.md, 2026-11-11; batches 2 and 3): each is pinned here as
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
    }
}
