using System;
using NumSharp.Backends;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The order NumPy 2.4.2 reports the problems of a call that involves text, and its answers for an unsized string
    ///     <c>dtype=</c> request (<c>'U'</c> / <c>'S'</c>): every expected text below was probed against NumPy 2.4.2. The
    ///     string oracle (<c>NumSharp.Tests.Oracle/Strings</c>, sections <c>strdtype</c> / <c>strprec</c> / <c>straxis</c> /
    ///     <c>strcov</c>) replays the full matrix; these tests pin one representative of each rule so a regression names
    ///     the rule it broke.
    /// </summary>
    /// <remarks>
    ///     A call NumPy answers by computing on text (concatenation, comparisons, sorting, text-to-number parsing) throws
    ///     <see cref="StringComputationNotSupportedException"/> in this phase; every NumPy refusal is reproduced with
    ///     NumPy's text. Exception types follow the house map: NumPy's <c>ValueError</c> for a failed broadcast is
    ///     <see cref="IncorrectShapeException"/> without out= / where= and <see cref="ArgumentException"/> with them, the
    ///     no-loop <c>TypeError</c> of a request NumPy cannot resolve at all is <see cref="IncorrectTypeException"/>, and a
    ///     non-bool where= mask is <see cref="ArgumentException"/>.
    /// </remarks>
    [TestClass]
    public class StringCallOrderTests
    {
        private static NDArray U() => np.array(new[] { "ab", "c", "de" });
        private static NDArray U2() => np.array(new[] { "x", "y", "z" });
        private static NDArray S() => U().astype("S2");
        private static NDArray F() => np.array(new[] { 1.5, 2.0, 3.0 });
        private static NDArray I() => np.array(new long[] { 1, 2, 3 });

        private static NDArray ReadOnlyZeros()
        {
            var ro = np.zeros(3);
            ro.setflags(write: false);
            return ro;
        }

        // ---------------------------------------------------------------- unsized string dtype= requests

        [TestMethod]
        public void UnaryRequest_NoStringLoop_IsTheNoLoopText()
        {
            // np.sqrt(f, dtype='U') -> TypeError: No loop matching the specified signature and casting was found for ufunc sqrt
            Action act = () => np.sqrt(F(), dtype: "U");
            act.Should().Throw<IncorrectTypeException>()
               .WithMessage("No loop matching the specified signature and casting was found for ufunc sqrt");
        }

        [TestMethod]
        public void UnaryRequest_ClassForm_NamesBothClasses()
        {
            // np.negative(f, dtype='U') names the operand CLASS and the requested CLASS.
            Action act = () => np.negative(F(), dtype: "U");
            act.Should().Throw<TypeError>().WithMessage(
                "ufunc 'negative' did not contain a loop with signature matching types <class 'numpy.dtypes.Float64DType'> -> " +
                "<class 'numpy.dtypes.StrDType'>");
        }

        [TestMethod]
        public void BinaryRequest_ClassForm_NamesTheOperandClasses()
        {
            Action act = () => np.subtract(F(), I(), dtype: "S");
            act.Should().Throw<TypeError>().WithMessage(
                "ufunc 'subtract' did not contain a loop with signature matching types (<class 'numpy.dtypes.Float64DType'>, " +
                "<class 'numpy.dtypes.Int64DType'>) -> <class 'numpy.dtypes.BytesDType'>");
        }

        [TestMethod]
        public void AddRequest_NumbersAreFormattedAsText_IsTheNextPhase()
        {
            // NumPy computes ['1.51.5', '2.02.0', '3.03.0'] (<U64): the string add loop, the next phase.
            Action act = () => np.add(F(), F(), dtype: "U");
            act.Should().Throw<StringComputationNotSupportedException>().WithMessage("*<U64*");
        }

        [TestMethod]
        public void Add_TwoStringClasses_IsRefused()
        {
            Action act = () => np.add(U(), S());
            act.Should().Throw<TypeError>()
               .WithMessage("ufunc 'add' cannot use operands with types dtype('<U2') and dtype('S2')");
        }

        [TestMethod]
        public void MultiplyRepeat_WithoutOut_IsTheOutKwargText()
        {
            Action act = () => np.multiply(U(), I(), dtype: "U");
            act.Should().Throw<TypeError>().WithMessage(
                "The 'out' kwarg is necessary when using the string multiply ufunc directly. Use numpy.strings.multiply to " +
                "multiply strings without specifying 'out'.");
        }

        // ---------------------------------------------------------------- out= / where= before the loop answer

        [TestMethod]
        public void ReadOnlyOut_IsReportedBeforeTheLoop()
        {
            Action act = () => np.add(U(), U2(), ReadOnlyZeros());
            act.Should().Throw<ValueError>().WithMessage("output array is read-only");
        }

        [TestMethod]
        public void NonBoolWhere_IsReportedBeforeTheLoop()
        {
            Action act = () => np.sqrt(U(), where: I());
            act.Should().Throw<ArgumentException>()
               .WithMessage("Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'*");
        }

        [TestMethod]
        public void ReadOnlyOut_WinsOverANonBoolWhere()
        {
            Action act = () => np.sqrt(U(), ReadOnlyZeros(), I());
            act.Should().Throw<ValueError>().WithMessage("output array is read-only");
        }

        [TestMethod]
        public void ComputingLoop_ReportsTheOutputCastThenTheBroadcast()
        {
            // The concatenation loop resolves <U3; a float64 out cannot take it (the house ArgumentException of a ufunc
            // output cast, NumPy's UFuncTypeError text).
            Action cast = () => np.add(U(), U2(), np.zeros(3));
            cast.Should().Throw<ArgumentException>().WithMessage(
                "Cannot cast ufunc 'add' output from dtype('<U3') to dtype('float64') with casting rule 'same_kind'");

            // Inputs that do not broadcast: NumPy's ufunc text after the loop answer.
            Action shapes = () => np.add(U(), U2()["0:2"]);
            shapes.Should().Throw<IncorrectShapeException>()
                  .WithMessage("operands could not be broadcast together with shapes (3,) (2,) ");

            // With an out=, NpyIter lists every operand.
            Action withOut = () => np.add(U(), U2(), np.zeros(4, np.dtype("<U9")));
            withOut.Should().Throw<ArgumentException>()
                   .WithMessage("operands could not be broadcast together with shapes (3,) (3,) (4,) *");
        }

        [TestMethod]
        public void DivMod_ParsesBothOutputsBeforeTheLoop()
        {
            var r = np.zeros(3);
            r.setflags(write: false);
            Action act = () => np.divmod(U(), U(), (np.zeros(3), r));
            act.Should().Throw<ValueError>().WithMessage("output array is read-only");
        }

        // ---------------------------------------------------------------- reductions

        [TestMethod]
        public void ReductionRequest_ReadsANumberAsTextOfItsRequiredLength()
        {
            // float64 reads as <U32; the concatenation loop's <U64 output is not reduce-compatible.
            Action act = () => np.sum(F(), (int?)null, np.dtype("U"));
            act.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U32'), dtype('<U32'), dtype('<U64'))");

            Action acc = () => np.cumsum(F(), (int?)null, np.dtype("S"));
            acc.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.accumulate. Resolved (dtype('S32'), dtype('S32'), dtype('S64'))");
        }

        [TestMethod]
        public void ProdRequest_Int64_IsTheRepeatLoop_OtherIntegers_NameNoLoop()
        {
            // int64 is the repeat loop's own operand type; another width needs a promoter, which a reduction does not use.
            Action i8 = () => np.prod(I(), (int?)null, np.dtype("U"));
            i8.Should().Throw<TypeError>().WithMessage("The 'out' kwarg is necessary*");

            Action i1 = () => np.prod(np.array(new sbyte[] { 1, 2, 3 }), (int?)null, np.dtype("U"));
            i1.Should().Throw<TypeError>().WithMessage(
                "ufunc 'multiply' did not contain a loop with signature matching types (<class 'numpy.dtypes.StrDType'>, " +
                "<class 'numpy.dtypes.Int8DType'>) -> <class 'numpy.dtypes.StrDType'>");
        }

        [TestMethod]
        public void SizedRequest_IsTheClassOnlyText_BeforeTheAxis_ForUfuncReduce()
        {
            Action act = () => np.sum(U(), 9, np.dtype("U3"));
            act.Should().Throw<TypeError>().WithMessage("The `dtype` and `signature` arguments to ufuncs only select*");
        }

        [TestMethod]
        public void SizedRequest_ComesAfterTheAxis_ForTheMeanFamily()
        {
            Action act = () => np.mean(U(), 9, np.dtype("U3"));
            act.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void NumericRequestOverText_AddFamilyIgnoresIt_OthersParse()
        {
            // The string add loop resolves from the text operands alone: the same refusal as without a dtype.
            Action sum = () => np.sum(U(), (int?)null, np.float64);
            sum.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U2'), dtype('<U2'), dtype('<U4'))");

            // np.prod(u, dtype='f8') parses the text (could not convert string to float: np.str_('ab')): the next phase.
            Action prod = () => np.prod(U(), (int?)null, np.float64);
            prod.Should().Throw<StringComputationNotSupportedException>();
        }

        [TestMethod]
        public void ReductionAxisRules()
        {
            var zeroD = NDArray.Scalar("ab");

            // ufunc.reduce: a 0-d input takes axis 0 as every axis, then refuses the text.
            Action sum0 = () => np.sum(zeroD, 0);
            sum0.Should().Throw<TypeError>().WithMessage("the resolved dtypes are not compatible with add.reduce*");

            // np.mean's family counts items first: a 0-d input refuses every axis.
            Action mean0 = () => np.mean(zeroD, 0);
            mean0.Should().Throw<AxisError>().WithMessage("axis 0 is out of bounds for array of dimension 0*");

            // The accumulations ravel a 0-d input first.
            Action cum1 = () => np.cumsum(zeroD, 1);
            cum1.Should().Throw<AxisError>().WithMessage("axis 1 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void TupleAxisRepeats_UseEachFunctionsText()
        {
            var u2d = np.array(new[,] { { "a", "b" }, { "c", "d" } });
            Action all = () => np.all(u2d, new[] { 0, 0 }, null);
            all.Should().Throw<ValueError>().WithMessage("duplicate value in 'axis'");
            Action median = () => np.median(u2d, new[] { 0, 0 });
            median.Should().Throw<ValueError>().WithMessage("repeated axis");
            Action average = () => np.average(u2d, new[] { 0, 0 });
            average.Should().Throw<ValueError>().WithMessage("repeated axis in `axis` argument");
            Action averageAxis = () => np.average(U(), 9);
            averageAxis.Should().Throw<AxisError>().WithMessage("axis: axis 9 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void ArgReductions_RefuseAnEmptyReducedLength()
        {
            var e1 = np.array(Array.Empty<string>()).astype("<U2");
            Action argmax = () => np.argmax(e1);
            argmax.Should().Throw<ValueError>().WithMessage("attempt to get argmax of an empty sequence");

            // nanargmin of text is argmin: the message names argmin; a (2, 0) array along axis 1 has an empty reduced axis.
            var e2 = np.empty(new Shape(2, 0), np.dtype("<U2"));
            Action nanargmin = () => np.nanargmin(e2, 1);
            nanargmin.Should().Throw<ValueError>().WithMessage("attempt to get argmin of an empty sequence");
        }

        [TestMethod]
        public void CountNonzero_EmptyText_ChecksTheAxis()
        {
            var e1 = np.array(Array.Empty<string>()).astype("<U2");
            Action act = () => np.count_nonzero(e1, 9);
            act.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 1*");
        }

        [TestMethod]
        public void CountNonzero_EmptyNumeric_NegativeAxisOutOfRange_IsAnAxisError()
        {
            // The empty-array shortcut used to wrap -3 into axis 0 and return a result (NumPy raises AxisError).
            Action act = () => np.count_nonzero(np.zeros(new Shape(0, 2)), -3);
            act.Should().Throw<AxisError>().WithMessage("axis -3 is out of bounds for array of dimension 2*");
        }

        // ---------------------------------------------------------------- the quantile family

        [TestMethod]
        public void Quantile_Interpolating_ScalarAndArrayResults()
        {
            // A scalar result subtracts two NumPy scalars (Python's TypeError); an array result subtracts arrays.
            Action scalar = () => np.quantile(U(), 0.5);
            scalar.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'numpy.str_' and 'numpy.str_'");

            var u2d = np.array(new[,] { { "a", "b" }, { "c", "d" } });
            Action array = () => np.quantile(u2d, 0.5, 0);
            array.Should().Throw<TypeError>().WithMessage(
                "ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U1'), dtype('<U1')) -> None");
        }

        [TestMethod]
        public void Quantile_EmptyText_IndexesAnEmptyAxis()
        {
            var e1 = np.array(Array.Empty<string>()).astype("<U2");
            Action lower = () => np.quantile(e1, 0.5, method: "lower");
            lower.Should().Throw<IndexError>().WithMessage("cannot do a non-empty take from an empty axes.");
            Action linear = () => np.quantile(e1, 0.5);
            linear.Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 0 with size 0");
            // The NaN forms hand an empty input to np.nanmean before any NaN test.
            Action nan = () => np.nanquantile(e1, 0.5);
            nan.Should().Throw<TypeError>().WithMessage("the resolved dtypes are not compatible with add.reduce*");
        }

        [TestMethod]
        public void Quantile_TheRangeOfQ_ComesFirst()
        {
            // The house ArgumentException of the q check (NumPy's ValueError text).
            Action act = () => np.quantile(U(), 1.5);
            act.Should().Throw<ArgumentException>().WithMessage("Quantiles must be in the range [0, 1]");
        }

        // ---------------------------------------------------------------- cov, einsum, gufuncs, clip, sort

        [TestMethod]
        public void Cov_EmptyText_IsAReshapeViewOfAnEmptyFloat64Array()
        {
            var c = np.cov(np.empty(new Shape(0, 2), np.dtype("<U2")));
            c.dtype.Should().Be(np.float64);
            c.Should().BeShaped(0, 0);
            c.flags.owndata.Should().BeFalse("NumPy returns np.array([]).reshape(0, 0), a view");
        }

        [TestMethod]
        public void Cov_Text_RefusesAtTheFirstReduction()
        {
            Action plain = () => np.cov(U());
            plain.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U32'), dtype('<U32'), dtype('<U64'))");

            // With weights, average's weight sum meets the SIZED string descriptor: the class-only text.
            Action weighted = () => np.cov(U(), fweights: np.array(new long[] { 1, 1, 1 }));
            weighted.Should().Throw<TypeError>().WithMessage("The `dtype` and `signature` arguments to ufuncs only select*");
        }

        [TestMethod]
        public void Einsum_ViewPathKeepsText_ThenCastsThenTheZeroFill()
        {
            var u = U();
            var view = np.einsum("i", new[] { u }, dtype: "U");
            view.dtype.str.Should().Be("<U2");
            view.flags.owndata.Should().BeFalse("the single-operand view path ignores dtype= and returns a view");
            np.shares_memory(view, u).Should().BeTrue();

            Action cast = () => np.einsum("i,i", new[] { U(), U() }, dtype: "U1", casting: "no");
            cast.Should().Throw<TypeError>().WithMessage(
                "Iterator operand 0 dtype could not be cast from dtype('<U2') to dtype('<U1') according to the rule 'no'");

            Action fill = () => np.einsum("i,i", new[] { U(), U() });
            fill.Should().Throw<TypeError>()
                .WithMessage("Cannot cast scalar from dtype('bool') to dtype('<U2') according to the rule 'safe'");
        }

        [TestMethod]
        public void Gufuncs_ResolveTheLoopFirst()
        {
            var u2d = np.array(new[,] { { "a", "b" }, { "c", "d" } });
            Action matmul = () => np.matmul(u2d, u2d);
            matmul.Should().Throw<TypeError>().WithMessage(
                "ufunc 'matmul' did not contain a loop with signature matching types (dtype('<U1'), dtype('<U1')) -> None");

            Action vecdot = () => np.vecdot(F(), F(), dtype: "U");
            vecdot.Should().Throw<TypeError>().WithMessage(
                "ufunc 'vecdot' did not contain a loop with signature matching types (<class 'numpy.dtypes.Float64DType'>, " +
                "<class 'numpy.dtypes.Float64DType'>) -> <class 'numpy.dtypes.StrDType'>");
        }

        [TestMethod]
        public void Clip_StringRequest_NamesTheRoutedUfunc()
        {
            Action both = () => np.clip(F(), F(), F(), dtype: "U");
            both.Should().Throw<TypeError>().WithMessage(
                "ufunc 'clip' did not contain a loop with signature matching types (<class 'numpy.dtypes.Float64DType'>, " +
                "<class 'numpy.dtypes.Float64DType'>, <class 'numpy.dtypes.Float64DType'>) -> <class 'numpy.dtypes.StrDType'>");

            // No bounds: clip routes to positive.
            Action none = () => np.clip(F(), null, null, dtype: "U");
            none.Should().Throw<TypeError>().WithMessage(
                "ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.Float64DType'> -> " +
                "<class 'numpy.dtypes.StrDType'>");
        }

        [TestMethod]
        public void SortAndPartition_CheckTheAxisAndKthFirst()
        {
            Action sort = () => np.sort(U(), 9);
            sort.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 1*");

            Action partition = () => np.partition(U(), 5);
            partition.Should().Throw<ValueError>().WithMessage("kth(=5) out of bounds (3)");

            // Past the checks, NumPy compares the text: the next phase.
            Action compare = () => np.partition(U(), 1);
            compare.Should().Throw<StringComputationNotSupportedException>();
        }

        [TestMethod]
        public void NanArgMax_ZeroD_AxisMessage_DependsOnTheDtypeFamily()
        {
            // An inexact input meets np.all(mask, axis) first (dimension 0); any other input meets argmax (ravelled: 1).
            Action f = () => np.nanargmax(NDArray.Scalar(5.0), 1);
            f.Should().Throw<AxisError>().WithMessage("axis 1 is out of bounds for array of dimension 0*");
            Action i = () => np.nanargmax(NDArray.Scalar(5L), 1);
            i.Should().Throw<AxisError>().WithMessage("axis 1 is out of bounds for array of dimension 1*");
            Action t = () => np.nanargmax(NDArray.Scalar("ab"), 1);
            t.Should().Throw<AxisError>().WithMessage("axis 1 is out of bounds for array of dimension 1*");
        }
    }
}
