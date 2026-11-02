using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — the NumPy parity QA journey of 2026-11-02 / 03
    // =====================================================================
    //
    // A manual QA pass over branch strings compared NumSharp against NumPy 2.4.2 with a probe harness (a C# probe and a
    // NumPy twin printing the same result lines, then a comparer). Every divergence it confirmed that no existing
    // [OpenBugs] / [Misaligned] entry already pins has an id QA-NN, an explanation in docs/bugs/parity-qa-2026-11-02.md
    // (the journal: symptom, NumPy's answer, root cause where known, severity, fix size) and a reproduction below through
    // public calls, with the smallest input that shows it. Nothing here is fixed yet: the owner picks which ones to fix.
    //
    // Every expected value is NumPy 2.4.2's answer for the same call (probed on win-amd64). EXPECTED (when fixed): each
    // test passes; remove its [OpenBugs], move it next to its API's tests, and update the journal entry.
    // =====================================================================
    public partial class OpenBugs
    {
        // ------------------------------------------------------------------
        // QA-01 — a slice with step 0 selects nothing instead of raising
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>a[::0]</c> is NumPy's <c>ValueError: slice step cannot be zero</c> (<c>PySlice_Unpack</c>), for a read and a
        ///     write alike. NumSharp's <c>Slice.ToSliceDef</c> (View/Slice.cs) answers a zero step with an EMPTY selection,
        ///     so the read silently returns a zero-length view and hides the caller's mistake.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA01")]
        public void ParityQa_Slice_ZeroStep_Get_ShouldRaiseValueError()
        {
            using var a = np.arange(10);

            Action stringSlice = () => a["::0"].Dispose();
            Action sliceObject = () => a[new Slice(0, 5, 0)].Dispose();

            stringSlice.Should().Throw<ValueError>().WithMessage("slice step cannot be zero");
            sliceObject.Should().Throw<ValueError>().WithMessage("slice step cannot be zero");
        }

        /// <summary>
        ///     <c>a[1:8:0] = 5</c> raises NumPy's <c>ValueError: slice step cannot be zero</c> before anything is written.
        ///     NumSharp's setter resolves the zero step to an empty selection and returns normally, so the assignment is
        ///     silently dropped (the array keeps every old value and the caller is never told).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA01")]
        public void ParityQa_Slice_ZeroStep_Set_ShouldRaiseValueError()
        {
            using var a = np.arange(10);

            Action act = () => a["1:8:0"] = 5;

            act.Should().Throw<ValueError>().WithMessage("slice step cannot be zero");
        }

        // ------------------------------------------------------------------
        // QA-02 — np.nonzero of a 0-d numeric array answers instead of raising
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 refuses <c>np.nonzero(np.array(5))</c> and <c>np.array(5).nonzero()</c> with
        ///     <c>ValueError: Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.</c>
        ///     (the 1.17 deprecation became an error in 2.1). NumSharp returns <c>(array([0]),)</c>: on purpose, per the
        ///     comment in <c>DefaultEngine.NonZero</c> (Backends/Default/Indexing/Default.NonZero.cs: "preserve that semantic,
        ///     otherwise this becomes a breaking-change PR"), while the text dtypes, <c>NDListArray</c> and numpy.ma already
        ///     raise NumPy's error. The project's principle 4 accepts breaking changes that align with NumPy 2.x, so the owner
        ///     decides whether the numeric dtypes follow.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA02")]
        public void ParityQa_Nonzero_ZeroDimNumeric_ShouldRaiseValueError()
        {
            using var five = np.array(5);
            using var zero = np.array(0.0);

            Action function = () => np.nonzero(five);
            Action member = () => zero.nonzero();

            const string text = "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.";
            function.Should().Throw<ValueError>().WithMessage(text);
            member.Should().Throw<ValueError>().WithMessage(text);
        }

        // ------------------------------------------------------------------
        // QA-03 — np.tril / np.triu of a 0-d array raise the wrong exception type
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>np.tril</c> / <c>np.triu</c> of a 0-d array leak <c>tri()</c>'s CPython <c>TypeError: tri() missing 1
        ///     required positional argument: 'N'</c>. NumSharp reproduces the text but throws <see cref="ArgumentException"/>
        ///     (<c>NDArray.TwoDimBase.cs</c>), so a caller catching NumPy's <see cref="TypeError"/> misses it;
        ///     <c>Indexing/np.tri.Test.cs</c> (<c>Tril_ZeroD_Throws_NumPysLeakedText</c>) asserts the current type and moves
        ///     with the fix.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA03")]
        public void ParityQa_TrilTriu_ZeroDim_ShouldRaiseTypeError()
        {
            using var five = np.array(5);

            Action tril = () => np.tril(five);
            Action triu = () => np.triu(five);

            tril.Should().Throw<TypeError>().WithMessage("tri() missing 1 required positional argument: 'N'");
            triu.Should().Throw<TypeError>().WithMessage("tri() missing 1 required positional argument: 'N'");
        }

        // ------------------------------------------------------------------
        // QA-04 — np.around of a bool array with decimals != 0 raises NotSupportedException
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.around(bool_array, 1)</c> is NumPy's <c>UFuncTypeError</c> (a <c>TypeError</c>): <c>Cannot cast ufunc
        ///     'multiply' output from dtype('float64') to dtype('bool') with casting rule 'same_kind'</c> (and <c>'divide'</c>
        ///     for a negative <c>decimals</c>). NumSharp has NumPy's text but throws <see cref="NotSupportedException"/>, so
        ///     neither a <see cref="TypeError"/> handler nor NumSharp's own house <see cref="ArgumentException"/> for ufunc cast
        ///     refusals catches it.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA04")]
        public void ParityQa_Around_BoolWithDecimals_ShouldRaiseTypeError()
        {
            using var b = np.array(new[] { true, false });

            Action up = () => np.around(b, 1);
            Action down = () => np.around(b, -1);

            up.Should().Throw<TypeError>()
                .WithMessage("Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('bool') with casting rule 'same_kind'");
            down.Should().Throw<TypeError>()
                .WithMessage("Cannot cast ufunc 'divide' output from dtype('float64') to dtype('bool') with casting rule 'same_kind'");
        }

        // ------------------------------------------------------------------
        // QA-05 — a computed NumPy scalar (a[i], a.sum()) is a WEAK operand in the binary ufuncs
        // ------------------------------------------------------------------

        /// <summary>
        ///     NEP 50 makes every NumPy scalar and 0-d array STRONG: <c>u8 + i64[1]</c> is int64 <c>[201, 202, 450]</c> in NumPy
        ///     2.4.2. NumSharp reads any 0-d operand beside an array as a Python literal, so the sum stays uint8 and the
        ///     third element silently wraps to 194. The rules record the 0-d rule as a known divergence because "a strong 0-d
        ///     NDArray cannot be told from a literal" (ufuncs.md, comparison.md), but since <c>51abf1ad</c> the implicit
        ///     conversions mark a C# literal's 0-d array <c>NDArray.WasPythonLiteral</c>, which can tell them apart: an element
        ///     read (<c>a[i]</c>) or a reduction result is not a literal. The owner decides whether the binary ufuncs read the
        ///     mark (the journal lists the affected calls: maximum, add, multiply, power, left_shift, clip, floor_divide).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA05")]
        public void ParityQa_ComputedScalarOperand_ShouldBeStrong_NoSilentWrap()
        {
            using var u8 = np.array(new byte[] { 1, 2, 250 });
            using var i64 = np.array(new long[] { 100, 200, 300 });
            using var element = i64[1];

            using var sum = u8 + element;

            sum.dtype.Should().Be(np.int64);
            sum.ToArray<long>().Should().Equal(201L, 202L, 450L);
        }

        /// <summary>
        ///     <c>f16 * f32.mean()</c>: NumPy's float32 scalar is strong, so the product is float32. NumSharp computes it in
        ///     float16 (the mean is read as a weak Python float), losing precision silently. The same weak reading turns
        ///     <c>np.left_shift(int8, np.array(9L))</c> into int8 zeros where NumPy returns int64 <c>[512, -1024, 1536]</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA05")]
        public void ParityQa_ReductionResultOperand_ShouldBeStrong()
        {
            using var f16 = np.array(new[] { 1.0, 2.5, 3.0 }).astype(np.float16);
            using var f32 = np.array(new[] { 1.5f, 2.0f, 0.25f });
            using var mean = f32.mean();
            using var i8 = np.array(new sbyte[] { 1, -2, 3 });
            using var nine = np.array(9L);

            using var product = f16 * mean;
            using var shifted = np.left_shift(i8, nine);

            product.dtype.Should().Be(np.float32);
            shifted.dtype.Should().Be(np.int64);
            shifted.ToArray<long>().Should().Equal(512L, -1024L, 1536L);
        }
    }
}
