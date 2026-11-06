using System;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Iteration;

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
        // QA-28 — np.asarray (and the rest of the as* family) refuses a Python list (object[])
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.asarray([1, 2.5, 3])</c> is NumPy's canonical array_like conversion (float64 <c>[1.0, 2.5, 3.0]</c>), and
        ///     NumSharp's <c>np.array(object[])</c> / <c>np.asanyarray(object[])</c> run NumPy's Python-list coercion for it.
        ///     <c>np.asarray</c> has no <c>object[]</c> overload, so C# binds <c>asarray(NDArray, …)</c> through the implicit
        ///     <c>Array</c> → <c>NDArray</c> conversion, which treats the <c>object[]</c> as an array of element type
        ///     <c>Object</c> and throws <c>NotSupportedException: Cannot interpret 'System.Object' as a data type</c>. The same
        ///     happens for <c>np.ascontiguousarray</c>, <c>np.asfortranarray</c>, <c>np.asarray_chkfinite</c>, <c>np.copy</c>
        ///     and <c>np.require</c>, and for every other array_like parameter typed <c>NDArray</c> (np.sum, np.sort,
        ///     np.unique, np.add, … in <c>p62</c>): the CLAUDE.md trap ("an object[] (a Python list) needs its own overload")
        ///     applied to np.array / np.asanyarray only.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA28")]
        public void ParityQa_Asarray_PythonList_ShouldCoerceLikeNpArray()
        {
            var list = new object[] { 1, 2.5, 3 };
            var nested = new object[] { new object[] { 1, 2 }, new object[] { 3, 4 } };

            using var flat = np.asarray(list);
            using var grid = np.asarray(nested);
            using var contiguous = np.ascontiguousarray(list);

            flat.dtype.Should().Be(np.float64);
            flat.GetDouble(1).Should().Be(2.5);
            grid.shape.Should().Equal(new long[] { 2, 2 });
            grid.dtype.Should().Be(np.int64);
            contiguous.GetDouble(2).Should().Be(3.0);
        }

        // ------------------------------------------------------------------
        // QA-41 — finfo's repr and the error texts follow the current culture
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's texts are culture-independent ASCII. <c>repr(np.finfo(np.float32))</c> is
        ///     <c>finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)</c> (the dtype's own float
        ///     repr), while NumSharp's <c>finfo.ToString()</c> (APIs/np.finfo.cs) interpolates <c>{resolution}</c> and
        ///     <c>{min:E}</c> / <c>{max:E}</c> with the current culture: <c>finfo(resolution=1E-06, min=-3.402823E+038,
        ///     max=3.402823E+038, dtype=float32)</c> under the invariant culture, a decimal comma under de-DE, U+2212 minus signs
        ///     under sv-SE, U+061C marks and Arabic separators under ar-SA (<c>rp6</c>, <c>rp8</c>). Every error text that
        ///     interpolates a negative number does the same (<c>rp7</c>, ar-SA): <c>np.arange(5.)[-6]</c> says
        ///     <c>index ؜-6 is out of bounds for axis 0 with size 5</c>, likewise the axis, linspace, kth, take and iinfo
        ///     texts (<c>iinfo(min=؜-128, …)</c>). Array printing, astype to text, savetxt, tofile, np.char.mod and the
        ///     text parsers are culture-invariant already.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA41")]
        public void ParityQa_FinfoReprAndErrorTexts_ShouldNotFollowTheCurrentCulture()
        {
            np.finfo(np.float32).ToString().Should().Be("finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)");

            var saved = System.Threading.Thread.CurrentThread.CurrentCulture;
            try
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.GetCultureInfo("ar-SA");
                using var a = np.arange(5.0);
                Action outOfBounds = () => a[-6].Dispose();

                outOfBounds.Should().Throw<IndexError>().WithMessage("index -6 is out of bounds for axis 0 with size 5");
            }
            finally
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = saved;
            }
        }

        // ------------------------------------------------------------------
        // QA-54 — np.cumsum / np.cumprod refuse outs NumPy's accumulate accepts
        // ------------------------------------------------------------------

        /// <summary>
        ///     Found 2026-11-06 while fixing QA-31 (scratchpad probe p168_numpy.py): NumPy's <c>PyUFunc_Accumulate</c> accepts
        ///     several outs whose shape is not the accumulation's. On its ITERATOR path (op_axes naming the input's axes) an
        ///     out with MORE dimensions whose leading ones fit is accepted, its extra trailing axes staying at index 0
        ///     (<c>np.cumsum(a(3,4), 0, out=zeros((3,4,1)))</c>); an input of extent 1 broadcasts against a longer out, and
        ///     along the accumulated axis only the input's extent is written (<c>np.cumsum(ones((3,1)), 0, out=(3,4))</c> fills
        ///     every column with 1, 2, 3; <c>np.cumsum(ones((1,4)), 0, out=(3,4))</c> writes row 0 and leaves the rest). On
        ///     its DIRECT 1-D path an empty input or an empty out returns the out untouched before the dimensions are compared
        ///     (<c>np.cumsum(zeros(0), out=zeros(5))</c>). NumSharp's numeric scans refuse all of them with <c>provided out is
        ///     the wrong size for the accumulation.</c> (<c>DefaultEngine.AccumulateOutShapeError</c>); its StringDType
        ///     accumulation (<c>NDVStringReduction.CumSum</c>) already accepts them.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA54")]
        public void ParityQa_CumsumCumprod_AcceptTheOutsNumPysAccumulateAccepts()
        {
            using var a = np.arange(12.0).reshape(3, 4);

            using var deeper = np.full(new Shape(3, 4, 1), -1.0);
            np.cumsum(a, 0, @out: deeper);
            deeper[":, :, 0"].ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0);

            using var columns = np.full(new Shape(3, 4), -1.0);
            using var col = np.ones(new Shape(3, 1));
            np.cumsum(col, 0, @out: columns);
            columns.ToArray<double>().Should().Equal(1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0);

            using var firstRow = np.full(new Shape(3, 4), -1.0);
            using var row = np.ones(new Shape(1, 4));
            np.cumprod(row, 0, @out: firstRow);
            firstRow.ToArray<double>().Should().Equal(1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);

            using var untouched = np.full(new Shape(5), -1.0);
            using var empty = np.zeros(new Shape(0));
            ReferenceEquals(np.cumsum(empty, @out: untouched), untouched).Should().BeTrue();
            untouched.ToArray<double>().Should().OnlyContain(x => x == -1.0);
        }

        // ------------------------------------------------------------------
        // QA-55 — np.median / np.ptp: the out's dtype rule is not NumPy's
        // ------------------------------------------------------------------

        /// <summary>
        ///     Found 2026-11-06 while fixing QA-31 (scratchpad probe p169_numpy.py): NumPy's np.median writes its out through
        ///     <c>np.mean(..., out=out)</c>, whose add.reduce and <c>true_divide(..., casting='unsafe')</c> store into an out of
        ///     any kind: <c>np.median(arange(12.).reshape(3, 4), axis=1, out=zeros(3, int64))</c> is <c>[1, 5, 9]</c> (1.5, 5.5,
        ///     9.5 truncated). NumSharp copies its median into the out under np.copyto's same_kind rule and raises
        ///     <c>InvalidCastException: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule
        ///     'same_kind'</c>. np.ptp into an integer out is NumPy's subtract refusal, <c>UFuncTypeError: Cannot cast ufunc
        ///     'subtract' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'</c> (a TypeError), where
        ///     NumSharp raises np.copyto's InvalidCastException text. (np.nanmedian's unsafe store is NumPy's since QA-31.)
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA55")]
        public void ParityQa_MedianAndPtp_IntegerOut_FollowNumPysCastRules()
        {
            using var a = np.arange(12.0).reshape(3, 4);

            using var medians = np.zeros(new Shape(3), np.int64);
            np.median(a, 1, medians);
            medians.ToArray<long>().Should().Equal(1L, 5L, 9L);

            using var ranges = np.zeros(new Shape(4), np.int64);
            new Action(() => np.ptp(a, 0, ranges)).Should().Throw<TypeError>()
                .WithMessage("Cannot cast ufunc 'subtract' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'");
        }
    }
}
