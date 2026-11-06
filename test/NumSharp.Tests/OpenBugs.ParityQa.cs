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
        // QA-08 — np.ediff1d's to_begin / to_end refusal is an ArgumentException with a parameter suffix
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>np.ediff1d(int8_array, to_begin=[1.5])</c> raises <c>TypeError: dtype of `to_begin` must be compatible
        ///     with input `ary` under the `same_kind` rule.</c> NumSharp throws <see cref="ArgumentException"/> built with a
        ///     parameter name, so the type differs and the message gains .NET's <c> (Parameter 'to_begin')</c> suffix.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA08")]
        public void ParityQa_Ediff1d_IncompatibleEnds_ShouldRaiseTypeErrorWithNumPysText()
        {
            using var a = np.array(new sbyte[] { 1, 2, 4 });
            using var begin = np.array(new[] { 1.5 });

            Action toBegin = () => np.ediff1d(a, to_begin: begin);
            Action toEnd = () => np.ediff1d(a, to_end: 2.5);

            toBegin.Should().Throw<TypeError>()
                .WithMessage("dtype of `to_begin` must be compatible with input `ary` under the `same_kind` rule.");
            toEnd.Should().Throw<TypeError>()
                .WithMessage("dtype of `to_end` must be compatible with input `ary` under the `same_kind` rule.");
        }

        // ------------------------------------------------------------------
        // QA-27 — np.divmod of a complex operand raises NotSupportedException instead of NumPy's TypeError
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy has no complex loop for <c>divmod</c> (nor for floor_divide / remainder) and raises
        ///     <c>TypeError: ufunc 'divmod' not supported for the input types, and the inputs could not be safely coerced to
        ///     any supported types according to the casting rule ''safe''</c> whenever either operand is complex. NumSharp's
        ///     np.floor_divide and np.remainder raise exactly that (their own names in it); np.divmod reaches the kernel and
        ///     throws <c>NotSupportedException: Operation DivMod not supported for Complex</c> — 25 of the 169 dtype pairs of
        ///     a 26-ufunc matrix (<c>p59</c>), the only refusal in it with the wrong type.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA27")]
        public void ParityQa_DivmodComplex_ShouldRaiseNumPysTypeError()
        {
            using var c = np.array(new[] { new Complex(1, 2), new Complex(3, -1) });
            using var f = np.array(new[] { 2.0, 1.0 });

            Action complexByFloat = () => np.divmod(c, f);

            complexByFloat.Should().Throw<TypeError>().WithMessage(
                "ufunc 'divmod' not supported for the input types, and the inputs could not be safely coerced to any " +
                "supported types according to the casting rule ''safe''");
        }

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
        // QA-30 — np.broadcast_to's refusal texts are not NumPy's
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's broadcast_to refuses through NpyIter, whose texts carry two quirks: the remapped-shapes list ends with a
        ///     space before " and requested shape " (two spaces) and a one-element shape prints as a Python tuple,
        ///     <c>operands could not be broadcast together with remapped shapes [original-&gt;remapped]: (3,)  and requested
        ///     shape (4,)</c>; a source with more axes than the target is a different message, <c>input operand has more
        ///     dimensions than allowed by the axis remapping</c>. <c>Shape.ValidateBroadcastTo</c> (View/Shape.Broadcasting.cs)
        ///     formats its own text with <c>string.Join</c>: one space, <c>(3)</c> / <c>(4)</c>, and the remapped-shapes
        ///     text for the more-axes case too. np.nditer and the gufunc guard already reproduce NumPy's formatter
        ///     (<c>np.nditer.cs</c>, <c>GufuncGuard.cs</c>). The exception type is the house <c>IncorrectShapeException</c>
        ///     for broadcast refusals, so only the texts are pinned.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA30")]
        public void ParityQa_BroadcastTo_RefusalTexts_ShouldBeNumPys()
        {
            using var row = np.zeros(new Shape(3));
            using var grid = np.zeros(new Shape(2, 3));

            Action lengthMismatch = () => np.broadcast_to(row, new Shape(4)).Dispose();
            Action moreAxes = () => np.broadcast_to(grid, new Shape(3)).Dispose();

            lengthMismatch.Should().Throw<Exception>().Which.Message.Should().Be(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (3,)  and requested shape (4,)");
            moreAxes.Should().Throw<Exception>().Which.Message.Should().Be(
                "input operand has more dimensions than allowed by the axis remapping");

            // np.broadcast_arrays / np.broadcast: NumPy 2.x names the two operands that clash (two spaces after the period).
            using var four = np.zeros(new Shape(4));
            Action arrays = () => np.broadcast_arrays(row, four);
            arrays.Should().Throw<Exception>().Which.Message.Should().Be(
                "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (3,) and arg 1 with shape (4,).");
        }

        // ------------------------------------------------------------------
        // QA-31 — out= of the wrong shape: np.cumsum (axis), np.ptp, np.median, np.nanmedian refuse with texts NumPy has not
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's refusals of an <c>out=</c> whose shape does not fit come from the iterator its statements run:
        ///     <c>np.cumsum(a, 0, out=(4, 3))</c> (accumulate over an axis) and <c>np.ptp(a, 0, out=(3,))</c>
        ///     (<c>maximum.reduce</c>) and <c>np.median(a, 0, out=(3,))</c> (<c>np.mean</c> of the partition) raise
        ///     <c>operands could not be broadcast together with remapped shapes [original-&gt;remapped]: …</c>; np.nanmedian
        ///     raises <c>could not broadcast input array from shape (4,) into shape (3,)</c>. Only the FLAT cumsum keeps
        ///     <c>provided out is the wrong size for the accumulation.</c>. NumSharp uses that flat text for the axis cumsum
        ///     too, its own <c>out has wrong shape; expected=[4] got=[3]</c> for ptp (<c>PtpOut</c>), and np.quantile's
        ///     <c>Wrong shape of argument 'out', shape=4 is required; got shape=3.</c> (right for quantile / percentile) for
        ///     median and nanmedian (<c>p70</c>). Everything else in the 26-call out= sweep matches.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA31")]
        public void ParityQa_OutShapeRefusals_CumsumAxisPtpMedian_ShouldUseNumPysTexts()
        {
            using var a = np.arange(12.0).reshape(3, 4);
            using var wrong43 = np.zeros(new Shape(4, 3));
            using var wrong3 = np.zeros(new Shape(3));

            Action cumsum = () => np.cumsum(a, 0, @out: wrong43);
            Action cumprod = () => np.cumprod(a, 0, @out: wrong43);
            Action ptp = () => np.ptp(a, 0, @out: wrong3);
            Action median = () => np.median(a, 0, @out: wrong3);

            cumsum.Should().Throw<Exception>().Which.Message.Should().Be(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (4,3)->(4,3) (3,4)->(3,4) ");
            cumprod.Should().Throw<Exception>().Which.Message.Should().Be(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (4,3)->(4,3) (3,4)->(3,4) ");
            ptp.Should().Throw<Exception>().Which.Message.Should().Be(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (3,)->(3) (3,4) ");
            median.Should().Throw<Exception>().Which.Message.Should().Be(
                "operands could not be broadcast together with remapped shapes [original->remapped]: (3,)->(3) (1,4) ");
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
        // QA-43 — np.vectorize's signature fallback wraps the callback's exception in TargetInvocationException
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's vectorize propagates whatever the Python function raises, unchanged. NumSharp's other callback routes
        ///     do the same (element-wise vectorize of any arity, frompyfunc with one or two outputs, the typed one- to
        ///     three-argument signature delegates, apply_along_axis, apply_over_axes, piecewise: <c>rp10</c>, <c>rp11</c>), but
        ///     <c>Vectorized.BuildSignatureInvoker</c>'s fallback (APIs/np.vectorize.cs: a multi-output signature such as
        ///     <c>"(n)->(),()"</c>, or more than three core arguments) calls the delegate through <c>DynamicInvoke</c>, so the
        ///     caller sees <see cref="System.Reflection.TargetInvocationException"/> "Exception has been thrown by the target of
        ///     an invocation." with the user's exception as its inner exception, and a <c>catch</c> for the user's own type
        ///     misses it.
        /// </summary>
        /// <exception cref="InvalidOperationException">Never escapes the test: the callback throws it inside
        ///     <c>CallMany</c>, where the assertion expects to observe it unwrapped.</exception>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA43")]
        public void ParityQa_VectorizeSignatureFallback_ShouldPropagateTheCallbacksException()
        {
            using var x = np.arange(4.0).reshape(2, 2);
            var twoOutputs = np.vectorize((Func<NDArray, (NDArray, NDArray)>)(v => throw new InvalidOperationException("boom")), "(n)->(),()");

            Action call = () => twoOutputs.CallMany(x);

            call.Should().Throw<InvalidOperationException>().WithMessage("boom");
        }

        // ------------------------------------------------------------------
        // QA-46 — DType == / != a string that names no dtype throws instead of answering
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>arraydescr_richcompare</c> converts the other operand and, when that fails, clears the error and
        ///     returns <c>NotImplemented</c>, so Python falls back to identity: <c>np.dtype('int32') == 'bogus'</c> is False and
        ///     <c>!= 'bogus'</c> True (likewise <c>''</c>, the structured spelling <c>'i4,i4'</c>, a number, a list). NumSharp's
        ///     <c>DType</c> operators take two <c>DType</c>s, so a string operand goes through the implicit string → DType
        ///     conversion first, which parses and throws: <c>NotSupportedException: data type 'bogus' not understood</c> (and
        ///     NumSharp's structured-dtype refusal for <c>'i4,i4'</c>) from both <c>==</c> and <c>!=</c> (<c>rp16</c>,
        ///     <c>rp17</c>). <c>DType.Equals("bogus")</c> already answers False, and every parseable spelling compares as NumPy's.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA46")]
        public void ParityQa_DTypeEqualityWithAnUnparseableString_ShouldAnswerNotThrow()
        {
            var int32 = np.dtype("int32");

            Func<bool> equal = () => int32 == "bogus";
            Func<bool> notEqual = () => int32 != "bogus";

            equal.Should().NotThrow().Which.Should().BeFalse();
            notEqual.Should().NotThrow().Which.Should().BeTrue();
        }
    }
}
