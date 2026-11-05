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
        // QA-09 — np.cumprod along a length-1 axis keeps a narrow integer dtype
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.cumprod(int32 array of shape (2, 1, 3), axis=1)</c> is int64 in NumPy (multiply.accumulate's NEP 50
        ///     accumulator: signed integers narrower than int64 → int64, unsigned → uint64), whatever the axis length.
        ///     NumSharp's length-1-axis shortcut returns the input dtype (int8 / uint8 / int16 / int32 / uint32 stay), through
        ///     <c>np.cumprod</c>, <c>ndarray.cumprod</c> and <c>np.nancumprod</c>; any longer axis promotes. np.cumsum had the
        ///     same shortcut and was fixed ("promoted to the accumulator dtype + copied", Default.Reduction.CumAdd.cs);
        ///     cumprod's twin was not.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA09")]
        public void ParityQa_CumprodLengthOneAxis_ShouldPromoteNarrowIntegers()
        {
            using var i32 = np.arange(6).astype(np.int32).reshape(2, 1, 3);
            using var u8 = np.arange(6).astype(np.uint8).reshape(2, 1, 3);

            using var signedResult = np.cumprod(i32, axis: 1);
            using var unsignedResult = np.nancumprod(u8, axis: 1);

            signedResult.dtype.Should().Be(np.int64);
            unsignedResult.dtype.Should().Be(np.uint64);
        }

        // ------------------------------------------------------------------
        // QA-17 — the float-tier binary ufuncs promote Char beside float16 / float32 to float64
        // ------------------------------------------------------------------

        /// <summary>
        ///     Char is documented as bit-identical to uint16 (its promotion was made "the uint16 masquerade" in 22589ad6), and
        ///     <c>np.result_type(char, float32)</c>, add / multiply / power / divide / fmod / maximum / fmax / float_power all give
        ///     uint16's answer, float32. The float-tier binary ufuncs give float64 instead, for float16 and float32 alike:
        ///     <c>np.hypot</c>, <c>np.arctan2</c>, <c>np.copysign</c>, <c>np.heaviside</c>, <c>np.logaddexp</c>,
        ///     <c>np.logaddexp2</c> and <c>np.nextafter</c> (DefaultEngine's float-tier dispatch ranks Char apart from uint16).
        ///     NumPy (uint16 + float32) is float32.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA17")]
        public void ParityQa_FloatTierUfuncs_CharBesideFloat32_ShouldPromoteLikeUInt16()
        {
            using var ch = np.array(new[] { 'a', 'b', 'c' });
            using var f32 = np.array(new[] { 1.5f, -2.0f, 0.25f });
            using var f16 = f32.astype(np.float16);

            using var hypot = np.hypot(ch, f32);
            using var arctan2 = np.arctan2(ch, f16);
            using var copysign = np.copysign(ch, f32);
            using var nextafter = np.nextafter(ch, f32);

            hypot.dtype.Should().Be(np.float32);
            arctan2.dtype.Should().Be(np.float32);
            copysign.dtype.Should().Be(np.float32);
            nextafter.dtype.Should().Be(np.float32);
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
        // QA-36 — fresh ufunc results over a permuted operand are C-ordered instead of NumPy's K order
        // ------------------------------------------------------------------

        /// <summary>
        ///     A ufunc allocates its output with the iterator's K order (<c>npyiter_new_temp_array</c> over
        ///     <c>npyiter_find_best_axis_ordering</c>): the axes sorted by the operands' strides, negative strides flipped.
        ///     For <c>p = np.arange(24.).reshape(3, 2, 4).transpose(1, 0, 2)</c> (strides <c>(32, 64, 8)</c>)
        ///     <c>np.negative(p)</c>, <c>p + p</c>, <c>np.where(p &gt; 3, p, 0)</c> and <c>np.cumsum(p, axis=2)</c> all keep
        ///     strides <c>(32, 64, 8)</c>, so <c>np.ravel(np.negative(p), order='K')</c> is <c>-0., -1., -2., …</c> (the
        ///     memory order). NumSharp's unary path keeps only a strictly F-contiguous layout
        ///     (<c>TryExecuteUnaryOpViaNDIter</c>: "input strict-F → allocate F-contig output … Otherwise C-contig", likewise
        ///     the binary engine's <c>bothF</c>) and returns C strides <c>(96, 32, 8)</c>, so the K-order ravel reads
        ///     <c>-0, -1, -2, -3, -8, -9, …</c> (<c>p91</c>, <c>p92</c>: every unary and binary ufunc over a permuted or a
        ///     flipped-F operand, cumsum / cumprod / nancumprod along an axis, diff, nan_to_num). Over a plain F operand the
        ///     float-tier binaries (arctan2, copysign, heaviside, hypot, logaddexp, nextafter) and cumprod / nancumprod
        ///     along an axis, diff and nan_to_num also return C where NumPy keeps F; the arithmetic ufuncs and cumsum keep F.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA36")]
        public void ParityQa_UfuncResult_PermutedOperand_ShouldKeepItsMemoryOrder()
        {
            using var p = np.arange(24.0).reshape(3, 2, 4).transpose(new[] { 1, 0, 2 });
            using var f = np.asfortranarray(np.arange(1.0, 25.0).reshape(2, 3, 4));

            using var neg = np.negative(p);
            using var sum = np.add(p, p);
            using var memoryOrder = np.ravel(neg, 'K');
            using var hyp = np.hypot(f, f);
            using var cp = np.cumprod(f, axis: 0);

            neg.strides.Should().Equal(32L, 64L, 8L);
            sum.strides.Should().Equal(32L, 64L, 8L);
            ((double)memoryOrder.GetValue(4)).Should().Be(-4.0, "a K-order ravel reads the result in memory order");
            hyp.flags.f_contiguous.Should().BeTrue("hypot of two F operands is F in NumPy");
            cp.flags.f_contiguous.Should().BeTrue("cumprod along an axis keeps an F operand's order");
        }

        // ------------------------------------------------------------------
        // QA-40 — print options flow into new threads; NumPy starts each thread at the defaults
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 keeps the print options in a <c>ContextVar</c>, and a new Python thread (3.12, the oracle's) starts
        ///     with an empty context: after <c>np.set_printoptions(precision=2)</c> — or inside <c>with np.printoptions(...)</c>
        ///     — a worker thread prints <c>np.array([1.23456789])</c> as <c>[1.23456789]</c>, the default. NumSharp keeps them in
        ///     an <c>AsyncLocal</c> (Backends/Printing/PrintOptions.cs), whose value .NET copies into every thread and task the
        ///     setting thread starts, so the worker prints <c>[1.23]</c> (<c>rp5</c>: <c>new Thread</c> and <c>Task.Run</c>
        ///     alike). A worker's own <c>set_printoptions</c> stays in the worker in both libraries, and <c>np.setbufsize</c>
        ///     (a [ThreadStatic]) already starts each thread at its default, as NumPy does.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA40")]
        public void ParityQa_PrintOptions_NewThread_ShouldStartAtTheDefaults()
        {
            string seen = null;
            using (np.printoptions(precision: 2))
            {
                var worker = new System.Threading.Thread(() =>
                {
                    using var value = np.array(new[] { 1.23456789 });
                    seen = value.ToString();
                });
                worker.Start();
                worker.Join();
            }

            seen.Should().Be("[1.23456789]", "a new thread starts with NumPy's default print options");
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

        // ------------------------------------------------------------------
        // QA-48 — every np.fft transform allocates its result in C order; NumPy keeps the input's memory order
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>_raw_fft</c> (<c>numpy/fft/_pocketfft.py</c>) allocates the result with
        ///     <c>empty_like(a, shape=..., dtype=...)</c>, whose default <c>order='K'</c> keeps the input's memory order: an
        ///     F-ordered input gives an F-ordered spectrum (<c>np.fft.fft(F, axis=1).strides == (16, 64)</c>), a permuted 3-D input
        ///     a spectrum permuted the same way, and the N-D forms keep it through every 1-D pass. NumSharp's
        ///     <c>PocketFFTDriver.Execute</c> allocates <c>TensorEngine.NewArray(outDtype, outShape, false)</c>, which is always C
        ///     order, so all 14 non-C layouts of the <c>p131</c> sweep (fft / ifft / rfft / irfft / hfft / ihfft / fft2 / rfft2 /
        ///     fftn, F, transposed and permuted inputs, any axis) come back C (<c>(96, 16)</c>). The values are identical; code
        ///     that reads the flags or strides, or passes the buffer on in memory order, sees a different array.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA48")]
        public void ParityQa_FftResult_FOrderInput_ShouldKeepTheInputsMemoryOrder()
        {
            using var c = np.arange(24.0).reshape(4, 6);
            using var f = np.asfortranarray(c);

            using var spectrum = np.fft.fft(f, axis: 1);
            using var half = np.fft.rfft(f, axis: 0);

            // NumPy: complex128 (4, 6) over an F input is (16, 64); rfft along axis 0 gives (3, 6) with strides (16, 48)
            spectrum.strides.Should().Equal(new long[] { 16, 64 }, "empty_like(a, ...) keeps an F input's order");
            spectrum.flags.f_contiguous.Should().BeTrue();
            half.strides.Should().Equal(new long[] { 16, 48 }, "the shorter axis keeps the same K order");
        }

        // ------------------------------------------------------------------
        // QA-51 — np.select / np.r_ promote their operands left to right (QA-07's bug in two more functions)
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>np.select</c> types its result <c>np.result_type(*choicelist, default)</c> and <c>np.r_</c>
        ///     <c>result_type(*arrays, *scalars)</c>: ONE order-independent NEP 50 reduction, so
        ///     <c>[int16, uint16, float16]</c> is float32 and <c>[uint8, int8, float16]</c> float16 in NumPy 2.4.2. Both
        ///     NumSharp functions fold <c>NDExprTypeRules.PromoteStrong</c> pairwise from the left
        ///     (<c>Manipulation/NDArray.FunctionBase.cs</c>, <c>Indexing/NDArray.IndexTricks.cs</c>), the fold np.choose had
        ///     until parity QA-07 replaced it with <c>NDExprTypeRules.PromoteStrongSequence</c>: float64 and float32 here.
        ///     Found on 2026-11-05 while fixing QA-07 (whose decision page had assumed np.select already matched).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA51")]
        public void ParityQa_SelectAndR_ThreeDtypes_ShouldPromoteLikeResultType()
        {
            using var i16 = np.array(new short[] { 1, 2, 3 });
            using var u16 = np.array(new ushort[] { 4, 5, 6 });
            using var f16 = np.array(new[] { 0.5, 1.5, 2.5 }).astype(np.float16);
            using var u8 = np.array(new byte[] { 1, 2, 3 });
            using var i8 = np.array(new sbyte[] { 4, 5, 6 });
            var conditions = new NDArray[] { np.array(new[] { true, false, false }), np.array(new[] { false, true, false }), np.array(new[] { false, false, true }) };

            using var selected = np.select(conditions, new object[] { i16, u16, f16 });
            using var selectedNarrow = np.select(conditions, new object[] { u8, i8, f16 });
            using var joined = np.r_[i16, u16, f16];

            selected.dtype.Should().Be(np.float32);
            selectedNarrow.dtype.Should().Be(np.float16);
            joined.dtype.Should().Be(np.float32);
        }
    }
}
