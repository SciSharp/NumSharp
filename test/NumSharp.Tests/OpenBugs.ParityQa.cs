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

        // ------------------------------------------------------------------
        // QA-06 — np.cumsum / np.nancumsum along an axis convert with .NET's saturating cast
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.cumsum([-3.0, 1.0], axis=0, dtype=np.uint8)</c> is <c>[253, 254]</c> in NumPy (its float → integer cast is
        ///     C's, which wraps on x86-64), and so is NumSharp's flat form (no <c>axis</c>) and every <c>np.sum</c> /
        ///     <c>np.prod</c> / <c>np.cumprod</c> route. With an explicit <c>axis</c> NumSharp returns <c>[0, 1]</c>: the axis
        ///     scan kernel (<c>ILKernelGenerator.ScanAddWiden</c> → <c>ConvIn</c>, ILKernelGenerator.Reduction.cs) converts each
        ///     element with <c>TAccum.CreateTruncating</c>, which SATURATES a floating-point source (-3.0 → 0, 300.0 → 127 for
        ///     int8, inf → MaxValue, NaN → 0), on net8.0 and net10.0 alike.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA06")]
        public void ParityQa_CumsumAxis_IntegerDtype_ShouldCastLikeNumPy()
        {
            using var neg = np.array(new[] { -3.0, 1.0 });
            using var big = np.array(new[,] { { 300.0, 1.0 } });

            using var unsigned = np.cumsum(neg, axis: 0, typeCode: np.uint8);
            using var signed = np.cumsum(big, axis: 1, typeCode: np.int8);

            unsigned.ToArray<byte>().Should().Equal((byte)253, (byte)254);
            signed.ToArray<sbyte>().Should().Equal((sbyte)44, (sbyte)45);
        }

        /// <summary>
        ///     <c>np.nancumsum([[-3.0, nan, 1.0]], axis=1, dtype=np.uint8)</c> is <c>[[253, 253, 254]]</c> in NumPy; NumSharp's
        ///     axis route reaches the same saturating <c>ConvIn</c> as <see cref="ParityQa_CumsumAxis_IntegerDtype_ShouldCastLikeNumPy"/>
        ///     and returns <c>[[0, 0, 1]]</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA06")]
        public void ParityQa_NancumsumAxis_IntegerDtype_ShouldCastLikeNumPy()
        {
            var nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            using var a = np.array(new[,] { { -3.0, nan, 1.0 } });

            using var r = np.nancumsum(a, axis: 1, dtype: np.uint8);

            r.ToArray<byte>().Should().Equal((byte)253, (byte)253, (byte)254);
        }

        // ------------------------------------------------------------------
        // QA-07 — np.choose promotes its choices left to right instead of NumPy's order-independent result_type
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.choose([0, 1, 2], [int16, uint16, float16])</c> is float32 in NumPy: <c>PyArray_ConvertToCommonType</c> runs
        ///     <c>PyArray_ResultType</c> over all choices at once (NEP 50's order-independent reduction, which NumSharp's own
        ///     <c>np.result_type</c>, <c>np.concatenate</c>, <c>np.stack</c> and <c>np.select</c> follow). NumSharp's
        ///     <c>np.choose</c> promotes pairwise from the left, so int16 + uint16 → int32 and int32 + float16 → float64; the
        ///     same choices in the order <c>[float16, int16, uint16]</c> give float32.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA07")]
        public void ParityQa_Choose_ThreeDtypes_ShouldPromoteLikeResultType()
        {
            using var index = np.array(new long[] { 0, 1, 2 });
            using var i16 = np.array(new short[] { 1, 2, 3 });
            using var u16 = np.array(new ushort[] { 4, 5, 6 });
            using var f16 = np.array(new[] { 0.5, 1.5, 2.5 }).astype(np.float16);

            using var r = np.choose(index, new NDArray[] { i16, u16, f16 });

            r.dtype.Should().Be(np.float32);
            r.ToArray<float>().Should().Equal(1f, 5f, 2.5f);
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
        // QA-10 — complex np.prod / np.cumprod multiply with the SIMD form, not NumPy's reduce / accumulate form
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's complex <c>multiply.reduce</c> (<c>np.prod</c>) and <c>multiply.accumulate</c> (<c>np.cumprod</c>) run
        ///     <c>CDOUBLE_multiply</c>'s <c>loop_scalar</c> (the output stride is 0, or the output overlaps the input), which
        ///     MSVC contracts to <c>im = fma(a.im, b.re, round(a.re * b.im))</c>; the elementwise form is <c>simd_cmul</c>,
        ///     <c>im = fma(a.re, b.im, round(a.im * b.re))</c>. NumSharp's plain <c>np.prod</c> and every <c>np.cumprod</c> use
        ///     the elementwise form, so imaginary parts differ by an ULP often (several of 64 random values in the journal's
        ///     probe; <c>np.prod</c> with <c>initial=</c> / <c>where=</c> already takes the reduce form,
        ///     Default.Reduction.AddMul.Keywords.cs). For <c>x = -3+1e10j</c>, <c>x**4</c>'s imaginary part is NumPy's
        ///     <c>1.2e+31</c> and NumSharp's <c>1.2000000000000001e+31</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA10")]
        public void ParityQa_ComplexProdAndCumprod_ShouldUseNumPysReduceMultiplyForm()
        {
            var x = new Complex(-3, 1e10);
            using var c = np.array(new[] { x, x, x, x });

            using var prod = np.prod(c);
            using var cumprod = np.cumprod(c);

            ((Complex)prod.GetValue()).Imaginary.Should().Be(1.2e31);
            ((Complex)cumprod.GetValue(3)).Imaginary.Should().Be(1.2e31);
        }

        // ------------------------------------------------------------------
        // QA-11 — np.compress checks a too-long condition even when an earlier axis is empty
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.compress([True, False, True], np.zeros((0, 2)), axis=1)</c> returns an empty <c>(0, 2)</c> array in NumPy:
        ///     compress is <c>take(a, nonzero(condition), axis)</c>, and <c>PyArray_TakeFrom</c> validates each index inside a
        ///     loop over the dimensions BEFORE the axis, which runs zero times when one of them is 0 (NumSharp's own
        ///     <c>np.take</c> already follows that). NumSharp's compress checks the condition's tail first
        ///     (<c>UnmanagedStorage.ItemSelection.cs</c>, "OOB-True detection") and throws
        ///     <see cref="IndexOutOfRangeException"/> "index 2 is out of bounds for axis 1 with size 2".
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA11")]
        public void ParityQa_Compress_EmptyLeadingAxis_LongCondition_ShouldReturnEmpty()
        {
            using var cond = np.array(new[] { true, false, true });
            using var a = np.zeros(new Shape(0, 2));

            using var r = np.compress(cond, a, axis: 1);

            r.shape.Should().Equal(0L, 2L);
        }

        // ------------------------------------------------------------------
        // QA-12 — NDArray.SetValue<T> / itemset<T> with another T write raw bytes at T-scaled offsets (Release)
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>a.SetValue(2.5, 0)</c> on a float32 array binds <c>SetValue&lt;double&gt;(T, params long[])</c> (C# prefers the
        ///     generic exact match over the converting <c>SetValue(object, ...)</c>), whose doc promises "value will be
        ///     converted"; the body is <c>*((T*)Address + offset) = value</c> behind a <c>Debug.Assert</c> only
        ///     (UnmanagedStorage.Setters.cs). In Release the double's 8 bytes land at element offset × 8: index 0 leaves
        ///     <c>[0, 2.0625, 3, 4]</c>, index 1 overwrites elements 2 and 3, an int into a uint8 array writes 4 bytes at
        ///     byte 4 × index, and the last element's write runs past the buffer. <c>itemset&lt;T&gt;</c> is the same body
        ///     (its own "//TODO! if T != dtype, we need to cast!"). The documented trap <c>SetAtIndex&lt;T&gt;</c>
        ///     (OpenBugs.Surface.cs) has the same cause; these two public members were not listed with it. EXPECTED (when
        ///     fixed): the value is cast into the addressed element (NumPy's <c>a[0] = 2.5</c>), or the call refuses, and no
        ///     neighbour changes.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA12")]
        public void ParityQa_SetValueGeneric_OtherElementType_ShouldCastNotWriteRawBytes()
        {
#if DEBUG
            // The Debug.Assert fires in a Debug build (and may end the test host): the defect is the Release behaviour.
            Assert.Inconclusive("Release-only defect: in Debug the type mismatch trips SetValue<T>'s Debug.Assert.");
#else
            using var a = np.array(new[] { 1f, 2f, 3f, 4f });
            using var b = np.array(new[] { 1f, 2f, 3f, 4f });
            try
            {
                a.SetValue(2.5, 0);
                b.itemset(new[] { 1 }, 2.5);
            }
            catch (Exception)
            {
                // A refusal is an acceptable fix; the neighbours must still be untouched (checked below).
            }

            var va = a.ToArray<float>();
            var vb = b.ToArray<float>();
            va[1].Should().Be(2f);
            va[0].Should().BeOneOf(1f, 2.5f);
            vb[2].Should().Be(3f);
            vb[3].Should().Be(4f);
            vb[1].Should().BeOneOf(2f, 2.5f);
#endif
        }

        /// <summary>
        ///     The getters share the raw access: <c>a.GetValue&lt;double&gt;(1)</c>, <c>a.GetAtIndex&lt;double&gt;(1)</c> and
        ///     <c>a.GetDouble(1)</c> on float32 <c>[1, 2, 3, 4]</c> return <c>512.0001225471497</c> (elements 2 and 3 read as one
        ///     double) in Release, and the last element's read runs past the buffer. <c>GetDouble</c>'s doc says it throws
        ///     <c>NullReferenceException</c> for another dtype; <c>a.item&lt;double&gt;(1)</c> converts correctly (2). EXPECTED
        ///     (when fixed): the element converted (2.0) or a refusal.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA12")]
        public void ParityQa_GetValueGeneric_OtherElementType_ShouldCastNotReadRawBytes()
        {
#if DEBUG
            Assert.Inconclusive("Release-only defect: in Debug the type mismatch trips GetValue<T>'s Debug.Assert.");
#else
            using var a = np.array(new[] { 1f, 2f, 3f, 4f });
            double generic = double.NaN, typed = double.NaN;
            try
            {
                generic = a.GetValue<double>(1);
                typed = a.GetDouble(1);
            }
            catch (Exception)
            {
                return; // a refusal is an acceptable fix
            }

            generic.Should().Be(2.0);
            typed.Should().Be(2.0);
#endif
        }

        // ------------------------------------------------------------------
        // QA-13 — np.broadcast_to accepts a negative target dimension
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.broadcast_to(np.arange(3), (-1, 3))</c> is NumPy's <c>ValueError: all elements of broadcast shape must be
        ///     non-negative</c>. NumSharp returns an array of shape <c>(-1, 3)</c> and size -3: <c>np.sum</c> of it answers 0,
        ///     its <c>ToString()</c> throws an IndexError ("index -2 is out of bounds for axis 0 with size -1"), its copy throws
        ///     "negative dimensions are not allowed", and <c>np.broadcast_arrays</c> hands the negative shape on to its other
        ///     operand. <c>Shape</c>'s constructor does not validate (only allocation does), and broadcast_to builds a view.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA13")]
        public void ParityQa_BroadcastTo_NegativeDimension_ShouldRaiseValueError()
        {
            using var a = np.arange(3);

            Action act = () => np.broadcast_to(a, new Shape(-1, 3)).Dispose();

            act.Should().Throw<ValueError>().WithMessage("all elements of broadcast shape must be non-negative");
        }

        // ------------------------------------------------------------------
        // QA-14 — the explicit (bool) conversion applies float()'s 0-d rule instead of NumPy's truth test
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>bool(arr)</c> is <c>array_nonzero</c>: any one-element array has a truth value
        ///     (<c>bool(np.array([5.0]))</c> is True, <c>bool(np.array([[0.0]]))</c> False), and only a larger or an empty array
        ///     is refused, with <c>ValueError: The truth value of an array with more than one element is ambiguous. Use a.any()
        ///     or a.all()</c>. NumSharp's explicit conversions all share <c>EnsureCastableToScalar</c>
        ///     (Casting/Implicit/NDArray.Implicit.ValueTypes.cs), the 0-d rule of NumPy 2.x's <c>float()</c> / <c>int()</c>,
        ///     so <c>(bool)</c> of a one-element 1-D or 2-D array throws <see cref="IncorrectShapeException"/> "only 0-d arrays
        ///     can be converted to scalar", and a larger array gets that text instead of NumPy's ValueError.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA14")]
        public void ParityQa_BoolConversion_ShouldFollowNumPysTruthTest()
        {
            using var one = np.array(new[] { 5.0 });
            using var oneZero = np.array(new[,] { { 0.0 } });
            using var two = np.array(new[] { 1.0, 2.0 });

            ((bool)one).Should().BeTrue();
            ((bool)oneZero).Should().BeFalse();
            Action ambiguous = () => _ = (bool)two;
            ambiguous.Should().Throw<ValueError>()
                .WithMessage("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
        }

        /// <summary>
        ///     NumPy 2.4.2's <c>float(np.array([5.5]))</c> raises <c>TypeError: only 0-dimensional arrays can be converted to
        ///     Python scalars</c>. NumSharp refuses the same calls (the explicit <c>(double)</c> / <c>(int)</c> / ... operators)
        ///     but with <see cref="IncorrectShapeException"/> and the text "only 0-d arrays can be converted to scalar", so a
        ///     caller catching NumPy's <see cref="TypeError"/> misses it.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA14")]
        public void ParityQa_ScalarConversion_NonZeroDim_ShouldRaiseNumPysTypeError()
        {
            using var one = np.array(new[] { 5.5 });

            Action act = () => _ = (double)one;

            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        // ------------------------------------------------------------------
        // QA-15 — np.nansum is not np.sum of the NaN-replaced array
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>nansum</c> is <c>a, mask = _replace_nan(a, 0); return np.sum(a, ...)</c>, so its bits are np.sum's.
        ///     NumSharp's np.sum matches NumPy bit for bit on these inputs, but np.nansum runs its own NaN-skipping kernels in
        ///     another order: float16 along an axis drifts by several ULP (column 0 of the (10, 10) case below is
        ///     <c>168.25</c>, NumPy <c>168.375</c>; a (1000, 10) column is <c>-3396</c> against <c>-3410</c>), float32 / float64
        ///     by one to three ULP flat and per axis, and np.nanprod likewise. The oracle excuses the whole cell as a known bug
        ///     (MisalignedRegistry "(W4-B) nansum/nanmean/nanprod … summation-order divergence [known bug]"; nanmean has
        ///     since left it), with no unit pin until this one. Inputs come from <c>RandomState(n).random_sample(n) * 200 -
        ///     100</c> with every 17th element NaN, which NumSharp reproduces bit for bit.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA15")]
        public void ParityQa_Nansum_ShouldBeSumOfTheNaNReplacedArray()
        {
            var nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            var rs100 = np.random.RandomState(100);
            using var x100 = rs100.random_sample(100L) * 200.0 - 100.0;
            x100["::17"] = nan;
            using var f16 = x100.astype(np.float16).reshape(10, 10);

            using var columns = np.nansum(f16, axis: 0);

            columns.ToArray<Half>().Select(h => (double)h).Should().Equal(
                168.375, -122.75, -216.75, -59.21875, -158.5, 190.25, 162.0, -149.875, -184.875, -48.21875);

            var rs1000 = np.random.RandomState(1000);
            using var x1000 = rs1000.random_sample(1000L) * 200.0 - 100.0;
            x1000["::17"] = nan;

            using var flat = np.nansum(x1000);

            BitConverter.DoubleToInt64Bits((double)flat).Should().Be(BitConverter.DoubleToInt64Bits(-2291.4343639665312));
        }

        // ------------------------------------------------------------------
        // QA-16 — the flat float16 np.prod accumulates wider than NumPy's HALF loop
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's float16 <c>multiply.reduce</c> (<c>HALF_multiply</c>'s reduce form) keeps a FLOAT32 accumulator, so
        ///     <c>np.prod(float16 [200] * 20 + [0])</c> overflows to inf at the 17th factor and the zero makes it NaN. NumSharp's
        ///     flat float16 product (plain float16 input, and <c>dtype=float16</c> over an integer input) accumulates wider,
        ///     never overflows, and returns 0. The axis forms and <c>np.cumprod</c> give NumPy's NaN, as do the float32 forms.
        ///     The same accumulator difference turns a product whose float32 partial overflows and later shrinks back into a
        ///     finite float16 where NumPy keeps inf.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA16")]
        public void ParityQa_ProdFlatFloat16_ShouldAccumulateInFloat32LikeNumPy()
        {
            var values = Enumerable.Repeat(200.0, 20).Concat(new[] { 0.0 }).ToArray();
            using var f64 = np.array(values);
            using var f16 = f64.astype(np.float16);
            using var u8 = f64.astype(np.uint8);

            using var plain = np.prod(f16);
            using var cast = np.prod(u8, dtype: np.float16);

            Half.IsNaN((Half)plain.GetValue()).Should().BeTrue();
            Half.IsNaN((Half)cast.GetValue()).Should().BeTrue();
        }

        /// <summary>
        ///     The other direction along a reduced axis: <c>np.prod(float16 [[300, 300, 0.001]], axis=1)</c> is
        ///     <c>[90.0625]</c> in NumPy (the reduce inner loop multiplies in float32: 90000 fits, then × 0.001); NumSharp
        ///     rounds each partial product to float16, so 90000 overflows to inf and the result is <c>[inf]</c>. And the flat
        ///     form's wider accumulator keeps <c>[60000] * 9 + [1e-4] * 10</c> finite (1009.5) where NumPy's float32
        ///     accumulator is already inf.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA16")]
        public void ParityQa_ProdFloat16_AxisAndFlat_ShouldUseNumPysFloat32Accumulator()
        {
            using var row = np.array(new[,] { { 300.0, 300.0, 0.001 } }).astype(np.float16);
            using var big = np.array(Enumerable.Repeat(60000.0, 9).Concat(Enumerable.Repeat(1e-4, 10)).ToArray()).astype(np.float16);

            using var axis = np.prod(row, axis: 1);
            using var flat = np.prod(big);

            ((double)(Half)axis.GetValue(0)).Should().Be(90.0625);
            Half.IsPositiveInfinity((Half)flat.GetValue()).Should().BeTrue();
        }
    }
}
