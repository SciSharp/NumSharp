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
        // QA-18 — complex np.maximum / np.fmax keep the second operand on a tie
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>CDOUBLE_maximum</c> / <c>CDOUBLE_fmax</c> (loops.c.src) keep <c>in1</c> when <c>CGE(in1, in2)</c>
        ///     holds, and <c>CGE</c> is non-strict (<c>xr == yr &amp;&amp; xi &gt;= yi</c>), so two operands that compare equal
        ///     return the FIRST one's bits: <c>np.maximum(0j, complex(-0.0, -0.0))</c> is <c>0j</c>. NumSharp's elementwise
        ///     kernel calls the clip helpers <c>ComplexMaxNaN</c> / <c>ComplexMaxNum</c>
        ///     (<c>DirectILKernelGenerator.Clip.cs</c>), whose last step <c>a.Imaginary &gt; b.Imaginary ? a : b</c> is strict,
        ///     and returns the SECOND operand: a silent sign-of-zero difference whenever the operands are equal but differ in
        ///     a zero's sign. np.minimum / np.fmin (non-strict on the other side) and the np.max / np.min reductions match.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA18")]
        public void ParityQa_ComplexMaximumFmax_Tie_ShouldKeepTheFirstOperand()
        {
            using var a = np.array(new[] { new Complex(0.0, 0.0), new Complex(0.0, -0.0) });
            using var b = np.array(new[] { new Complex(-0.0, -0.0), new Complex(-0.0, 0.0) });

            using var maximum = np.maximum(a, b);
            using var fmax = np.fmax(a, b);

            foreach (var result in new[] { maximum, fmax })
            {
                var first = (Complex)result.GetValue(0);
                var second = (Complex)result.GetValue(1);

                // NumPy returns a's bits: (+0, +0) and (+0, -0).
                double.IsNegative(first.Real).Should().BeFalse();
                double.IsNegative(first.Imaginary).Should().BeFalse();
                double.IsNegative(second.Real).Should().BeFalse();
                double.IsNegative(second.Imaginary).Should().BeTrue();
            }
        }

        // ------------------------------------------------------------------
        // QA-19 — complex np.clip propagates a bound's NaN imaginary part and keeps x on an upper-bound tie
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's complex clip (clip.cpp) is <c>MIN(MAX(x, lo), hi)</c> with <c>MAX(a, b) = isnan(a) || CGT(a, b) ? a : b</c>
        ///     and <c>CGT(p, q) = p.re == q.re ? p.im &gt; q.im : p.re &gt; q.re</c> (<c>MIN</c> the mirror with <c>CLT</c>). The
        ///     bound's NaN is never tested on its own: a bound whose IMAGINARY part is NaN only wins where the real parts tie,
        ///     so <c>np.clip([5+0j, 20+0j], complex(1, nan), 10+0j)</c> is <c>[5+0j, 10+0j]</c>. NumSharp's
        ///     <c>ComplexMaxNaN</c> / <c>ComplexMinNaN</c> return any NaN-bearing bound first and give
        ///     <c>[1+nanj, 1+nanj]</c>; an upper bound <c>10+nanj</c> likewise replaces every element. The tie rule differs too:
        ///     NumPy's <c>MIN</c> takes the bound on a tie (strict <c>CLT</c>), NumSharp keeps x, so
        ///     <c>np.clip(complex(0.0, -0.0), -5-5j, complex(-0.0, 0.0))</c> keeps x's zero signs where NumPy returns the
        ///     bound's.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA19")]
        public void ParityQa_ComplexClip_ShouldUseNumPysCgtCltSelection()
        {
            using var x = np.array(new[] { new Complex(5, 0), new Complex(20, 0) });
            using var nanLower = np.clip(x, new Complex(1, double.NaN), new Complex(10, 0));
            using var nanUpper = np.clip(x, new Complex(1, 0), new Complex(10, double.NaN));

            ((Complex)nanLower.GetValue(0)).Should().Be(new Complex(5, 0));
            ((Complex)nanLower.GetValue(1)).Should().Be(new Complex(10, 0));
            ((Complex)nanUpper.GetValue(0)).Should().Be(new Complex(5, 0));

            using var t = np.array(new[] { new Complex(0.0, -0.0) });
            using var lo = np.array(new[] { new Complex(-5, -5) });
            using var hi = np.array(new[] { new Complex(-0.0, 0.0) });
            using var tie = np.clip(t, lo, hi);

            // NumPy: CLT(x, hi) is false on the tie, so MIN returns hi = (-0, +0).
            var z = (Complex)tie.GetValue(0);
            double.IsNegative(z.Real).Should().BeTrue();
            double.IsNegative(z.Imaginary).Should().BeFalse();
        }

        // ------------------------------------------------------------------
        // QA-20 — float32 / float64 np.fmax / np.fmin signed-zero ties at NumPy's scalar positions
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's float <c>fmax</c> / <c>fmin</c> run the SIMD body (<c>npyv_maxp</c> / <c>npyv_minp</c>: the second
        ///     operand on a ±0 tie) only over whole vectors; every element outside them (an array shorter than one vector,
        ///     4 float64 / 8 float32 lanes on AVX2, and the tail after the last vector) calls the C library's
        ///     <c>fmax</c> / <c>fmin</c>, which on win-amd64 return <c>+0.0</c> / <c>-0.0</c> whatever the order. So
        ///     <c>np.fmax([0.0], [-0.0])</c> is <c>[0.0]</c> and <c>np.fmin([-0.0], [0.0])</c> is <c>[-0.0]</c>, and a 7-element
        ///     float32 fmax of +0 and -0 is +0 everywhere. NumSharp applies the second-operand rule at every position and
        ///     returns <c>-0.0</c> / <c>0.0</c>. np.maximum / np.minimum (second operand everywhere in NumPy too) and every
        ///     float16 form match. Partly known: the oracle ledger's S2 row excuses the float32 case on a reversed view.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA20")]
        public void ParityQa_FmaxFmin_SignedZeroTie_ShouldFollowNumPysScalarPositions()
        {
            using var pz = np.array(new[] { 0.0 });
            using var nz = np.array(new[] { -0.0 });
            using var pz32 = np.zeros(new Shape(7), np.float32);
            using var nz32 = np.negative(np.zeros(new Shape(7), np.float32));

            using var fmax = np.fmax(pz, nz);
            using var fmin = np.fmin(nz, pz);
            using var fmax32 = np.fmax(pz32, nz32);

            double.IsNegative((double)fmax.GetValue(0)).Should().BeFalse();
            double.IsNegative((double)fmin.GetValue(0)).Should().BeTrue();
            for (int i = 0; i < 7; i++)
                float.IsNegative((float)fmax32.GetValue(i)).Should().BeFalse();
        }

        // ------------------------------------------------------------------
        // QA-21 — np.evaluate's complex Max / Min root reductions fold in memory order and keep the later Max tie
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy reduces complex max / min with the scalar <c>CDOUBLE_maximum</c> / <c>CDOUBLE_minimum</c> loop in LOGICAL
        ///     order (a reduction never negates a stride), so the first NaN-bearing element met in logical order sticks
        ///     (<c>np.max(cn[::-1])</c> of <c>[nan+1j, 2, nan+2j, 1]</c> is <c>nan+2j</c>) and a tie keeps the earlier
        ///     element (<c>np.max([0j, complex(-0.0, -0.0)])</c> is <c>0j</c>). np.max / np.min match;
        ///     <c>np.evaluate(NDExpr.Max(x))</c> / <c>NDExpr.Min(x)</c> over complex fold through the engine's clamp
        ///     (<c>EmitScalarOperation(BinaryOp.Maximum)</c>, NDExpr.Evaluate.cs) walking MEMORY order, so a reversed view
        ///     returns <c>nan+1j</c>, and the clamp's Max returns the later operand on a tie (QA-18's helper). The float
        ///     forms, which run the ported <c>simd_reduce_c_max</c> schedule, match.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA21")]
        public void ParityQa_EvaluateComplexMaxMin_ShouldFoldInLogicalOrderKeepingTheEarlierTie()
        {
            using var cn = np.array(new[] { new Complex(double.NaN, 1), new Complex(2, 0), new Complex(double.NaN, 2), new Complex(1, 0) });
            using var reversed = cn["::-1"];

            using var max = np.evaluate(NDExpr.Max((NDExpr)reversed));
            using var min = np.evaluate(NDExpr.Min((NDExpr)reversed));

            // NumPy: the first NaN-bearing element in logical order of the reversed view is nan+2j.
            ((Complex)max.GetValue()).Imaginary.Should().Be(2);
            ((Complex)min.GetValue()).Imaginary.Should().Be(2);

            using var zeros = np.array(new[] { new Complex(0.0, 0.0), new Complex(-0.0, -0.0) });
            using var tie = np.evaluate(NDExpr.Max((NDExpr)zeros));

            // NumPy's CGE keeps the accumulator (the earlier 0j) on the tie.
            double.IsNegative(((Complex)tie.GetValue()).Real).Should().BeFalse();
        }

        // ------------------------------------------------------------------
        // QA-22 — np.expm1 / np.log1p cancel catastrophically for small arguments (known in float-math.md, unpinned)
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's float32 / float64 <c>expm1</c> / <c>log1p</c> call the C library's <c>expm1</c> / <c>log1p</c>, which
        ///     are accurate near zero: <c>np.expm1(1e-8)</c> is <c>1.000000005e-08</c>, <c>np.log1p(1e-300)</c> is
        ///     <c>1e-300</c>, both keep <c>-0.0</c>, and float32 <c>np.expm1(6e-8)</c> is <c>5.9999998e-08</c>. NumSharp
        ///     computes <c>Exp(x) - 1</c> / <c>Log(1 + x)</c> (.NET's <c>double.ExpM1</c> / <c>LogP1</c> are those
        ///     expressions), so the leading digits cancel: <c>expm1(1e-8)</c> is <c>9.99999993922529e-09</c>,
        ///     <c>expm1(1e-300)</c> and <c>log1p(1e-300)</c> are <c>0</c>, <c>-0.0</c> becomes <c>+0.0</c>, and the float32
        ///     <c>expm1(6e-8)</c> is <c>1.1920929e-07</c> (twice the true value). float-math.md calls the accuracy defect "a
        ///     separate, open bug"; the oracle excuses it (ledger S1, "≤ 2 ULP or ≤ ~ulp(1) abs"), and no pin existed.
        ///     The float16 loops are right (they widen to float32 and call NumPy's float32 routine).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA22")]
        public void ParityQa_Expm1Log1p_SmallArguments_ShouldNotCancel()
        {
            using var x = np.array(new[] { 1e-8, 1e-300, -0.0 });
            using var x32 = np.array(new[] { 6e-8f });

            using var expm1 = np.expm1(x);
            using var log1p = np.log1p(x);
            using var expm1f = np.expm1(x32);

            ((double)expm1.GetValue(0)).Should().Be(1.000000005e-08);
            ((double)log1p.GetValue(0)).Should().Be(9.999999950000001e-09);
            ((double)expm1.GetValue(1)).Should().Be(1e-300);
            ((double)log1p.GetValue(1)).Should().Be(1e-300);
            double.IsNegative((double)expm1.GetValue(2)).Should().BeTrue();
            double.IsNegative((double)log1p.GetValue(2)).Should().BeTrue();
            ((float)expm1f.GetValue(0)).Should().Be((float)5.99999978589949e-08);
        }

        // ------------------------------------------------------------------
        // QA-23 — a leading boolean mask over a strided view whose trailing extents are all 1 raises
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>a[mask, 0:1]</c> (a mask on axis 0, then a slice that keeps ONE column) is NumPy's <c>(k, 1)</c> selection:
        ///     <c>np.arange(6).reshape(2, 3)[[False, True], 0:1]</c> is <c>[[3]]</c>, and so are <c>a[mask, -1:]</c>,
        ///     <c>b3[mask, 2:3, 3:4]</c> and a 2-D mask followed by a one-element slice. NumSharp raises
        ///     <c>ValueError: non-broadcastable operand with shape (2,1) doesn't match the broadcast shape (2,2)</c> (the 2-D
        ///     mask form: <c>IncorrectShapeException</c>). The getter slices first (<c>this[basic]</c>, a strided
        ///     <c>(2, 1)</c> view) and hands the leading mask to <c>DefaultEngine.BooleanMask</c>, which uses the mask as a
        ///     FULL element mask whenever the trailing block size is 1 (<c>gatherMask = blockSize == 1 ? mask : …</c>,
        ///     Default.BooleanMask.cs); off the contiguous SIMD route, its NDIter gather then right-aligns the <c>(2,)</c>
        ///     mask against the <c>(2, 1)</c> operand. The same happens for a plain strided <c>(n, 1)</c> view masked by an
        ///     <c>(n,)</c> mask. A trailing extent above 1, a contiguous operand and the setter are fine.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA23")]
        public void ParityQa_LeadingMask_StridedViewWithUnitTrailingExtents_ShouldSelectRows()
        {
            using var a = np.arange(6L).reshape(new Shape(2, 3));
            using var mask = np.array(new[] { false, true });

            using var oneColumn = a[mask, "0:1"];
            using var lastColumn = a[mask, "-1:"];

            oneColumn.shape.Should().Equal(new long[] { 1, 1 });
            oneColumn.GetInt64(0, 0).Should().Be(3);
            lastColumn.GetInt64(0, 0).Should().Be(5);

            using var column = np.arange(6L).reshape(new Shape(3, 2))[":, :1"];
            using var rows = column[np.array(new[] { true, false, true })];

            rows.shape.Should().Equal(new long[] { 2, 1 });
            rows.GetInt64(1, 0).Should().Be(4);
        }

        // ------------------------------------------------------------------
        // QA-24 — float16 arctan2 / log1p / expm1 round a float64 result once where NumPy's HALF loops go through float32
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy computes float16 <c>arctan2</c>, <c>log1p</c> and <c>expm1</c> in float32 (<c>atan2f</c> /
        ///     <c>npy_log1pf</c> / <c>npy_expm1f</c>) and rounds that float32 result to float16, so a float32 result on a
        ///     float16 rounding midpoint ties to even. NumSharp computes them in float64 (<c>EmitATan2Operation</c>'s
        ///     <c>Math.Atan2</c>; the expm1 / log1p "double bridge", whose comment calls it byte-identical to NumPy's half
        ///     loop) and rounds once, so those inputs land on the other neighbour: over every float16 pattern (arctan2
        ///     against a seeded permutation) 13 arctan2, 2 log1p and 2 expm1 results differ by one float16 ULP.
        ///     <c>np.arctan2(float16 26.17, float16 -0.00011706)</c> is <c>1.57</c> (0x3E48; float32 <c>1.5708008</c> is the
        ///     midpoint) where NumSharp gives <c>1.571</c>; <c>np.log1p(float16 0.00587)</c> is <c>0.00585</c> (0x1DFE);
        ///     <c>np.expm1(float16 -0.663)</c> is <c>-0.4849</c> (0xB7C2). Rounding the float64 result to float32 first
        ///     reproduces NumPy on all 65,536 inputs for arctan2 and log1p (expm1 needs NumPy's float32 routine itself).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA24")]
        public void ParityQa_Float16Arctan2Log1pExpm1_ShouldComputeInFloat32LikeNumPysHalfLoops()
        {
            static NDArray H(ushort bits) => np.array(new[] { BitConverter.UInt16BitsToHalf(bits) });
            static ushort Bits(NDArray r) => BitConverter.HalfToUInt16Bits((Half)r.GetValue(0));

            using var y = H(0x4E8B);
            using var x = H(0x87AC);
            using var p = H(0x1E03);
            using var q = H(0xB94E);

            using var arctan2 = np.arctan2(y, x);
            using var log1p = np.log1p(p);
            using var expm1 = np.expm1(q);

            Bits(arctan2).Should().Be(0x3E48);
            Bits(log1p).Should().Be(0x1DFE);
            Bits(expm1).Should().Be(0xB7C2);
        }

        /// <summary>
        ///     QA-24's float32 sibling: NumPy's float32 <c>arctan2</c> is the C library's <c>atan2f</c>; NumSharp's
        ///     <c>EmitATan2Operation</c> widens float32 to double, calls <c>Math.Atan2</c> and rounds back, so where
        ///     <c>atan2f</c> is not the correctly rounded value the two differ by one float32 ULP: 122 of 2,097,152 dense
        ///     float32 pairs (<c>p75</c>). <c>np.arctan2(float32 0xF5B024C4, float32 0x7C96E3DC)</c> is <c>0xB8956C2E</c>
        ///     (-7.125024e-05) in NumPy and <c>0xB8956C2D</c> in NumSharp. <c>MathF.Atan2</c> calls the same <c>atan2f</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA24")]
        public void ParityQa_Float32Arctan2_ShouldBeAtan2fLikeNumPy()
        {
            using var y = np.array(new[] { BitConverter.UInt32BitsToSingle(0xF5B024C4) });
            using var x = np.array(new[] { BitConverter.UInt32BitsToSingle(0x7C96E3DC) });

            using var r = np.arctan2(y, x);

            BitConverter.SingleToUInt32Bits((float)r.GetValue(0)).Should().Be(0xB8956C2E);
        }

        // ------------------------------------------------------------------
        // QA-25 — complex np.log1p near the origin: the real part is up to 4.4e12 ULP from NumPy's
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>nc_log1p</c> is <c>real = log(hypot(x + 1, y))</c>, <c>imag = atan2(y, x + 1)</c>, and on win-amd64
        ///     <c>npy_hypot</c> is the UCRT's faithful <c>hypot</c>. NumSharp ports the same statements
        ///     (<c>NDComplexMath.Log1p</c>) but its <c>Hypot</c> is the correctly rounded one (the oracle documents the
        ///     float64 hypot's 1-ULP difference). Near the origin with <c>|y| ≫ |x|</c> the magnitude is <c>1 + ~4e-13</c>,
        ///     so a 1-ULP magnitude difference (2.2e-16) becomes a relative difference of 5e-4 after the log:
        ///     <c>np.log1p(complex(-1.3339345080004047e-16, 9.344550712680563e-07)).real</c> is <c>4.365396932825163e-13</c> in
        ///     NumPy and <c>4.3631764867759135e-13</c> in NumSharp (4.4e12 ULP apart). A dense 1M-value complex sample
        ///     (<c>p54</c>) finds 104 such real parts beyond the complex-unary family's documented ≤ 3-ULP envelope.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA25")]
        public void ParityQa_ComplexLog1p_NearOrigin_ShouldMatchNumPysHypotRounding()
        {
            using var z = np.array(new[] { new Complex(-1.3339345080004047e-16, 9.344550712680563e-07) });

            using var r = np.log1p(z);

            ((Complex)r.GetValue(0)).Real.Should().Be(4.365396932825163e-13);
        }

        // ------------------------------------------------------------------
        // QA-26 — the float32 / float64 value sort puts every -0.0 before every +0.0
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's comparators treat <c>-0.0</c> and <c>+0.0</c> as equal, so a stable sort keeps their input order:
        ///     <c>np.sort([0.0, -0.0, 1.0, -0.0, 0.0], kind='stable')</c> is <c>[0.0, -0.0, -0.0, 0.0, 1.0]</c> (timsort), and
        ///     the default kind leaves <c>[0.0, -0.0]</c> as it is. NumSharp's float value sort (<c>AxisSort.SortLineF32</c>
        ///     / <c>SortLineF64</c>) radix-sorts the raw-bit key <c>FKey32</c> / <c>FKey64</c>, which orders <c>-0.0</c> below
        ///     <c>+0.0</c> (it must rebuild each value from its key), so every kind returns <c>[-0.0, -0.0, 0.0, 0.0, 1.0]</c>
        ///     and <c>ndarray.sort()</c> does the same. The argsort was moved to a folded key for exactly this reason
        ///     (<c>FArgKey32</c>'s remarks: <c>np.argsort([0.0, -0.0])</c> was <c>[1, 0]</c>); the value sort was not. The
        ///     float16 and complex value sorts (comparison sorts) match.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA26")]
        public void ParityQa_SortFloat_SignedZeros_ShouldKeepInputOrderLikeNumPysStableSort()
        {
            using var a = np.array(new[] { 0.0, -0.0, 1.0, -0.0, 0.0 });
            using var pair32 = np.array(new[] { 0.0f, -0.0f });

            using var stable = np.sort(a, kind: "stable");
            using var stable32 = np.sort(pair32, kind: "stable");

            // NumPy: [0.0, -0.0, -0.0, 0.0, 1.0] and [0.0, -0.0].
            double.IsNegative((double)stable.GetValue(0)).Should().BeFalse();
            double.IsNegative((double)stable.GetValue(1)).Should().BeTrue();
            double.IsNegative((double)stable.GetValue(2)).Should().BeTrue();
            double.IsNegative((double)stable.GetValue(3)).Should().BeFalse();
            float.IsNegative((float)stable32.GetValue(0)).Should().BeFalse();
            float.IsNegative((float)stable32.GetValue(1)).Should().BeTrue();
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
        // QA-29 — arrays of more than 64 dimensions are built where NumPy refuses them
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy caps an array's rank at <c>NPY_MAXDIMS</c> = 64: <c>np.zeros((1,) * 65)</c>,
        ///     <c>np.expand_dims(a64, 0)</c> and a broadcast to 65 dimensions raise
        ///     <c>ValueError: maximum supported dimension for an ndarray is currently 64, found 65</c>, and <c>a64[np.newaxis]</c>
        ///     raises <c>IndexError: number of dimensions must be within [0, 64], indexing result would have 65</c>. NumSharp
        ///     has no such check (its only 64 is <c>np.nditer</c>'s op_axes emulation and a few fast paths that decline past
        ///     it), so every one of those calls returns a 65-dimensional array (<c>p68</c>) - which NumPy itself cannot
        ///     receive through the pythonnet bridge.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA29")]
        public void ParityQa_MoreThan64Dimensions_ShouldRaiseNumPysValueError()
        {
            var rank65 = Enumerable.Repeat(1L, 65).ToArray();
            using var rank64 = np.zeros(new Shape(Enumerable.Repeat(1L, 64).ToArray()));

            Action zeros = () => np.zeros(new Shape(rank65)).Dispose();
            Action expand = () => np.expand_dims(rank64, 0).Dispose();

            zeros.Should().Throw<ValueError>().WithMessage("maximum supported dimension for an ndarray is currently 64, found 65");
            expand.Should().Throw<ValueError>().WithMessage("maximum supported dimension for an ndarray is currently 64, found 65");
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
        // QA-32 — logaddexp / logaddexp2 leave their documented 2-ULP envelope where the result cancels toward zero
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>npy_logaddexp(x, y)</c> is <c>y + log1p(exp(x - y))</c> (larger argument first), with the UCRT's
        ///     <c>log1p</c>; NumSharp ports the statements over a managed fdlibm <c>log1p</c>, and the oracle registry
        ///     excuses the pair at ≤ 2 ULP per element ("a gross error still fails"). When the result is near zero — the
        ///     larger argument close to 0 below, the smaller one well below it — the final addition cancels, and the
        ///     one-ULP <c>log1p</c> difference becomes many result ULPs: over 2,097,152 dense pairs (<c>p75</c>) float32
        ///     logaddexp exceeds 2 ULP at 27 pairs (up to 64), float32 logaddexp2 at 21 (up to 32), float64 logaddexp at 35
        ///     (up to 128) and float64 logaddexp2 at 28 (up to 16,384). <c>np.logaddexp(float32 -3.0742855, float32
        ///     -0.048163086)</c> is <c>-0.0007993132</c> (<c>0xBA518900</c>) in NumPy, <c>-0.0007993169</c> in NumSharp;
        ///     <c>np.logaddexp2(-1.0835713039595367, -0.9211879047455193)</c> is <c>-9.616429790881842e-05</c> in NumPy,
        ///     <c>-9.616429790904046e-05</c> in NumSharp (closer to the truth: prefer-precise parity debt).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA32")]
        public void ParityQa_Logaddexp_Cancellation_ShouldStayWithinTwoUlpOfNumPy()
        {
            using var a32 = np.array(new[] { BitConverter.UInt32BitsToSingle(0xC044C118) });
            using var b32 = np.array(new[] { BitConverter.UInt32BitsToSingle(0xBD4546A8) });
            using var a64 = np.array(new[] { -1.0835713039595367 });
            using var b64 = np.array(new[] { -0.9211879047455193 });

            using var lae = np.logaddexp(a32, b32);
            using var lae2 = np.logaddexp2(a64, b64);

            // Both results are negative, so their bit patterns order like their magnitudes: the ULP distance is the
            // difference of the raw bits.
            long ulp32 = System.Math.Abs((long)BitConverter.SingleToUInt32Bits((float)lae.GetValue(0)) - 0xBA518900L);
            long ulp64 = System.Math.Abs(BitConverter.DoubleToInt64Bits((double)lae2.GetValue(0)) - BitConverter.DoubleToInt64Bits(-9.616429790881842e-05));
            ulp32.Should().BeLessThanOrEqualTo(2);
            ulp64.Should().BeLessThanOrEqualTo(2);
        }

        // ------------------------------------------------------------------
        // QA-33 — the typed coordinate getters / setters do not check their coordinates
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's coordinate access (<c>a.item(i, j)</c>, <c>a[i, j]</c>, <c>a[i, j] = v</c>) checks every coordinate
        ///     against its axis and wraps a negative one once: <c>np.arange(6.).reshape(2, 3).item(0, 5)</c> raises
        ///     <c>IndexError: index 5 is out of bounds for axis 1 with size 3</c> and <c>.item(1, -1)</c> is <c>5.0</c>.
        ///     NumSharp's indexer and <c>item</c> do the same, but <c>GetDouble</c> / <c>GetInt32</c> / … /
        ///     <c>GetValue</c> / <c>GetAtIndex</c> and <c>SetDouble</c> / <c>SetValue</c> / <c>SetAtIndex</c> /
        ///     <c>itemset</c> index the storage with <c>Shape.GetOffset(indices)</c> unchecked
        ///     (<c>_slices.Double[_shape.GetOffset(indices)]</c>, UnmanagedStorage.Getters.cs): <c>m.GetDouble(0, 5)</c>
        ///     returns the element of the NEXT row (5.0), <c>m.GetDouble(1, -1)</c> returns 2.0, and past the buffer the
        ///     read and the write leave it (<c>p84</c>, Release: <c>a.GetDouble(10)</c> on 3 elements returns 0.0,
        ///     <c>a.GetDouble(-1)</c> -3.85e-270, <c>a.SetDouble(9.0, 10)</c> / <c>SetValue</c> / <c>SetAtIndex</c> /
        ///     <c>itemset(10, …)</c> return normally having written outside the array; a Debug build fails
        ///     <c>ArraySlice</c>'s "index &lt; Count, Memory corruption expected." assertion instead). Only in-buffer
        ///     coordinates are used here, so the test cannot corrupt the host.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA33")]
        public void ParityQa_TypedCoordinateGetters_ShouldCheckAndWrapLikeItem()
        {
            using var m = np.arange(6.0).reshape(2, 3);

            Action pastTheAxis = () => m.GetDouble(0, 5);

            pastTheAxis.Should().Throw<IndexError>().WithMessage("index 5 is out of bounds for axis 1 with size 3");
            m.GetDouble(1, -1).Should().Be(5.0);
        }

        /// <summary>
        ///     QA-33 reaches the NumPy-named <c>ndarray.item</c>: with MORE than one coordinate it takes the same unchecked
        ///     offset (only the flat one-index form checks and wraps). <c>np.arange(6.).reshape(2, 3).item(1, -2)</c> is
        ///     <c>4.0</c> in NumPy and <c>1.0</c> in NumSharp; <c>.item(0, 4)</c> raises <c>IndexError: index 4 is out of
        ///     bounds for axis 1 with size 3</c> in NumPy and returns <c>4.0</c>; <c>.item(-1, -1)</c> and
        ///     <c>m.T.item(-1, 0)</c> read before the buffer (<c>p86</c>: denormal garbage in Release), the 3-D, typed
        ///     <c>item&lt;T&gt;</c> and reversed-view forms likewise. Only in-buffer coordinates are used here.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA33")]
        public void ParityQa_ItemWithSeveralCoordinates_ShouldCheckAndWrapEachAxis()
        {
            using var m = np.arange(6.0).reshape(2, 3);

            Action pastTheAxis = () => m.item(0, 4);

            ((double)m.item(1, -2)).Should().Be(4.0);
            pastTheAxis.Should().Throw<IndexError>().WithMessage("index 4 is out of bounds for axis 1 with size 3");
        }

        // ------------------------------------------------------------------
        // QA-34 — np.broadcast_arrays makes a read-only operand writeable
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>broadcast_arrays</c> (<c>lib/_stride_tricks_impl.py</c>) returns an operand whose shape already
        ///     matches as the operand itself and builds every other view with <c>_broadcast_to(..., readonly=False)</c>,
        ///     which sets <c>writeable = True</c> only <c>if array.flags._writeable_no_warn</c>: a read-only operand (one
        ///     whose WRITEABLE flag was cleared, or a <c>np.broadcast_to</c> view) gives a read-only view, and writing
        ///     through it raises <c>ValueError: assignment destination is read-only</c>. NumSharp's
        ///     <c>NDArray.broadcast_arrays</c> (Backends/NDArray.StrideTricks.cs) sets WRITEABLE on every view
        ///     unconditionally (<c>WithFlags(flagsToSet: ArrayFlags.WRITEABLE)</c>), so the view of a read-only array is
        ///     writeable and a write through it changes the array its owner made read-only (<c>p90</c>:
        ///     <c>b[0, 0, 0] = 99</c> turns the read-only <c>r[0, 0]</c> into 99). The params form, an operand already of
        ///     the broadcast shape, a <c>broadcast_to</c> operand and <c>np.meshgrid(copy: false)</c> (whose grids are
        ///     broadcast_arrays views) do the same.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA34")]
        public void ParityQa_BroadcastArrays_ReadOnlyOperand_ShouldStayReadOnly()
        {
            using var r = np.arange(6.0).reshape(2, 3);
            r.setflags(write: false);
            using var deeper = np.zeros(new Shape(2, 2, 1));
            using var sameShape = np.zeros(new Shape(2, 3));
            using var b0 = np.broadcast_to(np.arange(3.0), new Shape(2, 3));

            var (stretched, _) = np.broadcast_arrays(r, deeper);
            var (unchanged, _) = np.broadcast_arrays(r, sameShape);
            var (ofBroadcastTo, _) = np.broadcast_arrays(b0, deeper);
            Action writeThrough = () => stretched[0, 0, 0] = (NDArray)99.0;

            stretched.flags.writeable.Should().BeFalse("a view of a read-only operand stays read-only");
            unchanged.flags.writeable.Should().BeFalse("NumPy returns the matching operand itself");
            ofBroadcastTo.flags.writeable.Should().BeFalse("a broadcast_to view is not writeable");
            writeThrough.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
            r.GetDouble(0, 0).Should().Be(0.0);
        }

        // ------------------------------------------------------------------
        // QA-35 — np.array(ndarray) copies the source's read-only flag and C-orders a broadcast copy
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.array(a)</c> is a copy (<c>copy=True</c>, <c>order='K'</c>), and a copy NumPy allocates is always
        ///     writeable: <c>np.array(ro)[0, 0] = 99</c> works on a read-only <c>ro</c>. NumSharp's
        ///     <c>np.array(NDArray, bool copy)</c> (Creation/np.array.cs) returns <c>new NDArray(nd.Storage.Clone())</c>,
        ///     and <c>UnmanagedStorage.Clone</c>'s raw-layout branch (a C- or F-contiguous source at offset 0) builds the
        ///     clone's shape with <c>new Shape(_shape)</c>, which keeps the source's flags: a read-only C, F or 0-d source
        ///     gives a read-only copy, and the write raises <c>ValueError: assignment destination is read-only</c>
        ///     (<c>p90</c>; a strided source takes the other branch and comes back writeable). The other branch lays a
        ///     broadcast source out in C order, where NumPy's K-order copy of <c>broadcast_to(arange(4.), (3, 4))</c>
        ///     (strides <c>(0, 8)</c>) is F-contiguous (<c>p89</c>; np.copy and ndarray.copy already give NumPy's F).
        ///     <c>NDArray.Clone()</c> (not a NumPy API) shares the same storage clone.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA35")]
        public void ParityQa_ArrayOfNDArray_ShouldBeAWriteableKOrderCopy()
        {
            using var ro = np.arange(6.0).reshape(2, 3);
            ro.setflags(write: false);
            using var bcast = np.broadcast_to(np.arange(4.0), new Shape(3, 4));

            using var copy = np.array(ro);
            using var bcastCopy = np.array(bcast);
            Action write = () => copy[0, 0] = (NDArray)99.0;

            copy.flags.writeable.Should().BeTrue("a copy NumPy allocates is writeable");
            write.Should().NotThrow();
            ro.GetDouble(0, 0).Should().Be(0.0);
            bcastCopy.flags.f_contiguous.Should().BeTrue("order='K' keeps the stride-0 axis fastest");
            bcastCopy.flags.c_contiguous.Should().BeFalse();
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
        // QA-37 — np.isclose never calls an infinite difference close, whatever the tolerance
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>isclose</c> is <c>less_equal(abs(x - y), atol + rtol * abs(y)) &amp; isfinite(y) | (x == y)</c>:
        ///     with an infinite tolerance an infinite difference passes (<c>inf &lt;= inf</c>), so
        ///     <c>np.isclose([1e308, inf], [-1e308, 1.0], rtol=0, atol=inf)</c> is <c>[True, True]</c> (the difference of
        ///     the first pair overflows; the second has a finite reference). NumSharp's <c>DefaultEngine.IsClose</c>
        ///     (Backends/Default/Logic/Default.IsClose.cs) adds a third term, <c>IsFinite(diff)</c>, "because NumSharp's &lt;=
        ///     operator has a bug where NaN &lt;= value returns True" — the operator answers False for NaN today (<c>p94</c>),
        ///     and the extra term turns both answers False. The same happens with <c>rtol=inf</c> and with a finite
        ///     tolerance that overflows the computation dtype (<c>np.isclose(float32 [3e38], [-3e38], rtol=0, atol=1e300)</c>:
        ///     the float32 difference and the float32 atol are both inf, NumPy True); np.allclose shares the engine.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA37")]
        public void ParityQa_IsClose_InfiniteDifferenceWithInfiniteTolerance_ShouldBeClose()
        {
            using var x = np.array(new[] { 1e308, double.PositiveInfinity });
            using var y = np.array(new[] { -1e308, 1.0 });

            using var close = np.isclose(x, y, rtol: 0.0, atol: double.PositiveInfinity);

            close.ToArray<bool>().Should().Equal(true, true);
        }

        // ------------------------------------------------------------------
        // QA-38 — ndarray.view with a non-native numeric dtype ignores the byte order
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>a.view(dtype)</c> reads the same bytes through the new descriptor, byte order included:
        ///     <c>np.array([258], np.int32).view('&gt;i4')</c> is <c>[33619968]</c> (the little-endian bytes
        ///     <c>02 01 00 00</c> read big-endian). NumSharp keeps no non-native storage (documented: np.frombuffer /
        ///     np.fromfile / np.load decode a non-native descriptor into a native copy, correctly), but
        ///     <c>NDArray.view(DType)</c> (Backends/NDArray.cs) compares only the lane (<c>dtype.GetTypeCode() ==
        ///     Storage.TypeCode</c>) and returns a native alias: <c>[258]</c>, int32 (<c>p95</c>). The fixed-width string branch
        ///     of the same method refuses a non-native view with <see cref="NotSupportedException"/>; the numeric branch
        ///     silently reads native order instead. Either NumPy's values (a decoded copy, since no view can hold them) or the
        ///     string branch's refusal would be faithful; this test accepts both and fails only on the silent native read.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA38")]
        public void ParityQa_ViewWithNonNativeDtype_ShouldNotReadNativeOrderSilently()
        {
            using var a = np.array(new[] { 258 });

            NDArray v;
            try
            {
                v = a.view(np.dtype(">i4"));
            }
            catch (NotSupportedException)
            {
                return; // the refusal the string branch already gives is an acceptable fix
            }

            using (v)
                Convert.ToInt64(v.GetValue(0)).Should().Be(33619968L, "NumPy reads the bytes 02 01 00 00 big-endian");
        }

        // ------------------------------------------------------------------
        // QA-39 — a memory-mapped np.load locks its file against the other loads NumPy allows
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy opens a memmap's file through Python's <c>open()</c>, whose Windows share mode lets other handles read
        ///     and write, so while <c>a = np.load(p, mmap_mode='r+')</c> lives, <c>np.load(p)</c> reads the file (and sees
        ///     <c>a</c>'s writes: <c>[0, 99, 2, 3]</c>), and an <c>'r'</c> map and an <c>'r+'</c> map of one file coexist.
        ///     NumSharp's <c>NpyFormat.OpenMemmap</c> (IO/NpyFormat.cs) holds the file with <c>FileShare.Read</c> and the
        ///     plain <c>np.load</c> opens it with the default <c>FileShare.Read</c> too, so (<c>p103</c>, isolated pairs):
        ///     while an <c>'r+'</c> map lives every other load of the file (plain, <c>'r'</c>, <c>'r+'</c>) raises
        ///     <see cref="System.IO.IOException"/> "The process cannot access the file … because it is being used by another
        ///     process", and while an <c>'r'</c> or <c>'c'</c> map lives an <c>'r+'</c> load raises the same. Each succeeds
        ///     once the first array is disposed. <c>[DoNotParallelize]</c>: the cleanup may force a collection.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA39"), DoNotParallelize]
        public void ParityQa_LoadMmap_ShouldNotLockOutOtherLoadsOfTheFile()
        {
            string dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ns_qa39_" + Guid.NewGuid().ToString("N"));
            System.IO.Directory.CreateDirectory(dir);
            try
            {
                string p = System.IO.Path.Combine(dir, "a.npy");
                using (var seed = np.arange(4.0))
                    np.save(p, seed);

                using (var rw = (NDArray)np.load(p, "r+"))
                {
                    rw[1] = (NDArray)99.0;
                    using var plain = (NDArray)np.load(p);
                    plain.GetDouble(1).Should().Be(99.0, "NumPy reads the file while the r+ map lives, writes included");
                }

                using (var ro = (NDArray)np.load(p, "r"))
                {
                    Action readWriteToo = () => ((NDArray)np.load(p, "r+")).Dispose();
                    readWriteToo.Should().NotThrow("an 'r' map and an 'r+' map of one file coexist in NumPy");
                }
            }
            finally
            {
                // A map the failing assertion left open is released by the helper's collection before the delete.
                MappedFileCleanup.DeleteDirectory(dir);
            }
        }

        /// <summary>
        ///     QA-39 beyond memmaps: Python's <c>open()</c> shares read and write, so NumPy reads and overwrites a file that
        ///     another handle holds open — another program's writer, or a .NET <c>FileStream</c> opened with
        ///     <c>FileShare.ReadWrite</c>. NumSharp's readers and writers open with .NET's default share modes
        ///     (<c>FileShare.Read</c> to read, none to write), so with such a handle open <c>np.load</c> (.npy and .npz),
        ///     <c>np.loadtxt</c>, <c>np.save</c>, <c>np.savetxt</c> and <c>ndarray.tofile</c> raise <see cref="System.IO.IOException"/>
        ///     "The process cannot access the file … because it is being used by another process" (<c>rp3</c>; NumPy's twin
        ///     succeeds for all seven); only <c>np.fromfile</c> reads it.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QA39"), DoNotParallelize]
        public void ParityQa_LoadAndSave_ShouldShareAFileAnotherHandleHoldsOpen()
        {
            string dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ns_qa39b_" + Guid.NewGuid().ToString("N"));
            System.IO.Directory.CreateDirectory(dir);
            try
            {
                string p = System.IO.Path.Combine(dir, "a.npy");
                using (var seed = np.arange(3.0))
                    np.save(p, seed);

                using (var other = new System.IO.FileStream(p, System.IO.FileMode.Open, System.IO.FileAccess.ReadWrite, System.IO.FileShare.ReadWrite))
                {
                    Action load = () => ((NDArray)np.load(p)).Dispose();
                    load.Should().NotThrow("NumPy reads a file another handle shares");
                }
            }
            finally
            {
                MappedFileCleanup.DeleteDirectory(dir);
            }
        }
    }
}
