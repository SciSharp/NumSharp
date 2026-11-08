using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — the NumPy parity QA journey of 2026-11-08 / 09 (QB-NN)
    // =====================================================================
    //
    // The second manual QA pass, on branch journey5, compared NumSharp against NumPy 2.4.2 with a probe harness (one
    // probe file per direction, each line a C# expression and its NumPy twin; a runner builds the C# side, runs NumPy in
    // process and compares dtype, shape, every element bit for bit, flags, class and the error type and text). Every
    // divergence it confirmed that no existing [OpenBugs] / [Misaligned] entry already pins has an id QB-NN, an
    // explanation in docs/bugs/parity-qa-2026-11-08.md (the journal: symptom, NumPy's answer, root cause where known,
    // severity, fix size) and a reproduction below through public calls, with the smallest input that shows it.
    // Nothing here is fixed yet: the owner picks which ones to fix.
    //
    // Every expected value is NumPy 2.4.2's answer for the same call (probed on win-amd64). EXPECTED (when fixed): each
    // test passes; remove its [OpenBugs], move it next to its API's tests, and update the journal entry.
    // =====================================================================
    public partial class OpenBugs
    {
        // ------------------------------------------------------------------
        // QB-01 — complex np.sign divides by np.absolute's magnitude, not by npy_hypot
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>CDOUBLE_sign</c> (loops.c.src) computes <c>in1abs = npy_hypot(re, im)</c> and returns
        ///     <c>(re / in1abs, im / in1abs)</c>; on win-amd64 <c>npy_hypot</c> is the UCRT <c>hypot</c>, which NumSharp
        ///     already models exactly (<c>np.hypot</c>, parity QA-25). The complex branch of
        ///     <c>DirectILKernelGenerator.EmitSignCall</c> divides by <c>NDComplexMath.Abs</c> instead — np.absolute's
        ///     SIMD kernel, <c>larger * sqrt(fma(ratio, ratio, 1))</c> — which rounds differently: 4,552 of 20,000
        ///     log-uniform complex values differ by 1–2 ULP in a part. The oracle's "complex unary within 3 ULP" excuse
        ///     (written for the transcendental npy_c* ports) hides it, although this op has no transcendental step.
        ///     <c>np.real(z) / np.hypot(re, im)</c> computed with NumSharp's own kernels matches NumPy on all 20,000.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB01")]
        public void ParityQb_QB01_ComplexSign_DividesByNpyHypot()
        {
            using var z = np.array(new[] { new Complex(-2, 7), new Complex(1e-300, 1e-300) });
            using var s = np.sign(z);

            // NumPy 2.4.2: np.sign(np.array([-2+7j, 1e-300+1e-300j]))
            //   -> [-0.27472112789737807+0.9615239476408232j, 0.7071067811865476+0.7071067811865476j]
            var first = s.GetAtIndex<Complex>(0);
            var second = s.GetAtIndex<Complex>(1);
            first.Real.Should().Be(-0.27472112789737807);
            first.Imaginary.Should().Be(0.9615239476408232);
            second.Real.Should().Be(0.7071067811865476);
            second.Imaginary.Should().Be(0.7071067811865476);
        }

        // ------------------------------------------------------------------
        // QB-02 — np.arange's float overloads return float32 computed in float32
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.arange(float start, float stop, float step = 1)</c> and <c>np.arange(float stop)</c> (the "type inferred
        ///     from C# parameter types" overloads) pass <c>NPTypeCode.Single</c>, so <c>np.arange(0f, 1f, 0.1f)</c> is a
        ///     float32 array whose elements are float32 sums (<c>[3]</c> is <c>0.30000001192092896f</c>). NumPy 2.4.2 never
        ///     returns float32 from arange without a dtype: <c>np.arange(np.float32(0), np.float32(1), np.float32(0.1))</c>
        ///     is float64 <c>start + i*delta</c> in double (<c>[3]</c> is <c>0.30000000447034836</c>), and so is the house
        ///     map's reading of a C# <c>float</c> as a Python float (NumSharp's own <c>np.linspace(0f, 1f, 5)</c> is
        ///     float64). The int overloads already return NumPy's int64.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB02")]
        public void ParityQb_QB02_ArangeFloatOverloads_AreFloat64LikeNumPy()
        {
            using var r = np.arange(0f, 1f, 0.1f);
            using var s = np.arange(3.5f);

            // NumPy 2.4.2: np.arange(np.float32(0), np.float32(1), np.float32(0.1)) -> float64, 10 elements
            r.dtype.Should().Be(np.float64);
            r.size.Should().Be(10);
            r.GetDouble(3).Should().Be(0.30000000447034836);
            r.GetDouble(9).Should().Be(0.9000000134110451);
            // NumPy 2.4.2: np.arange(np.float32(3.5)) -> float64 [0., 1., 2., 3.]
            s.dtype.Should().Be(np.float64);
        }

        // ------------------------------------------------------------------
        // QB-03 — np.repeat's per-element negative-count text lacks NumPy's period
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>PyArray_Repeat</c> (item_selection.c) raises <c>ValueError: repeats may not contain negative
        ///     values.</c> for an array of counts holding a negative entry. NumSharp raises the same text without the final
        ///     period (<c>UnmanagedStorage.ItemSelection.cs</c> <c>ComputeCounts</c> and its 1-D twin). The broadcast
        ///     (scalar) count is a different, documented wording difference (NumPy: <c>negative dimensions are not
        ///     allowed</c>) and is not part of this pin.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB03")]
        public void ParityQb_QB03_RepeatNegativeCountArray_TextEndsWithPeriod()
        {
            using var a = np.array(new long[] { 5, 1, 4 });
            using var counts = np.array(new[] { 1, -1, 2 });

            Action act = () => np.repeat(a, counts);

            // NumPy 2.4.2: np.repeat([5, 1, 4], np.array([1, -1, 2])) -> ValueError
            act.Should().Throw<ArgumentException>().Which.Message.Should().Be("repeats may not contain negative values.");
        }

        // ------------------------------------------------------------------
        // QB-04 — ndarray.item() / MaskedArray.item() of a non-size-1 array: not NumPy's ValueError
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 raises <c>ValueError: can only convert an array of size 1 to a Python scalar</c> for
        ///     <c>a.item()</c> of an array whose size is not 1 (an empty one included), for ndarray and for
        ///     <c>numpy.ma.MaskedArray</c>. NumSharp's <c>NDArray.item()</c> and <c>item&lt;T&gt;()</c>
        ///     (<c>Manipulation/NDArray.item.cs</c>) throw <c>IncorrectSizeException</c>, which is not a
        ///     <c>ValueError</c> (it derives <c>NumSharpException</c>), and <c>NDMaskedArray.item()</c> throws
        ///     <c>InvalidOperationException</c>; NumSharp's own text arrays (<c>NDListArray</c>) already throw
        ///     <c>ValueError</c> with the same text. A caller catching NumSharp's <c>ValueError</c> (or
        ///     <c>ArgumentException</c>) misses both.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB04")]
        public void ParityQb_QB04_ItemOfANonSizeOneArray_RaisesValueError()
        {
            using var five = np.arange(5);
            using var empty = np.zeros(new Shape(0));
            var masked = np.ma.masked_array(np.arange(3.0));

            // NumPy 2.4.2: np.arange(5).item(), np.zeros(0).item(), np.ma.masked_array(np.arange(3.)).item() -> ValueError
            Action plain = () => five.item();
            Action typed = () => five.item<long>();
            Action none = () => empty.item();
            Action ma = () => masked.item();
            plain.Should().Throw<ValueError>().WithMessage("can only convert an array of size 1 to a Python scalar");
            typed.Should().Throw<ValueError>();
            none.Should().Throw<ValueError>();
            ma.Should().Throw<ValueError>();
        }

        // ------------------------------------------------------------------
        // QB-05 — complex log / arctan / arctanh / expm1 use stale private real log1p / expm1 helpers
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 on win-amd64 runs npymath's <c>npy_clog</c> / <c>npy_catanh</c> (MSVC's <c>clog</c> / <c>catan</c>
        ///     are blocklisted in <c>npy_config.h</c>) and <c>nc_expm1</c>, whose real steps are the UCRT's <c>log1p</c> /
        ///     <c>expm1</c>. NumSharp models those two exactly since parity QA-22 (<c>NDLog1pExpm1Math.Log1p</c> =
        ///     <c>log(u) - ((u - 1) - x) / u</c>; <c>Expm1</c> = <c>exp(x) - 1</c> outside [-ln 2, ln 1.5]), but
        ///     <c>NDComplexMath</c> still calls its own private helpers: <c>MathLog1p</c> (<c>log(w) * (u / (w - 1))</c>, another
        ///     rounding) in <c>Log</c>'s near-|z| = 1 branch and in <c>Catanh</c>, and <c>RealExpm1</c>
        ///     (<c>(e^x - 1) * x / log(e^x)</c>) in <c>Expm1</c>. Over 30,000 complex values: log differs on 334, arctan / arctanh
        ///     on ~7,970, expm1 on 1,487 (665 of them outside the band, up to 2,048 ULP in a cancelling real part). Rebuilt
        ///     from NumSharp's own exact real ufuncs (np.hypot, np.log1p, np.log, np.arctan2), the complex log matches NumPy on
        ///     all 30,000; a Python port of npy_catanh with the exact log1p model matches np.arctanh on all 30,000.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB05")]
        public void ParityQb_QB05_ComplexLogAtanhExpm1_UseTheExactRealLog1pExpm1()
        {
            using var log = np.log(np.array(new[] { new Complex(-1.0571132398591, -0.6968054066842436) }));
            using var atanh = np.arctanh(np.array(new[] { new Complex(79.5241198786932, -715.1693772939045) }));
            using var atan = np.arctan(np.array(new[] { new Complex(93.1830917281823, -8.418577625481108) }));
            using var expm1 = np.expm1(np.array(new[] { new Complex(0.6364991061529732, -0.6534557720485368) }));

            // NumPy 2.4.2, the same calls:
            log.GetAtIndex<Complex>(0).Real.Should().Be(0.23594660164716702);
            atanh.GetAtIndex<Complex>(0).Real.Should().Be(0.0001535830461655032);
            atan.GetAtIndex<Complex>(0).Imaginary.Should().Be(-0.000961579151786817);
            expm1.GetAtIndex<Complex>(0).Real.Should().Be(0.5005200535347242);
        }

        // ------------------------------------------------------------------
        // QB-06 — complex arcsin / arccos (and arcsinh / arccosh) compute through System.Numerics, not npymath's msun port
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>NDComplexMath.Asin</c> / <c>Acos</c> hand every finite input to <c>System.Numerics.Complex.Asin</c> /
        ///     <c>Acos</c> and fix only the zero signs; <c>Asinh</c> / <c>Acosh</c> derive from them. NumPy 2.4.2 on win-amd64
        ///     blocklists MSVC's <c>casin</c> / <c>cacos</c> / <c>casinh</c> / <c>cacosh</c> (<c>npy_config.h</c>) and runs
        ///     npymath's msun ports (<c>npy_casinh</c>'s <c>_do_hard_work</c>, <c>npy_cacos</c>), whose real steps are the UCRT
        ///     functions NumSharp already models (hypot, log1p, sqrt, asin, atan2). Over 30,000 complex values arcsin / arcsinh
        ///     differ on 7,574 / 7,717 and arccos / arccosh on 14,083, by up to 5 ULP, past the oracle's documented 3-ULP
        ///     envelope (which arcsin / arcsinh are not exempted from). A Python port of npy_casinh with those real models matches
        ///     np.arcsinh and np.arcsin on all 30,000. NumSharp's own comments call these "byte-exact ports".
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB06")]
        public void ParityQb_QB06_ComplexArcsinArccos_AreNpymathsMsunPorts()
        {
            using var asin = np.arcsin(np.array(new[] { new Complex(88.209178279867, 909.9421975228022) }));
            using var asinh = np.arcsinh(np.array(new[] { new Complex(909.9421975228022, 88.209178279867) }));
            using var acos = np.arccos(np.array(new[] { new Complex(-5.686795079249146, 243.19353968737) }));
            using var acosh = np.arccosh(np.array(new[] { new Complex(-5.686795079249146, 243.19353968737) }));

            // NumPy 2.4.2, the same calls:
            asin.GetAtIndex<Complex>(0).Real.Should().Be(0.09663730993736096);
            asinh.GetAtIndex<Complex>(0).Imaginary.Should().Be(0.09663730993736096);
            acos.GetAtIndex<Complex>(0).Real.Should().Be(1.5941756932192908);
            acosh.GetAtIndex<Complex>(0).Imaginary.Should().Be(1.5941756932192908);
        }

        // ------------------------------------------------------------------
        // QB-07 — a binary / comparison result over a reversed or stepped C-ordered view and an F array is F-ordered
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 allocates the result of <c>np.add(a[::-1], f)</c> (a C-ordered but non-contiguous view, and an
        ///     F-contiguous array of the same shape) through NpyIter's K order, whose axis sort compares |strides| and lets
        ///     C order win a conflict between operands (<c>npyiter_find_best_axis_ordering</c>): the result is C-contiguous,
        ///     strides <c>(32, 8)</c>. NumSharp's float64 binary routes and the comparison routes end with the legacy
        ///     "looser-F preservation" step (<c>ShouldProduceFContigOutput</c> in <c>DefaultEngine.BinaryOp.cs</c> and
        ///     <c>DefaultEngine.CompareOp.cs</c>): some operand strictly F and none strictly C → <c>result.copy('F')</c>. A
        ///     reversed or stepped view is neither, so it does not count, and the C result is copied to F (strides
        ///     <c>(8, 24)</c>). The K-order plan of the same routes (<c>IsKOrderCVoter</c>, which counts such a view as a C
        ///     voter) already gets it right; the int64 route matches NumPy. Visible through strides and flags, through
        ///     <c>ravel('K')</c> / <c>flatten('K')</c> (another element order) and as an extra copy.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB07")]
        public void ParityQb_QB07_BinaryOfAReversedViewAndAnFArray_IsCOrdered()
        {
            using var basis = np.arange(12.0).reshape(3, 4);
            using var reversed = basis["::-1"];
            using var f = np.asfortranarray(np.arange(12.0).reshape(3, 4));
            using var stepped = np.arange(24.0).reshape(3, 8)[":, ::2"];

            using var sum = np.add(reversed, f);
            using var less = np.less(reversed, f);
            using var product = np.multiply(stepped, f);
            using var k = sum.ravel('K');

            // NumPy 2.4.2: (32, 8) C-contiguous; np.less: (4, 1); stepped * F: (32, 8); ravel('K')[:4] = [8, 10, 12, 14]
            sum.strides.Should().Equal(32L, 8L);
            less.strides.Should().Equal(4L, 1L);
            product.strides.Should().Equal(32L, 8L);
            k.GetDouble(1).Should().Be(10.0);
        }

        // ------------------------------------------------------------------
        // QB-08 — zero-size results the allocation-strides fix (18314cac) does not reach
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 gives every zero-size array it ALLOCATES all-zero strides (<c>PyArray_NewFromDescr_int</c>), and
        ///     the ufunc result-layout re-port (<c>18314cac</c>, 2026-11-08) made NumSharp's allocation funnel
        ///     (<c>Shape.ForFreshAllocation</c>) do the same, closing the allocation half of the EmptyStrides pins. Some
        ///     producers still bypass it: <c>np.full_like(np.zeros((2, 0, 4)), 7.0)</c> has strides <c>(0, 32, 8)</c>
        ///     (NumPy <c>(0, 0, 0)</c>) and <c>np.unique</c> of an empty array <c>(8,)</c> (NumPy <c>(0,)</c>). The reverse
        ///     also happens where NumPy's result is a reshaped VIEW: <c>np.tile(np.arange(3), 0)</c> and
        ///     <c>np.fromstring("", sep=" ")</c> are <c>(8,)</c> in NumPy and <c>(0,)</c> in NumSharp. Only <c>strides</c>
        ///     observes it.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB08")]
        public void ParityQb_QB08_ZeroSizeResults_TakeNumPysStrides()
        {
            using var empty3 = np.zeros(new Shape(2, 0, 4));
            using var full = np.full_like(empty3, 7.0);
            using var unique = (NDArray)np.unique(empty3);
            using var tiled = np.tile(np.arange(3), 0);

            // NumPy 2.4.2: (0, 0, 0), (0,), (8,)
            full.strides.Should().Equal(0L, 0L, 0L);
            unique.strides.Should().Equal(0L);
            tiled.strides.Should().Equal(8L);
        }

        // ------------------------------------------------------------------
        // QB-09 — np.matmul (and 2-D row · column np.dot) keep another NaN than NumPy's ddot route
        // ------------------------------------------------------------------

        /// <summary>
        ///     Parity PA3b (<c>95191628</c>) made "which NaN survives where several meet" NumPy's for np.dot / np.vdot /
        ///     np.vecdot: NumPy's dot functions (<c>DOUBLE_dot</c> over cblas <c>ddot</c>) keep the first operand's NaN here,
        ///     so <c>np.dot([-nan, 1], [1, nan])</c> is <c>-nan</c> in both libraries. NumPy runs the same ddot for
        ///     <c>np.matmul</c> of two vectors and for a row × column product (<c>cblas_matrixproduct</c>'s vector-vector
        ///     case, and matmul's 1 × 1 output), so <c>np.matmul([-nan, 1], [1, nan])</c>,
        ///     <c>np.matmul(a.reshape(1, 2), b.reshape(2, 1))</c> and <c>np.dot</c> of those 2-D shapes are <c>-nan</c> too;
        ///     NumSharp's matmul and 2-D dot routes return the second operand's <c>+nan</c>, which <c>np.signbit</c> sees.
        ///     Payload-level as well: over (1, n) × (n, 1) / (1, n) × (n, 2) / (2, n) × (n, 2) shapes the surviving NaN's
        ///     payload differs from NumPy's (it matches for (2, n) × (n, 1), NumPy's gemv).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB09")]
        public void ParityQb_QB09_MatmulOfVectors_KeepsNumPysNaN()
        {
            using var a = np.array(new long[] { unchecked((long)0xfff8000000000000UL), 0x3ff0000000000000L }).view(np.float64);
            using var b = np.array(new long[] { 0x3ff0000000000000L, 0x7ff8000000000000L }).view(np.float64);
            using var dot = np.dot(a, b);
            using var mm = np.matmul(a, b);
            using var rowColumn = np.matmul(a.reshape(1, 2), b.reshape(2, 1));

            // NumPy 2.4.2: np.dot / np.matmul / row @ column of [-nan, 1] and [1, nan] are all -nan
            np.signbit(dot).GetBoolean().Should().BeTrue("np.dot already keeps NumPy's NaN (PA3b)");
            np.signbit(mm).GetBoolean().Should().BeTrue();
            np.signbit(rowColumn).GetBoolean(0, 0).Should().BeTrue();
        }

        // ------------------------------------------------------------------
        // QB-10 — real np.median / percentile / quantile / nanmedian return another NaN than NumPy
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>_median_nancheck</c> answers a slice holding a NaN with <c>data.take(-1)</c>, the LAST
        ///     element after its partition (the NaNs gather at the end in input order), and <c>_nanmedian1d</c> answers an
        ///     all-NaN slice with <c>arr1d[-1]</c>. So <c>np.median([-nan, 1, nan])</c> is <c>+nan</c> (as are
        ///     <c>np.percentile(…, 50)</c> and <c>np.quantile(…, 0.2)</c>), and <c>np.nanmedian([-nan])</c> is <c>-nan</c>.
        ///     NumSharp returns the FIRST NaN (<c>-nan</c>, so <c>np.signbit</c> differs) and, for nanmedian's all-NaN slice,
        ///     the canonical <c>+nan</c>. With distinct payloads the median family returns element 0's NaN where NumPy returns
        ///     the last NaN's. The complex median already follows NumPy's last-element rule (<c>c39a29bc</c>).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB10")]
        public void ParityQb_QB10_MedianFamily_ReturnsNumPysNaN()
        {
            using var mixed = np.array(new long[] { unchecked((long)0xfff8000000000000UL), 0x3ff0000000000000L, 0x7ff8000000000000L }).view(np.float64);
            using var negOnly = np.array(new long[] { unchecked((long)0xfff8000000000000UL) }).view(np.float64);

            using var median = np.median(mixed);
            using var percentile = np.percentile(mixed, 50);
            using var nanmedian = np.nanmedian(negOnly);

            // NumPy 2.4.2: np.signbit -> False, False, True
            np.signbit(median).GetBoolean().Should().BeFalse();
            np.signbit(percentile).GetBoolean().Should().BeFalse();
            np.signbit(nanmedian).GetBoolean().Should().BeTrue();
        }

        // ------------------------------------------------------------------
        // QB-11 — real np.correlate / np.convolve (small-kernel regime) keep another NaN than NumPy
        // ------------------------------------------------------------------

        /// <summary>
        ///     float-math.md calls the small-kernel regime of the real np.correlate / np.convolve byte-identical to NumPy
        ///     (each output's <c>Σ a[i+t]·k[t]</c> in t order, as NumPy's <c>small_correlate</c>). Where two NaNs meet the
        ///     bytes differ: <c>np.correlate([-nan, 1], [1, nan])</c> is <c>-nan</c> in NumPy 2.4.2 and <c>+nan</c> in
        ///     NumSharp, and <c>np.convolve([-nan, 1], [1, nan])</c> has signbits <c>[True, False, False]</c> in NumPy against
        ///     <c>[True, True, False]</c> (<c>np.convolve([1, nan], [-nan, 1])</c>: NumPy <c>[True, True, False]</c>, NumSharp
        ///     <c>[True, False, False]</c>). With distinct payloads 8 of 9 full-mode outputs keep another NaN for length 5,
        ///     63 of 65 for length 33. The complex path already follows NumPy's NaN choice (PA3b, <c>95191628</c>).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB11")]
        public void ParityQb_QB11_RealCorrelateConvolve_KeepNumPysNaN()
        {
            using var a = np.array(new long[] { unchecked((long)0xfff8000000000000UL), 0x3ff0000000000000L }).view(np.float64);
            using var v = np.array(new long[] { 0x3ff0000000000000L, 0x7ff8000000000000L }).view(np.float64);

            using var corr = np.correlate(a, v);
            using var conv = np.convolve(a, v);

            // NumPy 2.4.2: signbit(correlate) = [True]; signbit(convolve) = [True, False, False]
            np.signbit(corr).GetBoolean(0).Should().BeTrue();
            np.signbit(conv).GetBoolean(1).Should().BeFalse();
        }

        // ------------------------------------------------------------------
        // QB-12 — float32 np.exp2 turns every NaN into the canonical +NaN
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's win-amd64 wheel has no vector float32 exp2 (float-math.md), so <c>np.exp2</c> of a float32 is the
        ///     CRT's <c>exp2f</c>, which returns its NaN argument: <c>np.exp2(np.float32 0xffc00456)</c> is <c>0xffc00456</c>
        ///     (negative, payload kept; a signalling NaN comes back quiet with its payload). NumSharp's
        ///     <c>NDFloatMath.Exp2</c> sends every NaN to <c>Exp2NonFinite</c>, which answers the canonical
        ///     <c>0x7fc00000</c> (the rule of NumPy's SIMD exp / log / sin / cos kernels, which do blank NaN lanes, applied
        ///     to the one float32 function that has no such kernel), so the sign flips: <c>np.signbit</c> answers False
        ///     where NumPy answers True. float64 and float16 exp2 already propagate the input NaN.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB12")]
        public void ParityQb_QB12_Float32Exp2_PropagatesTheInputNaN()
        {
            using var x = np.array(new[] { unchecked((int)0xffc00456), 0x7f800789 }).view(np.float32);
            using var r = np.exp2(x);
            using var bits = r.view(np.int32);

            // NumPy 2.4.2: [0xffc00456, 0x7fc00789] (the signalling NaN quieted)
            bits.GetInt32(0).Should().Be(unchecked((int)0xffc00456));
            bits.GetInt32(1).Should().Be(0x7fc00789);
        }

        // ------------------------------------------------------------------
        // QB-13 — np.spacing of a NaN changes the NaN (float64 payload + 1, float32 sNaN left signalling)
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>npy_spacing</c> is <c>_next(x, 1) - x</c> after an infinity test, so a NaN propagates through
        ///     <c>nextafter</c> and the subtraction and comes back as the input NaN, quieted: <c>np.spacing</c> of float64
        ///     <c>0x7ff8000000000123</c> is <c>0x7ff8000000000123</c>, of float32 <c>0x7f800789</c> (signalling) is
        ///     <c>0x7fc00789</c>. NumSharp's SIMD body (every full vector: 4 float64 / 8 float32 lanes) treats the NaN's bits
        ///     as a finite magnitude and returns payload + 1 (<c>0x…124</c>; a signalling <c>0x7ff0…789</c> becomes
        ///     <c>0x7ff8…78a</c>), while its scalar tail returns a float64 quiet NaN unchanged (so an array's last elements
        ///     and its first ones disagree) but leaves a float32 signalling NaN signalling (<c>0x7f800789</c>). The sign
        ///     survives; only the payload / quiet bit differ. The registry holds spacing "bit-exact at every width", which the
        ///     corpus (NaN tokens) cannot see.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB13")]
        public void ParityQb_QB13_SpacingOfANaN_ReturnsTheInputNaNQuieted()
        {
            using var d = np.array(new[] { 0x7ff8000000000123L, 0x7ff8000000000123L, 0x7ff8000000000123L, 0x7ff8000000000123L }).view(np.float64);
            using var f = np.array(new[] { 0x7f800789 }).view(np.float32);
            using var sd = np.spacing(d).view(np.int64);
            using var sf = np.spacing(f).view(np.int32);

            // NumPy 2.4.2: every element 0x7ff8000000000123; the float32 signalling NaN quieted to 0x7fc00789
            sd.GetInt64(0).Should().Be(0x7ff8000000000123L);
            sf.GetInt32(0).Should().Be(0x7fc00789);
        }

        // ------------------------------------------------------------------
        // QB-14 — a signalling NaN through np.hypot stays signalling, through np.nextafter becomes quiet (NumPy: the reverse)
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 quiets a signalling NaN in <c>np.hypot</c> (<c>hypot(0x7ff0000000000789, 1.5)</c> is
        ///     <c>0x7ff8000000000789</c>) and returns it UNQUIETED from <c>np.nextafter</c> (the CRT's <c>nextafter</c> hands a
        ///     NaN argument back as it is: <c>0x7ff0000000000789</c>; float32 <c>0x7f800789</c> likewise). NumSharp does the
        ///     opposite in both: its hypot (<c>NDHypotMath</c>'s UCRT model) returns the signalling NaN unchanged and its
        ///     nextafter quiets it. Only the quiet bit differs (payload and sign survive); quiet NaNs match.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB14")]
        public void ParityQb_QB14_SignallingNaN_ThroughHypotAndNextafter()
        {
            using var snan = np.array(new[] { 0x7ff0000000000789L }).view(np.float64);
            using var one = np.array(new[] { 1.5 });
            using var hyp = np.hypot(snan, one).view(np.int64);
            using var next = np.nextafter(snan, one).view(np.int64);

            // NumPy 2.4.2: hypot quiets it, nextafter returns it signalling
            hyp.GetInt64(0).Should().Be(0x7ff8000000000789L);
            next.GetInt64(0).Should().Be(0x7ff0000000000789L);
        }
    }
}
