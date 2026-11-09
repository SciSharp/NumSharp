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

            // NumPy 2.4.2: np.arange(18446744073709551610, 18446744073709551615, 2) -> float64, 3 elements (the Python
            // ints' exact length); a C# ulong binds the float32 overload here and gives an empty float32 array.
            using var u = np.arange(18446744073709551610UL, 18446744073709551615UL, 2UL);
            u.dtype.Should().Be(np.float64);
            u.size.Should().Be(3);
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

            // NumPy 2.4.2: np.arange(6).reshape(2, 3).item(1, 1, 1) -> ValueError "incorrect number of indices for array"
            // (NumSharp: ArgumentException "Incorrect number of indices for array with 2 dimensions.")
            using var matrix = np.arange(6).reshape(2, 3);
            Action wrongCount = () => matrix.item(1, 1, 1);
            wrongCount.Should().Throw<ValueError>().WithMessage("incorrect number of indices for array");
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

        // ------------------------------------------------------------------
        // QB-15 — np.median / np.nanmedian / np.ma.median of ONE element return (x + x) / 2, which overflows
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's median of a one-element slice is <c>np.mean</c> of that element, the element itself:
        ///     <c>np.median(np.array([1e308]))</c> is <c>1e308</c> and <c>np.median(np.array([40000], np.float16))</c> is
        ///     <c>40000</c>. NumSharp returns <c>(x + x) / 2</c>, so every |x| above half the dtype's maximum (float64
        ///     8.99e307, float32 1.7e38, float16 32752) comes back as ±inf, on every route that reaches a one-element line:
        ///     a 0-d input, a one-element array, keepdims, <c>out=</c>, <c>overwrite_input</c>, a length-1 axis,
        ///     <c>np.nanmedian</c> of a slice with one non-NaN value, and <c>np.ma.median</c> of one unmasked value.
        ///     Root cause: <c>DirectILKernelGenerator.WriteCell</c> takes np.median's two-sample mean when
        ///     <c>gamma &gt;= 0.5</c>, and for a one-element line the plan's gamma is 1 (NumPy's <c>_get_indexes</c> moves an
        ///     index at the upper bound to -1, so <c>gamma = 0 - (-1)</c>): the "two central samples" are the same element
        ///     twice. np.percentile / np.quantile of one element take the lerp and are already right. A fix must keep
        ///     QB-17 in view: NumPy's nanmedian along a short axis DOES overflow a lone element (numpy.ma's median).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB15")]
        public void ParityQb_QB15_MedianOfOneElement_IsTheElement()
        {
            using var big = np.array(new[] { 1e308 });
            using var bigWithNaN = np.array(new[] { 1e308, double.NaN });
            using var half = np.array(new[] { (Half)40000 });
            var maskedPair = np.ma.masked_array(np.array(new[] { 1e308, 5.0 }), np.array(new[] { false, true }));

            using var median = np.median(big);
            using var nanmedian = np.nanmedian(bigWithNaN);
            using var medianHalf = np.median(half);
            var maMedian = np.ma.median(maskedPair);

            // NumPy 2.4.2: the element itself, never inf
            median.GetDouble().Should().Be(1e308);
            nanmedian.GetDouble().Should().Be(1e308);
            medianHalf.GetHalf().Should().Be((Half)40000);
            maMedian.data.GetDouble().Should().Be(1e308);
        }

        // ------------------------------------------------------------------
        // QB-16 — np.median's middle mean is not np.mean's: float16 sums in float16, complex skips Smith's division
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>np.median</c> is <c>np.mean(part[indexer])</c> of the middle element(s). For float16,
        ///     <c>_mean</c> sums in the float32 loop and narrows once, so <c>np.median(np.array([40000, 40000], np.float16))</c>
        ///     is <c>40000</c> (and the flat <c>np.nanmedian</c> of <c>[30000, 40000]</c> is <c>35008</c>); NumSharp's
        ///     <c>WriteCell</c> rounds the pair's sum to float16 first (its comment calls that NumPy's rule), so any pair
        ///     summing past 65504 is <c>inf</c>. For complex128, <c>_mean</c> divides the sum by <c>n + 0j</c> with
        ///     <c>CDOUBLE_divide</c>'s Smith algorithm, whose <c>in1i * 0</c> / <c>in1r * 0</c> terms turn the other part
        ///     NaN whenever a part is infinite: <c>np.median([1+infj])</c> is <c>nan+infj</c>,
        ///     <c>np.median([inf+1j, inf+0j, -1+0j])</c> is <c>inf+nanj</c>, and <c>np.median([1.7e308+0j, 1.7e308+1j])</c>
        ///     (an overflowing sum) is <c>inf+nanj</c>. NumSharp's complex route returns the middle element, or
        ///     <c>0.5·a + 0.5·b</c>, which never overflows. Same root cause as
        ///     <c>OpenBugsReductionTests.Median_OfNegativeZeros_IsPositiveZero</c> (the median is not np.mean of its middle),
        ///     whose remarks say the values differ only in a zero's sign; these are value differences.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB16")]
        public void ParityQb_QB16_MedianMiddleMean_IsNpMean()
        {
            using var half = np.array(new[] { (Half)40000, (Half)40000 });
            using var infImag = np.array(new[] { new Complex(1, double.PositiveInfinity) });
            using var overflowPair = np.array(new[] { new Complex(1.7e308, 0), new Complex(1.7e308, 1) });

            using var medianHalf = np.median(half);
            using var medianInfImag = np.median(infImag);
            using var medianOverflow = np.median(overflowPair);

            // NumPy 2.4.2: 40000, nan+infj, inf+nanj
            medianHalf.GetHalf().Should().Be((Half)40000);
            double.IsNaN(medianInfImag.GetComplex().Real).Should().BeTrue();
            double.IsPositiveInfinity(medianInfImag.GetComplex().Imaginary).Should().BeTrue();
            double.IsPositiveInfinity(medianOverflow.GetComplex().Real).Should().BeTrue();
            double.IsNaN(medianOverflow.GetComplex().Imaginary).Should().BeTrue();
        }

        // ------------------------------------------------------------------
        // QB-17 — np.nanmedian / np.ma.median along an axis: NumPy's numpy.ma median sums the duplicated middle
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>np.nanmedian</c> along an axis shorter than 600 is <c>_nanmedian_small</c>, i.e.
        ///     <c>np.ma.median</c>, whose N-D <c>_median</c> takes <c>low_high</c> (the middle element TWICE for an odd
        ///     count), sums it with <c>np.ma.sum</c> and halves it. So an odd-count slice whose middle exceeds half the
        ///     dtype's maximum is <c>inf</c>: <c>np.nanmedian(np.array([[1e308, 1e308, 1e308]]), axis=1)</c> is
        ///     <c>[inf]</c>, and so is <c>np.ma.median(..., axis=1)</c>. NumSharp returns the middle element. (NumPy's flat
        ///     nanmedian, and its axis form at 600 or more elements, go through <c>np.median</c> and stay finite, as
        ///     NumSharp does.) The quirk is NumPy's own, but it is its behavior: the axis route of both functions has to
        ///     add the middle to itself, where NumSharp's QB-15 fix must stop doing exactly that on the other routes.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB17")]
        public void ParityQb_QB17_NanmedianAlongAShortAxis_SumsTheDuplicatedMiddle()
        {
            using var rows = np.array(new double[,] { { 1e308, 1e308, 1e308 } });

            using var nanmedian = np.nanmedian(rows, 1);
            var maMedian = np.ma.median(np.ma.masked_array(rows), 1);

            // NumPy 2.4.2: [inf] for both
            double.IsPositiveInfinity(nanmedian.GetDouble(0)).Should().BeTrue();
            double.IsPositiveInfinity(maMedian.data.GetDouble(0)).Should().BeTrue();
        }

        // ------------------------------------------------------------------
        // QB-18 — signalling NaNs through the quantile family, one-element products and scans: the quiet bit differs
        // ------------------------------------------------------------------

        /// <summary>
        ///     Three places where NumPy 2.4.2 and NumSharp disagree only on a signalling NaN's quiet bit (sign and payload
        ///     survive):
        ///     <list type="bullet">
        ///         <item>float32 / float16 <c>np.median</c>, <c>np.percentile</c> and <c>np.quantile</c> (every method, the
        ///         selecting <c>lower</c> / <c>nearest</c> included, flat and per axis) return a slice's NaN quieted; NumPy
        ///         returns it untouched (<c>_median_nancheck</c>'s <c>data.take(-1)</c>, <c>_quantile</c>'s <c>arr[-1]</c>).
        ///         float64 already keeps it.</item>
        ///         <item><c>np.prod</c> of one element (flat, or along a length-1 axis) copies a signalling NaN; NumPy's
        ///         reduction multiplies it by the identity 1, which quiets it (the real-dtype face of
        ///         <c>OpenBugsReductionTests.Prod_OneComplexElement_IsTheIdentityTimesTheElement</c>).</item>
        ///         <item><c>np.cumsum</c> / <c>np.cumprod</c> quiet a signalling NaN in the first position; NumPy's accumulate
        ///         copies the first element (the root cause of
        ///         <c>OpenBugsReductionTests.CumSum_Flat_KeepsFirstNegativeZero_LikeNumPysAccumulate</c> and
        ///         <c>CumProd_Complex_CopiesTheFirstElement</c>, visible for real cumprod only through a signalling NaN).</item>
        ///     </list>
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB18")]
        public void ParityQb_QB18_SignallingNaN_ThroughQuantilesProductsAndScans()
        {
            using var f32 = np.array(new[] { unchecked((int)0xff800789) }).view(np.float32);
            using var f64 = np.array(new[] { unchecked((long)0xfff0000000000789UL) }).view(np.float64);
            using var lead = np.array(new[] { unchecked((long)0xfff0000000000789UL), 0x3ff0000000000000L, 0x4000000000000000L }).view(np.float64);

            using var median = np.median(f32).view(np.int32);
            using var prod = np.prod(f64).view(np.int64);
            using var cumsum = np.cumsum(lead).view(np.int64);

            // NumPy 2.4.2: the median keeps it signalling, the product quiets it, cumsum's first element is a copy
            median.GetInt32().Should().Be(unchecked((int)0xff800789));
            prod.GetInt64().Should().Be(unchecked((long)0xfff8000000000789UL));
            cumsum.GetInt64(0).Should().Be(unchecked((long)0xfff0000000000789UL));

            // NumPy 2.4.2: float16 cbrt / arcsinh / arccosh / arctanh hand a signalling NaN back untouched (the UCRT float
            // function returns its NaN argument; npy_half_to_float / float_to_half keep the quiet bit); NumSharp's float16
            // widen-unary path quiets it (0x7c01 -> 0x7e01). float16 sqrt / sinh quiet it on both sides.
            using var h = np.array(new short[] { 0x7c01 }).view(np.float16);
            using var cbrt = np.cbrt(h).view(np.int16);
            cbrt.GetInt16().Should().Be((short)0x7c01);
        }

        // ------------------------------------------------------------------
        // QB-19 — a fancy assignment whose VALUE shares memory with the destination reads it while writing
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>array_assign_subscript</c> (mapping.c) copies the value first when it may share memory with
        ///     the array (<c>solve_may_share_memory(self, tmp_arr, 1) != 0</c> → <c>PyArray_NewCopy</c>), so a fancy
        ///     assignment behaves as if the right-hand side were evaluated before any store:
        ///     <c>a = np.arange(10); a[[1, 2, 3, 4]] = a[:4]</c> gives <c>[0, 0, 1, 2, 3, 5, ...]</c>. NumSharp's fancy setter
        ///     (<c>NDArray.Indexing.Selection.Setter.cs</c>) has no overlap check: a forward (contiguous) view of the
        ///     destination is read while it is being written, so each store feeds the next and the result is
        ///     <c>[0, 0, 0, 0, 0, 5, ...]</c>. Every dtype, 1-D index arrays of any integer type or a raw <c>long[]</c>,
        ///     permutations, duplicates, negative indices, row indexing of a 2-D array and coordinate pairs; a reversed or
        ///     strided value happens to be materialized first and comes out right.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB19")]
        public void ParityQb_QB19_FancyAssignment_CopiesAnOverlappingValue()
        {
            using var a = np.arange(10);
            using var index = np.array(new long[] { 1, 2, 3, 4 });
            using var value = a[":4"];

            a[index] = value;

            // NumPy 2.4.2: [0, 0, 1, 2, 3, 5, 6, 7, 8, 9]
            a.GetInt64(2).Should().Be(1);
            a.GetInt64(3).Should().Be(2);
            a.GetInt64(4).Should().Be(3);
        }

        // ------------------------------------------------------------------
        // QB-20 — a fancy assignment whose INDEX array is a view of the destination reads the index while writing
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 also copies every index ARRAY that may share memory with the destination before storing
        ///     (<c>array_assign_subscript</c>: <c>solve_may_share_memory(self, indices[i].object, 1)</c> →
        ///     <c>PyArray_Copy</c>), so <c>a = np.array([2, 0, 1, 3]); a[a[:3]] = 7</c> is <c>[7, 7, 7, 3]</c> and
        ///     <c>a[a] = [5, 6, 7, 8]</c> over <c>[1, 2, 3, 0]</c> is <c>[8, 5, 6, 7]</c>. NumSharp reads the index array
        ///     lazily during the stores, so an index already overwritten points elsewhere: the first two raise
        ///     <c>IndexError: index 7 is out of bounds for axis 0 with size 4</c> (an error where NumPy computes), and
        ///     <c>a[a] = a</c> silently gives <c>[1, 1, 3, 3]</c> (NumPy <c>[0, 1, 2, 3]</c>). <c>np.put_along_axis</c>
        ///     (NumPy's <c>arr[_make_along_axis_idx(...)] = values</c>) inherits it: <c>np.put_along_axis(a, a, a * 10, 0)</c>
        ///     over <c>[1, 2, 3, 0]</c> is <c>[1, 10, 3, 30]</c> (NumPy <c>[0, 10, 20, 30]</c>). <c>np.put</c> and
        ///     <c>np.take(out=)</c> with aliasing indices already match.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB20")]
        public void ParityQb_QB20_FancyAssignment_CopiesAnIndexThatAliasesTheDestination()
        {
            using var a = np.array(new long[] { 2, 0, 1, 3 });
            using var index = a[":3"];
            using var b = np.array(new long[] { 1, 2, 3, 0 });
            using var b10 = b * 10;

            a[index] = 7;
            np.put_along_axis(b, b, b10, 0);

            // NumPy 2.4.2: [7, 7, 7, 3] and [0, 10, 20, 30]
            a.ToArray<long>().Should().Equal(7L, 7L, 7L, 3L);
            b.ToArray<long>().Should().Equal(0L, 10L, 20L, 30L);
        }

        // ------------------------------------------------------------------
        // QB-21 — a boolean-mask assignment copies a reversed overlapping value where NumPy reads through it
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's single-boolean-index assignment (<c>array_assign_subscript</c>'s <c>HAS_BOOL</c> branch →
        ///     <c>array_assign_boolean_subscript</c>) does NOT copy an overlapping value: it streams the value in C order while
        ///     storing, so <c>a = np.arange(6); a[a >= 0] = a[::-1]</c> reads elements it has already overwritten and gives
        ///     <c>[5, 4, 3, 3, 4, 5]</c>. NumSharp materializes the reversed view first and gives <c>[5, 4, 3, 2, 1, 0]</c>
        ///     (a forward view, <c>a[m] = a[:4]</c>, is read in place and already matches). So NumSharp decides copy-or-alias
        ///     by the VALUE's layout, NumPy by the INDEX kind (fancy: copy, QB-19; a single boolean array: read through).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB21")]
        public void ParityQb_QB21_MaskAssignment_ReadsAReversedOverlappingValueThrough()
        {
            using var a = np.arange(6);
            using var mask = a >= 0;
            using var reversed = a["::-1"];

            a[mask] = reversed;

            // NumPy 2.4.2: [5, 4, 3, 3, 4, 5]
            a.ToArray<long>().Should().Equal(5L, 4L, 3L, 3L, 4L, 5L);
        }

        // ------------------------------------------------------------------
        // QB-22 — flatiter assignment does not cycle, truncate or flatten its value as NumPy's iter_ass_subscript does
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>a.flat[index] = value</c> (<c>iter_ass_subscript</c>, iterators.c) reads the value as a FLAT
        ///     sequence and restarts it when it runs out (<c>if (val_it->index == val_it->size) PyArray_ITER_RESET(val_it)</c>),
        ///     np.put's rule: <c>a = np.zeros(6); a.flat[::2] = [1, 2]</c> is <c>[1, 0, 2, 0, 1, 0]</c>, a longer value is
        ///     cut (<c>a.flat[1:3] = [1, 2, 3]</c> stores 1, 2), an N-D value is read in C order and an empty value stores
        ///     nothing. NumSharp's <c>np.FlatIterator</c> (<c>a.flatiter</c>, the NumPy-faithful iterator) requires the exact
        ///     count and raises <c>ValueError: cannot assign 2 input values to the 3 output values where the mask is true</c>
        ///     (a boolean-mask text) for slices, index arrays and boolean masks alike.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB22")]
        public void ParityQb_QB22_FlatiterAssignment_CyclesTheValue()
        {
            using var a = np.zeros(new Shape(6));
            using var value = np.array(new double[] { 1, 2 });

            a.flatiter["::2"] = value;

            // NumPy 2.4.2: [1, 0, 2, 0, 1, 0]
            a.ToArray<double>().Should().Equal(1.0, 0.0, 2.0, 0.0, 1.0, 0.0);
        }

        // ------------------------------------------------------------------
        // QB-23 — flatiter indexing reads a boolean index array as integer indices 0 / 1
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's flatiter takes a boolean array as a MASK over the flat positions (<c>iter_subscript_Bool</c> /
        ///     <c>iter_ass_sub_Bool</c>): <c>np.arange(6).reshape(2, 3).flat[[True, False, True, False, True, False]]</c> is
        ///     <c>[0, 2, 4]</c>, and a mask of another length raises <c>IndexError: boolean index did not match indexed flat
        ///     iterator along axis 0; size of axis is 6 but size of corresponding boolean axis is 2</c>. NumSharp's
        ///     <c>np.FlatIterator</c> converts the mask to the integers 1 / 0 and fancy-indexes with them: the getter returns
        ///     <c>[1, 0, 1, 0, 1, 0]</c> (six elements read at positions 1 and 0), a short mask is accepted, and the setter
        ///     writes every value into positions 0 and 1 (<c>a.flat[a &gt;= 0] = ...</c> rewrites a[1] six times) — silent
        ///     wrong reads and writes.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB23")]
        public void ParityQb_QB23_FlatiterBooleanIndex_IsAMask()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var mask = np.array(new[] { true, false, true, false, true, false });

            using var picked = a.flatiter[mask];

            // NumPy 2.4.2: [0, 2, 4]
            picked.ToArray<long>().Should().Equal(0L, 2L, 4L);
        }

        // ------------------------------------------------------------------
        // QB-24 — flatiter indexing with an N-D index array returns a 1-D result
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>a.flat[ind]</c> with an integer index ARRAY returns an array of the INDEX's shape
        ///     (<c>iter_subscript_int</c> allocates with the index array's dimensions):
        ///     <c>np.arange(6).reshape(2, 3).flat[[[0, 5], [1, 2]]]</c> is the (2, 2) array <c>[[0, 5], [1, 2]]</c>.
        ///     NumSharp's <c>np.FlatIterator</c> returns the same values flattened to shape (4,).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB24")]
        public void ParityQb_QB24_FlatiterIndexArray_KeepsTheIndexShape()
        {
            using var a = np.arange(6).reshape(2, 3);
            using var index = np.array(new long[,] { { 0, 5 }, { 1, 2 } });

            using var picked = a.flatiter[index];

            // NumPy 2.4.2: shape (2, 2)
            picked.shape.Should().Equal(2L, 2L);
        }

        // ------------------------------------------------------------------
        // QB-25 — flatiter assignment copies an overlapping value (and index) where NumPy reads through them
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>iter_ass_subscript</c> never copies: the value and an index array are read through while
        ///     storing. <c>a = np.arange(10); a.flat[1:] = a[:-1]</c> is all zeros, <c>a.flat[::-1] = a</c> is
        ///     <c>[0, 1, 2, 3, 4, 4, 3, 2, 1, 0]</c>, <c>a.flat[[1, 2, 3, 4]] = a[:4]</c> is <c>[0, 0, 0, 0, 0, 5, ...]</c>
        ///     (the opposite of ndarray fancy assignment, QB-19), and <c>a.flat[a[:3]] = 7</c> over <c>[2, 0, 1, 3]</c> raises
        ///     <c>IndexError: index 7 is out of bounds for size 4</c>. NumSharp's <c>np.FlatIterator</c> copies the value and
        ///     the index first, so it returns the as-if-copied results (and <c>[7, 7, 7, 3]</c> for the last). NumPy's answers
        ///     are order-dependent aliasing, so the owner may prefer documenting these.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB25")]
        public void ParityQb_QB25_FlatiterAssignment_ReadsAnOverlappingValueThrough()
        {
            using var a = np.arange(10);
            using var shifted = a[":-1"];

            a.flatiter["1:"] = shifted;

            // NumPy 2.4.2: every element 0
            a.ToArray<long>().Should().OnlyContain(v => v == 0L);
        }

        // ------------------------------------------------------------------
        // QB-26 — np.isreal of a non-C real array and np.polyval of a broadcast x return C-ordered results
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>isreal(x)</c> is <c>imag(x) == 0</c>, a comparison ufunc whose fresh result takes NpyIter's K
        ///     order, so an F-ordered or transposed real input gives an F-ordered result (a (3, 4) bool: strides (1, 3)).
        ///     NumSharp's <c>DefaultEngine.Isreal</c> answers a real dtype with <c>NDArray.ones(dimensions, bool)</c>, C order
        ///     (strides (4, 1)); the complex branch already compares in K order. Likewise <c>np.polyval(p, x)</c> is
        ///     <c>y = zeros_like(x); y = y * x + pv</c>, so a row-broadcast x (strides (0, 8)) gives NumPy's K-order layout
        ///     of the broadcast, F (strides (8, 32) for (4, 3)); NumSharp's result is C (strides (24, 8)). The values match.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB26")]
        public void ParityQb_QB26_IsrealAndPolyval_KeepTheInputsLayout()
        {
            using var f = np.asfortranarray(np.arange(12).astype(np.float64).reshape(3, 4));
            using var row = np.array(new double[] { 3, 1, 2 });
            using var broadcast = np.broadcast_to(row, new Shape(4, 3));
            using var p = np.array(new double[] { 1, 2 });

            using var real = np.isreal(f);
            using var poly = np.polyval(p, broadcast);

            // NumPy 2.4.2: (1, 3) and (8, 32)
            real.strides.Should().Equal(1L, 3L);
            poly.strides.Should().Equal(8L, 32L);
        }

        // ------------------------------------------------------------------
        // QB-27 — np.array(a, copy=False) and squeeze() without a length-1 axis return a new object, not the input
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 returns the INPUT OBJECT itself from <c>np.array(a, copy=False)</c> when no conversion is needed
        ///     (<c>PyArray_CheckFromAny</c> hands the array back) and from <c>np.squeeze(a)</c> / <c>a.squeeze()</c> when there
        ///     is no length-1 axis to remove (<c>PyArray_Squeeze</c>: <c>if (unit_dims == 0) return self</c>), so
        ///     <c>r is a</c> holds and <c>r.flags.owndata</c> is <c>a</c>'s own flag. NumSharp returns a NEW object in both:
        ///     np.array's shares the storage AND reports <c>owndata</c> True (two objects claiming one buffer), squeeze's is a
        ///     view reporting <c>owndata</c> False where NumPy's True (an owning input). <c>np.asarray</c>,
        ///     <c>np.asanyarray</c>, <c>np.require</c>, <c>np.atleast_Nd</c>, <c>np.ascontiguousarray</c> /
        ///     <c>asfortranarray</c> and <c>astype(copy=False)</c> already return the input itself, as NumPy does.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB27")]
        public void ParityQb_QB27_ArrayCopyFalseAndSqueeze_ReturnTheInputItself()
        {
            using var a = np.arange(6.0).reshape(2, 3).copy();

            var noCopy = np.array(a, copy: false);
            var squeezed = np.squeeze(a);

            // NumPy 2.4.2: np.array(a, copy=False) is a -> True; np.squeeze(a) is a -> True
            ReferenceEquals(noCopy, a).Should().BeTrue();
            ReferenceEquals(squeezed, a).Should().BeTrue();
        }

        // ------------------------------------------------------------------
        // QB-28 — a C# multi-dimensional integer array (long[,] / int[,]) is refused as an index
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 reads a nested integer list as an N-D integer index array: <c>np.arange(10)[[[1, 2], [3, 4]]]</c>
        ///     is the (2, 2) array <c>[[1, 2], [3, 4]]</c>, and <c>a[[[0], [1], [2]], [0, 1, 2]] = [2, 3, 4]</c> broadcasts
        ///     the (3, 1) and (3,) index arrays (NumPy's own <c>test_indexing.py::test_index_is_larger</c>). The C# spelling of
        ///     that nested list, a <c>long[,]</c> or <c>int[,]</c>, raises <c>IndexError: only integers, slices (`:`), ellipsis
        ///     (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices</c>, alone or inside an
        ///     <c>object[]</c> tuple, while a <c>bool[,]</c> is accepted as a 2-D mask and a <c>long[]</c> as a 1-D index
        ///     array (wrapping the same values in <c>np.array(...)</c> works).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB28")]
        public void ParityQb_QB28_MultiDimensionalIntegerArray_IsAnIndexArray()
        {
            using var a = np.arange(10);

            using var picked = a[new long[,] { { 1, 2 }, { 3, 4 } }];

            // NumPy 2.4.2: shape (2, 2), [[1, 2], [3, 4]]
            picked.shape.Should().Equal(2L, 2L);
            picked.ToArray<long>().Should().Equal(1L, 2L, 3L, 4L);
        }

        // ------------------------------------------------------------------
        // QB-29 — C# integer literals bind the FLOAT overloads of np.mod / np.remainder / np.linspace
        // ------------------------------------------------------------------

        /// <summary>
        ///     C#'s better-conversion rule prefers <c>float</c> over <c>double</c> (and over a user-defined conversion to
        ///     <c>NDArray</c>) for an <c>int</c> / <c>long</c> / <c>uint</c> / <c>short</c> / <c>byte</c> argument, so the
        ///     <c>float</c> overloads <c>np.mod(NDArray, float)</c>, <c>np.remainder(NDArray, float)</c> and
        ///     <c>np.linspace(float, float, …)</c> capture every integer literal:
        ///     <list type="bullet">
        ///         <item><c>np.mod(np.array([7, -7]), 3)</c> is float64 <c>[1.0, 2.0]</c> (NumPy: int64 <c>[1, 2]</c>, the
        ///         Python int is weak); int8 / uint8 / bool arrays give float32; <c>np.mod([5], 0)</c> is NaN (NumPy's integer
        ///         loop: 0); <c>np.mod([16777217, 33554435], 16777217)</c> is <c>[1.0, 3.0]</c> (NumPy <c>[0, 1]</c>: the
        ///         divisor rounded to 16777216f). The <c>%</c> operator is right.</item>
        ///         <item><c>np.linspace(-20000001, 20000001, 3)</c> is <c>[-20000000, 0, 20000000]</c> (NumPy keeps the Python
        ///         ints exactly: <c>[-20000001, 0, 20000001]</c>); <c>np.linspace(0, 100000001, 5, endpoint: false)</c> steps
        ///         by 20000000 instead of 20000000.2. Any integer bound past 2^24 is rounded to float32 first.</item>
        ///     </list>
        ///     (np.arange's float overload is QB-02.)
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB29")]
        public void ParityQb_QB29_IntegerLiterals_DoNotBindTheFloatOverloads()
        {
            using var a = np.array(new long[] { 7, -7 });

            using var mod = np.mod(a, 3);
            using var space = np.linspace(-20000001, 20000001, 3);

            // NumPy 2.4.2: int64 [1, 2] and [-20000001., 0., 20000001.]
            mod.dtype.Should().Be(np.int64);
            space.GetDouble(0).Should().Be(-20000001.0);
            space.GetDouble(2).Should().Be(20000001.0);
        }

        // ------------------------------------------------------------------
        // QB-30 — the (NDArray, object) overloads read a boxed C# number as a STRONG array, not a weak Python scalar
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.power</c>, <c>np.floor_divide</c>, <c>np.fmod</c>, <c>np.float_power</c>, <c>np.left_shift</c> and
        ///     <c>np.right_shift</c> have an <c>(NDArray x1, object x2)</c> overload ("scalar or array-like") that converts
        ///     <paramref name="x2"/> with <c>np.asanyarray</c>, a STRONG 0-d array of the C# type. A C# number that reaches it
        ///     as an <c>object</c> (a boxed value, an <c>object[]</c> argument list, generic code) is therefore an int32 / float64
        ///     array operand: <c>np.power(int8 [2, 3], (object)2)</c> is int32 <c>[4, 9]</c>, where NumPy's Python int is weak
        ///     under NEP 50 and the result stays int8 (the same literal written directly binds the <c>NDArray</c> overload, whose
        ///     implicit conversion marks it weak, and is right). The library's own rule is that a C# number is a Python scalar
        ///     (<c>NDArray.PythonLiteral</c>); these overloads skip it.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB30")]
        public void ParityQb_QB30_ObjectOverloads_KeepABoxedNumberWeak()
        {
            using var a = np.array(new sbyte[] { 2, 3 });
            object two = 2;

            using var power = np.power(a, two);
            using var floor = np.floor_divide(a, two);

            // NumPy 2.4.2: np.power(int8, 2) and np.floor_divide(int8, 2) stay int8
            power.dtype.Should().Be(np.int8);
            floor.dtype.Should().Be(np.int8);
        }

        // ------------------------------------------------------------------
        // QB-31 and QB-32 — duplicates of earlier pins, found after pinning (journal: "Duplicates")
        // ------------------------------------------------------------------
        // QB-31 (iterating a (3, 0) array yields nothing) is OpenBugs.NDArrayIter_EmptySecondAxis_YieldsTheRows
        // (OpenBugs.cs, found 2026-11-05 by the np.mintypecode port); QB-32 (an object[] list refused by every array_like
        // parameter but np.array / np.asanyarray) is parity QA-28, OpenBugs.ParityQa_Asarray_PythonList_ShouldCoerceLikeNpArray
        // (OpenBugs.ParityQa.cs). Their pins were removed here; the journal keeps the extra call sites QB-32 reached (list
        // VALUES for setitem / put and list operands of the operators, `np.asarray(List<int>)` not compiling).

        // ------------------------------------------------------------------
        // QB-33 — np.nan is .NET's double.NaN, whose sign bit is SET; NumPy's np.nan is the positive quiet NaN
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>np.nan</c> is Python's <c>float('nan')</c>, the positive quiet NaN <c>0x7ff8000000000000</c>:
        ///     <c>np.signbit(np.nan)</c> is False and <c>np.copysign(1.0, np.nan)</c> is 1.0. NumSharp's <c>np.nan</c> (and
        ///     <c>np.NaN</c> / <c>np.NAN</c>, names NumPy 2.0 removed) is <c>double.NaN</c>, <c>0xfff8000000000000</c> on .NET:
        ///     <c>np.signbit(np.nan)</c> is True, <c>np.copysign(1.0, np.nan)</c> is -1.0, and every array built from it
        ///     (<c>np.full(n, np.nan)</c>, <c>np.where(..., np.nan, ...)</c>, <c>np.pad(..., constant_values: np.nan)</c>, a
        ///     float32 cast, a saved .npy file) carries the negative NaN. The library already knows the difference (it
        ///     writes NumPy's <c>NPY_NAN</c> as <c>0x7ff8…</c> where NumPy produces one), but the public constant was never
        ///     changed.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB33")]
        public void ParityQb_QB33_NpNan_IsThePositiveQuietNaN()
        {
            using var a = np.array(new[] { np.nan });
            using var bits = a.view(np.int64);
            using var sign = np.signbit(a);

            // NumPy 2.4.2: 0x7ff8000000000000, signbit False
            bits.GetInt64(0).Should().Be(0x7ff8000000000000L);
            sign.GetBoolean(0).Should().BeFalse();
        }

        // ------------------------------------------------------------------
        // QB-34 — ndarray.base is a NEW NDArray on every access, never the base object itself
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>ndarray.base</c> is the array object a view was made from (collapsed to the memory's owner):
        ///     <c>a[2:5].base is a</c>, <c>a[2:8][1:3].base is a</c>, <c>a.T.base is a</c>, <c>a.view(np.int32).base is a</c>,
        ///     <c>np.broadcast_to(a, …).base is a</c>, and <c>v.base is v.base</c>. NumSharp's <c>NDArray.@base</c> is
        ///     <c>new NDArray(Storage._baseStorage)</c>, a fresh wrapper on every read, so none of those identities holds
        ///     (the values, shape, owndata flag and write-through of the wrapper match), and NumPy's idiom for "is v a view
        ///     of a" (<c>v.base is a</c>) is always false.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB34")]
        public void ParityQb_QB34_Base_IsTheBaseObjectItself()
        {
            using var a = np.arange(10);
            using var v = a["2:5"];

            // NumPy 2.4.2: v.base is a -> True; v.base is v.base -> True
            ReferenceEquals(v.@base, a).Should().BeTrue();
            ReferenceEquals(v.@base, v.@base).Should().BeTrue();
        }

        // ------------------------------------------------------------------
        // QB-35 — a duplicate of an earlier pin, found after pinning (journal: "Duplicates")
        // ------------------------------------------------------------------
        // finfo(float16 / float32).resolution as the dtype's scalar is surface gap SU13,
        // OpenBugs.Surface_Finfo_Resolution_IsRoundedToTheDtype (OpenBugs.Surface.cs), with the same two assertions; the
        // QB-35 pin was removed here.

        // ------------------------------------------------------------------
        // QB-36 — the strides of LENGTH-1 axes differ from NumPy's (newaxis, (1, 1) slices, results over broadcasts)
        // ------------------------------------------------------------------

        /// <summary>
        ///     A length-1 axis can carry any stride without changing what an array addresses, so NumPy's values there are
        ///     a matter of its construction rules, visible through <c>.strides</c>: <c>np.newaxis</c> inserts stride 0
        ///     (<c>a[:, np.newaxis, :]</c> of a (3, 4) int64 array has strides (32, 0, 8)); a slice keeps its parent's stride
        ///     on every axis (<c>a[1:2, 2:3]</c> is (32, 8)); a ufunc allocates its result fresh (<c>np.negative</c> of
        ///     <c>np.broadcast_to(np.arange(3), (1, 3))</c> has strides (24, 8)). NumSharp gives (32, 32, 8), (8, 8) and (0, 8):
        ///     the computed C stride on the new axis, re-derived strides for an all-length-1 slice, and the input's stride
        ///     array reused by the elementwise result. Flags and values are unaffected (a length-1 axis is ignored by the
        ///     contiguity tests); the zero-size cases are the EmptyStrides pins, and broadcast_to's own extent-1 stride is a
        ///     documented divergence (ufuncs.md, np.clip).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB36")]
        public void ParityQb_QB36_LengthOneAxes_TakeNumPysStrides()
        {
            using var a = np.arange(12).reshape(3, 4);
            using var withAxis = a[":, np.newaxis, :"];
            using var oneOne = a["1:2, 2:3"];
            using var row = np.arange(3);
            using var broadcast = np.broadcast_to(row, new Shape(1, 3));
            using var negated = np.negative(broadcast);

            // NumPy 2.4.2: (32, 0, 8), (32, 8), (24, 8)
            withAxis.strides.Should().Equal(32L, 0L, 8L);
            oneOne.strides.Should().Equal(32L, 8L);
            negated.strides.Should().Equal(24L, 8L);
        }

        // ------------------------------------------------------------------
        // QB-37 — the bitwise ufuncs judge the inputs' own promotion before the dtype= loop
        // ------------------------------------------------------------------

        /// <summary>
        ///     With <c>dtype=</c>, NumPy 2.4.2 resolves the bitwise ufuncs' loop from the requested dtype and then checks each
        ///     input's same_kind cast into it: <c>np.bitwise_or(int64 [3], uint64 [2], dtype=np.int64)</c> computes int64
        ///     <c>[3]</c> (uint64 → int64 is same_kind); <c>np.bitwise_and(float64, float64, dtype=np.int64)</c> raises
        ///     <c>Cannot cast ufunc 'bitwise_and' input 0 from dtype('float64') to dtype('int64') with casting rule 'same_kind'</c>;
        ///     a float <c>dtype=</c> over a float input raises <c>No loop matching the specified signature and casting was found
        ///     for ufunc bitwise_and</c>. NumSharp checks the inputs' own promotion first (int64 with uint64 has no integer
        ///     result, a float input has no bitwise loop) and raises <c>ufunc 'bitwise_or' not supported for the input types,
        ///     and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''</c> in
        ///     every one of these cases — a refusal where NumPy computes for the first, the wrong text for the others
        ///     (bitwise_and / or / xor and the shifts; invert and an all-integer input with a float dtype= already match).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB37")]
        public void ParityQb_QB37_BitwiseDtypeLoop_DecidesBeforeTheInputPromotion()
        {
            using var a = np.array(new long[] { 3 });
            using var b = np.array(new ulong[] { 2 });

            using var r = np.bitwise_or(a, b, dtype: np.int64);

            // NumPy 2.4.2: int64 [3]
            r.dtype.Should().Be(np.int64);
            r.GetInt64(0).Should().Be(3L);
        }

        // ------------------------------------------------------------------
        // QB-38 — the printer's `min_val < 0.0001` exponential test runs in float64
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>FloatingFormat.fillFormat</c> switches to exponential notation when <c>min_val &lt; 0.0001</c>,
        ///     where <c>min_val</c> is the array's own scalar type (<c>np.min</c> of the absolute nonzero finite values) and the
        ///     Python float <c>0.0001</c> is a weak operand (NEP 50), so the test runs in the ARRAY'S dtype: for float32 it is
        ///     <c>float32(1e-4) &lt; float32(1e-4)</c>, false, and <c>np.array([1e-4, 1e-3, 1e-2], dtype=np.float32)</c> prints
        ///     <c>array([0.0001, 0.001 , 0.01  ], dtype=float32)</c>. NumSharp's <c>ElementFormatters.FillFormat</c> compares the
        ///     value widened to double with the double <c>0.0001</c>; float32's 1e-4 is 9.99999974737875e-05 there, below the
        ///     cutoff, so it prints <c>array([1.e-04, 1.e-03, 1.e-02], dtype=float32)</c> — every <c>str</c> / <c>repr</c> /
        ///     <c>array2string</c> of a float32 array whose smallest nonzero magnitude is exactly <c>float32(1e-4)</c>, the 0-d
        ///     repr included (the float32 just below prints exponential in both; float16 and float64 already match, their 1e-4
        ///     is not below the double cutoff; the max/min ratio test beside it already runs natively).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB38")]
        public void ParityQb_QB38_FloatPrinting_ExponentialCutoffComparesInTheArraysDtype()
        {
            using var steps = np.array(new float[] { 0.0001f, 0.001f, 0.01f });
            using var zeroD = NDArray.Scalar(0.0001f);

            // NumPy 2.4.2: repr(np.array([1e-4, 1e-3, 1e-2], dtype=np.float32)) and repr(np.array(np.float32(1e-4)))
            steps.ToString(true).Should().Be("array([0.0001, 0.001 , 0.01  ], dtype=float32)");
            zeroD.ToString(true).Should().Be("array(0.0001, dtype=float32)");
            np.array2string(steps).Should().Be("[0.0001 0.001  0.01  ]");
        }

        // ------------------------------------------------------------------
        // QB-39 — the printer's rstrip trims spaces only, Python's str.rstrip() every whitespace character
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>_formatArray</c> ends a row with <c>separator.rstrip()</c> (its <c>line_sep</c>), sizes an element
        ///     with <c>max(len(separator.rstrip()), len(']'))</c>, and breaks a wrapped line with <c>line.rstrip()</c>; Python's
        ///     <c>str.rstrip()</c> removes EVERY whitespace character (<c>\t \n \v \f \r</c>, U+001C–U+001F, U+0085, U+00A0, the
        ///     Unicode space separators, U+2028 / U+2029). NumSharp's <c>ArrayFormatter.RStrip</c> is <c>TrimEnd(' ')</c>, so a
        ///     tab, newline, carriage-return, vertical-tab, NBSP or ideographic-space separator stays at the end of each row of a
        ///     2-D+ array (<c>np.array2string(a2d, separator='\t')</c> is <c>[[0.\t1.\t2.]\t\n …</c> where NumPy prints
        ///     <c>[[0.\t1.\t2.]\n …</c>), and a wrapped 1-D line keeps its trailing tab and loses an element of width
        ///     (<c>separator=',\t', max_line_width=30</c> fits 5 elements per line where NumPy fits 6).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB39")]
        public void ParityQb_QB39_PrintingRStrip_StripsEveryWhitespaceCharacter()
        {
            using var grid = np.arange(6.0).reshape(2, 3);
            using var line = np.arange(12.0);

            // NumPy 2.4.2: np.array2string(np.arange(6.).reshape(2, 3), separator='\t')
            np.array2string(grid, separator: "\t").Should().Be("[[0.\t1.\t2.]\n [3.\t4.\t5.]]");
            // NumPy 2.4.2: np.array2string(np.arange(12.), separator=',\t', max_line_width=30)
            np.array2string(line, separator: ",\t", max_line_width: 30)
                .Should().Be("[ 0.,\t 1.,\t 2.,\t 3.,\t 4.,\t 5.,\n  6.,\t 7.,\t 8.,\t 9.,\t10.,\t11.]");
        }

        // ------------------------------------------------------------------
        // QB-40 — a negative precision is refused by the options instead of by the float formatter
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 checks <c>precision &gt;= 0</c> only where a FLOAT formatter is built (<c>FloatingFormat.__init__</c>'s
        ///     <c>_none_or_positive_arg</c>): <c>np.array2string(np.array([1.5]), precision=-1)</c> and <c>np.array_repr</c> /
        ///     <c>np.array_str</c> of a float or complex array raise <c>ValueError: precision must be &gt;= 0</c>. NumSharp's
        ///     <c>np.array2string</c> / <c>array_repr</c> / <c>array_str</c> never check (<c>BuildOptions</c> has no test), so
        ///     a float array prints as if the precision were valid. The other half of the same misplacement —
        ///     <c>set_printoptions</c> / <c>printoptions</c> refusing a negative precision NumPy stores — is the earlier pin
        ///     <c>Surface_SetPrintoptions_NegativePrecision_IsStoredUntilAFloatPrints</c> (gap SU22).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB40")]
        public void ParityQb_QB40_NegativePrecision_IsRefusedByTheFloatFormatter()
        {
            using var floats = np.array(new double[] { 1.5, 2.25 });

            // NumPy 2.4.2: ValueError: precision must be >= 0
            Action floatString = () => np.array2string(floats, precision: -1);
            floatString.Should().Throw<Exception>().WithMessage("precision must be >= 0");
        }

        // ------------------------------------------------------------------
        // QB-41 — a negative edgeitems drops the summary marker
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 does not validate <c>edgeitems</c>; a negative value goes through <c>_formatArray</c>'s arithmetic:
        ///     <c>2*edge_items &lt; a_len</c> keeps the summary, <c>range(edge_items)</c> and <c>range(edge_items, 1, -1)</c> are
        ///     empty, and the always-printed last child follows the marker, so <c>np.array2string(np.array([1.5, 2.25]),
        ///     edgeitems=-1, threshold=1)</c> is <c>[... 2.25]</c> (and a 3-D array prints <c>[...\n\n [...\n  [... 23]]]</c>).
        ///     NumSharp's <c>ArrayFormatter.Pieces</c> counts <c>leading + 1 + max(edgeItems, 1)</c> pieces with a negative
        ///     <c>leading</c>, which drops the marker: <c>[2.25]</c> and <c>[[[23]]]</c>. A degenerate option, but NumPy's text is
        ///     well-defined.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB41")]
        public void ParityQb_QB41_NegativeEdgeitems_KeepsTheSummaryMarker()
        {
            using var pair = np.array(new double[] { 1.5, 2.25 });

            // NumPy 2.4.2: np.array2string(np.array([1.5, 2.25]), edgeitems=-1, threshold=1)
            np.array2string(pair, edgeitems: -1, threshold: 1).Should().Be("[... 2.25]");
        }

        // ------------------------------------------------------------------
        // QB-42 — bool storage bytes other than 0 / 1: the hash / compare routes read the raw byte
        // ------------------------------------------------------------------

        /// <summary>
        ///     A bool array can hold bytes other than 0 and 1 when it is a view of other data
        ///     (<c>np.array([0, 2, 1, 255], dtype=np.uint8).view(np.bool_)</c>, <c>np.frombuffer(..., np.bool_)</c>); NumPy 2.4.2
        ///     reads every such byte as True (its loops test <c>byte != 0</c>), so <c>np.unique</c> of it is <c>[False, True]</c>,
        ///     <c>np.array_equal</c> with the canonical <c>[False, True, True, True]</c> is True, and <c>np.isin(b, [True])</c> is
        ///     True wherever the byte is nonzero. NumSharp's ufuncs, casts, reductions, masks and sorts already agree (79 + 62
        ///     probes), but the routes that hash or compare the RAW byte do not: the unique family and the set routines built
        ///     on it (<c>unique</c> with every return, <c>unique_values</c> / <c>unique_counts</c>, the axis form,
        ///     <c>intersect1d</c> / <c>union1d</c> / <c>setdiff1d</c>) count each distinct nonzero byte as its own True,
        ///     <c>isin</c> (every <c>kind</c>, <c>invert</c>) matches bytes, <c>array_equal</c> / <c>array_equiv</c> compare
        ///     bytes, and <c>bitwise_count</c> counts the byte's bits (255 → 8 where NumPy's bool loop gives 1). Two producers
        ///     also leak such bytes into their results where NumPy writes 1: the 2-D bool <c>dot</c> / <c>matmul</c> and
        ///     <c>np.min</c> along an axis.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB42")]
        public void ParityQb_QB42_NonCanonicalBoolBytes_ReadAsTrue()
        {
            using var bytes = np.array(new byte[] { 0, 2, 1, 255, 4, 0, 1, 128 });
            using var mask = bytes.view(np.@bool);
            using var canonical = np.array(new bool[] { false, true, true, true, true, false, true, true });
            using var trueOnly = np.array(new bool[] { true });

            // NumPy 2.4.2: np.unique(b) is [False, True]
            using (var unique = np.unique(mask).values)
                unique.size.Should().Be(2);

            // NumPy 2.4.2: np.array_equal(b, [F, T, T, T, T, F, T, T]) is True
            np.array_equal(mask, canonical).Should().BeTrue();

            // NumPy 2.4.2: np.isin(b, [True]) is True wherever the byte is nonzero
            using (var isin = np.isin(mask, trueOnly))
                isin.ToArray<bool>().Should().Equal(false, true, true, true, true, false, true, true);
        }

        // ------------------------------------------------------------------
        // QB-43 — a partition at SEVERAL kth is NumPy's scalar introselect_: its choice among equal values is defined
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2 dispatches a partition to x86-simd-sort only for ONE kth (<c>introselect_noarg</c>:
        ///     <c>nkth == 1 &amp;&amp; quickselect_dispatch(...)</c>); several kth — every np.median / np.quantile call
        ///     (<c>kth + [-1]</c>, <c>[0, -1, prev, next]</c>) and an explicit kth list — run the scalar <c>introselect_</c> on
        ///     every host, whose swaps decide which of several EQUAL values lands at a kth: <c>np.partition([0., -0., -2.],
        ///     [1, 2])</c> is <c>[-2., -0., 0.]</c> (dumb_select swaps the minimum to the front),
        ///     <c>np.argpartition([0., -0., -2.], [1, 2])</c> is <c>[2, 1, 0]</c>, and
        ///     <c>np.quantile([0., -0., 3., 0., 3., -2.], 0.6, method='lower')</c> is <c>-0.0</c>. NumSharp's
        ///     <c>Utilities/QuickSelect.cs</c> insertion-sorts a window of up to 16 elements before its dumb_select / maximum
        ///     shortcuts (a stable order, not dumb_select's swaps) and partitions longer lines its own way, so the tied
        ///     element differs: <c>[-2., 0., -0.]</c>, <c>[2, 0, 1]</c>, <c>0.0</c>. Visible as the SIGN of a zero result of
        ///     np.partition / the discontinuous quantile methods / np.median, and as argpartition's tied indices (in
        ///     <c>q104</c> 57 of 300 random lines of 3–64 elements; the kth VALUES always compare equal). quantile.md's "keep
        ///     NumPy's choice among ties" claim for the shortcuts does not survive the window sort.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB43")]
        public void ParityQb_QB43_MultiKthPartition_KeepsIntroselectsChoiceAmongTies()
        {
            using var zeros = np.array(new double[] { 0.0, -0.0, -2.0 });
            using var data = np.array(new double[] { 0.0, -0.0, 3.0, 0.0, 3.0, -2.0 });

            // NumPy 2.4.2: np.argpartition([0., -0., -2.], [1, 2]) is [2, 1, 0]
            using (var order = np.argpartition(zeros, new[] { 1, 2 }))
                order.ToArray<long>().Should().Equal(2L, 1L, 0L);

            // NumPy 2.4.2: np.partition([0., -0., -2.], [1, 2])[1] is -0.0
            using (var parted = np.partition(zeros, new[] { 1, 2 }))
                BitConverter.DoubleToInt64Bits(parted.GetDouble(1)).Should().Be(BitConverter.DoubleToInt64Bits(-0.0));

            // NumPy 2.4.2: np.quantile([0., -0., 3., 0., 3., -2.], 0.6, method='lower') is -0.0
            using (var lower = np.quantile(data, 0.6, method: "lower"))
                BitConverter.DoubleToInt64Bits(lower.GetDouble()).Should().Be(BitConverter.DoubleToInt64Bits(-0.0));
        }

        // ------------------------------------------------------------------
        // QB-44 — a C# long / ulong needle binds searchsorted's DOUBLE overload
        // ------------------------------------------------------------------

        /// <summary>
        ///     <c>np.searchsorted</c> and <c>ndarray.searchsorted</c> have scalar overloads for <see cref="int"/> and
        ///     <see cref="double"/> only; C#'s better-conversion rule sends a <see cref="long"/> or <see cref="ulong"/>
        ///     literal to the <see cref="double"/> one (an implicit numeric conversion beats the user-defined one to
        ///     <c>NDArray</c>), so a needle past 2^53 is rounded before the search: over int64
        ///     <c>[2^53, 2^53 + 1, 2^53 + 2]</c> the needle <c>9007199254740993L</c> searches as 2^53 and returns 0, and over
        ///     uint64 <c>[2^64 - 3, 2^64 - 2]</c> the needle <c>2^64 - 2</c> returns 0. NumPy 2.4.2 discovers a Python int
        ///     needle's dtype beside the array's (<c>PyArray_DescrFromObject(v, a.dtype)</c>) and searches exactly: 1 and 1.
        ///     The float needles are right (a float16 / float32 haystack searches the weak double as its own dtype); the 02:13
        ///     / 02:17 overload-capture sweeps did not list searchsorted's <c>(NDArray, double)</c> pair.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB44")]
        public void ParityQb_QB44_SearchsortedLongNeedle_IsNotRoundedToDouble()
        {
            using var signed = np.array(new long[] { 9007199254740992L, 9007199254740993L, 9007199254740994L });
            using var unsigned = np.array(new ulong[] { 18446744073709551613UL, 18446744073709551614UL });

            // NumPy 2.4.2: np.searchsorted([2**53, 2**53 + 1, 2**53 + 2], 2**53 + 1) is 1
            np.searchsorted(signed, 9007199254740993L).Should().Be(1L);
            signed.searchsorted(9007199254740993L).Should().Be(1L);
            // NumPy 2.4.2: np.searchsorted(uint64 [2**64 - 3, 2**64 - 2], 2**64 - 2) is 1
            np.searchsorted(unsigned, 18446744073709551614UL).Should().Be(1L);
        }

        // ------------------------------------------------------------------
        // QB-45 — einsum's default order 'K' over operands that are BOTH C- and F-contiguous gives an F-ordered result
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's default <c>np.einsum</c> (<c>c_einsum</c>) allocates its output through NpyIter in K order: the
        ///     operands' strides vote on the axis order, and where they do not decide — two 1-D vectors mapped to different
        ///     output axes, <c>(2, 1)</c> by <c>(1, 3)</c> — C order wins, so <c>np.einsum('i,j->ij', arange(3.), arange(4.))</c>
        ///     has strides <c>(32, 8)</c> and <c>'i,j,k->ijk'</c> over three vectors <c>(96, 32, 8)</c>. NumSharp's rule
        ///     ("'A' and 'K' give F-contiguous when every input is F-contiguous") counts a 1-D or single-row / single-column
        ///     operand as F-contiguous — it is both — and returns <c>(8, 24)</c> / <c>(8, 16, 48)</c>: the values match, the
        ///     layout (and every <c>ravel('K')</c>, <c>tobytes('A')</c>, flags read) differs. A non-contiguous vector
        ///     (<c>[::-1]</c>, <c>[::2]</c>) and <c>order='A'</c> (NumPy's F there too) already match; np.outer / tensordot /
        ///     a broadcast multiply are C on both sides.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB45")]
        public void ParityQb_QB45_EinsumKOrder_OverBothContiguousOperands_IsC()
        {
            using var a = np.arange(3.0);
            using var b = np.arange(4.0);
            using var outer = np.einsum("i,j->ij", a, b);

            // NumPy 2.4.2: (32, 8), C-contiguous
            outer.strides.Should().Equal(32L, 8L);
            outer.Shape.IsContiguous.Should().BeTrue();
        }

        // ------------------------------------------------------------------
        // QB-46 — einsum with optimize= returns NumPy's bmm_einsum result as it leaves it (a view of the matmul)
        // ------------------------------------------------------------------

        /// <summary>
        ///     With <c>optimize=True</c> (or a path name / an explicit path) and the default <c>order='K'</c>, NumPy 2.4.2
        ///     contracts each pair through <c>bmm_einsum</c>, whose <c>_parse_output_order('K')</c> is None: the result is what
        ///     the final <c>matmul</c> plus its <c>reshape</c> / <c>transpose(perm_ab)</c> leave, a NON-OWNING view —
        ///     <c>np.einsum('ij,jk->ik', a23, b34, optimize=True)</c> has strides <c>(8, 16)</c> and <c>owndata</c> False,
        ///     <c>'ij,jk->ki'</c> <c>(16, 8)</c>, the batch <c>'bij,bjk->bik'</c> <c>(120, 8, 24)</c>, a three-operand chain
        ///     <c>(8, 16)</c>. NumSharp ignores the path for the layout and allocates its own (C here: <c>(32, 8)</c>, owning,
        ///     <c>(8, 32)</c> for <c>'ki'</c>, <c>(120, 40, 8)</c> for the batch). Values match; <c>order='C'</c> matches.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB46")]
        public void ParityQb_QB46_EinsumOptimize_ReturnsTheBmmResultsLayout()
        {
            using var a = np.arange(6.0).reshape(2, 3);
            using var b = np.arange(12.0).reshape(3, 4);
            using var product = np.einsum("ij,jk->ik", new[] { a, b }, optimize: true);

            // NumPy 2.4.2: strides (8, 16), owndata False (a transposed view of the matmul result)
            product.strides.Should().Equal(8L, 16L);
            product.Shape.IsContiguous.Should().BeFalse();
        }

        // ------------------------------------------------------------------
        // QB-47 — a 0-d masked array prints its element in the wrong style
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>MaskedArray.__repr__</c> / <c>__str__</c> print <c>_insert_masked_print()</c>: with no mask it is
        ///     the DATA ARRAY, which <c>np.array2string</c> formats with the float formatter (a 0-d float64 <c>-0.0</c> is
        ///     <c>data=-0.</c>, <c>1/3</c> is <c>data=0.33333333</c>, <c>1e16</c> is <c>data=1.e+16</c>); with a mask it is an
        ///     OBJECT copy, whose element is the Python float <c>astype(object)</c> makes (a float32 <c>0.1</c> is
        ///     <c>0.10000000149011612</c>, a float16 <c>123.44</c> is <c>123.4375</c>, in <c>str</c> and <c>repr</c>). NumSharp's
        ///     0-d masked print does the reverse on both: <c>data=-0.0</c> / <c>0.3333333333333333</c> / <c>1e+16</c> for the
        ///     nomask array, and the narrow type's own shortest digits (<c>0.1</c>, <c>123.44</c>) for the masked one. N-D
        ///     masked arrays already print both ways right (masked-arrays.md's Python-float rule holds there).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB47")]
        public void ParityQb_QB47_ZeroDimMaskedArrayPrint_FollowsInsertMaskedPrint()
        {
            // NDMaskedArray is not IDisposable: its data and mask arrays are reclaimed with the wrapper.
            var nomask = np.ma.array(NDArray.Scalar(-0.0));
            var masked32 = np.ma.array(NDArray.Scalar(0.1f), mask: false);

            // NumPy 2.4.2: repr(np.ma.array(np.array(-0.0)))
            nomask.ToString(true).Should().Be("masked_array(data=-0.,\n             mask=False,\n       fill_value=1e+20)");
            // NumPy 2.4.2: str(np.ma.array(np.array(np.float32(0.1)), mask=False))
            masked32.ToString().Should().Be("0.10000000149011612");
        }

        // ------------------------------------------------------------------
        // QB-48 — np.empty_like(a, dtype, 'F') binds the order char to Shape: a (70,) array
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's third positional parameter of <c>np.empty_like(prototype, dtype=None, order='K', ...)</c> is <c>order</c>,
        ///     so <c>np.empty_like(a, None, 'F')</c> is an F-ordered array of <c>a</c>'s shape. NumSharp's first overload is
        ///     <c>empty_like(NDArray prototype, DType dtype = null, Shape shape = default, string device = null)</c>, and C#
        ///     converts a <see cref="char"/> to <see cref="int"/> and the <see cref="int"/> to <c>Shape</c> implicitly, so the
        ///     ported call <c>np.empty_like(a, null, 'F')</c> compiles and allocates a 1-D array of 70 elements ('F' is
        ///     U+0046); <c>'C'</c> gives 67, <c>'K'</c> 75, <c>'A'</c> 65. The order overload needs an explicit shape
        ///     (<c>np.empty_like(a, null, default, 'F')</c>). zeros_like / ones_like / full_like take the char as their order,
        ///     and np.empty / zeros / ones / full refuse it at compile time; the library already guards the same trap on
        ///     <c>np.ndarray(shape, dtype, buffer, 'F')</c> with an <c>[Obsolete(error: true)]</c> overload.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB48")]
        public void ParityQb_QB48_EmptyLikeOrderChar_IsNotAShape()
        {
            using var a = np.zeros(new Shape(2, 3));
            using var e = np.empty_like(a, null, 'F');

            // NumPy 2.4.2: np.empty_like(np.zeros((2, 3)), None, 'F') has shape (2, 3) and is F-ordered
            e.shape.Should().Equal(2L, 3L);
            e.Shape.IsFContiguous.Should().BeTrue();
        }

        // ------------------------------------------------------------------
        // QB-49 — np.ma.make_mask / np.ma.mask_or: NumPy's positional `copy` lands on C#'s `shrink`
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>make_mask(m, copy=False, shrink=True, dtype=np.bool)</c> and <c>mask_or(m1, m2, copy=False,
        ///     shrink=True)</c> take <c>copy</c> positionally before <c>shrink</c>, so <c>np.ma.make_mask(np.array([0, 0]),
        ///     False)</c> and <c>np.ma.mask_or(m, m, False)</c> still SHRINK an all-False mask to <c>nomask</c> (a 0-d False).
        ///     NumSharp's <c>make_mask(object m, bool shrink = true)</c> / <c>mask_or(object m1, object m2, bool shrink =
        ///     true)</c> have no <c>copy</c>, so the ported positional False binds <c>shrink</c> and the result is the full
        ///     <c>[False, False]</c> array. numpy-signature-diff.md lists both slots; the silent consequence is the finding
        ///     (the sibling slot mismatches of numpy.ma — <c>max(axis, out)</c>, <c>ma.array(data, dtype)</c>,
        ///     <c>masked_array(data, mask, dtype)</c> — fail loudly instead).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB49")]
        public void ParityQb_QB49_MakeMaskAndMaskOr_PositionalCopyIsNotShrink()
        {
            using var zeros = np.array(new long[] { 0, 0 });
            using var falses = np.array(new bool[] { false, false });

            // NumPy 2.4.2: np.ma.make_mask(np.array([0, 0]), False) and np.ma.mask_or(m, m, False) are nomask (0-d False)
            using (var made = np.ma.make_mask(zeros, false))
                made.ndim.Should().Be(0);
            using (var either = np.ma.mask_or(falses, falses, false))
                either.ndim.Should().Be(0);
        }

        // ------------------------------------------------------------------
        // QB-50 — complex cumprod over an axis of length 2: NumPy's one-element accumulate call runs simd_cmul
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>CDOUBLE_multiply</c> picks its arithmetic per call by the operands' memory overlap
        ///     (<c>is_mem_overlap</c>): an accumulate call <c>out[i] = out[i-1] * in[i]</c> over n ≥ 2 elements overlaps its
        ///     own output and runs the MSVC-contracted <c>loop_scalar</c> (<c>im = fma(a.im, b.re, a.re*b.im)</c>), but along an
        ///     accumulation axis of length 2 the call has ONE element, whose ranges do not overlap, so it runs
        ///     <c>simd_cmul</c> (<c>im = fma(a.re, b.im, a.im*b.re)</c>). With <c>t = 2·sign([3-1j, 3+1j])</c>,
        ///     <c>np.cumprod(t)[1]</c> is <c>4+3.473028842108373e-17j</c> (np.multiply's bits), while three or more elements
        ///     give the <c>-3.47e-17j</c> of loop_scalar; a (2, 2) array along axis 0 and a (1, 2) along axis 1 take the
        ///     one-element form too. NumSharp's complex cumprod multiplies with loop_scalar on every route (reductions.md:
        ///     "every multiply.accumulate step ... runs loop_scalar"), so a length-2 axis gets <c>4-3.47e-17j</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB50")]
        public void ParityQb_QB50_ComplexCumprodOfLengthTwo_RunsSimdCmul()
        {
            using var signs = np.sign(np.array(new[] { new Complex(3, -1), new Complex(3, 1) }));
            using var t = signs + signs;
            using var scan = np.cumprod(t);

            // NumPy 2.4.2: np.cumprod(t)[1].imag is +3.473028842108373e-17
            ((Complex)scan[1]).Imaginary.Should().Be(3.473028842108373e-17);
        }

        // ------------------------------------------------------------------
        // QB-51 — complex np.multiply over a single-element NpyIter iteration runs loop_scalar in NumPy
        // ------------------------------------------------------------------

        /// <summary>
        ///     NumPy 2.4.2's <c>CDOUBLE_multiply</c> skips its <c>simd_cmul</c> body when the destination stride is 0, which
        ///     NpyIter gives every operand of a single-element iteration (<c>nditer_constr.c</c>: <c>if (bshape == 1)
        ///     strides[iop] = 0</c>), reached when the non-0-d operands differ in ndim (<c>try_trivial_single_output_loop</c>
        ///     refuses) or an out of another shape is given: <c>np.multiply(t[0:1].reshape(1, 1), t[1:2])</c>,
        ///     <c>(1,) * (1, 1, 1)</c>, a <c>(1, 1)</c> <c>out=</c>, and <c>np.multiply(a[0:1], a[1:2], out=a[1:2])</c> run the
        ///     contracted <c>loop_scalar</c> (<c>im = fma(a.im, b.re, a.re*b.im)</c>): with <c>t = 2·sign([3-1j, 3+1j])</c> the
        ///     product is <c>4-3.473028842108373e-17j</c>. NumSharp's <c>NDComplexMath.Multiply</c> is always the simd form
        ///     and gives <c>4+3.47e-17j</c>. This confirms the open step 10 of docs/plans/numpy-polynomial-review.md ("probe
        ///     NumSharp np.multiply on complex128 for (1,1) x (1,), (1,) x (1,1,1) and an in-place overlap … if they differ,
        ///     it is a core fix"); QB-50 is the mirror case in cumprod. Equal-ndim one-element operands, a 0-d operand and
        ///     broadcast-to-3 operands already match.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "parity-qa-QB51")]
        public void ParityQb_QB51_ComplexMultiplyOfOneElementIteration_RunsLoopScalar()
        {
            using var signs = np.sign(np.array(new[] { new Complex(3, -1), new Complex(3, 1) }));
            using var t = signs + signs;
            using var left = t["0:1"].reshape(1, 1);
            using var right = t["1:2"];
            using var product = np.multiply(left, right);

            // NumPy 2.4.2: np.multiply(t[0:1].reshape(1, 1), t[1:2]) is [[4-3.473028842108373e-17j]]
            ((Complex)product[0, 0]).Imaginary.Should().Be(-3.473028842108373e-17);
        }
    }
}
