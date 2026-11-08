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
    }
}
