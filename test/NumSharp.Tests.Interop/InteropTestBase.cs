using System;
using System.Diagnostics;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Shared plumbing for all engine-backed interop tests.
    ///
    ///     <para><b>Leak gate:</b> every test captures the <see cref="NNDArrayInteropLiveExports"/> /
    ///     <see cref="NDArrayPythonInterop.LiveImports"/> baseline on entry, and the cleanup FAILS the test
    ///     unless the counters return to that baseline — so every single test doubles as a
    ///     no-leak / no-premature-free assertion.</para>
    ///
    ///     <para><b>GIL discipline:</b> pythonnet 3.0.x requires the GIL for <c>PyObject.Dispose</c>
    ///     (the final decref runs <c>PyErr_Fetch</c>), so every helper here that touches a
    ///     <see cref="PyObject"/> creates AND disposes it inside its own <see cref="Py.GIL"/> scope;
    ///     tests handle NumSharp arrays and plain CLR values only, unless they explicitly open
    ///     <see cref="Gil"/>.</para>
    ///
    ///     <para><b>GC realism:</b> collection-dependent lifecycles must run inside
    ///     <c>[MethodImpl(MethodImplOptions.NoInlining)]</c> helpers — debug-build JIT keeps untracked
    ///     temps of a frame's references alive until the method returns, so an NDArray created and
    ///     dropped in the test method itself is NOT collectable until the test ends.</para>
    /// </summary>
    public abstract class InteropTestBase
    {
        protected PyModule Scope;
        private int _baseExports, _baseImports;

        /// <summary>
        ///     MSTest's per-test context, injected before every test. Declared once here (derived classes
        ///     inherit it — redeclaring it would hide this one) because <see cref="InteropInit"/> reads
        ///     the running test's name from it to enforce the <see cref="PythonEcosystemAttribute"/>
        ///     drift guard; tests also use it for <c>TestContext.WriteLine</c> diagnostics.
        /// </summary>
        public TestContext TestContext { get; set; }

        /// <summary>
        ///     Whether the test that is running right now carries <see cref="PythonEcosystemAttribute"/>
        ///     on its method or class. Static because the package gates are static helpers; safe because
        ///     the assembly runs its tests sequentially (<c>[assembly: DoNotParallelize]</c>).
        /// </summary>
        private static bool _currentTestIsEcosystem;

        /// <summary>The running test's display name, for the drift guard's message.</summary>
        private static string _currentTestName;

        [TestInitialize]
        public void InteropInit()
        {
            // Recorded before anything can throw, so a package gate called from a derived class's own
            // [TestInitialize] (which MSTest runs after this one) sees the right test.
            RecordCurrentTest();

            PythonSession.EnsureOrInconclusive();

            Settle();
            _baseExports = NDArrayPythonInterop.LiveExports;
            _baseImports = NDArrayPythonInterop.LiveImports;

            using (Py.GIL())
            {
                Scope = Py.CreateScope();
                Scope.Exec("import numpy as np\nimport gc, array");
            }
        }

        [TestCleanup]
        public void InteropCleanup()
        {
            if (Scope is null)
                return;   // engine unavailable — test was Inconclusive

            using (Py.GIL())
                Scope.Dispose();
            Scope = null;

            bool settled = WaitFor(() => NDArrayPythonInterop.LiveExports <= _baseExports &&
                                         NDArrayPythonInterop.LiveImports <= _baseImports, 12_000);
            Assert.IsTrue(settled,
                $"interop leaked conversions: LiveExports {_baseExports} -> {NDArrayPythonInterop.LiveExports}, " +
                $"LiveImports {_baseImports} -> {NDArrayPythonInterop.LiveImports}");
        }

        // ---- python helpers (each opens/closes its own GIL scope) --------------------------------

        protected IDisposable Gil() => Py.GIL();

        /// <summary>
        ///     Gate for third-party-library claims (docs spec §8.3): attempts the import under the
        ///     GIL and reports Inconclusive when the package is absent — real proof wherever the
        ///     package exists, silent where it doesn't — unless
        ///     <c>NUMSHARP_PYTHONNET_REQUIRE_PACKAGES</c> makes an absent package a failure (CI's run
        ///     against the <c>ecosystem</c> environment, where every such package is installed on purpose).
        /// </summary>
        /// <param name="module">The importable module name (<c>scipy</c>, <c>PIL</c>, <c>cv2</c>, …).</param>
        /// <exception cref="AssertFailedException">
        ///     The running test lacks <see cref="PythonEcosystemAttribute"/> (see
        ///     <see cref="RequireEcosystemTag"/>), or the package is absent under
        ///     <c>NUMSHARP_PYTHONNET_REQUIRE_PACKAGES</c>.
        /// </exception>
        /// <exception cref="AssertInconclusiveException">The package is absent and absence is allowed.</exception>
        protected static void SkipUnless(string module)
        {
            RequireEcosystemTag(module);

            bool available;
            using (Py.GIL())
            {
                try
                {
                    using var m = Py.Import(module);
                    available = true;
                }
                catch (PythonException)
                {
                    available = false;
                }
            }

            if (!available)
                ReportMissingPackage($"python package '{module}' is not installed");
        }

        /// <summary>
        ///     The drift guard: fails the running test unless it carries <see cref="PythonEcosystemAttribute"/>.
        ///     Called by every package gate BEFORE it looks for the package, so an untagged test fails on
        ///     every machine — including a developer's with the package installed — rather than being
        ///     routed to the numpy-only environment, skipping there, and never running anywhere.
        /// </summary>
        /// <param name="package">The package the test asked for; named in the failure message.</param>
        /// <exception cref="AssertFailedException">The running test is not tagged.</exception>
        internal static void RequireEcosystemTag(string package)
        {
            if (!_currentTestIsEcosystem)
                Assert.Fail(
                    $"{_currentTestName ?? "this test"} needs the third-party python package '{package}' but carries no " +
                    "[PythonEcosystem] attribute (on the method or its class). " +
                    "CI runs untagged tests in the numpy-only 'parity' environment, where it would skip forever; " +
                    "tag it so it runs against the 'ecosystem' environment (python-envs/make_env.py).");
        }

        /// <summary>
        ///     Reports an absent (or wrong-version) optional package: Inconclusive by default, a failure
        ///     under <c>NUMSHARP_PYTHONNET_REQUIRE_PACKAGES</c>, where the environment was built to hold it.
        /// </summary>
        /// <param name="message">What is missing, e.g. <c>python package 'torch' is not installed</c>.</param>
        /// <exception cref="AssertFailedException">Absence is not allowed in this run.</exception>
        /// <exception cref="AssertInconclusiveException">Absence is allowed; the test is skipped.</exception>
        /// <remarks>Never returns — both outcomes throw — so callers may rely on it ending the test.</remarks>
        [System.Diagnostics.CodeAnalysis.DoesNotReturn]
        internal static void ReportMissingPackage(string message)
        {
            if (NumSharp.EnvVars.PythonnetRequirePackages)
                Assert.Fail($"{message} — NUMSHARP_PYTHONNET_REQUIRE_PACKAGES is set, so this environment " +
                            $"(interpreter: {PythonSession.Interpreter ?? "unknown"}) must provide it.");
            Assert.Inconclusive(message);
        }

        /// <summary>
        ///     Records the test about to run and whether it (or its class) carries
        ///     <see cref="PythonEcosystemAttribute"/>, for the static package gates.
        /// </summary>
        /// <remarks>
        ///     The method is found by NAME on the runtime type: MSTest's <see cref="TestContext.TestName"/>
        ///     is the method name (data rows share one method), and test methods are not overloaded. When
        ///     the context or the method cannot be resolved the class-level attribute still decides.
        /// </remarks>
        private void RecordCurrentTest()
        {
            Type type = GetType();
            string name = TestContext?.TestName;
            MethodInfo method = name is null
                ? null
                : Array.Find(type.GetMethods(BindingFlags.Public | BindingFlags.Instance), m => m.Name == name);

            _currentTestName = name is null ? type.Name : $"{type.Name}.{name}";
            _currentTestIsEcosystem = type.IsDefined(typeof(PythonEcosystemAttribute), inherit: true) ||
                                      (method?.IsDefined(typeof(PythonEcosystemAttribute), inherit: true) ?? false);
        }

        /// <summary>
        ///     Report Inconclusive on arm64 for a byte-exact cell that hits a genuine
        ///     cross-ARCHITECTURE last-bit difference no numpy pin or OpenBLAS version can remove:
        ///     NumSharp's managed <see cref="System.Numerics.Vector{T}"/> sliding-dot reduces at
        ///     NEON's 128-bit width where x64 uses AVX2's 256-bit, and Apple-silicon LAPACK gelsd
        ///     rounds the lstsq residual excess-rows 1 ULP off the x64 build. The byte-exact interop
        ///     gate is implicitly pinned to the x64 reference architecture — the same "Inconclusive
        ///     off the pinned host" model the offline matmul_parity / linalg_parity corpus tiers use.
        ///     x64 (Windows + Linux) stays STRICT byte-exact.
        /// </summary>
        protected static void SkipByteExactOnArm64(string cell)
        {
            if (System.Runtime.InteropServices.RuntimeInformation.OSArchitecture ==
                System.Runtime.InteropServices.Architecture.Arm64)
                Assert.Inconclusive(
                    $"{cell}: 1-ULP cross-architecture difference on arm64 (managed NEON reduction " +
                    "width / Apple-silicon LAPACK residual rounding); the byte-exact interop gate is " +
                    "pinned to the x64 reference architecture.");
        }

        /// <summary>
        ///     True when this process's NumPy rounds each product of a compiled <c>a*b + c*d</c>
        ///     expression separately, as NumSharp's managed kernels do (RyuJIT never contracts). That holds
        ///     for every x86/x64 wheel and does NOT hold on arm64, where the wheel's compiler fuses such an
        ///     expression into one multiply-add.
        /// </summary>
        /// <remarks>
        ///     <para>The arm64 wheels target a baseline ISA that includes fused multiply-add and are built
        ///     with the compiler's default floating-point contraction (clang's <c>-ffp-contract=on</c> on
        ///     macOS): each <c>x*y + z</c> or <c>x*y + u*v</c> written as ONE C expression becomes
        ///     <c>fmuladd(x, y, …)</c>, the left product fused and any right product rounded first. The
        ///     x86-64 wheels target an X86_V2 baseline with no FMA, so the same source cannot contract
        ///     there. The effect is the one <see cref="NumPyLegacyGaussianIsLiteral"/> documents for the
        ///     legacy Gaussian sampler, stated here for any compiled NumPy arithmetic a test compares.</para>
        ///     <para>Keyed off the PROCESS architecture because the in-process CPython loads the NumPy
        ///     binary built for it (an x64 process under emulation loads the x64 wheel). Arm (32-bit) and
        ///     every other non-x86 architecture fall on the fusing side: nothing guarantees their wheels
        ///     round each product, and a byte-exact claim needs that guarantee.</para>
        /// </remarks>
        protected static bool NumPyRoundsEachProduct =>
            System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture is
                System.Runtime.InteropServices.Architecture.X64 or System.Runtime.InteropServices.Architecture.X86;

        /// <summary>
        ///     The multiply-adds in NumPy's pocketfft (<c>numpy/fft/pocketfft/pocketfft_hdronly.h</c>) a
        ///     fusing wheel contracts — named in the Inconclusive message of every cell that compares a
        ///     result derived from <c>np.fft</c>.
        /// </summary>
        /// <remarks>
        ///     <para>Proven byte for byte against macos-latest (run 35890253503). A C# replica of
        ///     <c>rfftp</c> (factorization, <c>comp_twiddle</c>, <c>sincos_2pibyn</c>, <c>radf2/4/5</c>,
        ///     NumPy's r2c packing) was run in two modes over <c>SpectrumLiveParityTests</c>' input, which
        ///     is the same bytes on every host. The LITERAL mode is NumSharp's port. The FUSED mode fuses
        ///     exactly the expressions <c>-ffp-contract=on</c> fuses: the twiddle products in
        ///     <c>sincos_2pibyn::operator[]</c>, <c>MULPM</c> in every <c>radf</c> codelet, and the
        ///     <c>x + c*y (+ d*z)</c> rotations of <c>radf5</c>. Fed the twiddle values macOS's libm
        ///     returns, the literal mode reproduces the SHA-256 of NumSharp's arm64 <c>np.fft.rfft</c> and
        ///     the fused mode reproduces arm64 NumPy's, at n = 1000 (NumSharp <c>F494345710A79842</c>,
        ///     NumPy <c>483E089585F3D64E</c>, 882 of 1002 float64 lanes apart) and at n = 1024
        ///     (<c>319422554ED4EAB6</c>, <c>2FA2BA5D8FE0417A</c>, 748 of 1026). A one-ULP perturbation
        ///     search over the twiddle table recovered macOS's libm values. They are correctly rounded
        ///     except <c>sin(72·π/4096)</c> and <c>sin(216·π/4096)</c>, which are one ULP high. That
        ///     perturbation pair was the ONLY one among 140 candidates to reproduce NumSharp's hash, and it
        ///     then produced NumPy's hash with no further fitting.</para>
        ///     <para>The complex codelets behind Bluestein (prime lengths) are written the same way
        ///     (<c>special_mul</c>, <c>cmplx::operator*</c>); n = 1021 differs on 999 of 1022 lanes on
        ///     macos-latest, but no replica covers that path.</para>
        /// </remarks>
        protected const string PocketFftFusedArithmetic =
            "pocketfft's twiddle products (sincos_2pibyn::operator[]) and butterfly multiply-adds " +
            "(rfftp's MULPM and radf5 rotations, the complex codelets behind Bluestein)";

        /// <summary>
        ///     Runs a byte-exact assertion over a result that NumPy computes in compiled floating-point
        ///     arithmetic its arm64 wheels fuse: strict where <see cref="NumPyRoundsEachProduct"/> holds,
        ///     and on any other host a failed assertion becomes Inconclusive CARRYING the failure's own
        ///     message, so the CI log still records how far the two stacks differed.
        /// </summary>
        /// <remarks>
        ///     <para>Why not <see cref="SkipByteExactOnArm64"/>: that one skips before anything is
        ///     computed. Here the comparison always runs, so a fusing host whose result happens to match
        ///     still PASSES (a real byte-exact check), and one that does not match still reports the
        ///     measured difference — the evidence that the divergence is NumPy's contraction and not a
        ///     NumSharp regression hiding behind a skip.</para>
        ///     <para>Only <see cref="AssertFailedException"/> is converted (<c>Assert.Fail</c> and
        ///     AwesomeAssertions both raise it under MSTest); any other exception — a crash, a Python
        ///     error — propagates unchanged on every host, and on x86/x64 the failure itself propagates
        ///     untouched, stack trace included (the conversion is an exception filter).</para>
        ///     <para>Wrap ONLY the cells whose NumPy side goes through the named arithmetic; assert the
        ///     rest of a test (values NumPy does not compute with fusable expressions, or computes through
        ///     the same bundled OpenBLAS NumSharp calls) strictly on every host.</para>
        /// </remarks>
        /// <param name="cell">The compared cell, named first in the Inconclusive message.</param>
        /// <param name="fusedArithmetic">Which of NumPy's compiled expressions fuse on such a host
        ///     (for example <see cref="PocketFftFusedArithmetic"/>), so the message states the cause.</param>
        /// <param name="assertExact">The byte-exact assertion. It runs on every host.</param>
        /// <exception cref="AssertFailedException">The assertion failed on a host whose NumPy rounds each
        ///     product — a genuine parity failure.</exception>
        /// <exception cref="AssertInconclusiveException">The assertion failed on a host whose NumPy
        ///     fuses the named arithmetic.</exception>
        protected static void AssertExactUnlessNumPyFuses(string cell, string fusedArithmetic, Action assertExact)
        {
            try
            {
                assertExact();
            }
            catch (AssertFailedException failure) when (!NumPyRoundsEachProduct)
            {
                Assert.Inconclusive(
                    $"{cell} is not byte-identical on {System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture} " +
                    $"({failure.Message.Trim()}). This host's NumPy wheel targets a baseline ISA with fused " +
                    $"multiply-add and is compiled with the default floating-point contraction, so {fusedArithmetic} " +
                    "round once where NumSharp's port, and every x86-64 NumPy, round each product. The byte-exact " +
                    "gate for this cell is pinned to x64.");
            }
        }

        /// <summary>
        ///     True when this process's NumPy evaluates its legacy Gaussian sampler LITERALLY, so a seeded
        ///     <c>RandomState.randn</c> draws the same stream NumSharp does. That holds on every x86/x64
        ///     host and does NOT hold on arm64, where NumPy's own wheel fuses one line of the sampler.
        /// </summary>
        /// <remarks>
        ///     <para>NumPy's <c>legacy_gauss</c> (<c>numpy/random/src/legacy/legacy-distributions.c</c>)
        ///     computes <c>r2 = x1 * x1 + x2 * x2</c>. The arm64 wheels are built for an ASIMD baseline
        ///     (fused multiply-add is in the base ISA) with the compiler's default floating-point
        ///     contraction, so that line becomes one <c>fma(x1, x1, x2 * x2)</c>. The x86-64 wheels target
        ///     an X86_V2 baseline with no FMA and cannot contract it, and RyuJIT never contracts, so NumSharp
        ///     and x64 NumPy round both products separately. <c>log(r2)</c> magnifies that 1-ULP change near
        ///     the unit circle: about 14 % of draws differ, by 1 to ~50 000 ULP.</para>
        ///     <para>Evidence: a C# replica that fuses exactly that way reproduces the first mismatching
        ///     element macos-latest reported for every seeded test (seed 7 -> element 24, seed 23 -> 10,
        ///     seed 3 -> 44); the other operand order does not. The accept/reject decisions did not change
        ///     for those seeds, so the MT19937 position (and every later uniform draw) still agreed.</para>
        ///     <para>NumSharp deliberately keeps the literal stream on every architecture: it is what every
        ///     x86-64 NumPy returns, it keeps seeded NumSharp output portable, and it is what the
        ///     win-amd64-authored random corpus and unit tests pin. Keyed off the PROCESS architecture
        ///     because the in-process CPython loads the NumPy binary built for it (an x64 process under
        ///     emulation loads the x64 wheel).</para>
        /// </remarks>
        protected static bool NumPyLegacyGaussianIsLiteral =>
            System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture is
                System.Runtime.InteropServices.Architecture.X64 or System.Runtime.InteropServices.Architecture.X86;

        /// <summary>
        ///     Defines <c>legacy_randn(rs, *shape)</c> in this test's Python scope: the seeded Gaussian
        ///     draws NumSharp must reproduce, for tests that compare NumSharp's seeded stream with NumPy's.
        /// </summary>
        /// <remarks>
        ///     <para>Where <see cref="NumPyLegacyGaussianIsLiteral"/> holds, <c>legacy_randn</c> IS
        ///     <c>rs.randn</c>, so those hosts keep comparing against NumPy's own output.</para>
        ///     <para>Elsewhere it re-evaluates <c>legacy_gauss</c> exactly as its C source reads, over rs's
        ///     OWN live MT19937 doubles (<c>random_sample</c> is the same <c>next_double</c> the sampler
        ///     consumes). It rounds <c>x1*x1</c>, <c>x2*x2</c> and their sum separately as NumPy ufuncs and
        ///     takes <c>math.log</c> from the platform libm, the one .NET's <c>Math.Log</c> also calls. It
        ///     also honours and leaves rs's cached second Gaussian and advances rs by exactly the doubles
        ///     the literal evaluation consumed, so later draws on rs line up with NumSharp's generator.</para>
        ///     <para>Verified on win-amd64 NumPy 2.4.2, whose own randn IS literal, with the reference path
        ///     forced. Values, the full state tuple and the continuation were byte-identical to
        ///     <c>rs.randn</c> for 126 draws in 63 seeded sequences (9 seeds), including odd counts,
        ///     pre-cached Gaussians, empty draws and 1.28M-element draws. All 32 interop tests that use
        ///     it also passed with <see cref="NumPyLegacyGaussianIsLiteral"/> forced to false on x64.</para>
        ///     <para>It never patches NumPy (the pinned-source loader's rule, see
        ///     <c>KarpathyOriginalSource</c>). A pinned original that calls the global
        ///     <c>np.random.randn</c> therefore still draws NumPy's own stream, so on a fusing host its
        ///     output is compared against a <c>legacy_randn</c> evaluation of the same seed instead. Use
        ///     this only for a CLAIM about NumSharp's seeded stream. A test that feeds NumPy-drawn inputs
        ///     into NumSharp needs no change.</para>
        /// </remarks>
        protected void DefineLegacyRandn()
            => PyExec(LegacyRandnPython.Replace("__NUMPY_LEGACY_GAUSS_IS_LITERAL__", NumPyLegacyGaussianIsLiteral ? "True" : "False"));

        /// <summary>
        ///     The Python behind <see cref="DefineLegacyRandn"/>. The placeholder is replaced by
        ///     <see cref="NumPyLegacyGaussianIsLiteral"/>. Uses only public NumPy API.
        /// </summary>
        private const string LegacyRandnPython = """
            import math as _legacy_math

            _NUMPY_LEGACY_GAUSS_IS_LITERAL = __NUMPY_LEGACY_GAUSS_IS_LITERAL__

            def legacy_randn(rs, *shape):
                if _NUMPY_LEGACY_GAUSS_IS_LITERAL:
                    return rs.randn(*shape)
                n = 1
                for d in shape:
                    n *= int(d)
                if n == 0:
                    return np.empty(shape)            # randn consumes nothing for an empty draw
                name, key, pos, has_gauss, cached = rs.get_state()
                out = np.empty(n)
                filled = 0
                if has_gauss:                         # legacy_gauss hands out its cached value first
                    out[0] = cached
                    filled = 1
                pairs = (n - filled + 1) // 2
                if pairs == 0:
                    rs.set_state((name, key, pos, 0, 0.0))
                    return out.reshape(shape)
                probe = np.random.RandomState()
                probe.set_state((name, key, pos, 0, 0.0))
                xs, ys, consumed, accepted = [], [], 0, 0
                while accepted < pairs:
                    u = probe.random_sample(2 * (pairs - accepted) + 64)
                    x = 2.0 * u[0::2] - 1.0
                    y = 2.0 * u[1::2] - 1.0
                    r2 = x * x + y * y                # two rounded products, one rounded sum: never fused
                    ok = np.flatnonzero((r2 < 1.0) & (r2 != 0.0))[: pairs - accepted]
                    # Doubles consumed: through the last pair still needed, or the whole chunk.
                    consumed += 2 * (int(ok[-1]) + 1) if accepted + len(ok) == pairs else len(u)
                    xs.append(x[ok])
                    ys.append(y[ok])
                    accepted += len(ok)
                x = np.concatenate(xs)
                y = np.concatenate(ys)
                r2 = x * x + y * y
                f = np.sqrt(-2.0 * np.array([_legacy_math.log(v) for v in r2.tolist()]) / r2)
                g = np.empty(2 * pairs)
                g[0::2] = f * y                       # legacy_gauss returns f*x2 first ...
                g[1::2] = f * x                       # ... and caches f*x1 for the next call
                out[filled:] = g[: n - filled]
                advance = np.random.RandomState()
                advance.set_state((name, key, pos, 0, 0.0))
                advance.random_sample(consumed)
                s = advance.get_state()
                left = (n - filled) % 2 == 1          # an odd tail leaves the pair's second value cached
                rs.set_state((s[0], s[1], s[2], 1 if left else 0, float(g[n - filled]) if left else 0.0))
                return out.reshape(shape)
            """;

        protected void PyExec(string code)
        {
            using (Py.GIL()) Scope.Exec(code);
        }

        protected long PyLong(string expr)
        {
            using (Py.GIL()) { using var r = Scope.Eval(expr); return r.As<long>(); }
        }

        protected double PyFloat(string expr)
        {
            using (Py.GIL()) { using var r = Scope.Eval(expr); return r.As<double>(); }
        }

        protected string PyStr(string expr)
        {
            using (Py.GIL()) { using var r = Scope.Eval($"str({expr})"); return r.As<string>(); }
        }

        protected bool PyBool(string expr)
        {
            using (Py.GIL()) { using var r = Scope.Eval($"bool({expr})"); return r.As<bool>(); }
        }

        /// <summary>Zero-copy export of <paramref name="nd"/> bound to a scope name.</summary>
        protected void ExportTo(string name, NDArray nd)
        {
            using (Py.GIL()) { using var p = NDArrayPythonInterop.ToNumpy(nd); Scope.Set(name, p); }
        }

        /// <summary>Independent-copy export bound to a scope name.</summary>
        protected void ExportCopyTo(string name, NDArray nd)
        {
            using (Py.GIL()) { using var p = NDArrayPythonInterop.ToNumpyCopy(nd); Scope.Set(name, p); }
        }

        /// <summary>Raw-bytes memoryview export bound to a scope name.</summary>
        protected void ExportMemoryViewTo(string name, NDArray nd)
        {
            using (Py.GIL()) { using var p = NDArrayPythonInterop.ToMemoryView(nd); Scope.Set(name, p); }
        }

        /// <summary>Copy-import the result of a python expression.</summary>
        protected NDArray ImportOf(string expr)
        {
            using (Py.GIL()) { using var p = Scope.Eval(expr); return NDArrayPythonInterop.ToNDArray(p); }
        }

        /// <summary>Zero-copy view-import the result of a python expression.</summary>
        protected NDArray ViewOf(string expr, bool allowReadonly = false)
        {
            using (Py.GIL()) { using var p = Scope.Eval(expr); return NDArrayPythonInterop.ToNDArrayView(p, allowReadonly); }
        }

        // ---- direct-memory readers/writers (prove aliasing at the pointer level) -----------------

        protected static unsafe void WriteAt<T>(NDArray nd, T value, params long[] coords) where T : unmanaged
        {
            var sh = nd.Shape;
            long off = sh.Offset;
            for (int i = 0; i < coords.Length; i++) off += coords[i] * sh.Strides[i];
            *((T*)nd.Storage.Address + off) = value;
        }

        protected static unsafe T ReadAt<T>(NDArray nd, params long[] coords) where T : unmanaged
        {
            var sh = nd.Shape;
            long off = sh.Offset;
            for (int i = 0; i < coords.Length; i++) off += coords[i] * sh.Strides[i];
            return *((T*)nd.Storage.Address + off);
        }

        // ---- GC / drain pumping -------------------------------------------------------------------

        /// <summary>
        ///     One full release turn: CLR GC + finalizers (NDArray finalizers release ARC refs and
        ///     enqueue lease disposals), pythonnet's deferred-decref flush, a Python GC pass, and one
        ///     trivial conversion to run the interop's inline drain.
        /// </summary>
        protected static void Pump()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            using (Py.GIL())
            {
                Finalizer.Instance.Collect();
                PythonEngine.RunSimpleString("import gc; gc.collect()");
                using var t = NDArrayPythonInterop.ToNumpyCopy(np.arange(1));
            }
        }

        protected static bool WaitFor(Func<bool> condition, int timeoutMs = 10_000)
        {
            var sw = Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                if (condition()) return true;
                Pump();
                System.Threading.Thread.Sleep(20);
            }

            return condition();
        }

        /// <summary>Pump until the live counters stop changing (start-of-test quiescence).</summary>
        protected static void Settle()
        {
            int lastE = -1, lastI = -1;
            for (int i = 0; i < 40; i++)
            {
                int e = NDArrayPythonInterop.LiveExports, im = NDArrayPythonInterop.LiveImports;
                if (e == lastE && im == lastI && (e + im == 0 || i > 2))
                    return;
                lastE = e; lastI = im;
                Pump();
            }
        }
    }
}
