using System;
using System.Diagnostics;
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

        [TestInitialize]
        public void InteropInit()
        {
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
        ///     package exists, silent where it doesn't, so CI stays green on a bare image.
        /// </summary>
        protected static void SkipUnless(string module)
        {
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
                Assert.Inconclusive($"python package '{module}' is not installed");
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
