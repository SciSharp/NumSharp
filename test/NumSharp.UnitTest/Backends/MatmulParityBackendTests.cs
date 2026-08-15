using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.UnitTest.Backends
{
    /// <summary>
    ///     The contract around the optional <c>NumSharp.Interop.OpenBLAS</c> backend — that installing
    ///     it is opt-in, that the engine falls back to its own managed kernels for everything the
    ///     backend does not serve, and that a named library is never silently substituted. The BITS
    ///     are gated separately by the host-pinned <c>matmul_parity</c> corpus tier
    ///     (<c>FuzzCorpusTests.MatmulParity</c>); everything here must hold on any machine, and the
    ///     tests that need a real BLAS say so and go Inconclusive without one.
    /// </summary>
    [TestClass]
    public class MatmulParityBackendTests
    {
        [TestCleanup]
        public void Cleanup() => OpenBlasEngine.Disable();

        /// <summary>A BLAS-less machine must still be able to run these; skip loudly if so.</summary>
        private static void RequireBlas()
        {
            try
            {
                OpenBlasEngine.Enable();
            }
            catch (Exception e)
            {
                Assert.Inconclusive("no CBLAS library on this host: " + e.Message.Split('\n')[0]);
            }
        }

        [TestMethod]
        public void NotInstalled_ByDefault_TheEngineHasNoBlasBackend()
        {
            // NumSharp.Core is 100% managed: nothing may install a native backend implicitly. (The
            // test assembly suppresses the package's module-load auto-install — see
            // BlasEngineAutoInstallGuard — precisely so this stays observable.)
            OpenBlasEngine.Disable();
            Assert.IsFalse(OpenBlasEngine.Enabled);
            Assert.IsNull(BackendFactory.GetEngine().Blas);
        }

        [TestMethod]
        public void Enable_SetsTheBackendOnTheEngine_WithoutReplacingIt()
        {
            RequireBlas();
            Assert.IsTrue(OpenBlasEngine.Enabled);
            var engine = BackendFactory.GetEngine();
            // The engine itself is untouched — still the built-in one, now holding a backend.
            Assert.IsInstanceOfType(engine, typeof(DefaultEngine));
            Assert.IsInstanceOfType(engine.Blas, typeof(IBlasBackend));
            Assert.IsInstanceOfType(engine.Blas, typeof(OpenBlasBackend));
        }

        [TestMethod]
        public void Disable_ClearsTheBackend()
        {
            RequireBlas();
            OpenBlasEngine.Disable();
            Assert.IsFalse(OpenBlasEngine.Enabled);
            Assert.IsNull(BackendFactory.GetEngine().Blas);
            Assert.IsInstanceOfType(BackendFactory.GetEngine(), typeof(DefaultEngine));
        }

        [TestMethod]
        public void TheBackendIsAPlainSettableProperty()
        {
            // The seam is a property on the engine, not a type swap: Core exposes IBlasBackend and
            // nothing more, so any implementation can be dropped in or taken out at will.
            RequireBlas();
            var engine = BackendFactory.GetEngine();
            IBlasBackend backend = engine.Blas;
            Assert.IsNotNull(backend);
            StringAssert.Contains(backend.Info, "symbols");

            engine.Blas = null;
            Assert.IsFalse(OpenBlasEngine.Enabled);
            engine.Blas = backend;
            Assert.IsTrue(OpenBlasEngine.Enabled);
        }

        [TestMethod]
        public void Disable_IsAlwaysSafe_EvenWithNothingInstalled()
        {
            OpenBlasEngine.Disable();
            OpenBlasEngine.Disable();
            Assert.IsFalse(OpenBlasEngine.Enabled);
        }

        [TestMethod]
        public void NamedLibrary_IsBinding_NeverSilentlySubstituted()
        {
            // The whole claim is "the bits of THIS binary", so a named library that cannot be
            // loaded must fail — even on a machine where auto-discovery would happily find NumPy's
            // OpenBLAS and produce confidently wrong bits.
            var ex = Assert.ThrowsException<DllNotFoundException>(
                () => OpenBlasEngine.Enable("K:/definitely/not/a/blas/here.dll"));
            StringAssert.Contains(ex.Message, "no other one is substituted");
            Assert.IsFalse(OpenBlasEngine.Enabled, "a failed load must leave the backend uninstalled");
        }

        [TestMethod]
        public void NotACBlasProvider_Throws()
        {
            // The running test assembly is a perfectly loadable PE that exports no cblas_sgemm.
            var self = typeof(MatmulParityBackendTests).Assembly.Location;
            Assert.ThrowsException<EntryPointNotFoundException>(() => OpenBlasEngine.Enable(self));
            Assert.IsFalse(OpenBlasEngine.Enabled);
        }

        [TestMethod]
        public void TryEnable_ReportsFailureInsteadOfThrowing()
        {
            // The module-load path: referencing the package must never break an app that has no
            // native BLAS anywhere.
            Assert.IsFalse(OpenBlasEngine.TryEnable("K:/definitely/not/a/blas/here.dll"));
            Assert.IsFalse(OpenBlasEngine.Enabled);
        }

        [TestMethod]
        public void Info_DescribesTheLoadedLibrary_OrIsNull()
        {
            RequireBlas();
            var info = OpenBlasEngine.Info;
            Assert.IsNotNull(info);
            StringAssert.Contains(info, "symbols");
            StringAssert.Contains(info, "threads");
        }

        [TestMethod]
        public void Backend_ProducesTheSameSHAPE_AndDTYPE_AsTheManagedKernel()
        {
            // The backend hangs off the engine, and engines are cached singletons — so the SAME
            // arrays switch implementation when it is installed or cleared. (That is the advantage
            // of a property over swapping the engine: an NDArray binds its engine at construction,
            // so a swapped engine would only ever reach arrays created afterwards.)
            var a = np.arange(12).astype(NPTypeCode.Single).reshape(3, 4);
            var b = np.arange(20).astype(NPTypeCode.Single).reshape(4, 5);

            OpenBlasEngine.Disable();
            var native = np.dot(a, b);
            RequireBlas();
            var parity = np.dot(a, b);

            CollectionAssert.AreEqual(native.shape, parity.shape);
            Assert.AreEqual(native.typecode, parity.typecode);
        }

        [TestMethod]
        public void IntegerProducts_FallThroughToTheManagedKernel()
        {
            // Integer addition is associative even when it wraps, so summation order cannot change
            // an integer matrix product — the backend deliberately returns false for these and the
            // engine computes them itself.
            var a = np.arange(6).astype(NPTypeCode.Int32).reshape(2, 3);
            var b = np.arange(6).astype(NPTypeCode.Int32).reshape(3, 2);

            OpenBlasEngine.Disable();
            var native = np.matmul(a, b);
            RequireBlas();
            var parity = np.matmul(a, b);

            Assert.AreEqual(NPTypeCode.Int32, parity.typecode);
            CollectionAssert.AreEqual(native.ToArray<int>(), parity.ToArray<int>());
        }

        [TestMethod]
        public void EveryShapeRoute_RoundTripsThroughTheBackend()
        {
            // Values are not asserted here (that is the corpus tier's job, pinned to a host) — this
            // asserts the port never throws or mis-shapes on any of the routes it dispatches:
            // row*column dot, both gemv directions, column@row, scalar_vec, gemm, syrk, batched,
            // N-D dot, and the zero-sized fallbacks.
            RequireBlas();
            var v = np.arange(5).astype(NPTypeCode.Double);
            var m = np.arange(15).astype(NPTypeCode.Double).reshape(3, 5);
            var n = np.arange(20).astype(NPTypeCode.Double).reshape(5, 4);

            CollectionAssert.AreEqual(new long[0], np.dot(v, v).shape);
            CollectionAssert.AreEqual(new long[] { 3 }, np.dot(m, v).shape);
            CollectionAssert.AreEqual(new long[] { 4 }, np.dot(v, n).shape);
            CollectionAssert.AreEqual(new long[] { 3, 4 }, np.dot(m, n).shape);
            CollectionAssert.AreEqual(new long[] { 3, 4 }, np.matmul(m, n).shape);
            CollectionAssert.AreEqual(new long[] { 3, 3 }, np.dot(m, m.T).shape);          // syrk
            CollectionAssert.AreEqual(new long[] { 5, 5 }, np.dot(m.T, m).shape);          // syrk
            CollectionAssert.AreEqual(new long[] { 5, 4 },
                np.dot(v.reshape(5, 1), np.arange(4).astype(NPTypeCode.Double).reshape(1, 4)).shape);
            CollectionAssert.AreEqual(new long[] { 1, 1 },
                np.dot(v.reshape(1, 5), v.reshape(5, 1)).shape);
            CollectionAssert.AreEqual(new long[] { 2, 3, 4 },
                np.matmul(np.arange(30).astype(NPTypeCode.Double).reshape(2, 3, 5), n).shape);
            CollectionAssert.AreEqual(new long[] { 2, 3, 4 },
                np.dot(np.arange(30).astype(NPTypeCode.Double).reshape(2, 3, 5), n).shape);
            CollectionAssert.AreEqual(new long[] { 3, 4 },
                np.dot(np.zeros(new Shape(3, 0), NPTypeCode.Double),
                       np.zeros(new Shape(0, 4), NPTypeCode.Double)).shape);
        }

        [TestMethod]
        public void StridedAndTransposedViews_AreAccepted_NotSilentlyWrong()
        {
            RequireBlas();
            var baseA = np.arange(60).astype(NPTypeCode.Single).reshape(6, 10);
            var strided = baseA[":, ::2"];            // (6,5), last-axis stride 2 — not blasable
            var reversed = baseA["::-1"];             // negative row stride
            var b = np.arange(25).astype(NPTypeCode.Single).reshape(5, 5);

            CollectionAssert.AreEqual(new long[] { 6, 5 }, np.dot(strided, b).shape);
            CollectionAssert.AreEqual(new long[] { 6, 5 }, np.matmul(strided, b).shape);
            CollectionAssert.AreEqual(new long[] { 6, 6 }, np.dot(reversed, baseA.T).shape);
        }

        [TestMethod]
        public void MixedPrecisionOperands_PromoteLikeNumPy()
        {
            RequireBlas();
            var f = np.arange(6).astype(NPTypeCode.Single).reshape(2, 3);
            var d = np.arange(6).astype(NPTypeCode.Double).reshape(3, 2);
            Assert.AreEqual(NPTypeCode.Double, np.dot(f, d).typecode);
            Assert.AreEqual(NPTypeCode.Double, np.dot(d, f).typecode);
        }

        [TestMethod]
        public void EveryOtherOperation_StaysTheManagedImplementation()
        {
            // The backend answers two entry points; installing it must not change anything else.
            RequireBlas();
            var a = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
            Assert.AreEqual(15.0, (double)np.sum(a));
            Assert.AreEqual(2.0, (double)np.max(a[0]));
            CollectionAssert.AreEqual(new long[] { 3, 2 }, np.transpose(a).shape);
            CollectionAssert.AreEqual(new long[] { 2, 3 }, (a + a).shape);
            Assert.AreEqual(NPTypeCode.Double, np.sqrt(a).typecode);
        }

        [TestMethod]
        public void ThreadPin_IsReportedBack()
        {
            RequireBlas();
            OpenBlasEngine.Enable(threads: 1);
            var info = OpenBlasEngine.Info;
            // OpenBLAS reports the count back; a reference CBLAS has no such entry point (-1).
            Assert.IsTrue(info.Contains("threads 1") || info.Contains("threads -1"), info);
        }

        #region the bundled OpenBLAS runtime asset

        // The package ships the prebuilt OpenBLAS binaries NumPy itself pins (scipy-openblas
        // 0.3.31.22.0) as runtimes/<rid>/native/ assets, so a .NET app gets NumPy's exact matrix
        // products with no Python installed. These tests hold the packaging contract; the BITS are
        // gated by FuzzCorpusTests.MatmulParity, whose host pin now matches the library by CONTENT
        // HASH — which is what makes the bundled copy and NumPy's hash-mangled one the same library.

        /// <summary>The RID asset must reach the build output, or nothing below can load it.</summary>
        [TestMethod]
        public void BundledLibrary_ForThisRid_ReachesTheBuildOutput()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive(
                    "no bundled OpenBLAS asset staged for this RID — run " +
                    "`python src/NumSharp.Interop.OpenBLAS/tools/fetch_openblas.py`. The package builds " +
                    "without them and falls back to discovery, so this is a staging gap, not a defect.");

            Assert.IsTrue(new FileInfo(lib).Length > 1_000_000,
                "a real OpenBLAS build is tens of MB; this looks like a placeholder: " + lib);
        }

        /// <summary>
        ///     Auto-discovery takes the bundled library first — the whole point of bundling.
        /// </summary>
        /// <remarks>
        ///     Determinism is the reason this order is fixed rather than "whatever is found first":
        ///     a library's numeric output must not change because an unrelated `pip install` ran, and
        ///     an app with no python must not get a different backend from one that has it. Matching
        ///     some OTHER numpy stays possible, but only by NAMING its library — which is binding.
        /// </remarks>
        [TestMethod]
        public void AutoDiscovery_PrefersTheBundledLibrary()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            RequireBlas();
            Assert.IsTrue(OpenBlasEngine.IsBundledLibrary,
                "auto-discovery should bind the bundled asset, but loaded: " + OpenBlasEngine.LibraryPath);
            StringAssert.Contains(OpenBlasEngine.LibraryPath, "runtimes");
        }

        /// <summary>A named library still wins over the bundled one, and is never substituted.</summary>
        [TestMethod]
        public void NamedLibrary_OutranksTheBundledOne_AndAMissingOneDoesNotFallBackToIt()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            // Bundling must not weaken the binding guarantee: naming a library that cannot be
            // loaded has to fail, not quietly succeed against the copy sitting in the output folder.
            // That failure mode would be invisible — the products still compute, just with a
            // different binary than the caller pinned.
            Assert.ThrowsExactly<DllNotFoundException>(
                () => OpenBlasEngine.Enable(Path.Combine(AppContext.BaseDirectory, "no-such-blas-library.dll")));
            Assert.IsFalse(OpenBlasEngine.Enabled);
        }

        /// <summary>NUMSHARP_OPENBLAS_USE_BUNDLED=0 drops the bundled entry from discovery.</summary>
        [TestMethod]
        public void BundledLibrary_CanBeOptedOutOf()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            OpenBlasEngine.Disable();
            var previous = Environment.GetEnvironmentVariable("NUMSHARP_OPENBLAS_USE_BUNDLED");
            try
            {
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_USE_BUNDLED", "0");

                // Whether anything else is installed is host-dependent; the assertion is only that
                // the bundled asset is no longer what auto-discovery reaches for.
                bool loaded;
                try
                {
                    OpenBlasEngine.Enable();
                    loaded = true;
                }
                catch (DllNotFoundException)
                {
                    loaded = false;   // nothing else on this machine — the opt-out plainly worked
                }

                if (loaded && OpenBlasEngine.IsBundledLibrary && !CBlasWasAlreadyLoaded())
                    Assert.Fail("NUMSHARP_OPENBLAS_USE_BUNDLED=0 was ignored: " + OpenBlasEngine.LibraryPath);
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_USE_BUNDLED", previous);
            }
        }

        /// <summary>
        ///     A core type this CPU cannot execute is refused, not attempted.
        /// </summary>
        /// <remarks>
        ///     OPENBLAS_CORETYPE overrides OpenBLAS' own CPU detection, so an over-reach does not
        ///     degrade gracefully — it terminates the process with an illegal instruction (measured:
        ///     forcing SkylakeX on a Haswell host). There is no catching that, so it has to be
        ///     refused up front.
        /// </remarks>
        [TestMethod]
        public void CoreTypeAboveThisCpu_IsRefused_RatherThanCrashingTheProcess()
        {
            if (System.Runtime.Intrinsics.X86.Avx512F.IsSupported)
                Assert.Inconclusive("this CPU supports AVX-512, so SkylakeX is not an over-reach here.");
            if (!System.Runtime.Intrinsics.X86.Sse2.IsSupported)
                Assert.Inconclusive("not an x86 host; the core-type table does not apply.");

            OpenBlasEngine.Disable();
            Assert.ThrowsExactly<PlatformNotSupportedException>(() => OpenBlasEngine.Enable(coreType: "SkylakeX"));
        }

        /// <summary>The corpus's pinned kernel is the one the public constant names.</summary>
        [TestMethod]
        public void CoreType_MatchesTheCorpusHostPin()
        {
            var pin = Fuzz.MatmulParityPin.Load();
            if (string.IsNullOrEmpty(pin.Blas_Corename))
                Assert.Inconclusive("the committed host pin records no core name.");

            Assert.AreEqual(pin.Blas_Corename, OpenBlasEngine.CoreType, ignoreCase: true,
                "OpenBlasEngine.CoreType is what callers pin to reproduce the corpus on another " +
                "machine; it must name the kernel the corpus was actually generated with.");
        }

        /// <summary>The bundled binary is the one the checked-in pin manifest describes.</summary>
        /// <remarks>
        ///     This is the packaging half of the parity claim. The manifest is the version pin, and
        ///     an asset that does not hash to it is not the binary NumPy calls — which would make
        ///     every downstream "bit-identical" statement false while everything still ran.
        /// </remarks>
        [TestMethod]
        public void BundledLibrary_HashesToThePinnedManifest()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var manifest = FindManifest();
            if (manifest == null)
                Assert.Inconclusive("openblas-manifest.json is not reachable from the test output.");

            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(manifest));
            var runtimes = doc.RootElement.GetProperty("runtimes");

            string actual;
            using (var stream = File.OpenRead(lib))
                actual = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream))
                                .ToLowerInvariant();

            foreach (var rid in runtimes.EnumerateObject())
            {
                if (!string.Equals(rid.Value.GetProperty("file").GetString(),
                        Path.GetFileName(lib), StringComparison.OrdinalIgnoreCase))
                    continue;

                Assert.AreEqual(rid.Value.GetProperty("sha256").GetString(), actual, ignoreCase: true,
                    $"the staged {rid.Name} asset does not match the pinned manifest.");
                return;
            }

            Assert.Inconclusive("the staged asset is not named in the manifest: " + lib);
        }

        private static bool CBlasWasAlreadyLoaded() => OpenBlasEngine.LibraryPath != null;

        /// <summary>The bundled asset for the running RID, as laid down next to the test output.</summary>
        private static string FindBundledLibrary()
        {
            var root = Path.Combine(AppContext.BaseDirectory, "runtimes");
            if (!Directory.Exists(root))
                return null;

            foreach (var f in Directory.GetFiles(root, "*", SearchOption.AllDirectories))
            {
                var name = Path.GetFileName(f);
                if (name.IndexOf("openblas", StringComparison.OrdinalIgnoreCase) >= 0)
                    return f;
            }

            return null;
        }

        /// <summary>The checked-in pin manifest, walked up from the test output.</summary>
        private static string FindManifest()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null)
            {
                var candidate = Path.Combine(dir.FullName,
                    "src", "NumSharp.Interop.OpenBLAS", "tools", "openblas-manifest.json");
                if (File.Exists(candidate))
                    return candidate;

                dir = dir.Parent;
            }

            return null;
        }

        #endregion

        #region a failed Enable is a no-op

        // NamedLibrary_IsBinding_NeverSilentlySubstituted and NotACBlasProvider_Throws above assert
        // the right invariant, but only ever from a DISABLED start — which is the easy half. The
        // hard half is a failed Enable landing on a WORKING one: that used to unload the good
        // library first and leave the backend installed over nothing, so OpenBlasEngine.Enabled kept saying
        // true while every product quietly reverted to the managed kernel and stayed plausible.

        [TestMethod]
        public void FailedEnable_OverAWorkingLibrary_ChangesNothing()
        {
            RequireBlas();
            var before = OpenBlasEngine.Info;
            var expected = np.dot(Ramp(48, 32), Ramp(32, 48));

            Assert.ThrowsException<DllNotFoundException>(
                () => OpenBlasEngine.Enable("K:/definitely/not/a/blas/here.dll"));

            Assert.IsTrue(OpenBlasEngine.Enabled, "a failed Enable must not uninstall a working backend");
            Assert.AreEqual(before, OpenBlasEngine.Info, "a failed Enable must not unload the working library");
            AssertBitEqual(expected, np.dot(Ramp(48, 32), Ramp(32, 48)));
        }

        [TestMethod]
        public void FailedEnable_OnANonCBlasFile_OverAWorkingLibrary_ChangesNothing()
        {
            RequireBlas();
            var before = OpenBlasEngine.Info;
            var self = typeof(MatmulParityBackendTests).Assembly.Location;

            Assert.ThrowsException<EntryPointNotFoundException>(() => OpenBlasEngine.Enable(self));

            Assert.IsTrue(OpenBlasEngine.Enabled);
            Assert.AreEqual(before, OpenBlasEngine.Info);
        }

        [TestMethod]
        public void Enabled_MeansInstalledAND_Bound_NotMerelyInstalled()
        {
            // The property answers "are my products going through a BLAS?", so an installed backend
            // with nothing to call must report false — it declines every operand, and reporting
            // true there would claim a parity it is not delivering.
            OpenBlasEngine.Disable();
            Assert.IsFalse(OpenBlasEngine.Enabled);
            BackendFactory.GetEngine().Blas = new DeclineEverythingBackend();
            Assert.IsFalse(OpenBlasEngine.Enabled, "a foreign backend is not this package's backend");
            BackendFactory.GetEngine().Blas = null;
        }

        #endregion

        #region the batched entry point

        [TestMethod]
        public void StackedProducts_GoThroughTheBatchedEntryPoint_Once()
        {
            RequireBlas();
            var spy = new CountingBackend(BackendFactory.GetEngine().Blas);
            BackendFactory.GetEngine().Blas = spy;
            try
            {
                np.matmul(Ramp(4, 9, 11), Ramp(4, 11, 6));
                Assert.AreEqual(1, spy.Batched, "the whole stack must be offered in one call");
                Assert.AreEqual(0, spy.TwoD, "no per-element fallback when the stack is served");
            }
            finally
            {
                BackendFactory.GetEngine().Blas = null;
            }
        }

        [TestMethod]
        public void BatchedEntryPoint_IsBitIdenticalToThePerElementRoute()
        {
            // The hoist is a PERFORMANCE change only: the plan is a pure function of the trailing
            // strides and dims, which do not vary across a stack, so both routes reach the same
            // MatmulCore call with the same pointers. Prove it by declining the batched entry point
            // through a decorator and comparing bytes.
            RequireBlas();
            var real = BackendFactory.GetEngine().Blas;
            var perElement = new CountingBackend(real) { DeclineBatched = true };

            var pairs = new[]
            {
                (Ramp(4, 9, 11), Ramp(4, 11, 6)),
                (Ramp(64, 8, 8), Ramp(64, 8, 8)),
                (Ramp(2, 3, 33, 17), Ramp(2, 3, 17, 29)),
                (Ramp(4, 11, 9).transpose(new[] { 0, 2, 1 }), Ramp(4, 11, 6)),
                (Ramp(4, 1, 11), Ramp(4, 11, 1)),
            };

            try
            {
                foreach (var (l, r) in pairs)
                {
                    BackendFactory.GetEngine().Blas = perElement;
                    var viaElements = np.matmul(l, r);
                    BackendFactory.GetEngine().Blas = real;
                    var viaStack = np.matmul(l, r);
                    AssertBitEqual(viaElements, viaStack);
                }
            }
            finally
            {
                BackendFactory.GetEngine().Blas = null;
            }
        }

        [TestMethod]
        public void BatchedEntryPoint_DeclinesZeroSizedExtents_SoBehaviourIsUnchanged()
        {
            // A backend may change WHICH implementation runs and how fast — never WHETHER the call
            // works. NumPy excludes zero dims from its own blas routes (the `any_zero_dim` arm of
            // @TYPE@_matmul forces matmul_inner_noblas), so the backend declines them and the engine
            // answers from its own degenerate-stack shortcut.
            //
            // Asserting only that the two outcomes AGREE is too weak — it held while both sides
            // threw. So pin the NumPy answer itself as well: a zero stack dim gives an empty
            // (0,3,5), and k == 0 gives a NON-empty (2,3,5) of exact zeros (the empty sum).
            RequireBlas();
            var zeroBatch = (Func<NDArray>)(() => np.matmul(
                np.zeros(new Shape(0, 3, 4), NPTypeCode.Single),
                np.zeros(new Shape(0, 4, 5), NPTypeCode.Single)));
            var zeroK = (Func<NDArray>)(() => np.matmul(
                np.zeros(new Shape(2, 3, 0), NPTypeCode.Single),
                np.zeros(new Shape(2, 0, 5), NPTypeCode.Single)));

            foreach (var (call, expected) in new[] { (zeroBatch, "Single(0,3,5)"), (zeroK, "Single(2,3,5)") })
            {
                OpenBlasEngine.Disable();
                var withoutBackend = Outcome(call);
                Assert.AreEqual(expected, withoutBackend,
                    "the managed engine does not give NumPy's answer for a zero-sized stacked matmul");

                OpenBlasEngine.Enable();
                Assert.AreEqual(withoutBackend, Outcome(call),
                    "installing the backend changed the outcome of a zero-sized stacked matmul");

                // k == 0 is the one that produces values rather than an empty view — prove they are
                // exactly zero on BOTH sides, not merely that the shapes line up.
                var withBackend = call();
                for (long i = 0; i < withBackend.size; i++)
                    Assert.AreEqual(0, BitConverter.SingleToInt32Bits((float)withBackend.GetAtIndex(i)),
                        $"element {i} of a zero-sized stacked matmul is not exactly 0.0f");
            }
        }

        [TestMethod]
        public void ABackendThatKnowsNothingOfBatching_StillWorks()
        {
            // TryMatMulBatched arrived as a DEFAULT interface method, so a backend written against
            // the original two-member interface keeps compiling and running untouched — the engine
            // simply falls back to its per-element loop. This is the extensibility claim, executed.
            var legacy = new TwoMemberBackend();
            BackendFactory.GetEngine().Blas = legacy;
            try
            {
                var r = np.matmul(Ramp(3, 4, 5), Ramp(3, 5, 6));
                CollectionAssert.AreEqual(new long[] { 3, 4, 6 }, r.shape);
                Assert.AreEqual(3, legacy.TwoD, "one 2-D call per batch element");
                AssertBitEqual(Ramp(3, 4, 5).astype(NPTypeCode.Single), Ramp(3, 4, 5));
            }
            finally
            {
                BackendFactory.GetEngine().Blas = null;
            }
        }

        #endregion

        #region helpers

        private static NDArray Ramp(params long[] dims)
        {
            long n = 1;
            foreach (var d in dims)
                n *= d;
            return (np.arange(n).astype(NPTypeCode.Single) / np.array(7.0f) - np.array(3.0f)).reshape(dims);
        }

        private static string Outcome(Func<NDArray> call)
        {
            try
            {
                var r = call();
                return $"{r.dtype.Name}({string.Join(",", r.shape)})";
            }
            catch (Exception e)
            {
                return e.GetType().Name;
            }
        }

        private static void AssertBitEqual(NDArray expected, NDArray actual)
        {
            Assert.AreEqual(expected.typecode, actual.typecode);
            CollectionAssert.AreEqual(expected.shape, actual.shape);
            var a = expected.ToArray<float>();
            var b = actual.ToArray<float>();
            for (int i = 0; i < a.Length; i++)
                Assert.AreEqual(BitConverter.SingleToInt32Bits(a[i]), BitConverter.SingleToInt32Bits(b[i]),
                    $"element {i}: {a[i]} vs {b[i]}");
        }

        /// <summary>Forwards to a real backend, counting calls; can decline the batched route.</summary>
        private sealed class CountingBackend : IBlasBackend
        {
            private readonly IBlasBackend _inner;
            public int TwoD, Batched;
            public bool DeclineBatched;
            public CountingBackend(IBlasBackend inner) => _inner = inner;
            public string Info => _inner.Info;
            public bool TryDot(NDArray l, NDArray r, out NDArray result) => _inner.TryDot(l, r, out result);
            public bool TryMatMul2D(NDArray l, NDArray r, NDArray result) { TwoD++; return _inner.TryMatMul2D(l, r, result); }
            public bool TryMatMulBatched(NDArray l, NDArray r, NDArray result)
            {
                Batched++;
                return !DeclineBatched && _inner.TryMatMulBatched(l, r, result);
            }
        }

        /// <summary>A backend written against the ORIGINAL two-member interface — nothing else.</summary>
        private sealed unsafe class TwoMemberBackend : IBlasBackend
        {
            public int TwoD;
            public string Info => "two-member backend";

            public bool TryDot(NDArray left, NDArray right, out NDArray result)
            {
                result = null;
                return false;
            }

            public bool TryMatMul2D(NDArray left, NDArray right, NDArray result)
            {
                if (result.typecode != NPTypeCode.Single) return false;
                if (left.typecode != NPTypeCode.Single || right.typecode != NPTypeCode.Single) return false;
                TwoD++;

                long m = left.shape[0], k = left.shape[1], n = right.shape[1];
                float* pa = (float*)left.GetData().Address + left.Shape.Offset;
                float* pb = (float*)right.GetData().Address + right.Shape.Offset;
                float* pc = (float*)result.GetData().Address + result.Shape.Offset;
                long a0 = left.Shape.Strides[0], a1 = left.Shape.Strides[1];
                long b0 = right.Shape.Strides[0], b1 = right.Shape.Strides[1];
                long c0 = result.Shape.Strides[0], c1 = result.Shape.Strides[1];
                for (long i = 0; i < m; i++)
                    for (long j = 0; j < n; j++)
                    {
                        float sum = 0f;
                        for (long p = 0; p < k; p++)
                            sum += pa[i * a0 + p * a1] * pb[p * b0 + j * b1];
                        pc[i * c0 + j * c1] = sum;
                    }

                return true;
            }
        }

        private sealed class DeclineEverythingBackend : IBlasBackend
        {
            public string Info => "declines everything";
            public bool TryDot(NDArray l, NDArray r, out NDArray result) { result = null; return false; }
            public bool TryMatMul2D(NDArray l, NDArray r, NDArray result) => false;
        }

        #endregion
    }
}
