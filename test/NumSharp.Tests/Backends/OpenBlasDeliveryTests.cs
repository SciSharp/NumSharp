using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     The delivery/discovery model of <c>docs/OPENBLAS_DELIVERY_DESIGN.md</c>, runtime side:
    ///     the <c>BundleAutoinstall</c> rename (+ deprecated env alias), the source marker
    ///     (<c>openblas.source.json</c>) flipping a staged folder between "required version
    ///     override" and "bundle", the hard-required no-fall-through contract, and the removal of
    ///     the <c>numpy.libs</c> discovery tier. The BITS stay gated by the host-pinned
    ///     <c>matmul_parity</c> corpus; the packaging contract by <c>MatmulParityBackendTests</c>.
    /// </summary>
    [TestClass]
    public class OpenBlasDeliveryTests
    {
        private const string MarkerFileName = "openblas.source.json";

        private static string MarkerPath => Path.Combine(AppContext.BaseDirectory, MarkerFileName);

        private string _previousOpenBlasPath;
        private string _previousParityBlas;

        // OPENBLAS_HOME/OPENBLAS_ROOT (+ the mixed-case spellings) are now promoted ABOVE the bundle,
        // so a machine-level value would bind before the marker/bundle tiers these tests exercise —
        // park them too.
        private static readonly string[] ExplicitRootEnvVars =
            { "OPENBLAS_HOME", "OPENBLAS_ROOT", "OpenBLAS_HOME", "OpenBLAS_ROOT" };
        private readonly System.Collections.Generic.Dictionary<string, string> _previousRoots = new();

        [TestInitialize]
        public void Init()
        {
            // A machine-level NUMSHARP_OPENBLAS_SEARCH_PATH is tier 2a and would bind BEFORE the marker
            // tiers these tests exercise, and NUMSHARP_OPENBLAS_LIBRARY makes discovery strict and
            // bypasses them entirely — park both for the duration. The explicit OPENBLAS_HOME/
            // OPENBLAS_ROOT roots are now promoted above the bundle, so park those too.
            _previousOpenBlasPath = Environment.GetEnvironmentVariable("NUMSHARP_OPENBLAS_SEARCH_PATH");
            _previousParityBlas = Environment.GetEnvironmentVariable("NUMSHARP_OPENBLAS_LIBRARY");
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_SEARCH_PATH", null);
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_LIBRARY", null);
            foreach (var name in ExplicitRootEnvVars)
            {
                _previousRoots[name] = Environment.GetEnvironmentVariable(name);
                Environment.SetEnvironmentVariable(name, null);
            }
            File.Delete(MarkerPath);
        }

        [TestCleanup]
        public void Cleanup()
        {
            File.Delete(MarkerPath);
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_SEARCH_PATH", _previousOpenBlasPath);
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_LIBRARY", _previousParityBlas);
            foreach (var kv in _previousRoots)
                Environment.SetEnvironmentVariable(kv.Key, kv.Value);
            OpenBlasEngine.Disable();
        }

        #region the BundleAutoinstall rename

        [TestMethod]
        public void BundleAutoinstall_IsTheModuleInitializer_AndAutoInstallIsGone()
        {
            var t = typeof(OpenBlasEngine);
            var renamed = t.GetMethod("BundleAutoinstall", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.IsNotNull(renamed, "OpenBlasEngine.BundleAutoinstall must exist (delivery design §4 rename)");
            Assert.IsNotNull(renamed.GetCustomAttribute<ModuleInitializerAttribute>(),
                "BundleAutoinstall must still be the [ModuleInitializer] — referencing the package is the opt-in");
            Assert.IsNull(t.GetMethod("AutoInstall", BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Public),
                "the old AutoInstall name must be gone, not kept alongside");
        }

        [TestMethod]
        public void BundleAutoinstall_HonorsTheOptOut()
        {
            // The guard sets it to "0" for the whole run; exercise the spelling in isolation.
            var prev = Environment.GetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL");
            try
            {
                OpenBlasEngine.Disable();
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", "0");
                OpenBlasEngine.BundleAutoinstall();
                Assert.IsFalse(OpenBlasEngine.Enabled, "NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0 must suppress the install");
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", prev);
            }
        }

        [TestMethod]
        public void RetiredAutoinstallAlias_NoLongerSuppresses()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            // Two pre-rename spellings are retired (each rename shipped unreleased): the original
            // NUMSHARP_BLAS_AUTOINSTALL and the interim NUMSHARP_BLAS_BUNDLE_AUTOINSTALL. Setting
            // either must be a no-op: autoinstall proceeds regardless — only the current
            // NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0 suppresses it. This pins the REMOVAL — a
            // reintroduced alias would turn this test red.
            var prevNew = Environment.GetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL");
            var prevInterim = Environment.GetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL");
            var prevOld = Environment.GetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL");
            try
            {
                OpenBlasEngine.Disable();
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", null);
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL", "0");
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL", "0");

                OpenBlasEngine.BundleAutoinstall();
                Assert.IsTrue(OpenBlasEngine.Enabled,
                    "the retired NUMSHARP_BLAS_AUTOINSTALL and NUMSHARP_BLAS_BUNDLE_AUTOINSTALL " +
                    "spellings must be ignored — only NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0 " +
                    "suppresses the install");
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", prevNew);
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL", prevInterim);
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL", prevOld);
                OpenBlasEngine.Disable();
            }
        }

        [TestMethod]
        public void BundleAutoinstall_WithNoOptOut_InstallsTheBackend()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var prev = Environment.GetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL");
            try
            {
                OpenBlasEngine.Disable();
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", null);

                OpenBlasEngine.BundleAutoinstall();
                Assert.IsTrue(OpenBlasEngine.Enabled, "with no opt-out and a loadable bundle, autoinstall must wire the backend");
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", prev);
                OpenBlasEngine.Disable();
            }
        }

        #endregion

        #region the source marker — version mode is a hard requirement

        [TestMethod]
        public void VersionMarker_IsHardRequired_NeverFallsThroughToTheBundle()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID — the fall-through " +
                                    "temptation this test refuses does not exist here.");

            var emptyDir = MakeEmptyDir();
            var before = OpenBlasEngine.LibraryPath;
            try
            {
                WriteMarker(new { mode = "version", distribution = "scipy-openblas64", version = "9.9.9.9", path = emptyDir });

                // The bundle IS loadable on this machine — that is exactly what makes this a real
                // test: a version override whose staged binary is gone must throw, not quietly bind
                // the bundle (same layout, plausibly same bytes, NOT the pinned contract).
                var ex = Assert.ThrowsException<OpenBlasRequiredOverrideException>(
                    () => OpenBlasNative.Load(null));
                StringAssert.Contains(ex.Message, "9.9.9.9");
                StringAssert.Contains(ex.Message, "hard requirement");
                Assert.AreEqual(before, OpenBlasEngine.LibraryPath, "a failed required override must change nothing");
            }
            finally
            {
                File.Delete(MarkerPath);
                Directory.Delete(emptyDir);
            }
        }

        [TestMethod]
        public void VersionMarker_BindsTheStagedDirectory_WhichIsNotTheBundle()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var stagedDir = Path.GetDirectoryName(lib);
            try
            {
                WriteMarker(new { mode = "version", distribution = "scipy-openblas64", version = "0.3.31.22.0", path = stagedDir });

                OpenBlasNative.Load(null);
                Assert.IsNotNull(OpenBlasEngine.LibraryPath);
                StringAssert.Contains(Path.GetFullPath(OpenBlasEngine.LibraryPath), Path.GetFullPath(stagedDir));
                Assert.IsFalse(OpenBlasEngine.IsBundledLibrary,
                    "a folder the marker declares a version override is an OVERRIDE, not the bundle — " +
                    "the marker is the only thing that can tell them apart (same layout, same bytes)");
            }
            finally
            {
                File.Delete(MarkerPath);
            }

            // With the marker gone the same folder reads as the bundle again.
            Assert.IsTrue(OpenBlasEngine.IsBundledLibrary, "without a marker the runtimes/<rid>/native folder is the bundle");
        }

        [TestMethod]
        public void VersionMarker_WithMismatchedSha_RefusesTheFile()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var before = OpenBlasEngine.LibraryPath;
            try
            {
                // The staged file EXISTS and is loadable — but it does not hash to the pin, so it is
                // not the pinned binary and must never be silently substituted (design §10.6).
                WriteMarker(new
                {
                    mode = "version",
                    version = "0.3.31.22.0",
                    path = Path.GetDirectoryName(lib),
                    sha256 = new string('0', 64),
                });

                var ex = Assert.ThrowsException<OpenBlasRequiredOverrideException>(
                    () => OpenBlasNative.Load(null));
                StringAssert.Contains(ex.Message, "sha256");
                Assert.AreEqual(before, OpenBlasEngine.LibraryPath);
            }
            finally
            {
                File.Delete(MarkerPath);
            }
        }

        [TestMethod]
        public void BundleAutoinstall_OnARequiredOverrideMiss_IsLoudButDoesNotThrow()
        {
            var emptyDir = MakeEmptyDir();
            var prev = Environment.GetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL");
            try
            {
                WriteMarker(new { mode = "version", version = "9.9.9.9", path = emptyDir });
                OpenBlasEngine.Disable();
                // Autoinstall runs at PROCESS START, before anything is loaded; an earlier test's
                // library would make Enable() reuse it and skip discovery (its documented already-
                // loaded short-circuit). Unload to reproduce the state autoinstall actually sees —
                // safe here: tests run sequentially and nothing is executing in the BLAS.
                OpenBlasNative.Unload();
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", null);

                // Referencing the package must never break the app — the module-load path reports
                // the broken pin to stderr and leaves the backend UNINSTALLED (never substituted).
                OpenBlasEngine.BundleAutoinstall();
                Assert.IsFalse(OpenBlasEngine.Enabled,
                    "a required override that cannot load must leave the backend uninstalled, not substituted");
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", prev);
                File.Delete(MarkerPath);
                Directory.Delete(emptyDir);
            }
        }

        #endregion

        #region the source marker — path mode is non-binding

        [TestMethod]
        public void PathMarker_IsNonBinding_FallsThroughToTheBundle()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var emptyDir = MakeEmptyDir();
            try
            {
                // A path is "look here", not a contract (design §5.2): an empty override dir falls
                // through and the bundle serves — and stays reported as the bundle, because the
                // path marker declares only ITS directory an override location.
                WriteMarker(new { mode = "path", path = emptyDir });

                OpenBlasNative.Load(null);
                Assert.IsTrue(OpenBlasEngine.IsBundledLibrary,
                    "an empty path-mode override must fall through to the bundle, but bound: " + OpenBlasEngine.LibraryPath);
            }
            finally
            {
                File.Delete(MarkerPath);
                Directory.Delete(emptyDir);
            }
        }

        [TestMethod]
        public void PathMarker_BindsItsDirectory_WhenItHoldsALibrary()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            try
            {
                WriteMarker(new { mode = "path", path = Path.GetDirectoryName(lib) });

                OpenBlasNative.Load(null);
                Assert.IsNotNull(OpenBlasEngine.LibraryPath);
                Assert.IsFalse(OpenBlasEngine.IsBundledLibrary,
                    "a library read from a marker-declared path is an override read-in-place, not the bundle");
            }
            finally
            {
                File.Delete(MarkerPath);
            }
        }

        #endregion

        #region adversarial — priority pins and hostile markers

        [TestMethod]
        public void Binding_OutranksABrokenVersionMarker()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var emptyDir = MakeEmptyDir();
            try
            {
                // Tier 1 over tier 2b: an explicit binding must bypass the marker tiers
                // entirely — a broken pin cannot take down a caller who NAMED their binary.
                WriteMarker(new { mode = "version", version = "9.9.9.9", path = emptyDir });
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_LIBRARY", lib);

                OpenBlasNative.Load(null);
                Assert.IsNotNull(OpenBlasEngine.LibraryPath);
                Assert.AreEqual(Path.GetFullPath(lib), Path.GetFullPath(OpenBlasEngine.LibraryPath));
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_LIBRARY", null);
                File.Delete(MarkerPath);
                Directory.Delete(emptyDir);
            }
        }

        [TestMethod]
        public void OverridePathEnv_OutranksTheVersionMarker()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var emptyDir = MakeEmptyDir();
            try
            {
                // Tier 2a over tier 2b (env wins over build metadata, design §3): when the env
                // path holds a loadable BLAS, a version marker with a dead staged dir must not
                // throw — the caller's environment already answered.
                WriteMarker(new { mode = "version", version = "9.9.9.9", path = emptyDir });
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_SEARCH_PATH", Path.GetDirectoryName(lib));

                OpenBlasNative.Load(null);
                Assert.IsNotNull(OpenBlasEngine.LibraryPath);
                StringAssert.Contains(Path.GetFullPath(OpenBlasEngine.LibraryPath),
                    Path.GetFullPath(Path.GetDirectoryName(lib)));
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_SEARCH_PATH", null);
                File.Delete(MarkerPath);
                Directory.Delete(emptyDir);
            }
        }

        [TestMethod]
        public void VersionMarker_RequiredFalse_FallsThroughToTheBundle()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var emptyDir = MakeEmptyDir();
            try
            {
                // The schema's escape hatch: a build may stage a preferred-but-optional binary.
                WriteMarker(new { mode = "version", version = "9.9.9.9", required = false, path = emptyDir });

                OpenBlasNative.Load(null);
                Assert.IsTrue(OpenBlasEngine.IsBundledLibrary,
                    "required:false downgrades a version-marker miss to fall-through, but bound: " + OpenBlasEngine.LibraryPath);
            }
            finally
            {
                File.Delete(MarkerPath);
                Directory.Delete(emptyDir);
            }
        }

        [TestMethod]
        public void Marker_UnknownMode_IsIgnored()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            try
            {
                // Forward compatibility: a mode this runtime does not know is neither an
                // override nor an error — the folder reads as the ordinary bundle.
                WriteMarker(new { mode = "banana", version = "1.2.3" });

                OpenBlasNative.Load(null);
                Assert.IsTrue(OpenBlasEngine.IsBundledLibrary,
                    "an unknown marker mode must not turn the bundle into an override");
            }
            finally
            {
                File.Delete(MarkerPath);
            }
        }

        [TestMethod]
        public void Marker_CorruptJson_IsIgnored_AndDiscoveryProceeds()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            try
            {
                // A half-written or vandalized marker must not brick discovery (TryFind runs on
                // the module-initializer path): one stderr line, then the bundle serves.
                File.WriteAllText(MarkerPath, "{ this is not json");

                OpenBlasNative.Load(null);
                Assert.IsTrue(OpenBlasEngine.IsBundledLibrary,
                    "a corrupt marker must be ignored, not treated as an override or an error");
            }
            finally
            {
                File.Delete(MarkerPath);
            }
        }

        [TestMethod]
        public void VersionMarker_RelativePath_ResolvesAgainstTheMarkersDirectory()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            try
            {
                // The build writes marker and binary together; a relative path must survive the
                // pair being xcopy-deployed anywhere (it resolves against the MARKER, not CWD).
                var relative = Path.GetRelativePath(AppContext.BaseDirectory, Path.GetDirectoryName(lib));
                WriteMarker(new { mode = "version", version = "0.3.31.22.0", path = relative });

                OpenBlasNative.Load(null);
                Assert.IsNotNull(OpenBlasEngine.LibraryPath);
                Assert.IsFalse(OpenBlasEngine.IsBundledLibrary, "a marker-declared override is not the bundle");
            }
            finally
            {
                File.Delete(MarkerPath);
            }
        }

        [TestMethod]
        public void RidDirMarker_WithoutARootMarker_IsFound()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            // The packed-dependent layout (§6.2): OpenBlasDelivery=package bakes the marker NEXT
            // TO the binary inside runtimes/<rid>/native/ — no root marker exists in the consumer.
            var ridMarker = Path.Combine(Path.GetDirectoryName(lib), "openblas.source.json");
            try
            {
                File.WriteAllText(ridMarker, JsonSerializer.Serialize(
                    new { mode = "version", version = "0.3.31.22.0" }));

                OpenBlasNative.Load(null);
                Assert.IsNotNull(OpenBlasEngine.LibraryPath);
                Assert.IsFalse(OpenBlasEngine.IsBundledLibrary,
                    "a rid-dir marker (the packed-dependent layout) must flip the folder to an override");
            }
            finally
            {
                File.Delete(ridMarker);
            }
        }

        [TestMethod]
        public void VersionMarker_ShaComparison_IsCaseInsensitive()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            string shaUpper;
            using (var stream = File.OpenRead(lib))
                shaUpper = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream));

            try
            {
                WriteMarker(new
                {
                    mode = "version",
                    version = "0.3.31.22.0",
                    path = Path.GetDirectoryName(lib),
                    sha256 = shaUpper, // Convert.ToHexString yields UPPERCASE
                });

                OpenBlasNative.Load(null);
                Assert.IsNotNull(OpenBlasEngine.LibraryPath);
                StringAssert.Contains(Path.GetFullPath(OpenBlasEngine.LibraryPath), Path.GetFullPath(Path.GetDirectoryName(lib)));
            }
            finally
            {
                File.Delete(MarkerPath);
            }
        }

        [TestMethod]
        public void PathMarker_MayNameTheLibraryFile_AndStillReadsAsAnOverride()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            try
            {
                // Expand accepts a file as well as a directory, so a marker may name the library
                // itself — and the loaded result must still be recognized as the declared
                // override, not misread as the bundle.
                WriteMarker(new { mode = "path", path = lib });

                OpenBlasNative.Load(null);
                Assert.AreEqual(Path.GetFullPath(lib), Path.GetFullPath(OpenBlasEngine.LibraryPath));
                Assert.IsFalse(OpenBlasEngine.IsBundledLibrary,
                    "a library bound through a file-valued path marker is an override, not the bundle");
            }
            finally
            {
                File.Delete(MarkerPath);
            }
        }

        #endregion

        #region numpy.libs is gone

        [TestMethod]
        public void NumpyLibs_DiscoveryTier_IsDeleted()
        {
            // Design Goal 5: never grab OpenBLAS out of a numpy installation. The tier was a method;
            // its absence is the contract (a conda/system openblas remains machine tooling).
            Assert.IsNull(typeof(OpenBlasNative).GetMethod("PythonLibDirectories",
                    BindingFlags.NonPublic | BindingFlags.Static),
                "PythonLibDirectories (the numpy.libs scan) must not come back");
        }

        #endregion

        #region helpers

        private static void WriteMarker(object marker)
            => File.WriteAllText(MarkerPath, JsonSerializer.Serialize(marker));

        private static string MakeEmptyDir()
        {
            var dir = Path.Combine(Path.GetTempPath(), "numsharp-blas-test-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            return dir;
        }

        /// <summary>The bundled asset for the running RID, as laid down next to the test output.</summary>
        private static string FindBundledLibrary()
        {
            var root = Path.Combine(AppContext.BaseDirectory, "runtimes");
            if (!Directory.Exists(root))
                return null;

            return Directory.GetFiles(root, "*", SearchOption.AllDirectories)
                .FirstOrDefault(f => Path.GetFileName(f).IndexOf("openblas", StringComparison.OrdinalIgnoreCase) >= 0);
        }

        #endregion
    }
}
