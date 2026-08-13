using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.UnitTest.Backends
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

        [TestInitialize]
        public void Init()
        {
            // A machine-level NUMSHARP_OPENBLAS_PATH is tier 2a and would bind BEFORE the marker
            // tiers these tests exercise, and NUMSHARP_PARITY_BLAS makes discovery strict and
            // bypasses them entirely — park both for the duration.
            _previousOpenBlasPath = Environment.GetEnvironmentVariable("NUMSHARP_OPENBLAS_PATH");
            _previousParityBlas = Environment.GetEnvironmentVariable("NUMSHARP_PARITY_BLAS");
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_PATH", null);
            Environment.SetEnvironmentVariable("NUMSHARP_PARITY_BLAS", null);
            File.Delete(MarkerPath);
        }

        [TestCleanup]
        public void Cleanup()
        {
            File.Delete(MarkerPath);
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_PATH", _previousOpenBlasPath);
            Environment.SetEnvironmentVariable("NUMSHARP_PARITY_BLAS", _previousParityBlas);
            Blas.Disable();
        }

        #region the BundleAutoinstall rename

        [TestMethod]
        public void BundleAutoinstall_IsTheModuleInitializer_AndAutoInstallIsGone()
        {
            var t = typeof(Blas);
            var renamed = t.GetMethod("BundleAutoinstall", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.IsNotNull(renamed, "Blas.BundleAutoinstall must exist (delivery design §4 rename)");
            Assert.IsNotNull(renamed.GetCustomAttribute<ModuleInitializerAttribute>(),
                "BundleAutoinstall must still be the [ModuleInitializer] — referencing the package is the opt-in");
            Assert.IsNull(t.GetMethod("AutoInstall", BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Public),
                "the old AutoInstall name must be gone, not kept alongside");
        }

        [TestMethod]
        public void BundleAutoinstall_HonorsTheNewOptOut_AndTheDeprecatedAlias()
        {
            // The guard sets both to "0" for the whole run; exercise each spelling alone.
            var prevNew = Environment.GetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL");
            var prevOld = Environment.GetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL");
            try
            {
                Blas.Disable();

                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL", "0");
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL", null);
                Blas.BundleAutoinstall();
                Assert.IsFalse(Blas.Enabled, "NUMSHARP_BLAS_BUNDLE_AUTOINSTALL=0 must suppress the install");

                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL", null);
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL", "0");
                Blas.BundleAutoinstall();
                Assert.IsFalse(Blas.Enabled,
                    "the deprecated NUMSHARP_BLAS_AUTOINSTALL=0 alias must keep suppressing for one release");
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL", prevNew);
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL", prevOld);
            }
        }

        [TestMethod]
        public void BundleAutoinstall_WithNoOptOut_InstallsTheBackend()
        {
            if (FindBundledLibrary() == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var prevNew = Environment.GetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL");
            var prevOld = Environment.GetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL");
            try
            {
                Blas.Disable();
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL", null);
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL", null);

                Blas.BundleAutoinstall();
                Assert.IsTrue(Blas.Enabled, "with no opt-out and a loadable bundle, autoinstall must wire the backend");
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL", prevNew);
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL", prevOld);
                Blas.Disable();
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
            var before = Blas.LibraryPath;
            try
            {
                WriteMarker(new { mode = "version", distribution = "scipy-openblas64", version = "9.9.9.9", path = emptyDir });

                // The bundle IS loadable on this machine — that is exactly what makes this a real
                // test: a version override whose staged binary is gone must throw, not quietly bind
                // the bundle (same layout, plausibly same bytes, NOT the pinned contract).
                var ex = Assert.ThrowsException<BlasRequiredOverrideException>(
                    () => CBlasNative.Load(null));
                StringAssert.Contains(ex.Message, "9.9.9.9");
                StringAssert.Contains(ex.Message, "hard requirement");
                Assert.AreEqual(before, Blas.LibraryPath, "a failed required override must change nothing");
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

                CBlasNative.Load(null);
                Assert.IsNotNull(Blas.LibraryPath);
                StringAssert.Contains(Path.GetFullPath(Blas.LibraryPath), Path.GetFullPath(stagedDir));
                Assert.IsFalse(Blas.IsBundledLibrary,
                    "a folder the marker declares a version override is an OVERRIDE, not the bundle — " +
                    "the marker is the only thing that can tell them apart (same layout, same bytes)");
            }
            finally
            {
                File.Delete(MarkerPath);
            }

            // With the marker gone the same folder reads as the bundle again.
            Assert.IsTrue(Blas.IsBundledLibrary, "without a marker the runtimes/<rid>/native folder is the bundle");
        }

        [TestMethod]
        public void VersionMarker_WithMismatchedSha_RefusesTheFile()
        {
            var lib = FindBundledLibrary();
            if (lib == null)
                Assert.Inconclusive("no bundled OpenBLAS asset staged for this RID.");

            var before = Blas.LibraryPath;
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

                var ex = Assert.ThrowsException<BlasRequiredOverrideException>(
                    () => CBlasNative.Load(null));
                StringAssert.Contains(ex.Message, "sha256");
                Assert.AreEqual(before, Blas.LibraryPath);
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
            var prevNew = Environment.GetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL");
            var prevOld = Environment.GetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL");
            try
            {
                WriteMarker(new { mode = "version", version = "9.9.9.9", path = emptyDir });
                Blas.Disable();
                // Autoinstall runs at PROCESS START, before anything is loaded; an earlier test's
                // library would make Enable() reuse it and skip discovery (its documented already-
                // loaded short-circuit). Unload to reproduce the state autoinstall actually sees —
                // safe here: tests run sequentially and nothing is executing in the BLAS.
                CBlasNative.Unload();
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL", null);
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL", null);

                // Referencing the package must never break the app — the module-load path reports
                // the broken pin to stderr and leaves the backend UNINSTALLED (never substituted).
                Blas.BundleAutoinstall();
                Assert.IsFalse(Blas.Enabled,
                    "a required override that cannot load must leave the backend uninstalled, not substituted");
            }
            finally
            {
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_BUNDLE_AUTOINSTALL", prevNew);
                Environment.SetEnvironmentVariable("NUMSHARP_BLAS_AUTOINSTALL", prevOld);
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

                CBlasNative.Load(null);
                Assert.IsTrue(Blas.IsBundledLibrary,
                    "an empty path-mode override must fall through to the bundle, but bound: " + Blas.LibraryPath);
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

                CBlasNative.Load(null);
                Assert.IsNotNull(Blas.LibraryPath);
                Assert.IsFalse(Blas.IsBundledLibrary,
                    "a library read from a marker-declared path is an override read-in-place, not the bundle");
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
            Assert.IsNull(typeof(CBlasNative).GetMethod("PythonLibDirectories",
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
