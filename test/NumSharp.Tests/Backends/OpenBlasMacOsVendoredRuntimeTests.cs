using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Backends
{
    /// <summary>
    ///     The macOS vendored-runtime materializer behind the bundled OpenBLAS dylib
    ///     (<c>MacOsVendoredRuntime</c> + <c>MachOImage</c>), pinned on EVERY platform.
    /// </summary>
    /// <remarks>
    ///     The defect this guards: scipy-openblas' macOS dylib names its Fortran runtime through
    ///     <c>@loader_path/../.dylibs/…</c>, NuGet delivers native assets only from under
    ///     <c>runtimes/&lt;rid&gt;/native/</c>, so the deps are staged NESTED (<c>native/.dylibs/</c>)
    ///     and the loader must put them where dyld looks. The logic is pure path/file work over a
    ///     parsed Mach-O header, so it is exercised here with synthetic Mach-O images on Windows and
    ///     Linux too — a macOS-only gate would have hidden this bug exactly the way the
    ///     ProjectReference build did. The real dylib, when staged, is parsed as well.
    /// </remarks>
    [TestClass]
    public class OpenBlasMacOsVendoredRuntimeTests
    {
        private string _root;
        private string _previousCacheDir;

        [TestInitialize]
        public void Init()
        {
            _root = Path.Combine(Path.GetTempPath(), "numsharp-macho-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_root);
            _previousCacheDir = Environment.GetEnvironmentVariable("NUMSHARP_OPENBLAS_CACHE_DIR");
        }

        [TestCleanup]
        public void Cleanup()
        {
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_CACHE_DIR", _previousCacheDir);
            try
            {
                Directory.Delete(_root, true);
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }

        #region MachOImage — the load-command reader

        [TestMethod]
        public void MachO_ReadsInstallName_Dependencies_AndRunPaths_ThinLittleEndian()
        {
            var file = Path.Combine(_root, "thin.dylib");
            File.WriteAllBytes(file, SyntheticMachO.Dylib(
                installName: "@rpath/libmain.dylib",
                loads: new[] { "@loader_path/../.dylibs/libgfortran.5.dylib", "/usr/lib/libSystem.B.dylib" },
                rpaths: new[] { "@loader_path/" }));

            Assert.IsTrue(MachOImage.TryRead(file, out var image));
            Assert.AreEqual("@rpath/libmain.dylib", image.InstallName);
            CollectionAssert.AreEqual(
                new[] { "@loader_path/../.dylibs/libgfortran.5.dylib", "/usr/lib/libSystem.B.dylib" },
                image.Dependencies.ToArray());
            CollectionAssert.AreEqual(new[] { "@loader_path/" }, image.RunPaths.ToArray());
        }

        [TestMethod]
        public void MachO_ReadsBigEndian32Bit_AndWeakReexportUpward()
        {
            var file = Path.Combine(_root, "be32.dylib");
            File.WriteAllBytes(file, SyntheticMachO.Dylib(
                installName: "/DLC/x/.dylibs/libx.dylib",
                loads: new[] { "@loader_path/liba.dylib" },
                rpaths: Array.Empty<string>(),
                bigEndian: true, is64: false,
                extraLoads: new (uint, string)[]
                {
                    (0x80000018, "@loader_path/libweak.dylib"),  // LC_LOAD_WEAK_DYLIB
                    (0x8000001F, "@loader_path/libre.dylib"),    // LC_REEXPORT_DYLIB
                    (0x80000023, "@loader_path/libup.dylib"),    // LC_LOAD_UPWARD_DYLIB
                }));

            Assert.IsTrue(MachOImage.TryRead(file, out var image));
            Assert.AreEqual("/DLC/x/.dylibs/libx.dylib", image.InstallName);
            CollectionAssert.AreEqual(
                new[] { "@loader_path/liba.dylib", "@loader_path/libweak.dylib", "@loader_path/libre.dylib", "@loader_path/libup.dylib" },
                image.Dependencies.ToArray());
        }

        [TestMethod]
        public void MachO_FatFile_ReportsTheUnionOfSlices()
        {
            var a = SyntheticMachO.Dylib("id", new[] { "@loader_path/libonlyA.dylib", "/usr/lib/libSystem.B.dylib" }, Array.Empty<string>());
            var b = SyntheticMachO.Dylib("id", new[] { "@loader_path/libonlyB.dylib", "/usr/lib/libSystem.B.dylib" }, Array.Empty<string>());
            var file = Path.Combine(_root, "fat.dylib");
            File.WriteAllBytes(file, SyntheticMachO.Fat(a, b));

            Assert.IsTrue(MachOImage.TryRead(file, out var image));
            CollectionAssert.AreEquivalent(
                new[] { "@loader_path/libonlyA.dylib", "/usr/lib/libSystem.B.dylib", "@loader_path/libonlyB.dylib" },
                image.Dependencies.ToArray());
        }

        [TestMethod]
        public void MachO_RejectsNonMachO_AndTruncatedHeaders_WithoutThrowing()
        {
            var elf = Path.Combine(_root, "not.so");
            File.WriteAllBytes(elf, new byte[] { 0x7F, (byte)'E', (byte)'L', (byte)'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
            Assert.IsFalse(MachOImage.TryRead(elf, out _));

            var pe = Path.Combine(_root, "not.dll");
            File.WriteAllBytes(pe, Encoding.ASCII.GetBytes("MZ this is not a mach-o file at all, just bytes"));
            Assert.IsFalse(MachOImage.TryRead(pe, out _));

            // A valid magic whose command table runs past EOF must decline, never read garbage.
            var truncated = Path.Combine(_root, "truncated.dylib");
            var whole = SyntheticMachO.Dylib("id", new[] { "@loader_path/liba.dylib" }, Array.Empty<string>());
            File.WriteAllBytes(truncated, whole.Take(40).ToArray());
            Assert.IsFalse(MachOImage.TryRead(truncated, out _));

            // Hostile ncmds/sizeofcmds must not become a multi-GB allocation.
            var hostile = whole.ToArray();
            BitConverter.GetBytes(0x7FFFFFFFu).CopyTo(hostile, 16); // ncmds
            BitConverter.GetBytes(0x7FFFFFFFu).CopyTo(hostile, 20); // sizeofcmds
            var hostileFile = Path.Combine(_root, "hostile.dylib");
            File.WriteAllBytes(hostileFile, hostile);
            Assert.IsFalse(MachOImage.TryRead(hostileFile, out _));

            Assert.IsFalse(MachOImage.TryRead(Path.Combine(_root, "missing.dylib"), out _));
        }

        /// <summary>The real bundled dylib, when staged, carries exactly the load command this whole design rests on.</summary>
        [TestMethod]
        public void MachO_TheStagedBundledDylib_NamesItsVendoredRuntimeThroughLoaderPath()
        {
            var staged = FindStagedMacOsDylibs();
            if (staged.Count == 0)
                Assert.Inconclusive("no osx-* OpenBLAS asset staged (python tools/fetch_openblas.py).");

            foreach (var main in staged)
            {
                Assert.IsTrue(MachOImage.TryRead(main, out var image), main);
                Assert.IsTrue(image.Dependencies.Any(d => d.StartsWith("@loader_path/../.dylibs/", StringComparison.Ordinal)),
                    $"{main}: expected an @loader_path/../.dylibs/ dependency, got: {string.Join(", ", image.Dependencies)}");

                // The deps are staged NESTED (native/.dylibs/) — the layout NuGet delivers.
                var nested = Path.Combine(Path.GetDirectoryName(main), ".dylibs");
                Assert.IsTrue(Directory.Exists(nested), "vendored runtime must be staged under native/.dylibs/: " + nested);
                foreach (var dep in image.Dependencies.Where(d => d.StartsWith("@loader_path/../.dylibs/", StringComparison.Ordinal)))
                    Assert.IsTrue(File.Exists(Path.Combine(nested, Path.GetFileName(dep))), "missing staged dep " + dep);

                // And NOT as the sibling the pack glob would sweep in as dead weight.
                var sibling = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(main)), ".dylibs");
                Assert.IsFalse(Directory.Exists(sibling), "the pre-fix sibling layout must be gone: " + sibling);
            }
        }

        #endregion

        #region planning — which files are missing, and where they are

        [TestMethod]
        public void Plan_NestedLayout_MapsEveryLoaderPathDep_AndFollowsTheDepsOwnDeps()
        {
            // runtimes/osx-arm64/native/libmain.dylib  ->  @loader_path/../.dylibs/libgfortran.5.dylib
            // native/.dylibs/libgfortran.5.dylib       ->  @loader_path/libquadmath.0.dylib, @loader_path/libgcc_s.1.1.dylib
            var native = MakeNuGetLayout(out _);
            var main = WriteMain(native, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib");
            WriteDep(Path.Combine(native, ".dylibs"), "libgfortran.5.dylib", "@loader_path/libquadmath.0.dylib", "@loader_path/libgcc_s.1.1.dylib");
            WriteDep(Path.Combine(native, ".dylibs"), "libquadmath.0.dylib");
            WriteDep(Path.Combine(native, ".dylibs"), "libgcc_s.1.1.dylib");

            Assert.IsTrue(MachOImage.TryRead(main, out var image));
            var plan = MacOsVendoredRuntime.Plan(native, image, out var missing);

            Assert.IsNotNull(plan, "missing: " + missing);
            CollectionAssert.AreEquivalent(
                new[] { "../.dylibs/libgfortran.5.dylib", "../.dylibs/libquadmath.0.dylib", "../.dylibs/libgcc_s.1.1.dylib" },
                plan.Select(p => p.RelativeTarget).ToArray());
            foreach (var p in plan)
                Assert.AreEqual(Path.Combine(native, ".dylibs", Path.GetFileName(p.RelativeTarget)), p.Source);
        }

        [TestMethod]
        public void Plan_FlattenedLayout_FindsTheDepsBesideTheMain()
        {
            // A RID-specific publish flattens native/.dylibs/x to <app>/x.
            var app = Path.Combine(_root, "app");
            Directory.CreateDirectory(app);
            var main = WriteMain(app, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib");
            WriteDep(app, "libgfortran.5.dylib", "@loader_path/libquadmath.0.dylib");
            WriteDep(app, "libquadmath.0.dylib");

            Assert.IsTrue(MachOImage.TryRead(main, out var image));
            var plan = MacOsVendoredRuntime.Plan(app, image, out var missing);

            Assert.IsNotNull(plan, "missing: " + missing);
            CollectionAssert.AreEquivalent(
                new[] { "../.dylibs/libgfortran.5.dylib", "../.dylibs/libquadmath.0.dylib" },
                plan.Select(p => p.RelativeTarget).ToArray());
            Assert.AreEqual(Path.Combine(app, "libgfortran.5.dylib"), plan.Single(p => p.RelativeTarget.EndsWith("libgfortran.5.dylib")).Source);
        }

        [TestMethod]
        public void Plan_IsEmpty_WhenEveryDependencyAlreadyResolves()
        {
            var native = MakeNuGetLayout(out var rid);
            var main = WriteMain(native, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib");
            WriteDep(Path.Combine(rid, ".dylibs"), "libgfortran.5.dylib"); // already at the expected sibling

            Assert.IsTrue(MachOImage.TryRead(main, out var image));
            var plan = MacOsVendoredRuntime.Plan(native, image, out _);
            Assert.IsNotNull(plan);
            Assert.AreEqual(0, plan.Count, "nothing to materialize: dyld's failure has another cause");

            // ... and TryPrepare therefore declines without a note (it is not a layout problem).
            Assert.IsFalse(MacOsVendoredRuntime.TryPrepare(main, out var image2, out var note));
            Assert.IsNull(image2);
            Assert.IsNull(note);
        }

        [TestMethod]
        public void Plan_DeclinesAndNamesTheDep_WhenItIsStagedNowhere()
        {
            var native = MakeNuGetLayout(out _);
            var main = WriteMain(native, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib");

            Assert.IsTrue(MachOImage.TryRead(main, out var image));
            Assert.IsNull(MacOsVendoredRuntime.Plan(native, image, out var missing));
            Assert.AreEqual("libgfortran.5.dylib", missing);

            Assert.IsFalse(MacOsVendoredRuntime.TryPrepare(main, out _, out var note));
            StringAssert.Contains(note, "libgfortran.5.dylib");
        }

        [TestMethod]
        public void Plan_IgnoresAbsoluteAndRpathDependencies()
        {
            var native = MakeNuGetLayout(out _);
            var main = WriteMain(native, "libmain.dylib", "/usr/lib/libSystem.B.dylib", "@rpath/libother.dylib");

            Assert.IsTrue(MachOImage.TryRead(main, out var image));
            var plan = MacOsVendoredRuntime.Plan(native, image, out _);
            Assert.AreEqual(0, plan.Count);
        }

        [TestMethod]
        public void NormalizeRelative_CollapsesDotsButKeepsClimbing()
        {
            Assert.AreEqual("../.dylibs/x", MacOsVendoredRuntime.NormalizeRelative("../.dylibs/x"));
            Assert.AreEqual("../.dylibs/y", MacOsVendoredRuntime.NormalizeRelative("../.dylibs/./y"));
            Assert.AreEqual("../y", MacOsVendoredRuntime.NormalizeRelative("../.dylibs/../y"));
            Assert.AreEqual("../../y", MacOsVendoredRuntime.NormalizeRelative("../../y"));
            Assert.AreEqual("x", MacOsVendoredRuntime.NormalizeRelative("./a/../x"));
            Assert.AreEqual(string.Empty, MacOsVendoredRuntime.NormalizeRelative(""));
        }

        [TestMethod]
        public void Layout_Classification()
        {
            var native = MakeNuGetLayout(out var rid);
            Assert.IsTrue(MacOsVendoredRuntime.IsNuGetNativeLayout(native));
            Assert.IsFalse(MacOsVendoredRuntime.IsNuGetNativeLayout(rid));
            Assert.IsFalse(MacOsVendoredRuntime.IsNuGetNativeLayout(Path.Combine(_root, "app")));
            Assert.IsFalse(MacOsVendoredRuntime.IsNuGetNativeLayout(null));

            var inside = new List<MacOsVendoredRuntime.VendoredFile> { new("../.dylibs/a.dylib", "src") };
            var escaping = new List<MacOsVendoredRuntime.VendoredFile> { new("../../.dylibs/a.dylib", "src") };
            Assert.IsTrue(MacOsVendoredRuntime.AllTargetsInside(rid, native, inside));
            Assert.IsFalse(MacOsVendoredRuntime.AllTargetsInside(rid, native, escaping));
        }

        [TestMethod]
        public void FindStagedDependency_PrefersNested_ThenFlat_AndRefusesPaths()
        {
            var dir = Path.Combine(_root, "d");
            Directory.CreateDirectory(Path.Combine(dir, ".dylibs"));
            File.WriteAllText(Path.Combine(dir, ".dylibs", "a.dylib"), "nested");
            File.WriteAllText(Path.Combine(dir, "a.dylib"), "flat");
            File.WriteAllText(Path.Combine(dir, "b.dylib"), "flat");

            Assert.AreEqual(Path.Combine(dir, ".dylibs", "a.dylib"), MacOsVendoredRuntime.FindStagedDependency(dir, "a.dylib"));
            Assert.AreEqual(Path.Combine(dir, "b.dylib"), MacOsVendoredRuntime.FindStagedDependency(dir, "b.dylib"));
            Assert.IsNull(MacOsVendoredRuntime.FindStagedDependency(dir, "c.dylib"));
            Assert.IsNull(MacOsVendoredRuntime.FindStagedDependency(dir, "../a.dylib"));
            Assert.IsNull(MacOsVendoredRuntime.FindStagedDependency(dir, "sub/a.dylib"));
        }

        #endregion

        #region materialization — in place

        [TestMethod]
        public void InPlace_NuGetLayout_MaterializesTheSibling_AndIsIdempotent()
        {
            var native = MakeNuGetLayout(out var rid);
            var main = WriteMain(native, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib");
            WriteDep(Path.Combine(native, ".dylibs"), "libgfortran.5.dylib", "@loader_path/libquadmath.0.dylib");
            WriteDep(Path.Combine(native, ".dylibs"), "libquadmath.0.dylib");

            Assert.IsTrue(MacOsVendoredRuntime.TryPrepare(main, out var image, out var note), note);
            Assert.AreEqual(Path.GetFullPath(main), image, "in place: the library itself is what gets loaded");
            StringAssert.Contains(note, "in place");

            // Every file dyld will ask for now exists at the path the load commands expand to
            // (through the directory symlink when one could be made, as copies otherwise).
            var sibling = Path.Combine(rid, ".dylibs");
            foreach (var name in new[] { "libgfortran.5.dylib", "libquadmath.0.dylib" })
            {
                var expected = Path.Combine(sibling, name);
                Assert.IsTrue(File.Exists(expected), "missing " + expected);
                Assert.IsTrue(MacOsVendoredRuntime.FilesIdentical(expected, Path.Combine(native, ".dylibs", name)));
            }

            // Nothing was written outside runtimes/<rid>/.
            Assert.IsFalse(Directory.Exists(Path.Combine(_root, ".dylibs")));
            Assert.IsFalse(Directory.Exists(Path.Combine(Path.GetDirectoryName(rid), ".dylibs")));

            // A second call finds every dependency already resolving: nothing to do, no note.
            Assert.IsFalse(MacOsVendoredRuntime.TryPrepare(main, out _, out var second));
            Assert.IsNull(second);
        }

        [TestMethod]
        public void InPlace_CompletesAPartialRealSiblingFolder_ByCopying()
        {
            var native = MakeNuGetLayout(out var rid);
            var main = WriteMain(native, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib");
            WriteDep(Path.Combine(native, ".dylibs"), "libgfortran.5.dylib", "@loader_path/libquadmath.0.dylib");
            WriteDep(Path.Combine(native, ".dylibs"), "libquadmath.0.dylib");
            // An older ProjectReference build left a REAL sibling folder with one dep only.
            Directory.CreateDirectory(Path.Combine(rid, ".dylibs"));
            File.Copy(Path.Combine(native, ".dylibs", "libgfortran.5.dylib"), Path.Combine(rid, ".dylibs", "libgfortran.5.dylib"));

            Assert.IsTrue(MacOsVendoredRuntime.TryPrepare(main, out var image, out var note), note);
            Assert.AreEqual(Path.GetFullPath(main), image);
            StringAssert.Contains(note, "copied 1 file");
            Assert.IsTrue(File.Exists(Path.Combine(rid, ".dylibs", "libquadmath.0.dylib")));
        }

        #endregion

        #region materialization — the per-user cache

        [TestMethod]
        public void Cache_FlattenedLayout_RelocatesMainAndDeps_KeyedByContentHash()
        {
            var cache = Path.Combine(_root, "cache");
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_CACHE_DIR", cache);

            var app = Path.Combine(_root, "app");
            Directory.CreateDirectory(app);
            var main = WriteMain(app, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib");
            WriteDep(app, "libgfortran.5.dylib", "@loader_path/libquadmath.0.dylib");
            WriteDep(app, "libquadmath.0.dylib");

            Assert.IsTrue(MacOsVendoredRuntime.TryPrepare(main, out var image, out var note), note);
            StringAssert.Contains(note, "relocated");
            StringAssert.Contains(note, "cache entry created");

            var hash = MacOsVendoredRuntime.HashFile(main);
            var entry = Path.Combine(cache, MacOsVendoredRuntime.CacheSubdirectory, hash);
            Assert.AreEqual(Path.Combine(entry, MacOsVendoredRuntime.ImageSubdirectory, "libmain.dylib"), image);
            Assert.IsTrue(File.Exists(image));
            Assert.IsTrue(File.Exists(Path.Combine(entry, ".dylibs", "libgfortran.5.dylib")), "the dep must sit where @loader_path/../.dylibs expands to");
            Assert.IsTrue(File.Exists(Path.Combine(entry, ".dylibs", "libquadmath.0.dylib")), "and so must the dep's own dep");
            Assert.IsTrue(MacOsVendoredRuntime.FilesIdentical(image, main));

            // Nothing was written next to, or above, the app.
            Assert.IsFalse(Directory.Exists(Path.Combine(_root, ".dylibs")));
            Assert.IsFalse(Directory.Exists(Path.Combine(app, ".dylibs")));
            Assert.AreEqual(0, Directory.GetDirectories(Path.Combine(cache, MacOsVendoredRuntime.CacheSubdirectory), "*.tmp").Length, "no temp directory may survive");

            // Second process: a verified hit, same image.
            Assert.IsTrue(MacOsVendoredRuntime.TryPrepare(main, out var again, out var note2), note2);
            Assert.AreEqual(image, again);
            StringAssert.Contains(note2, "cache hit");
        }

        [TestMethod]
        public void Cache_RebuildsAStaleOrTamperedEntry_AndNeverTrustsItsName()
        {
            var cache = Path.Combine(_root, "cache");
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_CACHE_DIR", cache);

            var app = Path.Combine(_root, "app");
            Directory.CreateDirectory(app);
            var main = WriteMain(app, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib");
            WriteDep(app, "libgfortran.5.dylib");

            Assert.IsTrue(MacOsVendoredRuntime.TryPrepare(main, out var image, out _));
            var entry = Path.GetDirectoryName(Path.GetDirectoryName(image));

            // Tamper with the cached dep: the entry name still matches the main, but the entry is
            // not the staged bytes any more — it must be rebuilt, not mapped.
            File.WriteAllText(Path.Combine(entry, ".dylibs", "libgfortran.5.dylib"), "garbage");
            Assert.IsTrue(MacOsVendoredRuntime.TryPrepare(main, out var image2, out var note), note);
            StringAssert.Contains(note, "cache entry created");
            Assert.AreEqual(image, image2);
            Assert.IsTrue(MacOsVendoredRuntime.FilesIdentical(Path.Combine(entry, ".dylibs", "libgfortran.5.dylib"), Path.Combine(app, "libgfortran.5.dylib")));

            // Tamper with the cached MAIN: same outcome.
            File.WriteAllBytes(image, new byte[] { 1, 2, 3 });
            Assert.IsTrue(MacOsVendoredRuntime.TryPrepare(main, out _, out var note2), note2);
            StringAssert.Contains(note2, "cache entry created");
            Assert.IsTrue(MacOsVendoredRuntime.FilesIdentical(image, main));
        }

        [TestMethod]
        public void Cache_TwoDifferentBinaries_NeverAlias()
        {
            var cache = Path.Combine(_root, "cache");
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_CACHE_DIR", cache);

            var appA = Path.Combine(_root, "appA");
            var appB = Path.Combine(_root, "appB");
            Directory.CreateDirectory(appA);
            Directory.CreateDirectory(appB);
            var mainA = WriteMain(appA, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib");
            var mainB = WriteMain(appB, "libmain.dylib", "@loader_path/../.dylibs/libgfortran.5.dylib", "@loader_path/../.dylibs/libextra.dylib");
            WriteDep(appA, "libgfortran.5.dylib");
            WriteDep(appB, "libgfortran.5.dylib");
            WriteDep(appB, "libextra.dylib");

            Assert.IsTrue(MacOsVendoredRuntime.TryPrepare(mainA, out var imageA, out _));
            Assert.IsTrue(MacOsVendoredRuntime.TryPrepare(mainB, out var imageB, out _));
            Assert.AreNotEqual(imageA, imageB);
            Assert.AreNotEqual(Path.GetDirectoryName(Path.GetDirectoryName(imageA)), Path.GetDirectoryName(Path.GetDirectoryName(imageB)));
        }

        [TestMethod]
        public void Cache_RefusesADependencyThatEscapesTheEntry()
        {
            var cache = Path.Combine(_root, "cache");
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_CACHE_DIR", cache);

            var app = Path.Combine(_root, "app");
            Directory.CreateDirectory(app);
            // ../../ climbs above the cache entry's own root: never written.
            var main = WriteMain(app, "libmain.dylib", "@loader_path/../../escape/libgfortran.5.dylib");
            WriteDep(app, "libgfortran.5.dylib");

            Assert.IsFalse(MacOsVendoredRuntime.TryPrepare(main, out var image, out var note));
            Assert.IsNull(image);
            StringAssert.Contains(note, "escapes");
            Assert.IsFalse(Directory.Exists(Path.Combine(cache, "escape")));
        }

        [TestMethod]
        public void CacheRoot_HonorsTheEnvVariable_ThenThePlatformDefaults()
        {
            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_CACHE_DIR", Path.Combine(_root, "explicit"));
            Assert.AreEqual(Path.GetFullPath(Path.Combine(_root, "explicit")), MacOsVendoredRuntime.ResolveCacheRoot());

            Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_CACHE_DIR", null);
            var root = MacOsVendoredRuntime.ResolveCacheRoot();
            Assert.IsNotNull(root);
            StringAssert.EndsWith(root.Replace('\\', '/'), "NumSharp/openblas");
        }

        #endregion

        #region the loader seam

        [TestMethod]
        public void LoadedImagePath_IsExposed_AndTracksTheLoadedLibrary()
        {
            // Not macOS-specific: everywhere else the mapped file IS the library discovery chose.
            if (OpenBlasEngine.LibraryPath == null)
            {
                Assert.IsNull(OpenBlasEngine.LoadedImagePath);
                if (!OpenBlasEngine.TryEnable())
                    Assert.Inconclusive("no CBLAS library could be loaded on this host.");
            }

            try
            {
                Assert.IsNotNull(OpenBlasEngine.LoadedImagePath);
                if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                    Assert.AreEqual(OpenBlasEngine.LibraryPath, OpenBlasEngine.LoadedImagePath);
                else
                    Assert.IsTrue(File.Exists(OpenBlasEngine.LoadedImagePath), OpenBlasEngine.LoadedImagePath);

                // Whatever was mapped is byte-for-byte the file LibraryPath names — the parity pin
                // hashes LibraryPath, so this is what keeps that pin honest.
                if (File.Exists(OpenBlasEngine.LibraryPath))
                    Assert.IsTrue(MacOsVendoredRuntime.FilesIdentical(OpenBlasEngine.LibraryPath, OpenBlasEngine.LoadedImagePath));
                StringAssert.Contains(OpenBlasEngine.Info, OpenBlasEngine.LibraryPath);
            }
            finally
            {
                OpenBlasEngine.Disable();
            }
        }

        #endregion

        #region helpers

        private string MakeNuGetLayout(out string ridDir)
        {
            ridDir = Path.Combine(_root, "app", "runtimes", "osx-arm64");
            var native = Path.Combine(ridDir, "native");
            Directory.CreateDirectory(native);
            return native;
        }

        private static string WriteMain(string dir, string name, params string[] loads)
        {
            var path = Path.Combine(dir, name);
            File.WriteAllBytes(path, SyntheticMachO.Dylib("@rpath/" + name, loads, Array.Empty<string>()));
            return path;
        }

        private static string WriteDep(string dir, string name, params string[] loads)
        {
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, name);
            File.WriteAllBytes(path, SyntheticMachO.Dylib("/DLC/scipy_openblas64/.dylibs/" + name, loads, new[] { "@loader_path/" }));
            return path;
        }

        private static List<string> FindStagedMacOsDylibs()
        {
            var result = new List<string>();
            var root = Path.Combine(AppContext.BaseDirectory, "runtimes");
            if (Directory.Exists(root))
                foreach (var rid in Directory.GetDirectories(root, "osx-*"))
                    result.AddRange(Directory.GetFiles(Path.Combine(rid, "native"), "*openblas*.dylib"));

            // The in-repo staging tree, when the tests run from the repo (CI stages it there).
            var repoRuntimes = FindRepoRuntimes();
            if (repoRuntimes != null)
                foreach (var rid in Directory.GetDirectories(repoRuntimes, "osx-*"))
                {
                    var native = Path.Combine(rid, "native");
                    if (Directory.Exists(native))
                        result.AddRange(Directory.GetFiles(native, "*openblas*.dylib"));
                }

            return result.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        }

        private static string FindRepoRuntimes()
        {
            var dir = AppContext.BaseDirectory;
            for (int i = 0; i < 8 && dir != null; i++)
            {
                var candidate = Path.Combine(dir, "src", "NumSharp.Interop.OpenBLAS", "runtimes");
                if (Directory.Exists(candidate))
                    return candidate;
                dir = Path.GetDirectoryName(dir);
            }

            return null;
        }

        /// <summary>Builds minimal but structurally valid Mach-O images: header + dylib/rpath load commands.</summary>
        private static class SyntheticMachO
        {
            private const uint LC_ID_DYLIB = 0xD, LC_LOAD_DYLIB = 0xC, LC_RPATH = 0x8000001C;

            internal static byte[] Dylib(string installName, string[] loads, string[] rpaths,
                                         bool bigEndian = false, bool is64 = true,
                                         (uint cmd, string name)[] extraLoads = null)
            {
                var commands = new List<byte[]>();
                if (installName != null)
                    commands.Add(DylibCommand(LC_ID_DYLIB, installName, bigEndian));
                foreach (var l in loads)
                    commands.Add(DylibCommand(LC_LOAD_DYLIB, l, bigEndian));
                if (extraLoads != null)
                    foreach (var (cmd, name) in extraLoads)
                        commands.Add(DylibCommand(cmd, name, bigEndian));
                foreach (var r in rpaths)
                    commands.Add(RpathCommand(r, bigEndian));

                int sizeofcmds = commands.Sum(c => c.Length);
                var ms = new MemoryStream();
                void U32(uint v)
                {
                    var b = BitConverter.GetBytes(v);
                    if (BitConverter.IsLittleEndian == bigEndian) Array.Reverse(b);
                    ms.Write(b, 0, 4);
                }

                U32(is64 ? 0xFEEDFACFu : 0xFEEDFACEu);
                U32(is64 ? 0x0100000Cu : 0x0000000Cu); // cputype (arm64 / arm)
                U32(0);                                // cpusubtype
                U32(6);                                // MH_DYLIB
                U32((uint)commands.Count);
                U32((uint)sizeofcmds);
                U32(0);                                // flags
                if (is64) U32(0);                      // reserved
                foreach (var c in commands)
                    ms.Write(c, 0, c.Length);
                // A little "text" so the file is not just a header.
                var pad = Encoding.ASCII.GetBytes("synthetic mach-o body for tests " + Guid.NewGuid());
                ms.Write(pad, 0, pad.Length);
                return ms.ToArray();
            }

            internal static byte[] Fat(params byte[][] slices)
            {
                var ms = new MemoryStream();
                void BE(uint v)
                {
                    var b = BitConverter.GetBytes(v);
                    if (BitConverter.IsLittleEndian) Array.Reverse(b);
                    ms.Write(b, 0, 4);
                }

                BE(0xCAFEBABE);
                BE((uint)slices.Length);
                uint offset = (uint)(8 + 20 * slices.Length);
                // Slices sit at 4096-aligned offsets, like real fat files.
                var offsets = new List<uint>();
                foreach (var s in slices)
                {
                    offset = (offset + 4095) & ~4095u;
                    offsets.Add(offset);
                    BE(0x0100000C); BE(0); BE(offset); BE((uint)s.Length); BE(12);
                    offset += (uint)s.Length;
                }

                for (int i = 0; i < slices.Length; i++)
                {
                    while (ms.Length < offsets[i]) ms.WriteByte(0);
                    ms.Write(slices[i], 0, slices[i].Length);
                }

                return ms.ToArray();
            }

            private static byte[] DylibCommand(uint cmd, string name, bool bigEndian)
            {
                // dylib_command: cmd, cmdsize, name offset (24), timestamp, current_version, compatibility_version, then the string
                var str = Encoding.UTF8.GetBytes(name + "\0");
                int cmdsize = (24 + str.Length + 7) & ~7;
                var b = new byte[cmdsize];
                Put(b, 0, cmd, bigEndian);
                Put(b, 4, (uint)cmdsize, bigEndian);
                Put(b, 8, 24, bigEndian);
                Put(b, 12, 2, bigEndian);
                Put(b, 16, 0x00010000, bigEndian);
                Put(b, 20, 0x00010000, bigEndian);
                Array.Copy(str, 0, b, 24, str.Length);
                return b;
            }

            private static byte[] RpathCommand(string path, bool bigEndian)
            {
                var str = Encoding.UTF8.GetBytes(path + "\0");
                int cmdsize = (12 + str.Length + 7) & ~7;
                var b = new byte[cmdsize];
                Put(b, 0, LC_RPATH, bigEndian);
                Put(b, 4, (uint)cmdsize, bigEndian);
                Put(b, 8, 12, bigEndian);
                Array.Copy(str, 0, b, 12, str.Length);
                return b;
            }

            private static void Put(byte[] b, int at, uint v, bool bigEndian)
            {
                var bytes = BitConverter.GetBytes(v);
                if (BitConverter.IsLittleEndian == bigEndian) Array.Reverse(bytes);
                Array.Copy(bytes, 0, b, at, 4);
            }
        }

        #endregion
    }
}
