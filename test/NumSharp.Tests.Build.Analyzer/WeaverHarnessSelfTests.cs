using System;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Proves the in-process weaver harness has teeth before anything is built on it: a plain
    ///     method is NOT woven and the file is untouched, an attributed one IS (scope local, the
    ///     try/finally Dispose seam, one Open, one Returns) and its temporaries are really reclaimed at
    ///     runtime, a second weave is a skipped no-op, an error shape is rejected without rewriting,
    ///     hand-scoped bodies are skipped, symbols survive, and both Release and Debug IL weave.
    /// </summary>
    [TestClass]
    public class WeaverHarnessSelfTests
    {
        private const string Prelude = "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n";

        [TestMethod]
        public void Unattributed_IsNotWoven_FileUntouched()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class P { public static NDArray M(NDArray a) { var t = a + 1.0; return t * 2.0; } }\n", "Plain");
            run.AssertWoven(0);
            Assert.IsFalse(run.Rewrote, "a target-free assembly must not be rewritten:\n" + run.Report);
            Assert.IsFalse(run.Result.Wrote);
            StringAssert.Contains(run.Stdout, "nothing to do");
            Assert.IsFalse(run.HasScopeLocal("P", "M"));
        }

        [TestMethod]
        public void Attributed_IsWoven_AndReclaimsAtRuntime()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class S { public static NDArray Last;\n" +
                "  [NDScoped] public static NDArray M(NDArray a) { var t = a + 1.0; Last = t; return t * 2.0; } }\n", "Scoped");
            run.AssertWoven(1);
            Assert.IsTrue(run.Rewrote && run.Result.Wrote, "a woven assembly is written back");
            Assert.IsTrue(run.Result.SymbolsWritten, "the portable PDB the fixture was compiled with must be rewritten alongside");
            Assert.IsTrue(run.HasScopeLocal("S", "M"), "the woven method carries an NDScope local");
            Assert.IsTrue(run.HasFinallyDispose("S", "M"), "the body is wrapped in try/finally { scope.Dispose() }");
            Assert.AreEqual(1, run.CountScopeCalls("S", "M", "Open"), "exactly one NDScope.Open");
            Assert.AreEqual(1, run.CountScopeCalls("S", "M", "Returns"), "the single return is routed through Returns");

            var a = np.arange(3).astype(np.float64);
            var result = (NDArray)run.Invoke("S", "M", null, a);
            CollectionAssert.AreEqual(new[] { 2.0, 4.0, 6.0 }, result.ToArray<double>());
            var temp = (NDArray)run.GetStatic("S", "Last");
            Assert.IsTrue(temp.IsDisposed, "the dropped temporary is reclaimed at scope exit");
            Assert.IsFalse(result.IsDisposed, "the yielded result survives");
            Assert.IsFalse(a.IsDisposed, "the input is never touched");
        }

        [TestMethod]
        public void SecondWeave_IsIdempotent_NoRewrite()
        {
            var fixture = WeaverTestHarness.Compile(Prelude +
                "public static class S { [NDScoped] public static NDArray M(NDArray a) { var t = a + 1.0; return t * 2.0; } }\n", "Twice");
            var first = WeaverTestHarness.Weave(fixture).AssertWoven(1);
            var second = WeaverTestHarness.Weave(fixture);
            Assert.AreEqual(0, second.Result.Errors, second.Report);
            Assert.AreEqual(0, second.Result.Woven, "already woven — nothing to weave");
            Assert.AreEqual(1, second.Result.Skipped, "the woven method is reported as already scoped");
            Assert.IsFalse(second.Rewrote, "an idempotent pass must not rewrite the file");
            Assert.AreEqual(1, second.CountScopeCalls("S", "M", "Open"), "still exactly one Open — never double-wrapped");
            Assert.IsTrue(first.Rewrote);
        }

        [TestMethod]
        public void ErrorShape_IsRejected_FileUntouched()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class E { [NDScoped] public static List<NDArray> M(NDArray a) => new List<NDArray> { a + 1.0 }; }\n", "Err");
            run.AssertRejected("NDW003");
            Assert.AreEqual(0, run.Result.Woven);
        }

        [TestMethod]
        public void HandScoped_IsSkipped_NotDoubleWrapped()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class H { [NDScoped] public static NDArray M(NDArray a) { using var s = NDScope.Open(); var t = a + 1.0; return s.Returns(t * 2.0); } }\n", "Hand");
            Assert.AreEqual(0, run.Result.Errors, run.Report);
            Assert.AreEqual(0, run.Result.Woven);
            Assert.AreEqual(1, run.Result.Skipped, "a body that already opens an NDScope is skipped");
            Assert.IsFalse(run.Rewrote);
            Assert.AreEqual(1, run.CountScopeCalls("H", "M", "Open"), "the hand-written Open is the only one");
        }

        [TestMethod]
        public void ReleaseAndDebugIL_BothWeave_SameResult()
        {
            foreach (var run in WeaverTestHarness.CompileAndWeaveBoth(Prelude +
                "public static class S { public static NDArray Last;\n" +
                "  [NDScoped] public static NDArray M(NDArray a) { var t = a + 1.0; Last = t; var u = t * 2.0; return u - 1.0; } }\n", "Opt"))
            {
                run.AssertWoven(1);
                Assert.IsTrue(run.HasScopeLocal("S", "M"), run.Report);
                var result = (NDArray)run.Invoke("S", "M", null, np.arange(3).astype(np.float64));
                CollectionAssert.AreEqual(new[] { 1.0, 3.0, 5.0 }, result.ToArray<double>(), $"optimized={run.Fixture.Optimized}");
                Assert.IsTrue(((NDArray)run.GetStatic("S", "Last")).IsDisposed, $"temp reclaimed (optimized={run.Fixture.Optimized})");
                Assert.IsFalse(result.IsDisposed);
            }
        }

        [TestMethod]
        public void Harness_ReportsCompileErrors_Loudly()
        {
            var ex = Assert.ThrowsException<System.InvalidOperationException>(() =>
                WeaverTestHarness.Compile(Prelude + "public static class B { public static NDArray M(NDArray a) { return a + ; } }\n", "Broken"));
            StringAssert.Contains(ex.Message, "does not compile");
        }

        // ---------------------------------------------------------------- the remaining weaver error codes

        [TestMethod]
        public void ReferenceListIsAdvisory_InProcess_NumSharpResolvesThroughTheHostsTrustedAssemblies()
        {
            // Documents a harness limit rather than a weaver rule: Mono.Cecil's resolver also probes the
            // running process's TRUSTED_PLATFORM_ASSEMBLIES list, and this test host's list carries
            // NumSharp.dll — so a reference-less IN-PROCESS weave still resolves NDScope. The real tool
            // runs in its own process whose list holds only the tool and Cecil; NDW001 is therefore
            // exercised by WeaverCliTests, which spawns the tool exactly as the MSBuild target does.
            var fixture = WeaverTestHarness.Compile(Prelude +
                "public static class S { [NDScoped] public static NDArray M(NDArray a) { var t = a + 1.0; return t * 2.0; } }\n", "NoRefs");
            var run = WeaverTestHarness.Weave(fixture, System.Array.Empty<string>(), null);
            Assert.AreEqual(0, run.Result.Errors, "in-process, NumSharp is reachable through the host's trusted assemblies:\n" + run.Report);
            Assert.AreEqual(1, run.Result.Woven);
        }

        [TestMethod]
        public void NDW004_UnrecognizedStateMachine_IsRejected()
        {
            // A hand-rolled [AsyncStateMachine] naming a struct with no MoveNext / builder: the analyzer
            // passes it (a Task-returning method under [NDScopedAsync] is valid source), the weaver
            // refuses loudly rather than mis-weave it.
            var run = WeaverTestHarness.CompileAndWeave(Prelude + "using System.Threading.Tasks;\n" +
                "public static class S {\n" +
                "  [NDScopedAsync] [System.Runtime.CompilerServices.AsyncStateMachine(typeof(Bogus))]\n" +
                "  public static Task<NDArray> M(NDArray a) => Task.FromResult(a.copy());\n" +
                "  private struct Bogus { }\n" +
                "}\n", "Bogus");
            run.AssertRejected("NDW004");
        }

        [TestMethod]
        public void NDW014_UnsupportedExitParameter_IsRejected()
        {
            var scalar = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class S { public static int Kept; public static void Keep([NDScopedExit] int x) => Kept = x; }\n", "ExitScalar");
            scalar.AssertRejected("NDW014");
            var byRef = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class S { public static NDArray Kept; public static void Keep([NDScopedExit] ref NDArray x) => Kept = x; }\n", "ExitRef");
            byRef.AssertRejected("NDW014");
        }

        [TestMethod]
        public void NDW015_UnsupportedOutCarrier_IsRejected()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class S { [NDScoped] public static void M(NDArray a, out List<NDArray> r) { r = new List<NDArray> { a + 1.0 }; } }\n", "OutList");
            run.AssertRejected("NDW015");
        }

        [TestMethod]
        public void NDW002_HiddenRefEgress_IsRejected()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class S { [NDScoped] public static void M(ref NDArray a) { a = a + 1.0; } }\n", "RefEgress");
            run.AssertRejected("NDW002");
        }

        [TestMethod]
        public void NDW009_NDW010_WrongAttribute_IsRejected()
        {
            var sync = WeaverTestHarness.CompileAndWeave(Prelude + "using System.Threading.Tasks;\n" +
                "public static class S { [NDScoped] public static async Task<NDArray> M(NDArray a) { await Task.Yield(); return a.copy(); } }\n", "WrongSync");
            sync.AssertRejected("NDW009");
            var async = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class S { [NDScopedAsync] public static NDArray M(NDArray a) => a.copy(); }\n", "WrongAsync");
            async.AssertRejected("NDW010");
        }

        // ---------------------------------------------------------------- re-signing and determinism

        [TestMethod]
        public void StrongNamedFixture_IsResignedWithItsKey_AndRuns()
        {
            var snk = RepoFile("Open.snk");
            var fixture = WeaverTestHarness.CompileSigned(Prelude +
                "public static class S { public static NDArray Last; [NDScoped] public static NDArray M(NDArray a) { var t = a + 1.0; Last = t; return t * 2.0; } }\n",
                "Signed", snk);
            var refs = AnalyzerTestHarness.ReferenceFilePaths;
            var run = WeaverTestHarness.Weave(fixture, refs, File.ReadAllBytes(snk));
            run.AssertWoven(1);
            Assert.IsTrue(run.Rewrote);

            var identity = System.Reflection.AssemblyName.GetAssemblyName(fixture.DllPath);
            var token = string.Concat(identity.GetPublicKeyToken().Select(b => b.ToString("x2")));
            Assert.AreEqual("cc7b13ffcd2ddd51", token, "the woven assembly keeps the repo key's identity (re-signed after the rewrite)");

            var result = (NDArray)run.Invoke("S", "M", null, np.arange(3).astype(np.float64));
            CollectionAssert.AreEqual(new[] { 2.0, 4.0, 6.0 }, result.ToArray<double>());
            Assert.IsTrue(((NDArray)run.GetStatic("S", "Last")).IsDisposed);
        }

        [TestMethod]
        public void IdenticalInputs_ProduceByteIdenticalOutputs()
        {
            // Reproducible builds: the same compiled bytes woven twice must yield the same bytes — the
            // rewrite introduces no timestamp, MVID or ordering nondeterminism of its own. The re-weave
            // runs at the SAME path (the PE's debug directory embeds the PDB path, so a copy elsewhere
            // legitimately differs): weave, keep the output, restore the original input, weave again.
            var a = WeaverTestHarness.Compile(Prelude +
                "public static class S { [NDScoped] public static NDArray M(NDArray a) { var t = a + 1.0; return t * 2.0; }\n" +
                "  [NDScoped] public static (NDArray, NDArray) N(NDArray a) { var t = a + 1.0; return (t, a.copy()); } }\n", "Determ");
            var pdb = Path.ChangeExtension(a.DllPath, ".pdb");
            var originalDll = File.ReadAllBytes(a.DllPath);
            var originalPdb = File.ReadAllBytes(pdb);

            WeaverTestHarness.Weave(a).AssertWoven(2);
            var firstDll = File.ReadAllBytes(a.DllPath);
            var firstPdb = File.ReadAllBytes(pdb);
            Assert.IsFalse(originalDll.AsSpan().SequenceEqual(firstDll), "the weave changed the DLL");

            File.WriteAllBytes(a.DllPath, originalDll);
            File.WriteAllBytes(pdb, originalPdb);
            WeaverTestHarness.Weave(a).AssertWoven(2);
            CollectionAssert.AreEqual(firstDll, File.ReadAllBytes(a.DllPath), "woven DLLs differ for identical inputs");
            CollectionAssert.AreEqual(firstPdb, File.ReadAllBytes(pdb), "rewritten PDBs differ for identical inputs");
        }

        private static string RepoFile(string relative)
        {
            for (var dir = new DirectoryInfo(System.AppContext.BaseDirectory); dir != null; dir = dir.Parent)
                if (File.Exists(Path.Combine(dir.FullName, "SciSharp.NumSharp.sln")))
                    return Path.Combine(dir.FullName, relative);
            Assert.Inconclusive("repo root (SciSharp.NumSharp.sln) not found above the test assembly");
            return null;
        }

        [TestMethod]
        public void Fixtures_GetUniqueAssemblies_AndPortablePdbs()
        {
            var f1 = WeaverTestHarness.Compile(Prelude + "public static class U { }\n", "Same");
            var f2 = WeaverTestHarness.Compile(Prelude + "public static class U { }\n", "Same");
            Assert.AreNotEqual(f1.AssemblyName, f2.AssemblyName, "same base name, distinct assembly identities");
            Assert.IsTrue(File.Exists(Path.ChangeExtension(f1.DllPath, ".pdb")), "a portable PDB is emitted beside the fixture");
            Assert.IsTrue(AnalyzerTestHarness.ReferenceFilePaths.Any(p => p.EndsWith("NumSharp.dll", System.StringComparison.OrdinalIgnoreCase)),
                "the weaver's --refs list carries NumSharp");
        }
    }
}
