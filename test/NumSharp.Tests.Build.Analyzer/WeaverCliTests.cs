using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Mono.Cecil;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The weaver's PROCESS surface — <c>Program.Main</c>, what the MSBuild target actually runs
    ///     (<c>dotnet exec NumSharp.Build.dll &lt;assembly&gt; [--refs rsp] [--snk key] [--verbose]</c>):
    ///     argument parsing, the response-file reference list, exit codes (0 woven / 1 NDW errors /
    ///     2 usage or unhandled), the summary line, re-signing, and the ONE failure the in-process harness
    ///     cannot reproduce — NDW001, an assembly whose NumSharp cannot be resolved — because the tool's
    ///     own process has no NumSharp on its trusted-assembly list while the test host's does. Uses the
    ///     Release tool NumSharp.Core's self-weave builds (Inconclusive when it is absent).
    /// </summary>
    [TestClass]
    public class WeaverCliTests
    {
        private const string Prelude = "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n";

        private sealed class CliRun
        {
            public int ExitCode;
            public string Stdout;
            public string Stderr;
            public override string ToString() => $"exit {ExitCode}\n--- stdout\n{Stdout}--- stderr\n{Stderr}";
        }

        private static CliRun Run(string workingDirectory, params string[] args)
        {
            var dotnet = RepoPaths.Dotnet();
            var tool = RepoPaths.ToolDll;
            var psi = new ProcessStartInfo(dotnet)
            {
                WorkingDirectory = workingDirectory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            psi.ArgumentList.Add("exec");
            psi.ArgumentList.Add(tool);
            foreach (var a in args)
                psi.ArgumentList.Add(a);
            using var p = Process.Start(psi);
            var stdout = p.StandardOutput.ReadToEndAsync();
            var stderr = p.StandardError.ReadToEndAsync();
            if (!p.WaitForExit(120_000))
            {
                try { p.Kill(); } catch { }
                Assert.Inconclusive("the weaver process timed out");
            }

            return new CliRun { ExitCode = p.ExitCode, Stdout = stdout.Result, Stderr = stderr.Result };
        }

        private static string WriteRefs(CompiledFixture fixture)
        {
            var rsp = Path.Combine(fixture.Directory, "refs.rsp");
            File.WriteAllLines(rsp, AnalyzerTestHarness.ReferenceFilePaths.AddRange(fixture.ExtraReferencePaths));
            return rsp;
        }

        private static CompiledFixture ScopedFixture(string name) => WeaverTestHarness.Compile(Prelude +
            "public static class S { [NDScoped] public static NDArray M(NDArray a) { var t = a + 1.0; return t * 2.0; } }\n", name);

        [TestMethod]
        public void NoRefs_NumSharpUnresolvable_IsNDW001_ExitCode1_FileUntouched()
        {
            var fixture = ScopedFixture("CliNoRefs");
            var before = File.ReadAllBytes(fixture.DllPath);
            var run = Run(fixture.Directory, fixture.DllPath);
            Assert.AreEqual(1, run.ExitCode, "NDW errors exit 1:\n" + run);
            StringAssert.Contains(run.Stderr, "error NDW001", run.ToString());
            StringAssert.Contains(run.Stderr, "--refs", "the message tells the caller how to fix the invocation");
            CollectionAssert.AreEqual(before, File.ReadAllBytes(fixture.DllPath), "an erroring weave leaves the assembly untouched");
        }

        [TestMethod]
        public void WithRefs_Weaves_ExitCode0_SummaryLine()
        {
            var fixture = ScopedFixture("CliRefs");
            var run = Run(fixture.Directory, fixture.DllPath, "--refs", WriteRefs(fixture), "--verbose");
            Assert.AreEqual(0, run.ExitCode, run.ToString());
            StringAssert.Contains(run.Stdout, "woven 1, already-scoped 0", run.ToString());
            StringAssert.Contains(run.Stdout, "symbols updated", "the portable PDB beside the fixture is rewritten");
            StringAssert.Contains(run.Stdout, "NumSharp.Build: woven: ", "--verbose lists each woven target");
            Assert.AreEqual("", run.Stderr.Trim(), "a clean weave writes nothing to stderr");

            using var asm = AssemblyDefinition.ReadAssembly(fixture.DllPath, new ReaderParameters { InMemory = true });
            Assert.IsTrue(WeaveRun.HasScopeLocal(WeaveRun.Find(asm.MainModule, "S", "M")), "the process-woven method carries the scope local");
        }

        [TestMethod]
        public void SecondRun_ReportsAlreadyScoped_AssemblyUnchanged()
        {
            var fixture = ScopedFixture("CliTwice");
            var rsp = WriteRefs(fixture);
            Assert.AreEqual(0, Run(fixture.Directory, fixture.DllPath, "--refs", rsp).ExitCode);
            var after = File.ReadAllBytes(fixture.DllPath);
            var second = Run(fixture.Directory, fixture.DllPath, "--refs", rsp);
            Assert.AreEqual(0, second.ExitCode, second.ToString());
            StringAssert.Contains(second.Stdout, "woven 0, already-scoped 1, assembly unchanged", second.ToString());
            CollectionAssert.AreEqual(after, File.ReadAllBytes(fixture.DllPath), "the idempotent pass must not rewrite");
        }

        [TestMethod]
        public void TargetFree_NothingToDo_ExitCode0_NoRewrite()
        {
            var fixture = WeaverTestHarness.Compile(Prelude + "public static class P { public static NDArray M(NDArray a) => a + 1.0; }\n", "CliPlain");
            var before = File.ReadAllBytes(fixture.DllPath);
            var run = Run(fixture.Directory, fixture.DllPath, "--refs", WriteRefs(fixture));
            Assert.AreEqual(0, run.ExitCode, run.ToString());
            StringAssert.Contains(run.Stdout, "nothing to do", run.ToString());
            StringAssert.Contains(run.Stdout, "woven 0, already-scoped 0, assembly unchanged", run.ToString());
            CollectionAssert.AreEqual(before, File.ReadAllBytes(fixture.DllPath));
        }

        [TestMethod]
        public void ErrorShape_IsNDW003_ExitCode1()
        {
            var fixture = WeaverTestHarness.Compile(Prelude +
                "public static class E { [NDScoped] public static List<NDArray> M(NDArray a) => new List<NDArray> { a + 1.0 }; }\n", "CliErr");
            var run = Run(fixture.Directory, fixture.DllPath, "--refs", WriteRefs(fixture));
            Assert.AreEqual(1, run.ExitCode, run.ToString());
            StringAssert.Contains(run.Stderr, "error NDW003", run.ToString());
        }

        [TestMethod]
        public void InheritedTarget_AcrossAssemblies_IsWovenByTheProcess()
        {
            // The real consumer path end to end: the library with the contract is a separate assembly
            // listed in the rsp, and the attribute-free override in the woven assembly is found through it.
            var lib = WeaverTestHarness.Compile(Prelude +
                "namespace Lib { public abstract class Base { [NDScoped] public abstract NDArray Compute(NDArray a); public abstract void Adopt([NDScopedExit] NDArray w); } }\n", "CliLib");
            var app = WeaverTestHarness.Compile(Prelude +
                "public class D : Lib.Base { public NDArray Kept; public override NDArray Compute(NDArray a) { var t = a + 1.0; return t.copy(); } public override void Adopt(NDArray w) => Kept = w; }\n",
                "CliApp", true, lib.DllPath);
            var run = Run(app.Directory, app.DllPath, "--refs", WriteRefs(app), "--verbose");
            Assert.AreEqual(0, run.ExitCode, run.ToString());
            StringAssert.Contains(run.Stdout, "woven 2, already-scoped 0", run.ToString());
            StringAssert.Contains(run.Stdout, "inherited from Lib.Base::Compute", run.ToString());
            StringAssert.Contains(run.Stdout, "inherited from Lib.Base::Adopt", run.ToString());
        }

        [TestMethod]
        public void Snk_ResignsAStrongNamedAssembly()
        {
            var snk = RepoPaths.File("Open.snk");
            var fixture = WeaverTestHarness.CompileSigned(Prelude +
                "public static class S { [NDScoped] public static NDArray M(NDArray a) { var t = a + 1.0; return t * 2.0; } }\n", "CliSigned", snk);
            var run = Run(fixture.Directory, fixture.DllPath, "--refs", WriteRefs(fixture), "--snk", snk);
            Assert.AreEqual(0, run.ExitCode, run.ToString());
            StringAssert.Contains(run.Stdout, "re-signed", run.ToString());
            var token = string.Concat(System.Reflection.AssemblyName.GetAssemblyName(fixture.DllPath).GetPublicKeyToken().Select(b => b.ToString("x2")));
            Assert.AreEqual("cc7b13ffcd2ddd51", token);
        }

        [TestMethod]
        public void EmptySnkValue_MeansNoKey()
        {
            // The packaged target expands $(KeyOriginatorFile), which an unsigned consumer leaves blank.
            var fixture = ScopedFixture("CliEmptySnk");
            var run = Run(fixture.Directory, fixture.DllPath, "--refs", WriteRefs(fixture), "--snk", "");
            Assert.AreEqual(0, run.ExitCode, run.ToString());
            StringAssert.Contains(run.Stdout, "woven 1", run.ToString());
            Assert.IsFalse(run.Stdout.Contains("re-signed"), "no key, no re-signing");
        }

        [TestMethod]
        public void UsageErrors_ExitCode2_WithNDW000()
        {
            var fixture = ScopedFixture("CliUsage");
            var missing = Run(fixture.Directory, Path.Combine(fixture.Directory, "does-not-exist.dll"));
            Assert.AreEqual(2, missing.ExitCode, missing.ToString());
            StringAssert.Contains(missing.Stderr, "NDW000", "a missing assembly is a usage error");

            var noArgs = Run(fixture.Directory);
            Assert.AreEqual(2, noArgs.ExitCode, noArgs.ToString());
            StringAssert.Contains(noArgs.Stderr, "no assembly path given");

            var badOption = Run(fixture.Directory, fixture.DllPath, "--frobnicate");
            Assert.AreEqual(2, badOption.ExitCode, badOption.ToString());
            StringAssert.Contains(badOption.Stderr, "unexpected argument");

            var danglingRefs = Run(fixture.Directory, fixture.DllPath, "--refs");
            Assert.AreEqual(2, danglingRefs.ExitCode, danglingRefs.ToString());
            StringAssert.Contains(danglingRefs.Stderr, "missing value for --refs");

            var missingRsp = Run(fixture.Directory, fixture.DllPath, "--refs", Path.Combine(fixture.Directory, "nope.rsp"));
            Assert.AreEqual(2, missingRsp.ExitCode, missingRsp.ToString());
            StringAssert.Contains(missingRsp.Stderr, "reference list not found");

            var missingKey = Run(fixture.Directory, fixture.DllPath, "--snk", Path.Combine(fixture.Directory, "nope.snk"));
            Assert.AreEqual(2, missingKey.ExitCode, missingKey.ToString());
            StringAssert.Contains(missingKey.Stderr, "strong-name key not found");

            var help = Run(fixture.Directory, "--help");
            Assert.AreEqual(0, help.ExitCode, help.ToString());
            StringAssert.Contains(help.Stderr, "usage:");
        }

        [TestMethod]
        public void MalformedRspLines_AreTolerated()
        {
            // A blank line, whitespace, and a path that does not exist must not kill the weave — the
            // resolver skips what it cannot use and the real references still resolve.
            var fixture = ScopedFixture("CliRsp");
            var rsp = Path.Combine(fixture.Directory, "refs.rsp");
            var lines = new StringBuilder();
            lines.AppendLine("");
            lines.AppendLine("   ");
            lines.AppendLine(Path.Combine(fixture.Directory, "missing-reference.dll"));
            foreach (var p in AnalyzerTestHarness.ReferenceFilePaths)
                lines.AppendLine("  " + p + "  ");
            File.WriteAllText(rsp, lines.ToString());
            var run = Run(fixture.Directory, fixture.DllPath, "--refs", rsp);
            Assert.AreEqual(0, run.ExitCode, run.ToString());
            StringAssert.Contains(run.Stdout, "woven 1", run.ToString());
        }
    }
}
