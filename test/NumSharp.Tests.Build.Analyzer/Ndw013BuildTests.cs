using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The MSBuild half of NDW013: NumSharp.Core's <c>build/NumSharp.targets</c> (shipped in the NumSharp
    ///     package, so it is present when the NumSharp.Build package is ABSENT) declares the two knobs the
    ///     analyzer reads and carries a metadata text-scan FALLBACK that warns project-wide when the
    ///     analyzer is not applied to the compile — and yields to the analyzer (the primary, per-member
    ///     reporter; see <see cref="Ndw013AnalyzerTests"/>) when it is. Both are MSBuild behaviour, so —
    ///     unlike the in-process analyzer tests — they can only be exercised by a real build. These tests
    ///     generate a tiny project that imports that targets file and references the built NumSharp
    ///     assembly (a bare Reference, so NuGet applies no analyzer unless the test adds one), then invoke
    ///     <c>dotnet build</c>. Categorised <c>AnalyzerBuild</c> so the fast in-process unit run can exclude
    ///     them.
    /// </summary>
    [TestClass]
    [TestCategory("AnalyzerBuild")]
    public class Ndw013BuildTests
    {
        /// <summary>The project-wide sentence the MSBuild fallback prints — distinct from the analyzer's per-member "[NDScoped] on '…'" form.</summary>
        private const string ScanForm = "T uses [NDScoped]/[NDScopedAsync]/[NDScopedExit]";

        [TestMethod]
        public void Ndw013_ComesFromTheAnalyzer_WhenItIsApplied_AndTheScanYields()
        {
            // The real product path: the NumSharp package applies the analyzer to the consumer's compile.
            // Here the built analyzer DLL is added explicitly, which puts it in @(Analyzer) exactly as
            // NuGet would. The analyzer reports NDW013 AT the attributed member (a source location), and
            // the MSBuild fallback — seeing NumSharp.Build.Analyzer among the analyzers — stays silent, so
            // the project sees the finding once, not once per member plus once project-wide.
            var analyzerDll = typeof(NumSharp.Build.Analyzer.NDScopedTargetAnalyzer).Assembly.Location;
            RunConsumer(
                "using NumSharp;\npublic static class S { [NDScoped] public static NDArray F(NDArray a) => a + 1.0; }\n",
                "",
                output =>
                {
                    Assert.IsTrue(System.Text.RegularExpressions.Regex.IsMatch(output, @"S\.cs\(2,\d+\): warning NDW013"),
                        "the analyzer's NDW013 carries the attributed member's source location:\n" + output);
                    StringAssert.Contains(output, "[NDScoped] on 'S.F(NumSharp.NDArray)'", "the analyzer's per-member sentence");
                    Assert.IsFalse(output.Contains(ScanForm), "the MSBuild fallback must yield when the analyzer is applied");
                },
                extraItems: $@"<Analyzer Include=""{analyzerDll}"" />");

            // ... and the analyzer honours the same knob: weaver active -> silent.
            RunConsumer(
                "using NumSharp;\npublic static class S { [NDScoped] public static NDArray F(NDArray a) => a + 1.0; }\n",
                "-p:NumSharpBuildActive=true",
                output => Assert.IsFalse(output.Contains("NDW013"), "NDW013 must be silent when NumSharpBuildActive=true, analyzer applied"),
                extraItems: $@"<Analyzer Include=""{analyzerDll}"" />");
        }

        [TestMethod]
        public void Ndw013_Fallback_Fires_ForExitOnlyConsumer()
        {
            // [NDScopedExit] is weaver-dependent too (without the weaver the argument is never detached), so
            // the fallback scan matches its type name as well — no analyzer here, the scan is the reporter.
            RunConsumer(
                "using NumSharp;\npublic class H : System.IDisposable { private NDArray _w; public void Adopt([NDScopedExit] NDArray w) => _w = w; public void Dispose() => _w?.Dispose(); }\n",
                "",
                output =>
                {
                    StringAssert.Contains(output, "NDW013", "NDW013 must fire for an [NDScopedExit]-only consumer when the weaver is absent");
                    StringAssert.Contains(output, ScanForm, "without an analyzer the MSBuild fallback is the reporter");
                });
        }
        /// <summary>
        ///     The generated consumer must target the TFM of the RUNNING test host, because it
        ///     references the host's own NumSharp.dll (<c>typeof(NDArray).Assembly.Location</c>) —
        ///     a hardcoded net8.0 project handed the net10.0 build fails reference resolution
        ///     outright, which is exactly how these tests were silently broken on the net10.0
        ///     test target while CI (which excludes AnalyzerBuild) stayed green.
        /// </summary>
        private static string HostTfm => $"net{Environment.Version.Major}.0";

        [TestMethod]
        public void Ndw013_FiresWithoutWeaver_And_IsSilentWhenActive()
        {
            var dotnet = FindDotnet();
            if (dotnet == null)
                Assert.Inconclusive("dotnet CLI not found on PATH — skipping the NDW013 build test");

            var targets = RepoFile("src/NumSharp.Core/build/NumSharp.targets");
            var numsharpDll = typeof(NumSharp.NDArray).Assembly.Location;

            var work = Path.Combine(Path.GetTempPath(), "ndw013_" + Guid.NewGuid().ToString("N").Substring(0, 8));
            Directory.CreateDirectory(work);
            try
            {
                File.WriteAllText(Path.Combine(work, "T.csproj"), $@"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>{HostTfm}</TargetFramework>
    <Nullable>disable</Nullable>
    <SignAssembly>false</SignAssembly>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include=""NumSharp""><HintPath>{numsharpDll}</HintPath></Reference>
  </ItemGroup>
  <Import Project=""{targets.Replace("\\", "/")}"" />
</Project>");
                File.WriteAllText(Path.Combine(work, "S.cs"),
                    "using NumSharp;\npublic static class S { [NDScoped] public static NDArray F(NDArray a) => a + 1.0; }\n");

                // (a) no weaver active -> NDW013 fires
                var without = Build(dotnet, work, "");
                StringAssert.Contains(without, "NDW013",
                    "NDW013 must fire when [NDScoped] is used and the weaver is not active");

                // (b) weaver active -> NDW013 silent
                var with = Build(dotnet, work, "-p:NumSharpBuildActive=true");
                Assert.IsFalse(with.Contains("NDW013"),
                    "NDW013 must be silent when NumSharpBuildActive=true");
            }
            finally
            {
                try { Directory.Delete(work, recursive: true); } catch { /* best effort */ }
            }
        }

        [TestMethod]
        public void Ndw013_DoesNotFire_ForCoveredAttributeOnly()
        {
            // [NDScopedCovered] is an analyzer-only hint that needs NO weaver (it is inert at runtime),
            // so the "you used [NDScoped] but the weaver is absent" guard must stay silent for a
            // consumer that uses ONLY it — even though 'NDScopedCoveredAttribute' shares the 'NDScoped'
            // prefix. This pins the precise-name scan (NDScopedAttribute / NDScopedAsyncAttribute).
            var dotnet = FindDotnet();
            if (dotnet == null)
                Assert.Inconclusive("dotnet CLI not found on PATH — skipping the NDW013 helper test");

            var targets = RepoFile("src/NumSharp.Core/build/NumSharp.targets");
            var numsharpDll = typeof(NumSharp.NDArray).Assembly.Location;

            var work = Path.Combine(Path.GetTempPath(), "ndw013h_" + Guid.NewGuid().ToString("N").Substring(0, 8));
            Directory.CreateDirectory(work);
            try
            {
                File.WriteAllText(Path.Combine(work, "T.csproj"), $@"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>{HostTfm}</TargetFramework>
    <Nullable>disable</Nullable>
    <SignAssembly>false</SignAssembly>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include=""NumSharp""><HintPath>{numsharpDll}</HintPath></Reference>
  </ItemGroup>
  <Import Project=""{targets.Replace("\\", "/")}"" />
</Project>");
                File.WriteAllText(Path.Combine(work, "S.cs"),
                    "using NumSharp;\npublic static class S { [NDScopedCovered] public static NDArray F(NDArray a) => a + 1.0; }\n");

                var output = Build(dotnet, work, "");
                Assert.IsFalse(output.Contains("NDW013"),
                    "[NDScopedCovered] needs no weaver, so NDW013 must NOT fire for a helper-only consumer");
            }
            finally
            {
                try { Directory.Delete(work, recursive: true); } catch { /* best effort */ }
            }
        }

        [TestMethod]
        public void Ndw013_IsSilent_WhenSuppressionKnobSet()
        {
            // N13-3: the documented opt-out -p:NumSharpDisableWeaverMissingWarning=true silences NDW013
            // even when [NDScoped] is used and the weaver is absent.
            RunConsumer(
                "using NumSharp;\npublic static class S { [NDScoped] public static NDArray F(NDArray a) => a + 1.0; }\n",
                "-p:NumSharpDisableWeaverMissingWarning=true",
                output => Assert.IsFalse(output.Contains("NDW013"),
                    "the suppression knob must silence NDW013"));
        }

        [TestMethod]
        public void Ndw013_Fires_ForAsyncOnlyConsumer()
        {
            // N13-7: an [NDScopedAsync]-only consumer (no [NDScoped]) is equally inert without the
            // weaver, so the shared-prefix scan must still fire NDW013.
            RunConsumer(
                "using NumSharp;\nusing System.Threading.Tasks;\n" +
                "public static class S { [NDScopedAsync] public static async Task<NDArray> F(NDArray a) { await Task.Yield(); return a + 1.0; } }\n",
                "",
                output => StringAssert.Contains(output, "NDW013",
                    "NDW013 must fire for an [NDScopedAsync]-only consumer when the weaver is absent"));
        }

        /// <summary>
        ///     Builds a throwaway consumer project (referencing the built NumSharp + importing the NDW013
        ///     targets; <paramref name="extraItems"/> is spliced into its ItemGroup — e.g. an
        ///     <c>&lt;Analyzer Include&gt;</c>) and hands the build output to <paramref name="assert"/>.
        /// </summary>
        private static void RunConsumer(string source, string buildArgs, Action<string> assert, string extraItems = "")
        {
            var dotnet = FindDotnet();
            if (dotnet == null)
                Assert.Inconclusive("dotnet CLI not found on PATH — skipping the NDW013 build test");

            var targets = RepoFile("src/NumSharp.Core/build/NumSharp.targets");
            var numsharpDll = typeof(NumSharp.NDArray).Assembly.Location;
            var work = Path.Combine(Path.GetTempPath(), "ndw013m_" + Guid.NewGuid().ToString("N").Substring(0, 8));
            Directory.CreateDirectory(work);
            try
            {
                File.WriteAllText(Path.Combine(work, "T.csproj"), $@"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>{HostTfm}</TargetFramework>
    <Nullable>disable</Nullable>
    <SignAssembly>false</SignAssembly>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include=""NumSharp""><HintPath>{numsharpDll}</HintPath></Reference>
    {extraItems}
  </ItemGroup>
  <Import Project=""{targets.Replace("\\", "/")}"" />
</Project>");
                File.WriteAllText(Path.Combine(work, "S.cs"), source);
                assert(Build(dotnet, work, buildArgs));
            }
            finally
            {
                try { Directory.Delete(work, recursive: true); } catch { /* best effort */ }
            }
        }

        private static string Build(string dotnet, string work, string extraArgs)
        {
            var psi = new ProcessStartInfo(dotnet,
                $"build \"{Path.Combine(work, "T.csproj")}\" -c Release -t:Rebuild -v n --nologo {extraArgs}")
            {
                WorkingDirectory = work,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            using var p = Process.Start(psi);
            var sb = new StringBuilder();
            sb.Append(p.StandardOutput.ReadToEnd());
            sb.Append(p.StandardError.ReadToEnd());
            if (!p.WaitForExit(180_000))
            {
                try { p.Kill(); } catch { }
                Assert.Inconclusive("dotnet build timed out");
            }
            var output = sb.ToString();
            Assert.AreEqual(0, p.ExitCode, "the generated project must build (NDW013 is only a warning):\n" + output);
            return output;
        }

        private static string RepoFile(string relative)
        {
            for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            {
                if (File.Exists(Path.Combine(dir.FullName, "SciSharp.NumSharp.sln")))
                {
                    var path = Path.Combine(dir.FullName, relative.Replace('/', Path.DirectorySeparatorChar));
                    if (!File.Exists(path))
                        Assert.Inconclusive($"repo file not found: {path}");
                    return path;
                }
            }
            Assert.Inconclusive("repo root (SciSharp.NumSharp.sln) not found above the test assembly");
            return null; // unreachable
        }

        private static string FindDotnet()
        {
            var name = OperatingSystem.IsWindows() ? "dotnet.exe" : "dotnet";
            var envDir = Environment.GetEnvironmentVariable("DOTNET_ROOT");
            if (!string.IsNullOrEmpty(envDir) && File.Exists(Path.Combine(envDir, name)))
                return Path.Combine(envDir, name);
            foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
            {
                try { if (File.Exists(Path.Combine(dir, name))) return Path.Combine(dir, name); }
                catch { /* malformed PATH entry */ }
            }
            return null;
        }
    }
}
