using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Analyzer
{
    /// <summary>
    ///     NDW013 is an MSBuild warning shipped in NumSharp.Core's <c>build/NumSharp.targets</c> (it must
    ///     fire when the NumSharp.Weaver package is ABSENT), so — unlike the Roslyn analyzers — it can
    ///     only be exercised by a real build. These tests generate a tiny project that imports that
    ///     targets file and references the built NumSharp assembly, then invoke <c>dotnet build</c>.
    ///     Categorised <c>AnalyzerBuild</c> so the fast in-process unit run can exclude them.
    /// </summary>
    [TestClass]
    [TestCategory("AnalyzerBuild")]
    public class Ndw013BuildTests
    {
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
    <TargetFramework>net8.0</TargetFramework>
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
                var with = Build(dotnet, work, "-p:NumSharpWeaverActive=true");
                Assert.IsFalse(with.Contains("NDW013"),
                    "NDW013 must be silent when NumSharpWeaverActive=true");
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
    <TargetFramework>net8.0</TargetFramework>
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
