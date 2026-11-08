using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     NDW021 (P5 of the 0.80.0-beta1 bridge QA, the owner's decision of 2026-11-05; numbered NDW020 until the merge into
    ///     strings, where the culture rule of parity QA-41 had taken that id): NumSharp.Core's
    ///     <c>build/NumSharp.targets</c> warns at BUILD time when a project that references NumSharp sets
    ///     <c>PublishAot=true</c>, because that setting switches dynamic code off - in a plain build and
    ///     <c>dotnet run</c> too, and by default in every .NET 10 file-based app - and NumSharp generates its array
    ///     kernels at run time, so the first array arithmetic throws <c>PlatformNotSupportedException</c> after a
    ///     clean build. MSBuild behaviour, so - like <see cref="Ndw013BuildTests"/> - exercised by real builds of a
    ///     generated project that imports the targets file and references the built NumSharp assembly. Categorised
    ///     <c>AnalyzerBuild</c> so the fast in-process unit run can exclude them.
    /// </summary>
    [TestClass]
    [TestCategory("AnalyzerBuild")]
    public class Ndw021BuildTests
    {
        /// <summary>
        ///     The generated consumer targets the TFM of the RUNNING test host, because it references the host's own
        ///     NumSharp.dll (see <see cref="Ndw013BuildTests"/>: a hardcoded TFM fails reference resolution).
        /// </summary>
        private static string HostTfm => $"net{Environment.Version.Major}.0";

        /// <summary>
        ///     With <c>PublishAot=true</c> the build warns NDW021 - naming the project, the one-line fix for a
        ///     file-based app and the help page - and still succeeds (a warning only). The test also pins the premise
        ///     the warning rests on: the SDK writes <c>IsDynamicCodeSupported=false</c> into the built app's
        ///     runtimeconfig, so even <c>dotnet run</c> (a JIT run, not a Native AOT publish) has no dynamic code.
        /// </summary>
        [TestMethod]
        public void Ndw021_Fires_WhenPublishAotIsOn_NamingTheFix()
        {
            RunConsumer("-p:PublishAot=true", (output, work) =>
            {
                StringAssert.Contains(output, "warning NDW021", "PublishAot=true must warn NDW021:\n" + output);
                StringAssert.Contains(output, "T sets PublishAot=true", "the message names the project");
                StringAssert.Contains(output, "'#:property PublishAot=false'", "the message names the file-based app's fix");
                StringAssert.Contains(output, "absolute-basics.html#ndw020", "the message links its help section");

                string runtimeConfig = Path.Combine(work, "bin", "Release", HostTfm, "T.runtimeconfig.json");
                Assert.IsTrue(File.Exists(runtimeConfig), "the Exe build must write its runtimeconfig: " + runtimeConfig);
                string config = File.ReadAllText(runtimeConfig);
                StringAssert.Contains(config, "\"System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeSupported\": false",
                    "the warning's premise: PublishAot=true switches dynamic code off even in a plain build:\n" + config);
            });
        }

        /// <summary>An ordinary project (PublishAot unset) builds without NDW021.</summary>
        [TestMethod]
        public void Ndw021_IsSilent_WhenPublishAotIsOff()
        {
            RunConsumer("", (output, _) =>
                Assert.IsFalse(output.Contains("NDW021"), "NDW021 must be silent when PublishAot is not set:\n" + output));
            RunConsumer("-p:PublishAot=false", (output, _) =>
                Assert.IsFalse(output.Contains("NDW021"), "NDW021 must be silent when PublishAot=false:\n" + output));
        }

        /// <summary>
        ///     The opt-out: a project that only uses NumSharp's types, and knows it never computes, silences NDW021 with
        ///     <c>NumSharpDisableAotWarning=true</c> - the knob the targets file documents.
        /// </summary>
        [TestMethod]
        public void Ndw021_IsSilent_WithTheOptOut()
        {
            RunConsumer("-p:PublishAot=true -p:NumSharpDisableAotWarning=true", (output, _) =>
                Assert.IsFalse(output.Contains("NDW021"), "NumSharpDisableAotWarning=true must silence NDW021:\n" + output));
        }

        /// <summary>
        ///     Builds a throwaway console project - referencing the built NumSharp assembly and importing
        ///     <c>NumSharp.targets</c>, as the NumSharp package's consumers do - with <paramref name="buildArgs"/>, and
        ///     hands the build output and the project folder to <paramref name="assert"/>. The folder is deleted after.
        /// </summary>
        /// <param name="buildArgs">Extra <c>dotnet build</c> arguments (the properties under test).</param>
        /// <param name="assert">The checks, given the build output and the project folder.</param>
        private static void RunConsumer(string buildArgs, Action<string, string> assert)
        {
            var dotnet = FindDotnet();
            if (dotnet == null)
                Assert.Inconclusive("dotnet CLI not found on PATH — skipping the NDW021 build test");

            var targets = RepoFile("src/NumSharp.Core/build/NumSharp.targets");
            var numsharpDll = typeof(NumSharp.NDArray).Assembly.Location;
            var work = Path.Combine(Path.GetTempPath(), "ndw020_" + Guid.NewGuid().ToString("N").Substring(0, 8));
            Directory.CreateDirectory(work);
            try
            {
                // An Exe, because PublishAot is an application setting: the SDK writes the runtimeconfig the
                // premise check reads only for an app.
                File.WriteAllText(Path.Combine(work, "T.csproj"), $@"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>{HostTfm}</TargetFramework>
    <Nullable>disable</Nullable>
    <SignAssembly>false</SignAssembly>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include=""NumSharp""><HintPath>{numsharpDll}</HintPath></Reference>
  </ItemGroup>
  <Import Project=""{targets.Replace("\\", "/")}"" />
</Project>");
                File.WriteAllText(Path.Combine(work, "Program.cs"), "System.Console.WriteLine(typeof(NumSharp.NDArray).Name);\n");
                assert(Build(dotnet, work, buildArgs), work);
            }
            finally
            {
                try { Directory.Delete(work, recursive: true); } catch { /* best effort */ }
            }
        }

        /// <summary>Runs <c>dotnet build</c> on the generated project and returns its output; the build must succeed.</summary>
        /// <param name="dotnet">The dotnet CLI.</param>
        /// <param name="work">The project folder.</param>
        /// <param name="extraArgs">The properties under test.</param>
        /// <returns>The build's standard output and error.</returns>
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
            Assert.AreEqual(0, p.ExitCode, "the generated project must build (NDW021 is only a warning):\n" + output);
            return output;
        }

        /// <summary>A file of this repository, found by walking up from the test assembly to the solution file.</summary>
        /// <param name="relative">The path from the repository root, with forward slashes.</param>
        /// <returns>The file's full path; the test is Inconclusive when the repository cannot be found.</returns>
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

        /// <summary>The dotnet CLI: <c>DOTNET_ROOT</c> first, then <c>PATH</c>; null when neither has it.</summary>
        /// <returns>The CLI's path, or null.</returns>
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
