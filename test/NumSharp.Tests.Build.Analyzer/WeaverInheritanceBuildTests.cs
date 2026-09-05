using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The WEAVER half of declaration-site inheritance, proven on a real build: a library declares
    ///     <c>[NDScoped]</c> on abstract, virtual, property and interface members (and an
    ///     <c>[NDScopedExit]</c> parameter on an abstract one); a SECOND project overrides and
    ///     implements them with no attribute of its own — no attribute text anywhere in its assembly —
    ///     and after the packaged <c>NDScopeWeave</c> target runs, every inheriting member carries the
    ///     woven scope (an <c>NDScope</c> local — the <c>NDScopeWeaveTests</c> signal), the inherited
    ///     exit parameter is detached, the abstract declarations are contracts (no NDW005), and the
    ///     woven code runs correctly. This is also the gate for the pre-scan change in
    ///     <c>build/NumSharp.Build.targets</c>: a consumer with NO attribute text used to be skipped
    ///     before the tool ever ran, which would silently leave every inherited target unwoven.
    ///     Categorised <c>AnalyzerBuild</c> like the NDW013 build tests (real <c>dotnet build</c>).
    /// </summary>
    [TestClass]
    [TestCategory("AnalyzerBuild")]
    public class WeaverInheritanceBuildTests
    {
        private static string HostTfm => $"net{Environment.Version.Major}.0";

        private const string LibSource = @"using NumSharp;
using System.Threading.Tasks;
namespace Lib
{
    public interface IOp
    {
        [NDScoped] NDArray Apply(NDArray a);
    }

    public abstract class Base
    {
        [NDScoped] public abstract NDArray Compute(NDArray a);
        [NDScoped] public virtual NDArray Twice(NDArray a) { var t = a * 2.0; return t.copy(); }
        [NDScopedAsync] public abstract Task<NDArray> ComputeAsync(NDArray a);
        [NDScoped] public abstract NDArray Value { get; }
        public virtual NDArray Plain(NDArray a) => a.copy();
        public abstract void Adopt([NDScopedExit] NDArray w);
    }

    public abstract class GBase<T>
    {
        [NDScoped] public virtual NDArray Map(T x, NDArray a) => a.copy();
    }
}
";

        // Deliberately attribute-FREE: the weaver must find every target through the referenced
        // declarations (the text pre-scan sees nothing here).
        private const string AppSource = @"using NumSharp;
using System.Threading.Tasks;
public class Derived : Lib.Base, Lib.IOp
{
    public NDArray Kept;
    public override NDArray Compute(NDArray a) { var t = a + 1.0; return t.copy(); }
    public override NDArray Twice(NDArray a) { var t = a * 3.0; return t.copy(); }
    public override async Task<NDArray> ComputeAsync(NDArray a) { var t = a + 2.0; await Task.Yield(); return t.copy(); }
    public override NDArray Value { get { var t = np.arange(3) + 5.0; return t.copy(); } }
    public override NDArray Plain(NDArray a) { var t = a + 1.0; return t.copy(); }
    public NDArray Apply(NDArray a) { var t = a - 1.0; return t.copy(); }
    public override void Adopt(NDArray w) => Kept = w;
}
public class Leaf : Lib.GBase<int>
{
    public override NDArray Map(int x, NDArray a) { var t = a + x; return t.copy(); }
}
";

        [TestMethod]
        public void InheritedTargets_AreWoven_AcrossAssemblies()
        {
            var dotnet = FindDotnet();
            if (dotnet == null)
                Assert.Inconclusive("dotnet CLI not found on PATH — skipping the weaver inheritance build test");

            var targets = RepoFile("tools/NumSharp.Build/build/NumSharp.Build.targets");
            var tool = RepoFile("tools/NumSharp.Build/bin/Release/net8.0/NumSharp.Build.dll");
            var numsharpDll = typeof(NumSharp.NDArray).Assembly.Location;

            var work = Path.Combine(Path.GetTempPath(), "ndwinh_" + Guid.NewGuid().ToString("N").Substring(0, 8));
            var lib = Path.Combine(work, "Lib");
            var app = Path.Combine(work, "App");
            Directory.CreateDirectory(lib);
            Directory.CreateDirectory(app);
            try
            {
                File.WriteAllText(Path.Combine(lib, "Lib.csproj"), Project(targets, tool, numsharpDll, "Lib", null));
                File.WriteAllText(Path.Combine(lib, "Base.cs"), LibSource);
                File.WriteAllText(Path.Combine(app, "App.csproj"), Project(targets, tool, numsharpDll, "App", "../Lib/Lib.csproj"));
                File.WriteAllText(Path.Combine(app, "Derived.cs"), AppSource);

                var output = Build(dotnet, Path.Combine(app, "App.csproj"));

                // The library: its two attributed BODIES (Base.Twice, GBase<T>.Map) are woven; the
                // abstract/interface declarations are contracts, not NDW005s.
                Assert.IsFalse(output.Contains("NDW005"), "abstract/interface declarations carrying the attribute are contracts, never NDW005:\n" + output);
                StringAssert.Matches(output, new Regex(@"Lib\.dll .*woven 2,"), "Lib weaves exactly its two attributed bodies (Twice, Map):\n" + output);

                // The app: NOTHING is attributed, yet Compute/Twice/ComputeAsync/get_Value/Apply/Map
                // inherit a scope (6) and Adopt inherits the [NDScopedExit] detach (1) = 7 woven — and
                // the pre-scan did not skip it.
                Assert.IsFalse(Regex.IsMatch(output, @"usage detected in App"), "an attribute-free consumer that references NumSharp must still spawn the weaver (inherited targets):\n" + output);
                StringAssert.Matches(output, new Regex(@"App\.dll .*woven 7,"), "the app's seven inheriting members are woven:\n" + output);

                var appDll = Path.Combine(app, "bin", "Release", HostTfm, "App.dll");
                Assert.IsTrue(File.Exists(appDll), "built App.dll not found at " + appDll);
                VerifyWovenAssembly(appDll);
            }
            finally
            {
                try { Directory.Delete(work, recursive: true); } catch { /* best effort */ }
            }
        }

        private static void VerifyWovenAssembly(string appDll)
        {
            // The app references the host's own NumSharp.dll (Private=false, not copied beside it), so
            // LoadFrom resolves NumSharp to the copy this test process already has; Lib.dll sits beside
            // App.dll and resolves through the LoadFrom directory probe.
            var asm = Assembly.LoadFrom(appDll);
            var derived = asm.GetType("Derived", throwOnError: true);
            var leaf = asm.GetType("Leaf", throwOnError: true);
            const BindingFlags all = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly;

            foreach (var name in new[] { "Compute", "Twice", "ComputeAsync", "get_Value", "Apply" })
            {
                var m = derived.GetMethod(name, all);
                Assert.IsNotNull(m, "method not found: " + name);
                Assert.IsTrue(HasScopeLocal(m), $"Derived.{name} inherits [NDScoped]/[NDScopedAsync] from Lib but carries no NDScope local — not woven");
            }

            Assert.IsTrue(HasScopeLocal(leaf.GetMethod("Map", all)), "Leaf.Map inherits [NDScoped] from GBase<T>.Map through the generic instantiation — not woven");
            Assert.IsFalse(HasScopeLocal(derived.GetMethod("Plain", all)), "Derived.Plain overrides an UNSCOPED member and must NOT be woven");
            Assert.IsTrue(CallsDetach(derived.GetMethod("Adopt", all)), "Derived.Adopt inherits the [NDScopedExit] parameter from Lib.Base.Adopt and must call NDScope.Detach");

            // And the woven code runs: values survive their scopes, the retained argument survives the caller's.
            var instance = Activator.CreateInstance(derived);
            var a = np.arange(3).astype(np.float64);
            var compute = (NDArray)derived.GetMethod("Compute", all).Invoke(instance, new object[] { a });
            CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, compute.ToArray<double>(), "Compute (woven, inherited) returns a+1");
            var twice = (NDArray)derived.GetMethod("Twice", all).Invoke(instance, new object[] { a });
            CollectionAssert.AreEqual(new[] { 0.0, 3.0, 6.0 }, twice.ToArray<double>(), "the OVERRIDE's body runs (a*3), not the base's");
            var apply = (NDArray)derived.GetMethod("Apply", all).Invoke(instance, new object[] { a });
            CollectionAssert.AreEqual(new[] { -1.0, 0.0, 1.0 }, apply.ToArray<double>(), "Apply (woven, inherited from the interface) returns a-1");
            var value = (NDArray)derived.GetProperty("Value", all).GetValue(instance);
            CollectionAssert.AreEqual(new[] { 5.0, 6.0, 7.0 }, value.ToArray<double>(), "the getter (woven from the base property's attribute) returns its value");
            var map = (NDArray)leaf.GetMethod("Map", all).Invoke(Activator.CreateInstance(leaf), new object[] { 10, a });
            CollectionAssert.AreEqual(new[] { 10.0, 11.0, 12.0 }, map.ToArray<double>(), "Map (woven through the generic base) returns a+x");

            NDArray kept;
            using (NDScope.Open())
            {
                var w = a + 100.0; // tracked by this scope — Adopt's inherited detach must free it from the scope
                derived.GetMethod("Adopt", all).Invoke(instance, new object[] { w });
                kept = (NDArray)derived.GetField("Kept").GetValue(instance);
            }

            CollectionAssert.AreEqual(new[] { 100.0, 101.0, 102.0 }, kept.ToArray<double>(), "the retained argument survived the caller's scope (inherited [NDScopedExit])");
        }

        private static bool HasScopeLocal(MethodInfo m)
        {
            var smType = m.GetCustomAttributes<System.Runtime.CompilerServices.StateMachineAttribute>(inherit: false)
                          .FirstOrDefault()?.StateMachineType;
            var body = smType != null
                ? smType.GetMethod("MoveNext", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly)?.GetMethodBody()
                : m.GetMethodBody();
            if (body is null)
                return false;
            foreach (var local in body.LocalVariables)
                if (local.LocalType.FullName == "NumSharp.NDScope")
                    return true;
            return false;
        }

        /// <summary>Scans the method's IL for a <c>call NDScope.Detach(...)</c> (opcode 0x28 + a MethodDef/MemberRef token).</summary>
        private static bool CallsDetach(MethodInfo m)
        {
            var il = m.GetMethodBody()?.GetILAsByteArray();
            if (il is null)
                return false;
            for (int i = 0; i + 4 < il.Length; i++)
            {
                if (il[i] != 0x28)
                    continue;
                int token = BitConverter.ToInt32(il, i + 1);
                try
                {
                    var target = m.Module.ResolveMethod(token);
                    if (target != null && target.Name == "Detach" && target.DeclaringType?.FullName == "NumSharp.NDScope")
                        return true;
                }
                catch
                {
                    // not a method token at this offset (an operand byte that happens to be 0x28)
                }
            }

            return false;
        }

        private static string Project(string targets, string tool, string numsharpDll, string name, string projectReference)
            => $@"<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>{HostTfm}</TargetFramework>
    <AssemblyName>{name}</AssemblyName>
    <Nullable>disable</Nullable>
    <SignAssembly>false</SignAssembly>
    <!-- Source-mode consumption of the weaver: the packaged targets imported straight from the repo,
         the tool taken from its Release build output (NumSharp.Core's self-weave builds it). -->
    <NumSharpBuildToolDll>{tool.Replace("\\", "/")}</NumSharpBuildToolDll>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include=""NumSharp""><HintPath>{numsharpDll}</HintPath><Private>false</Private></Reference>
    {(projectReference is null ? "" : $@"<ProjectReference Include=""{projectReference}"" />")}
  </ItemGroup>
  <Import Project=""{targets.Replace("\\", "/")}"" />
</Project>";

        private static string Build(string dotnet, string projectPath)
        {
            var psi = new ProcessStartInfo(dotnet,
                $"build \"{projectPath}\" -c Release -t:Rebuild -v n --nologo")
            {
                WorkingDirectory = Path.GetDirectoryName(projectPath),
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            using var p = Process.Start(psi);
            var sb = new StringBuilder();
            sb.Append(p.StandardOutput.ReadToEnd());
            sb.Append(p.StandardError.ReadToEnd());
            if (!p.WaitForExit(300_000))
            {
                try { p.Kill(); } catch { }
                Assert.Inconclusive("dotnet build timed out");
            }

            var output = sb.ToString();
            Assert.AreEqual(0, p.ExitCode, "the generated projects must build (the weave included):\n" + output);
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
                        Assert.Inconclusive($"repo file not found: {path} (build NumSharp.Core first — its self-weave builds the Release weaver)");
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
