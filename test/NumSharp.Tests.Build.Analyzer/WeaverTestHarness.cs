using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Emit;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Mono.Cecil;
using Mono.Cecil.Cil;
using NumSharp.Build;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The IN-PROCESS weaver harness: compiles a fixture source with Roslyn (framework + NumSharp
    ///     references, a portable PDB, Release or Debug IL) to a throwaway directory, runs
    ///     <see cref="ScopeWeaver.WeaveAssembly"/> on it exactly as the MSBuild target would (the
    ///     compile's reference list as <c>--refs</c>, no strong-name key), and hands back the outcome
    ///     three ways: the <see cref="WeaveResult"/> tallies + captured stdout/stderr, the woven IL read
    ///     back through Mono.Cecil (scope locals, the try/finally, every <c>NDScope</c> call), and the
    ///     woven assembly LOADED and executed so reclamation can be asserted on real arrays
    ///     (<c>IsDisposed</c>). It is the fast, fine-grained twin of the real-build gates
    ///     (<see cref="WeaverInheritanceBuildTests"/>, <c>tools/verify_build_package.sh</c>,
    ///     <c>tools/stress_weaver.sh</c>): a few milliseconds per fixture, so the weaver can be pinned
    ///     across hundreds of shapes instead of a handful of scripted consumers.
    /// </summary>
    internal static class WeaverTestHarness
    {
        private static readonly Lazy<string> LazyRoot = new(() =>
        {
            var root = Path.Combine(Path.GetTempPath(), "nsweave_" + Guid.NewGuid().ToString("N").Substring(0, 8));
            Directory.CreateDirectory(root);
            return root;
        });

        private static int s_sequence;

        /// <summary>
        ///     Compiles <paramref name="source"/> into <c>{assemblyName}_{n}.dll</c> (+ portable PDB) in its
        ///     own directory. The sequence suffix keeps every fixture's assembly identity unique — a
        ///     loaded assembly can never be replaced in the default load context, and two tests naming
        ///     their fixture the same must not collide. <paramref name="extraReferencePaths"/> are
        ///     other fixtures (a "base library") this one compiles against.
        /// </summary>
        public static CompiledFixture Compile(string source, string assemblyName, bool optimize = true, params string[] extraReferencePaths)
            => CompileCore(source, assemblyName, optimize, null, extraReferencePaths);

        /// <summary>
        ///     <see cref="Compile"/> with a strong-name key file — a full-signed fixture, for the weaver's
        ///     re-signing path. A DISTINCT name on purpose: an optional key-file parameter beside the
        ///     <c>params</c> reference list would let a reference path bind as the key.
        /// </summary>
        public static CompiledFixture CompileSigned(string source, string assemblyName, string strongNameKeyFile, bool optimize = true, params string[] extraReferencePaths)
            => CompileCore(source, assemblyName, optimize, strongNameKeyFile, extraReferencePaths);

        private static CompiledFixture CompileCore(string source, string assemblyName, bool optimize, string strongNameKeyFile, string[] extraReferencePaths)
        {
            var name = $"{assemblyName}_{Interlocked.Increment(ref s_sequence):D3}";
            var dir = Path.Combine(LazyRoot.Value, name);
            Directory.CreateDirectory(dir);

            // An explicit encoding: a portable PDB embeds the source's checksum, and Roslyn refuses to
            // emit debug information for encoding-less text (CS8055).
            var parse = new CSharpParseOptions(LanguageVersion.Latest);
            var tree = CSharpSyntaxTree.ParseText(SourceText.From(source, System.Text.Encoding.UTF8), parse, path: Path.Combine(dir, name + ".cs"));
            var options = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary,
                allowUnsafe: true,
                optimizationLevel: optimize ? OptimizationLevel.Release : OptimizationLevel.Debug);
            if (strongNameKeyFile != null)
                options = options.WithCryptoKeyFile(strongNameKeyFile)
                                 .WithStrongNameProvider(new DesktopStrongNameProvider(ImmutableArray<string>.Empty));
            var references = AnalyzerTestHarness.References
                .AddRange(extraReferencePaths.Select(p => (MetadataReference)MetadataReference.CreateFromFile(p)));
            var comp = CSharpCompilation.Create(name, new[] { tree }, references, options);

            var dll = Path.Combine(dir, name + ".dll");
            var pdb = Path.Combine(dir, name + ".pdb");
            using (var peStream = File.Create(dll))
            using (var pdbStream = File.Create(pdb))
            {
                var emit = comp.Emit(peStream, pdbStream,
                    options: new EmitOptions(debugInformationFormat: DebugInformationFormat.PortablePdb));
                if (!emit.Success)
                    throw new InvalidOperationException(
                        $"fixture '{name}' does not compile:\n  " +
                        string.Join("\n  ", emit.Diagnostics.Where(d => d.Severity == DiagnosticSeverity.Error)));
            }

            return new CompiledFixture(name, dll, dir, extraReferencePaths.ToImmutableArray(), optimize);
        }

        /// <summary>Runs the weaver over a compiled fixture in-process, capturing everything it reports.</summary>
        public static WeaveRun Weave(CompiledFixture fixture, bool verbose = true)
            => Weave(fixture, AnalyzerTestHarness.ReferenceFilePaths.AddRange(fixture.ExtraReferencePaths), null, verbose);

        /// <summary>
        ///     <see cref="Weave(CompiledFixture,bool)"/> with an explicit reference list (what the MSBuild
        ///     target's <c>--refs</c> response file would carry — empty to simulate a mis-configured
        ///     invocation) and an optional strong-name key blob (the <c>--snk</c> re-signing path).
        /// </summary>
        public static WeaveRun Weave(CompiledFixture fixture, IReadOnlyList<string> referencePaths, byte[] snkBlob, bool verbose = true)
        {
            var before = File.ReadAllBytes(fixture.DllPath);
            var stdout = new StringWriter();
            var stderr = new StringWriter();
            var result = ScopeWeaver.WeaveAssembly(fixture.DllPath, snkBlob, referencePaths, verbose, stdout, stderr);
            var after = File.ReadAllBytes(fixture.DllPath);
            return new WeaveRun(fixture, result, stdout.ToString(), stderr.ToString(), !before.AsSpan().SequenceEqual(after));
        }

        public static WeaveRun CompileAndWeave(string source, string assemblyName, bool optimize = true, params string[] extraReferencePaths)
            => Weave(Compile(source, assemblyName, optimize, extraReferencePaths));

        /// <summary>The same fixture compiled Release AND Debug — the two IL shapes every real build produces.</summary>
        public static IEnumerable<WeaveRun> CompileAndWeaveBoth(string source, string assemblyName, params string[] extraReferencePaths)
        {
            yield return CompileAndWeave(source, assemblyName + "R", optimize: true, extraReferencePaths);
            yield return CompileAndWeave(source, assemblyName + "D", optimize: false, extraReferencePaths);
        }
    }

    /// <summary>Repository-relative paths for the gates that need the checked-in key or the built Release tool.</summary>
    internal static class RepoPaths
    {
        /// <summary>A file under the repo root (located by walking up from the test assembly to the solution file), or Inconclusive.</summary>
        public static string File(string relative)
        {
            for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            {
                if (System.IO.File.Exists(Path.Combine(dir.FullName, "SciSharp.NumSharp.sln")))
                {
                    var path = Path.Combine(dir.FullName, relative.Replace('/', Path.DirectorySeparatorChar));
                    if (!System.IO.File.Exists(path))
                        Assert.Inconclusive($"repo file not found: {path}");
                    return path;
                }
            }

            Assert.Inconclusive("repo root (SciSharp.NumSharp.sln) not found above the test assembly");
            return null;
        }

        /// <summary>The Release weaver tool NumSharp.Core's self-weave builds (the same DLL the NumSharp.Build package ships under tools/).</summary>
        public static string ToolDll => File("tools/NumSharp.Build/bin/Release/net8.0/NumSharp.Build.dll");

        /// <summary>The dotnet host, DOTNET_ROOT first then PATH — Inconclusive when absent.</summary>
        public static string Dotnet()
        {
            var name = OperatingSystem.IsWindows() ? "dotnet.exe" : "dotnet";
            var root = Environment.GetEnvironmentVariable("DOTNET_ROOT");
            if (!string.IsNullOrEmpty(root) && System.IO.File.Exists(Path.Combine(root, name)))
                return Path.Combine(root, name);
            foreach (var dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
            {
                try { if (System.IO.File.Exists(Path.Combine(dir, name))) return Path.Combine(dir, name); }
                catch { /* malformed PATH entry */ }
            }

            Assert.Inconclusive("dotnet CLI not found on PATH");
            return null;
        }
    }

    internal sealed class CompiledFixture
    {
        public readonly string AssemblyName;
        public readonly string DllPath;
        public readonly string Directory;
        public readonly ImmutableArray<string> ExtraReferencePaths;
        public readonly bool Optimized;

        public CompiledFixture(string assemblyName, string dllPath, string directory, ImmutableArray<string> extraReferencePaths, bool optimized)
        {
            AssemblyName = assemblyName;
            DllPath = dllPath;
            Directory = directory;
            ExtraReferencePaths = extraReferencePaths;
            Optimized = optimized;
        }
    }

    /// <summary>One weave over one fixture: the tallies, the reports, the woven IL, and the loaded assembly.</summary>
    internal sealed class WeaveRun
    {
        private const string NDScopeFullName = "NumSharp.NDScope";

        public readonly CompiledFixture Fixture;
        public readonly WeaveResult Result;
        public readonly string Stdout;
        public readonly string Stderr;
        /// <summary>Whether the file on disk changed — the weaver must NOT rewrite a target-free or error assembly.</summary>
        public readonly bool Rewrote;

        private Assembly _loaded;

        public WeaveRun(CompiledFixture fixture, WeaveResult result, string stdout, string stderr, bool rewrote)
        {
            Fixture = fixture;
            Result = result;
            Stdout = stdout;
            Stderr = stderr;
            Rewrote = rewrote;
        }

        public string Report => $"[weave {Fixture.AssemblyName}: woven {Result.Woven}, skipped {Result.Skipped}, errors {Result.Errors}, rewrote {Rewrote}]\n--- stdout\n{Stdout}--- stderr\n{Stderr}";

        /// <summary>Asserts the weave reported no error and wove exactly <paramref name="woven"/> targets.</summary>
        public WeaveRun AssertWoven(int woven)
        {
            Assert.AreEqual(0, Result.Errors, "weave reported errors:\n" + Report);
            Assert.AreEqual(woven, Result.Woven, "woven count:\n" + Report);
            return this;
        }

        /// <summary>Asserts the weave reported exactly the given NDW codes (each at least once) on stderr and did not rewrite the file.</summary>
        public WeaveRun AssertRejected(params string[] codes)
        {
            Assert.IsTrue(Result.Errors > 0, "expected a weave error:\n" + Report);
            foreach (var code in codes)
                StringAssert.Contains(Stderr, "error " + code, $"expected {code}:\n" + Report);
            Assert.IsFalse(Rewrote, "an erroring weave must leave the assembly untouched:\n" + Report);
            return this;
        }

        // ---------------------------------------------------------------- the woven IL (Mono.Cecil)

        /// <summary>Reads the woven assembly through the weaver's own resolver (so cross-assembly types resolve as they did during the weave).</summary>
        public AssemblyDefinition ReadCecil()
        {
            var refs = AnalyzerTestHarness.ReferenceFilePaths.AddRange(Fixture.ExtraReferencePaths);
            var resolver = new WeaverAssemblyResolver(Fixture.Directory, refs);
            return AssemblyDefinition.ReadAssembly(Fixture.DllPath, new ReaderParameters { AssemblyResolver = resolver, InMemory = true });
        }

        /// <summary>Every method of every type (nested included) in the woven module.</summary>
        public static IEnumerable<MethodDefinition> AllMethods(ModuleDefinition module)
        {
            var stack = new Stack<TypeDefinition>(module.Types);
            while (stack.Count > 0)
            {
                var t = stack.Pop();
                foreach (var m in t.Methods)
                    yield return m;
                foreach (var nested in t.NestedTypes)
                    stack.Push(nested);
            }
        }

        /// <summary>Finds a method by its type's full name (Cecil spelling: <c>Ns.Outer/Nested</c>) and name; <paramref name="paramCount"/> disambiguates overloads.</summary>
        public static MethodDefinition Find(ModuleDefinition module, string typeFullName, string methodName, int? paramCount = null)
        {
            var hits = AllMethods(module)
                .Where(m => m.DeclaringType.FullName == typeFullName && m.Name == methodName &&
                            (paramCount is null || m.Parameters.Count == paramCount.Value))
                .ToList();
            Assert.AreEqual(1, hits.Count, $"expected exactly one method {typeFullName}::{methodName}({(paramCount?.ToString() ?? "*")}), found {hits.Count}");
            return hits[0];
        }

        /// <summary>The body the weave lands in: the method's own, or its compiler state machine's <c>MoveNext</c>.</summary>
        public static Mono.Cecil.Cil.MethodBody WovenBody(MethodDefinition m)
        {
            var kind = ScopeWeaver.GetStateMachineKind(m, out var smType);
            if (kind != ScopeWeaver.StateMachineKind.None && smType != null)
            {
                var moveNext = smType.Methods.FirstOrDefault(x => x.Name == "MoveNext" || x.Name.EndsWith(".MoveNext", StringComparison.Ordinal));
                return moveNext?.Body;
            }

            return m.HasBody ? m.Body : null;
        }

        /// <summary>The weaver's own coverage signal: a local of type <c>NDScope</c> in the woven body.</summary>
        public static bool HasScopeLocal(MethodDefinition m)
        {
            var body = WovenBody(m);
            return body != null && body.Variables.Any(v => v.VariableType.FullName == NDScopeFullName);
        }

        public bool HasScopeLocal(string typeFullName, string methodName, int? paramCount = null)
        {
            using var asm = ReadCecil();
            return HasScopeLocal(Find(asm.MainModule, typeFullName, methodName, paramCount));
        }

        /// <summary>Counts <c>call</c>s to <c>NDScope.&lt;name&gt;</c> in the woven body.</summary>
        public static int CountScopeCalls(MethodDefinition m, string name)
        {
            var body = WovenBody(m);
            if (body is null)
                return 0;
            return body.Instructions.Count(i => i.OpCode.FlowControl == FlowControl.Call &&
                                                i.Operand is MethodReference mr &&
                                                mr.Name == name && mr.DeclaringType.FullName == NDScopeFullName);
        }

        public int CountScopeCalls(string typeFullName, string methodName, string name, int? paramCount = null)
        {
            using var asm = ReadCecil();
            return CountScopeCalls(Find(asm.MainModule, typeFullName, methodName, paramCount), name);
        }

        /// <summary>A <c>finally</c> handler whose body calls <c>NDScope.Dispose</c> — the synchronous weave's reclamation seam.</summary>
        public static bool HasFinallyDispose(MethodDefinition m)
        {
            var body = WovenBody(m);
            if (body is null)
                return false;
            foreach (var h in body.ExceptionHandlers)
            {
                if (h.HandlerType != ExceptionHandlerType.Finally)
                    continue;
                for (var i = h.HandlerStart; i != null && i != h.HandlerEnd; i = i.Next)
                    if (i.OpCode.FlowControl == FlowControl.Call && i.Operand is MethodReference mr &&
                        mr.Name == "Dispose" && mr.DeclaringType.FullName == NDScopeFullName)
                        return true;
            }

            return false;
        }

        public bool HasFinallyDispose(string typeFullName, string methodName, int? paramCount = null)
        {
            using var asm = ReadCecil();
            return HasFinallyDispose(Find(asm.MainModule, typeFullName, methodName, paramCount));
        }

        /// <summary>Whether the method's OWN body (never MoveNext — the detach lands in the stub) calls <c>NDScope.Detach</c>.</summary>
        public bool CallsDetach(string typeFullName, string methodName, int? paramCount = null)
        {
            using var asm = ReadCecil();
            var m = Find(asm.MainModule, typeFullName, methodName, paramCount);
            return m.HasBody && m.Body.Instructions.Any(i => i.OpCode.FlowControl == FlowControl.Call &&
                                                            i.Operand is MethodReference mr &&
                                                            mr.Name == "Detach" && mr.DeclaringType.FullName == NDScopeFullName);
        }

        // ---------------------------------------------------------------- the woven code, executed

        /// <summary>
        ///     Loads the woven assembly. NumSharp resolves to the copy this test process already has
        ///     (same identity); a base-library fixture this one references is copied beside it so the
        ///     load-from directory probe finds it.
        /// </summary>
        public Assembly Load()
        {
            if (_loaded != null)
                return _loaded;
            foreach (var extra in Fixture.ExtraReferencePaths)
            {
                var target = Path.Combine(Fixture.Directory, Path.GetFileName(extra));
                if (!File.Exists(target))
                    File.Copy(extra, target);
            }

            return _loaded = Assembly.LoadFrom(Fixture.DllPath);
        }

        public Type LoadType(string fullName)
        {
            var t = Load().GetType(fullName.Replace('/', '+'), throwOnError: true);
            return t;
        }

        private const BindingFlags All = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        /// <summary>Invokes a method (static, or on <paramref name="instance"/>) by name, unwrapping reflection's exception wrapper.</summary>
        public object Invoke(string typeFullName, string methodName, object instance, params object[] args)
        {
            var type = LoadType(typeFullName);
            var candidates = type.GetMethods(All).Where(m => m.Name == methodName && m.GetParameters().Length == args.Length).ToList();
            Assert.AreEqual(1, candidates.Count, $"expected exactly one runtime method {typeFullName}.{methodName}/{args.Length}, found {candidates.Count}");
            try
            {
                return candidates[0].Invoke(instance, args);
            }
            catch (TargetInvocationException e) when (e.InnerException != null)
            {
                throw e.InnerException;
            }
        }

        public object GetProperty(string typeFullName, string propertyName, object instance)
            => LoadType(typeFullName).GetProperty(propertyName, All).GetValue(instance);

        public object New(string typeFullName, params object[] args) => Activator.CreateInstance(LoadType(typeFullName), args);

        public object GetStatic(string typeFullName, string fieldName) => LoadType(typeFullName).GetField(fieldName, All).GetValue(null);

        /// <summary>Reads an instance field, declared on the instance's type or inherited from a base (no DeclaredOnly).</summary>
        public object GetField(object instance, string fieldName)
        {
            var field = instance.GetType().GetField(fieldName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.IsNotNull(field, $"field '{fieldName}' not found on {instance.GetType().Name} or its bases");
            return field.GetValue(instance);
        }
    }
}
