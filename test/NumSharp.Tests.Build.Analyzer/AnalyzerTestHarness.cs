using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Build.Analyzer;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Compiles a piece of C# in-process, runs all three NumSharp analyzers over it (the NDW012
    ///     leak pass, the [NDScoped]-target gate, and the NDW016/NDW017 ownership pass), and exposes
    ///     the resulting NDW diagnostics — the engine every analyzer test drives. References are the
    ///     running framework's trusted-platform set plus NumSharp itself (so <c>NDArray</c>/<c>NDScope</c>/
    ///     <c>np</c> resolve exactly as they do in a real build).
    /// </summary>
    internal static class AnalyzerTestHarness
    {
        private static readonly Lazy<ImmutableArray<MetadataReference>> LazyRefs = new(BuildReferences);

        private static ImmutableArray<MetadataReference> BuildReferences()
        {
            var refs = new List<MetadataReference>();
            var tpa = (string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? "";
            foreach (var path in tpa.Split(Path.PathSeparator))
                if (path.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) && File.Exists(path))
                    refs.Add(MetadataReference.CreateFromFile(path));
            // NumSharp is a test dependency, not on the TPA list — add it explicitly.
            refs.Add(MetadataReference.CreateFromFile(typeof(NumSharp.NDArray).Assembly.Location));
            return refs.ToImmutableArray();
        }

        // The framework refs WITHOUT NumSharp — for the "NumSharp not referenced -> analyzer no-ops"
        // path (KnownTypes.Resolve returns null). NumSharp is not on the TPA list, so the TPA set alone
        // is a NumSharp-free compilation.
        private static readonly Lazy<ImmutableArray<MetadataReference>> LazyRefsNoNumSharp =
            new(() =>
            {
                var refs = new List<MetadataReference>();
                var tpa = (string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") ?? "";
                foreach (var path in tpa.Split(Path.PathSeparator))
                    if (path.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) && File.Exists(path))
                        refs.Add(MetadataReference.CreateFromFile(path));
                return refs.ToImmutableArray();
            });

        /// <summary>Runs both analyzers over source that does NOT reference NumSharp (the no-op guard path).</summary>
        public static async Task<AnalyzerResult> RunWithoutNumSharpAsync(string source, string displayPath)
        {
            var tree = CSharpSyntaxTree.ParseText(SourceText.From(source), path: displayPath);
            var options = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true);
            var comp = CSharpCompilation.Create(
                "NoNumSharp_" + Path.GetFileNameWithoutExtension(displayPath),
                new[] { tree }, LazyRefsNoNumSharp.Value, options);
            var compileErrors = comp.GetDiagnostics()
                .Where(d => d.Severity == DiagnosticSeverity.Error)
                .ToImmutableArray();
            var analyzers = ImmutableArray.Create<DiagnosticAnalyzer>(
                new NDArrayLeakAnalyzer(), new NDScopedTargetAnalyzer(), new NDArrayHolderAnalyzer());
            var all = await comp.WithAnalyzers(analyzers).GetAnalyzerDiagnosticsAsync();
            var ndw = all.Where(d => d.Id.StartsWith("NDW", StringComparison.Ordinal)).ToImmutableArray();
            return new AnalyzerResult(ndw, compileErrors);
        }

        /// <summary>Locates a fixture .cs file copied next to the test assembly (see the csproj glob).</summary>
        public static string FixturePath(string fileName)
        {
            var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", fileName);
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"fixture '{fileName}' not found at '{path}'. It should be copied from " +
                    "NumSharp.Tests.Build.Analyzer.Fixtures by the test csproj — rebuild the test project.");
            return path;
        }

        /// <summary>Runs both analyzers over <paramref name="source"/> and returns (NDW diagnostics, C# compile errors).</summary>
        public static async Task<AnalyzerResult> RunAsync(string source, string displayPath)
        {
            var tree = CSharpSyntaxTree.ParseText(SourceText.From(source), path: displayPath);
            var options = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true);
            var comp = CSharpCompilation.Create(
                "AnalyzerFixture_" + Path.GetFileNameWithoutExtension(displayPath),
                new[] { tree }, LazyRefs.Value, options);

            var compileErrors = comp.GetDiagnostics()
                .Where(d => d.Severity == DiagnosticSeverity.Error)
                .ToImmutableArray();

            var analyzers = ImmutableArray.Create<DiagnosticAnalyzer>(
                new NDArrayLeakAnalyzer(), new NDScopedTargetAnalyzer(), new NDArrayHolderAnalyzer());
            var all = await comp.WithAnalyzers(analyzers).GetAnalyzerDiagnosticsAsync();
            var ndw = all.Where(d => d.Id.StartsWith("NDW", StringComparison.Ordinal))
                         .OrderBy(LineOf).ThenBy(d => d.Id, StringComparer.Ordinal)
                         .ToImmutableArray();
            return new AnalyzerResult(ndw, compileErrors);
        }

        public static Task<AnalyzerResult> RunFileAsync(string fileName)
        {
            var path = FixturePath(fileName);
            return RunAsync(File.ReadAllText(path), path);
        }

        public static int LineOf(Diagnostic d) => d.Location.GetLineSpan().StartLinePosition.Line + 1;

        /// <summary>
        ///     Asserts the analyzers produce EXACTLY the diagnostics tagged in the fixture — every
        ///     <c>// [NDWxxx]</c> tag is matched by a diagnostic on that line, and no diagnostic is
        ///     produced that is not tagged. Both a missing and an unexpected diagnostic fail.
        /// </summary>
        public static async Task AssertExactAsync(string fileName)
        {
            var path = FixturePath(fileName);
            var result = await RunAsync(File.ReadAllText(path), path);

            Assert.IsTrue(result.CompileErrors.IsEmpty,
                $"fixture '{fileName}' does not compile:\n  " +
                string.Join("\n  ", result.CompileErrors.Select(e => e.ToString())));

            var expected = DiagnosticMarkers.Parse(path);
            var actual = result.Ndw.Select(d => new Marker(LineOf(d), d.Id)).ToList();

            var missing = Multiset.Except(expected, actual); // tagged, but the analyzer did not produce
            var extra = Multiset.Except(actual, expected);   // produced, but not tagged

            if (missing.Count == 0 && extra.Count == 0)
                return;

            var sb = new StringBuilder();
            sb.AppendLine($"analyzer output for '{fileName}' did not match the tags:");
            if (missing.Count > 0)
                sb.AppendLine("  MISSING (tagged, not produced): " + string.Join(", ", missing));
            if (extra.Count > 0)
                sb.AppendLine("  UNEXPECTED (produced, not tagged): " + string.Join(", ", extra));
            sb.AppendLine("  full actual: " + (actual.Count == 0 ? "(none)" : string.Join(", ", actual)));
            Assert.Fail(sb.ToString());
        }
    }

    internal readonly struct AnalyzerResult
    {
        public readonly ImmutableArray<Diagnostic> Ndw;
        public readonly ImmutableArray<Diagnostic> CompileErrors;
        public AnalyzerResult(ImmutableArray<Diagnostic> ndw, ImmutableArray<Diagnostic> compileErrors)
        {
            Ndw = ndw;
            CompileErrors = compileErrors;
        }
        public int CountOf(string id) => Ndw.Count(d => d.Id == id);
    }

    internal readonly struct Marker : IEquatable<Marker>
    {
        public readonly int Line;
        public readonly string Code;
        public Marker(int line, string code) { Line = line; Code = code; }
        public bool Equals(Marker other) => Line == other.Line && Code == other.Code;
        public override bool Equals(object obj) => obj is Marker m && Equals(m);
        public override int GetHashCode() => (Line * 397) ^ (Code?.GetHashCode() ?? 0);
        public override string ToString() => $"{Code}@L{Line}";
    }

    /// <summary>Parses the <c>// [NDWxxx]</c> (or <c>// [NDWxxx, NDWyyy]</c>) tags from a fixture file.</summary>
    internal static class DiagnosticMarkers
    {
        private static readonly Regex Tag =
            new(@"\[\s*(NDW\d{3}(?:\s*,\s*NDW\d{3})*)\s*\]", RegexOptions.Compiled);

        public static List<Marker> Parse(string path)
        {
            var markers = new List<Marker>();
            var lines = File.ReadAllLines(path);
            for (int i = 0; i < lines.Length; i++)
            {
                // Only scan the comment tail, so a tag can never be confused with real code.
                int comment = lines[i].IndexOf("//", StringComparison.Ordinal);
                if (comment < 0)
                    continue;
                foreach (Match m in Tag.Matches(lines[i].Substring(comment)))
                    foreach (var code in m.Groups[1].Value.Split(','))
                        markers.Add(new Marker(i + 1, code.Trim()));
            }
            return markers;
        }
    }

    /// <summary>Multiset difference (so a line legitimately carrying two diagnostics is handled).</summary>
    internal static class Multiset
    {
        public static List<Marker> Except(IEnumerable<Marker> from, IEnumerable<Marker> remove)
        {
            var counts = new Dictionary<Marker, int>();
            foreach (var m in remove)
                counts[m] = counts.TryGetValue(m, out var c) ? c + 1 : 1;
            var result = new List<Marker>();
            foreach (var m in from)
            {
                if (counts.TryGetValue(m, out var c) && c > 0)
                    counts[m] = c - 1;
                else
                    result.Add(m);
            }
            return result;
        }
    }
}
