using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The overload-resolution proof of API-parity batch 6's axis overloads, against the REAL NumSharp members. The owner
    ///     asked to keep the <c>int</c> and <c>int[]</c> overloads beside every <c>NDAxis?</c> axis ("Can you prove that they
    ///     will resolve correctly before taking that"): each spelling a user can write is compiled at C# 12 (what a net8.0
    ///     consumer compiles with; OverloadResolutionPriority does not exist there) and the overload C# binds is read from the
    ///     semantic model - its axis parameter's type is asserted.
    /// </summary>
    /// <remarks>
    ///     The rules the test pins (proven first on mimics, scratchpad probe_axis): a function whose NumPy axis takes a tuple
    ///     has <c>int axis</c> and <c>int[] axis</c> overloads WITHOUT defaults beside <c>NDAxis? axis = null</c>; an int binds
    ///     <c>int</c>, an <c>int[]</c> and a bare <c>null</c> bind <c>int[]</c>, everything else (tuples, NDAxis values,
    ///     <c>int?</c> variables, an omitted axis) binds <c>NDAxis?</c>. A one-integer axis (argmax / argmin / nanargmax /
    ///     nanargmin) has an <c>int?</c> overload instead, which <c>null</c> binds. <c>f(a, default)</c> of a three-overload
    ///     function is a compile error (CS0121) - never a silent answer.
    /// </remarks>
    [TestClass]
    public class AxisOverloadResolutionTests
    {
        /// <summary>The declarations every case can use: arrays, axis values of each C# kind, a reduction delegate.</summary>
        private const string Prelude = @"
using System;
using NumSharp;
/// <summary>The compiled proof snippet: one assignment per case, never run.</summary>
public static class Calls
{
    /// <summary>Holds the cases' calls; only the semantic model reads them.</summary>
    public static void Run()
    {
        NDArray a = np.arange(24.0).reshape(2, 3, 4);
        NDArray q = 50.0;
        NDCharArray c = np.@char.array(new[] { ""b"", ""a"" });
        Func<NDArray, int, NDArray> f = (x, i) => np.sum(x, i, keepdims: true);
        int ax = 1;
        short sh = 1;
        int? nax = 1;
        int[] arr = { 0, 2 };
        NDAxis ndx = (0, 2);
        object sink;
";

        /// <summary>
        ///     Every spelling and the axis-parameter type its bound overload must have (a parameter named <c>axis</c>,
        ///     <c>axes</c> or <c>source</c>).
        /// </summary>
        private static readonly (string Call, string Expected)[] Cases =
        {
            // np.sum's shape (the axis second, keywords after it) - and every reduction like it.
            ("np.sum(a)", "NDAxis?"), ("np.sum(a, 1)", "int"), ("np.sum(a, -1)", "int"), ("np.sum(a, ax)", "int"),
            ("np.sum(a, sh)", "int"), ("np.sum(a, nax)", "NDAxis?"), ("np.sum(a, null)", "int[]"),
            ("np.sum(a, axis: null)", "int[]"), ("np.sum(a, new[] { 0, 2 })", "int[]"), ("np.sum(a, arr)", "int[]"),
            ("np.sum(a, (0, 2))", "NDAxis?"), ("np.sum(a, ndx)", "NDAxis?"), ("np.sum(a, new NDAxis())", "NDAxis?"),
            ("np.sum(a, 1, keepdims: true)", "int"), ("np.sum(a, keepdims: true)", "NDAxis?"),
            ("np.sum(a, null, null, null, true)", "int[]"), ("np.sum(a, 0, null, null, true)", "int"),
            ("np.prod(a, 1)", "int"), ("np.mean(a, (0, 2))", "NDAxis?"), ("np.var(a, 1, ddof: 1)", "int"),
            ("np.std(a, new[] { 0, 2 })", "int[]"), ("np.max(a, 1)", "int"), ("np.amax(a, null)", "int[]"),
            ("np.min(a, (0, 2))", "NDAxis?"), ("np.amin(a, 1)", "int"), ("np.ptp(a, 1)", "int"),
            ("np.nansum(a, 1)", "int"), ("np.nanprod(a, null)", "int[]"), ("np.nanmean(a, (0, 2))", "NDAxis?"),
            ("np.nanvar(a, 1)", "int"), ("np.nanstd(a, new[] { 1 })", "int[]"), ("np.nanmax(a, 1)", "int"),
            ("np.nanmin(a, (0, 1))", "NDAxis?"), ("np.median(a, 1)", "int"), ("np.nanmedian(a, null)", "int[]"),
            ("np.percentile(a, q, 1)", "int"), ("np.nanpercentile(a, q, (0, 2))", "NDAxis?"),
            ("np.quantile(a, 0.5, new[] { 0, 2 })", "int[]"), ("np.nanquantile(a, 0.5, null)", "int[]"),
            ("np.average(a, 1)", "int"), ("np.average_returned(a, (0, 2))", "NDAxis?"),
            ("np.all(a, 1)", "int"), ("np.any(a, null)", "int[]"), ("np.count_nonzero(a, 1)", "int"),
            ("np.count_nonzero(a, (0, 2))", "NDAxis?"),
            // Manipulation.
            ("np.flip(a, 1)", "int"), ("np.flip(a)", "NDAxis?"), ("np.squeeze(a, 0)", "int"), ("np.squeeze(a, null)", "int[]"),
            ("np.expand_dims(a, 0)", "int"), ("np.expand_dims(a, (0, 2))", "NDAxis"), ("np.expand_dims(a, new[] { 0 })", "int[]"),
            ("np.moveaxis(a, 0, -1)", "int"), ("np.moveaxis(a, new[] { 0 }, new[] { -1 })", "int[]"),
            ("np.moveaxis(a, (0, 1), (1, 0))", "NDAxis"), ("np.moveaxis(a, 0, new[] { 1 })", "NDAxis"),
            ("np.roll(a, 1, 0)", "int"), ("np.roll(a, 1, (0, 1))", "NDAxis?"), ("np.roll(a, 1)", "NDAxis?"),
            ("np.trim_zeros(a, \"fb\", 0)", "int"), ("np.trim_zeros(a)", "NDAxis?"),
            ("np.size(a, 1)", "int"), ("np.size(a)", "NDAxis?"),
            ("np.gradient(a, null, 0)", "int"), ("np.gradient(a, axis: 0)", "NDAxis?"), ("np.gradient(a, 2.0, (0, 1))", "NDAxis?"),
            // ndarray methods.
            ("a.sum(1)", "int"), ("a.sum(null)", "int[]"), ("a.sum((0, 2))", "NDAxis?"), ("a.sum(keepdims: true)", "NDAxis?"),
            ("a.mean(1)", "int"), ("a.max(new[] { 0, 2 })", "int[]"), ("a.all(1)", "int"), ("a.squeeze(0)", "int"),
            // One-integer axes: the kept overload is int?, so null and default stay None.
            ("np.argmax(a, 1)", "int?"), ("np.argmax(a, null)", "int?"), ("np.argmax(a, nax)", "int?"),
            ("np.argmax(a, (0, 1))", "NDAxis?"), ("np.argmax(a, keepdims: true)", "NDAxis?"), ("np.argmin(a, 1)", "int?"),
            ("np.nanargmax(a, 1)", "int?"), ("np.nanargmax(a, default)", "int?"), ("np.nanargmin(a, (0, 1))", "NDAxis?"),
            ("a.argmax(1)", "int?"), ("a.argmin(null)", "int?"), ("c.argmax(0)", "int?"), ("c.argmin((0, 1))", "NDAxis?"),
            // Batch 6's own functions.
            ("np.fft.fftshift(a, 1)", "int"), ("np.fft.fftshift(a, null)", "int[]"), ("np.fft.fftshift(a, (0, 1))", "NDAxis?"),
            ("np.fft.ifftshift(a)", "NDAxis?"), ("np.linalg.norm(a, null, 1)", "int"), ("np.linalg.norm(a, null, null)", "int[]"),
            ("np.linalg.norm(a, ord: null, axis: null, keepdims: false)", "int[]"), ("np.linalg.norm(a, null, (0, 2))", "NDAxis?"),
            ("np.linalg.norm(a, axis: 1)", "NDAxis?"), ("np.linalg.vector_norm(a, 1)", "int"),
            ("np.linalg.vector_norm(a, (0, 2))", "NDAxis?"), ("np.linalg.vector_norm(a)", "NDAxis?"),
            ("np.apply_over_axes(f, a, 1)", "int"), ("np.apply_over_axes(f, a, new[] { 0, 2 })", "int[]"),
            ("np.apply_over_axes(f, a, (0, 2))", "NDAxis"),
        };

        /// <summary>The spellings that must NOT bind (each a compile error code).</summary>
        private static readonly (string Call, string ErrorId)[] Refused =
        {
            ("np.sum(a, default)", "CS0121"), ("np.mean(a, axis: default)", "CS0121"), ("np.fft.fftshift(a, default)", "CS0121"),
            ("np.linalg.norm(a, null, default)", "CS0121"), ("np.sum(a, 1L)", "CS1503"),
        };

        /// <summary>
        ///     Compiles every case in one C# 12 compilation against the real NumSharp and checks the bound overload of each
        ///     call: no compile error, and the axis parameter of the bound member has the expected type.
        /// </summary>
        [TestMethod]
        public void EverySpelling_BindsTheDesignedOverload_AtCSharp12()
        {
            var source = Prelude + string.Concat(Cases.Select((c, i) => $"        sink = {c.Call}; // case {i}\n")) +
                         "        GC.KeepAlive(sink);\n    }\n}\n";
            var tree = CSharpSyntaxTree.ParseText(source, new CSharpParseOptions(LanguageVersion.CSharp12));
            var compilation = CSharpCompilation.Create("AxisOverloadProof", new[] { tree }, AnalyzerTestHarness.References,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true));
            var errors = compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
            Assert.AreEqual(0, errors.Count, string.Join("\n", errors.Take(10)));

            var model = compilation.GetSemanticModel(tree);
            var invocations = tree.GetRoot().DescendantNodes().OfType<AssignmentExpressionSyntax>()
                .Where(s => s.Left is IdentifierNameSyntax { Identifier.Text: "sink" })
                .Select(s => (InvocationExpressionSyntax)s.Right)
                .ToList();
            Assert.AreEqual(Cases.Length, invocations.Count);

            var wrong = new List<string>();
            for (int i = 0; i < Cases.Length; i++)
            {
                var method = (IMethodSymbol)model.GetSymbolInfo(invocations[i]).Symbol;
                var axis = method?.Parameters.FirstOrDefault(p => p.Name is "axis" or "axes" or "source");
                string bound = axis is null ? "(no axis parameter)" : Display(axis.Type);
                if (bound != Cases[i].Expected)
                    wrong.Add($"{Cases[i].Call}: bound {bound} ({method}), expected {Cases[i].Expected}");
            }
            Assert.AreEqual(0, wrong.Count, string.Join("\n", wrong));
        }

        /// <summary>
        ///     The spellings the three overloads refuse at compile time: <c>default</c> fits both <c>int</c> and
        ///     <c>int[]</c> (CS0121, never a silent axis 0), and a <c>long</c> axis never converted.
        /// </summary>
        [TestMethod]
        public void RefusedSpellings_AreCompileErrors_AtCSharp12()
        {
            foreach (var (call, errorId) in Refused)
            {
                var source = Prelude + $"        sink = {call};\n        GC.KeepAlive(sink);\n    }}\n}}\n";
                var tree = CSharpSyntaxTree.ParseText(source, new CSharpParseOptions(LanguageVersion.CSharp12));
                var compilation = CSharpCompilation.Create("AxisOverloadRefusal", new[] { tree }, AnalyzerTestHarness.References,
                    new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true));
                var ids = compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).Select(d => d.Id).ToList();
                CollectionAssert.Contains(ids, errorId, call);
            }
        }

        /// <summary>The type as the cases spell it: NumSharp types without their namespace, C# keywords for the rest.</summary>
        /// <param name="type">The parameter type.</param>
        /// <returns>For example <c>int</c>, <c>int[]</c>, <c>int?</c>, <c>NDAxis</c> or <c>NDAxis?</c>.</returns>
        private static string Display(ITypeSymbol type) => type.ToDisplayString().Replace("NumSharp.", "");
    }
}
