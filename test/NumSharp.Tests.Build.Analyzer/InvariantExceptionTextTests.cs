using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
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
    ///     NDW020, the rule that keeps NumSharp's exception texts culture-independent (parity QA-41): what it flags inside
    ///     an exception message (interpolation holes, <c>+</c> operands, <c>ToString()</c>, <c>string.Format</c> /
    ///     <c>Join</c>, <c>StringBuilder</c> appends, an <c>ArgumentOutOfRangeException</c> actualValue), how far it follows
    ///     the text (locals, <c>: base(...)</c> calls, the methods a message calls, transitively, and the
    ///     <c>ToString()</c> overrides it formats), what it accepts (a provider chosen by <c>string.Create</c>,
    ///     <c>FormattableString.Invariant</c>, an explicit <c>CultureInfo.InvariantCulture</c>), and that a consumer never
    ///     sees it (off by default; NumSharp.Core turns it on in its <c>.editorconfig</c>).
    /// </summary>
    [TestClass]
    public class InvariantExceptionTextTests
    {
        private const string Usings =
            "using System;\nusing System.Collections.Generic;\nusing System.Globalization;\nusing System.Linq;\nusing System.Text;\n";

        /// <summary>
        ///     Compiles <paramref name="body"/> (appended to <see cref="Usings"/>) and runs the NDW020 analyzer alone.
        /// </summary>
        /// <param name="body">The fixture's declarations.</param>
        /// <param name="enabled">True to turn NDW020 on as an error (NumSharp.Core's <c>.editorconfig</c>); false leaves
        ///     the rule's default, which is off.</param>
        /// <returns>The NDW diagnostics, ordered by line, and the fixture's compile errors.</returns>
        /// <exception cref="AssertFailedException">The fixture does not compile, or the analyzer threw (AD0001).</exception>
        private static async Task<AnalyzerResult> Run(string body, bool enabled = true)
        {
            var tree = CSharpSyntaxTree.ParseText(SourceText.From(Usings + body), path: "ndw020.cs");
            var options = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true);
            if (enabled)
                options = options.WithSpecificDiagnosticOptions(new Dictionary<string, ReportDiagnostic> { ["NDW020"] = ReportDiagnostic.Error });
            var comp = CSharpCompilation.Create("Ndw020Fixture", new[] { tree }, AnalyzerTestHarness.References, options);
            var compileErrors = comp.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).ToImmutableArray();
            Assert.IsTrue(compileErrors.IsEmpty, "fixture must compile:\n" + string.Join("\n", compileErrors));
            var all = await comp.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new InvariantExceptionTextAnalyzer()))
                                .GetAnalyzerDiagnosticsAsync();
            var crashes = all.Where(d => d.Id == "AD0001").ToList();
            Assert.AreEqual(0, crashes.Count, "the analyzer threw:\n" + string.Join("\n", crashes.Select(d => d.GetMessage())));
            var ndw = all.Where(d => d.Id.StartsWith("NDW", StringComparison.Ordinal))
                         .OrderBy(AnalyzerTestHarness.LineOf).ThenBy(d => d.Location.SourceSpan.Start).ToImmutableArray();
            return new AnalyzerResult(ndw, compileErrors);
        }

        /// <summary>The line of <paramref name="marker"/> in a fixture body, counted the way diagnostics are (1-based, usings included).</summary>
        /// <param name="body">The fixture body.</param>
        /// <param name="marker">A text that occurs once in the body.</param>
        /// <returns>The 1-based line.</returns>
        /// <exception cref="AssertFailedException">The marker does not occur exactly once.</exception>
        private static int LineOf(string body, string marker)
        {
            string text = Usings + body;
            int at = text.IndexOf(marker, StringComparison.Ordinal);
            Assert.IsTrue(at >= 0 && text.IndexOf(marker, at + 1, StringComparison.Ordinal) < 0, $"marker '{marker}' must occur once");
            return text.Substring(0, at).Count(c => c == '\n') + 1;
        }

        /// <summary>Asserts the NDW020 diagnostics sit exactly on the lines of <paramref name="markers"/>, one each.</summary>
        /// <param name="result">The analyzer run.</param>
        /// <param name="body">The fixture body the markers are found in.</param>
        /// <param name="markers">One text per expected diagnostic.</param>
        /// <exception cref="AssertFailedException">A diagnostic is missing, extra, or on another line.</exception>
        private static void AssertLines(AnalyzerResult result, string body, params string[] markers)
        {
            var expected = markers.Select(m => LineOf(body, m)).OrderBy(l => l).ToArray();
            var actual = result.Ndw.Where(d => d.Id == "NDW020").Select(AnalyzerTestHarness.LineOf).OrderBy(l => l).ToArray();
            CollectionAssert.AreEqual(expected, actual,
                "NDW020 lines: expected [" + string.Join(", ", expected) + "], got [" + string.Join(", ", actual) + "]:\n  "
                + string.Join("\n  ", result.Ndw.Select(d => AnalyzerTestHarness.LineOf(d) + ": " + d.GetMessage())));
        }

        // ---------------------------------------------------------------- the descriptors

        /// <summary>
        ///     The two NDW020 descriptors: one id, off by default (the analyzer ships in the NumSharp package), an
        ///     error where NumSharp turns it on, the help anchor, the ordered fixes with the sentence that blocks the
        ///     wrong one, a one-line message within the build-line budget, and the compilation-end tag on the producer
        ///     text (RS1037).
        /// </summary>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public void Descriptors_AreOffByDefault_ErrorsWhenOn_AndCarryTheFixProcedure()
        {
            var descriptors = new InvariantExceptionTextAnalyzer().SupportedDiagnostics;
            Assert.AreEqual(2, descriptors.Length, "a direct text and a compilation-end text");
            foreach (var d in descriptors)
            {
                Assert.AreEqual("NDW020", d.Id);
                Assert.IsFalse(d.IsEnabledByDefault, "off by default: the analyzer ships to consumers, whose texts are theirs");
                Assert.AreEqual(DiagnosticSeverity.Error, d.DefaultSeverity, "an error where NumSharp turns it on");
                Assert.IsTrue(d.HelpLinkUri.EndsWith("exceptions.html#culture-independent-texts", StringComparison.Ordinal));
                string message = d.MessageFormat.ToString();
                StringAssert.Contains(message, "string.Create(CultureInfo.InvariantCulture, $\"...\")", "fix 1, the mechanical one");
                StringAssert.Contains(message, "ToString(CultureInfo.InvariantCulture)", "fix 3");
                StringAssert.Contains(message, "Never change CultureInfo.CurrentCulture", "blocks the wrong fix");
                Assert.IsTrue(message.Length <= 760, $"a build line, not a page ({message.Length} chars)");
                Assert.IsFalse(message.Contains('\n'), "one build-log line");
                Assert.IsTrue(message.EndsWith(".", StringComparison.Ordinal), "a multi-sentence message ends with a period (RS1032)");
                Assert.IsTrue(d.Description.ToString().EndsWith(".", StringComparison.Ordinal), "RS1033");
            }
            Assert.IsTrue(descriptors.Single(d => d.MessageFormat.ToString().Contains("{1}")).CustomTags.Contains(WellKnownDiagnosticTags.CompilationEnd),
                "the producer text is reported from the compilation-end action (RS1037)");
        }

        /// <summary>
        ///     Without the <c>.editorconfig</c> line NumSharp.Core carries, a flagged construct draws nothing: a
        ///     consumer's exception texts are not NumSharp's business.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task OffByDefault_AConsumerNeverSeesIt()
        {
            const string body = "public static class C { public static void F(int n) => throw new ArgumentException($\"bad {n}\"); }\n";
            var r = await Run(body, enabled: false);
            Assert.AreEqual(0, r.CountOf("NDW020"), "without NumSharp.Core's .editorconfig line the rule never runs");
        }

        // ---------------------------------------------------------------- interpolation holes

        /// <summary>
        ///     A hole of a number (signed, unsigned, floating, nullable), <c>object</c> or a type parameter is
        ///     flagged; text types (string, char, bool, an enum, <c>Type</c>, NumSharp's <c>DType</c> / <c>Shape</c>)
        ///     are not, because their text does not depend on the culture or the provider cannot change it.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task Holes_OfANumberOrAnUnknownType_AreFlagged_TextTypesAreNot()
        {
            const string body =
                "public enum Mode { A }\n" +
                "public static class C {\n" +
                "  public static Exception I(int n) => new InvalidOperationException($\"int {n}\");\n" +
                "  public static Exception L(long n) => new ArgumentException($\"long {n}\");\n" +
                "  public static Exception D(double x) => new ArgumentException($\"double {x:F2}\");\n" +
                "  public static Exception U(ulong x) => new ArgumentException($\"ulong {x}\");\n" +
                "  public static Exception O(object o) => new ArgumentException($\"object {o}\");\n" +
                "  public static Exception G<T>(T v) => new ArgumentException($\"generic {v}\");\n" +
                "  public static Exception N(int? v) => new ArgumentException($\"nullable {v}\");\n" +
                "  public static Exception Text(string s, char c, bool b, Mode m, Type t) => new ArgumentException($\"{s} {c} {b} {m} {t}\");\n" +
                "  public static Exception Nd(NumSharp.DType dt, NumSharp.Shape sh) => new ArgumentException($\"{dt} {sh}\");\n" +
                "}\n";
            var r = await Run(body);
            AssertLines(r, body, "int {n}", "long {n}", "double {x:F2}", "ulong {x}", "object {o}", "generic {v}", "nullable {v}");
            StringAssert.Contains(r.Ndw[0].GetMessage(), "the interpolation {n} (int) formats a number with the current culture inside an exception message");
        }

        /// <summary>
        ///     <c>string.Create(CultureInfo.InvariantCulture, ...)</c> (one string or a <c>+</c> chain of them) and
        ///     <c>FormattableString.Invariant</c> chose their culture; a <c>$"..."</c> nested in a hole of such a
        ///     string does not inherit the provider, so it is still flagged.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task AChosenProvider_IsAccepted_ButANestedInterpolationIsStillChecked()
        {
            const string body =
                "public static class C {\n" +
                "  public static Exception Create(int n) => new ArgumentException(string.Create(CultureInfo.InvariantCulture, $\"ok {n}\"));\n" +
                "  public static Exception Chain(int n, int m) => new ArgumentException(string.Create(CultureInfo.InvariantCulture, $\"a {n} \" + $\"b {m}\"));\n" +
                "  public static Exception Invariant(int n) => new ArgumentException(FormattableString.Invariant($\"ok {n}\"));\n" +
                "  public static Exception Nested(int n, bool b) => new ArgumentException(string.Create(CultureInfo.InvariantCulture, $\"x {(b ? $\"inner {n}\" : \"\")}\"));\n" +
                "}\n";
            var r = await Run(body);
            AssertLines(r, body, "inner {n}");
        }

        // ---------------------------------------------------------------- concatenation and formatting calls

        /// <summary>
        ///     A <c>+</c> operand, <c>ToString(format)</c>, <c>Convert.ToString(object)</c>, <c>string.Format</c> and
        ///     <c>string.Join</c> over numbers are flagged; the same calls given <c>CultureInfo.InvariantCulture</c>,
        ///     <c>Convert.ToString(int, toBase)</c>, a char or bool operand, and a <c>(object[])</c> cast of a
        ///     <c>Type[]</c> (whose elements stay types) are not.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task Concatenation_ToString_Format_Join_AreFlagged_WithoutAProvider()
        {
            const string body =
                "public static class C {\n" +
                "  public static Exception Plus(int n) => new ArgumentException(\"plus \" + n);\n" +
                "  public static Exception Str(double x) => new ArgumentException(\"str \" + x.ToString(\"R\"));\n" +
                "  public static Exception Conv(object o) => new ArgumentException(\"conv \" + Convert.ToString(o));\n" +
                "  public static Exception Fmt(long n) => new ArgumentException(string.Format(\"fmt {0}\", n));\n" +
                "  public static Exception Join(long[] dims) => new ArgumentException(\"join (\" + string.Join(\", \", dims) + \")\");\n" +
                "  public static Exception Fine(long n, long[] dims, double x, int b) => new ArgumentException(\n" +
                "      \"fine \" + n.ToString(CultureInfo.InvariantCulture) + x.ToString(\"R\", CultureInfo.InvariantCulture)\n" +
                "      + string.Format(CultureInfo.InvariantCulture, \"{0}\", n) + Convert.ToString(b, 16)\n" +
                "      + string.Join(\", \", Array.ConvertAll(dims, d => d.ToString(CultureInfo.InvariantCulture))) + 'c' + true);\n" +
                "  public static Exception Types(Type[] types) => new ArgumentException(\"types \" + string.Join(\", \", (object[])types));\n" +
                "}\n";
            var r = await Run(body);
            AssertLines(r, body, "\"plus \"", "\"str \"", "\"conv \"", "\"fmt {0}\"", "\"join (\"");
        }

        /// <summary>
        ///     The text a lambda builds inside a message (<c>Select(d =&gt; d.ToString())</c> under
        ///     <c>string.Join</c>) is part of the message, so its provider-less <c>ToString()</c> is flagged.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task ALambdaInsideAMessage_IsPartOfTheMessage()
        {
            const string body =
                "public static class C {\n" +
                "  public static Exception F(long[] dims) => new ArgumentException(\"(\" + string.Join(\",\", dims.Select(d => d.ToString())) + \")\");\n" +
                "}\n";
            var r = await Run(body);
            AssertLines(r, body, "dims.Select");
            StringAssert.Contains(r.Ndw.Single().GetMessage(), "'d.ToString()' (long) without a format provider");
        }

        // ---------------------------------------------------------------- locals

        /// <summary>
        ///     A message read from a local is followed to the local's initializer, assignments and <c>+=</c>, and a
        ///     <c>StringBuilder</c> local to its appends down a fluent chain (a provider-less <c>Append($"...")</c>
        ///     handler included); a local that never reaches a message is not.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task Locals_AreFollowedToTheirWrites_AndStringBuilderAppends()
        {
            const string body =
                "public static class C {\n" +
                "  public static void Local(int n) { var msg = $\"local {n}\"; throw new ArgumentException(msg); }\n" +
                "  public static void Assigned(int n, bool b) { string msg = \"x\"; if (b) msg = $\"assigned {n}\"; throw new ArgumentException(msg); }\n" +
                "  public static void Compound(int n) { string msg = \"x\"; msg += n; throw new ArgumentException(msg); }\n" +
                "  public static void Builder(long[] dims) {\n" +
                "    var sb = new StringBuilder(\"(\");\n" +
                "    for (int i = 0; i < dims.Length; i++) sb.Append(dims[i]).Append(',');\n" +
                "    sb.Append($\"handler {dims.Length}\");\n" +
                "    sb.Append(CultureInfo.InvariantCulture, $\"chosen {dims.Length}\");\n" +
                "    sb.Append(dims[0].ToString(CultureInfo.InvariantCulture));\n" +
                "    throw new ArgumentException(sb.ToString());\n" +
                "  }\n" +
                "  public static string NotAMessage(int n) { var s = $\"free {n}\"; return s; }\n" +
                "}\n";
            var r = await Run(body);
            AssertLines(r, body, "local {n}", "assigned {n}", "msg += n", "sb.Append(dims[i])", "handler {dims.Length}");
        }

        // ---------------------------------------------------------------- base calls and actualValue

        /// <summary>
        ///     A <c>: base(...)</c> call of an exception constructor is a message too, and a number passed as an
        ///     <c>ArgumentOutOfRangeException</c> actualValue is flagged (the BCL appends it with the current
        ///     culture); invariant text in that slot is accepted.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task ABaseCall_AndAnActualValue_AreMessages()
        {
            const string body =
                "public class MyError : ArgumentOutOfRangeException {\n" +
                "  public MyError(int axis) : base(\"axis\", axis, $\"axis {axis} out of range\") { }\n" +
                "  public MyError(long n, bool _) : base(\"n\", n.ToString(CultureInfo.InvariantCulture), \"fine\") { }\n" +
                "}\n" +
                "public static class C {\n" +
                "  public static Exception Range(int width) => new ArgumentOutOfRangeException(nameof(width), width, \"a width\");\n" +
                "  public static Exception RangeFine(int width) => new ArgumentOutOfRangeException(nameof(width), width.ToString(CultureInfo.InvariantCulture), \"a width\");\n" +
                "}\n";
            var r = await Run(body);
            // The base call carries two findings on one line: the actualValue and the interpolation.
            var lines = r.Ndw.Where(d => d.Id == "NDW020").Select(AnalyzerTestHarness.LineOf).ToArray();
            CollectionAssert.AreEqual(new[] { LineOf(body, "base(\"axis\""), LineOf(body, "base(\"axis\""), LineOf(body, "nameof(width), width, ") }, lines,
                string.Join("\n", r.Ndw.Select(d => AnalyzerTestHarness.LineOf(d) + ": " + d.GetMessage())));
            Assert.IsTrue(r.Ndw.Any(d => d.GetMessage().StartsWith("the actualValue width (int), which ArgumentOutOfRangeException appends to its text,", StringComparison.Ordinal)));
        }

        // ---------------------------------------------------------------- producers (compilation end)

        /// <summary>
        ///     A source method whose returned text reaches a message (directly or through another method) is checked
        ///     at the compilation end and named in the text; an equal method no message calls is not flagged.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task AMethodAMessageCalls_IsChecked_Transitively_AndNamed()
        {
            const string body =
                "public static class C {\n" +
                "  private static string Inner(long n) => $\"inner {n}\";\n" +
                "  private static string Outer(long n) => \"outer \" + Inner(n);\n" +
                "  private static string Direct(int n) { var sb = new StringBuilder(); sb.Append(n); return sb.ToString(); }\n" +
                "  private static string Unused(int n) => $\"unused {n}\";\n" +
                "  public static Exception F(long n) => new ArgumentException(Outer(n) + Direct((int)n));\n" +
                "}\n";
            var r = await Run(body);
            AssertLines(r, body, "inner {n}", "sb.Append(n)");
            var inner = r.Ndw.First(d => d.GetMessage().Contains("inner {n}") || d.GetMessage().Contains("{n} (long)"));
            StringAssert.Contains(inner.GetMessage(), "in 'C.Inner', whose text reaches an exception message");
        }

        /// <summary>
        ///     A source type's <c>ToString()</c> override runs when a message formats a value of that type, so its
        ///     text is checked; the same override of a type no message formats is not flagged.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task AToStringOverride_AMessageFormats_IsChecked()
        {
            const string body =
                "public sealed class Slice { public int Step; public override string ToString() => $\"::{Step}\"; }\n" +
                "public sealed class Quiet { public int Step; public override string ToString() => $\"quiet {Step}\"; }\n" +
                "public static class C {\n" +
                "  public static Exception F(Slice s) => new ArgumentException($\"bad slice {s}\");\n" +
                "  public static string G(Quiet q) => $\"{q}\";\n" +
                "}\n";
            var r = await Run(body);
            AssertLines(r, body, "::{Step}");
            StringAssert.Contains(r.Ndw.Single().GetMessage(), "in 'Slice.ToString'");
        }

        /// <summary>
        ///     Text that never reaches an exception (a label, console output, a list item) is outside the rule.
        /// </summary>
        /// <returns>The asynchronous test.</returns>
        /// <exception cref="AssertFailedException">The analyzer's answer differs from the one described.</exception>
        [TestMethod]
        public async Task TextOutsideAnExceptionMessage_IsNotFlagged()
        {
            const string body =
                "public static class C {\n" +
                "  public static string Label(int n) => $\"label {n}\";\n" +
                "  public static void Print(int n) => Console.WriteLine($\"print {n}\");\n" +
                "  public static List<string> Items(int n) => new List<string> { $\"item {n}\" };\n" +
                "}\n";
            var r = await Run(body);
            Assert.AreEqual(0, r.CountOf("NDW020"), string.Join("\n", r.Ndw.Select(d => d.GetMessage())));
        }
    }
}
