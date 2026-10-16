using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Build.Analyzer;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Pins the analyzer CONTRACT — most importantly that a leak (NDW012) is reported as a
    ///     <b>Warning</b>, never an Error. A leak is a performance nudge (the finalizer still reclaims
    ///     the buffer), so it must never break a build; the target-gate diagnostics, by contrast, ARE
    ///     errors because a mis-woven method would silently misbehave.
    /// </summary>
    [TestClass]
    public class AnalyzerContractTests
    {
        /// <summary>
        ///     Both NDW012 descriptors (the dropped-NDArray text and the dropped-holder text) are default-on
        ///     warnings that link the NDW012 anchor: a leak is a nudge, never a build-breaker.
        /// </summary>
        [TestMethod]
        public void Ndw012_DefaultSeverity_IsWarning_And_EnabledByDefault()
        {
            var ds = new NDArrayLeakAnalyzer().SupportedDiagnostics.Where(x => x.Id == "NDW012").ToList();
            Assert.AreEqual(2, ds.Count, "one NDW012 text for a dropped NDArray, one for a dropped holder");
            foreach (var d in ds)
            {
                Assert.AreEqual(DiagnosticSeverity.Warning, d.DefaultSeverity,
                    $"NDW012 '{d.Title}' must be a WARNING, never an Error — a leak is a nudge, not a build-breaker");
                Assert.IsTrue(d.IsEnabledByDefault, $"NDW012 '{d.Title}' must be enabled by default");
                Assert.IsTrue(d.HelpLinkUri.EndsWith("#ndw012"), $"NDW012 '{d.Title}' links its anchor on the build-compiler page");
            }
        }

        /// <summary>
        ///     Pins the fix procedure the NDW012 texts carry, because coding agents act on the build-log line
        ///     alone. The dropped-NDArray text must: say what the scope attribute disposes; forbid adding it as a
        ///     reflex; give the false-positive exit; prefer <c>[NDScopedCovered]</c> when every call is scoped,
        ///     with the ways a search for callers goes wrong (delegates, interfaces, base virtuals, threads, an
        ///     await); name every shape where a scope disposes a live array or stops reclaiming one (an array kept
        ///     past the call, a static cache, a holder inside a tuple, a pointer, a long loop, a lambda or task
        ///     that runs elsewhere or later, a constructor); and end in explicit disposal that never touches a
        ///     returned, stored, given or passthrough array. The holder text must say that no scope attribute
        ///     fixes it. Wording may change; these facts may not.
        /// </summary>
        [TestMethod]
        public void Ndw012_Messages_CarryTheFixProcedure()
        {
            string array = NDArrayLeakAnalyzer.Nw012.MessageFormat.ToString();
            foreach (var fact in new[]
                     {
                         "cannot see a scope that a caller opened",    // why a covered helper is flagged
                         "except those it returns, yields, or assigns to out parameters", // what a scope keeps
                         "Do not add {0} just to silence this warning", // no reflex attribute
                         "act on the first case that holds",           // an ordered procedure
                         "an @out argument that the call returns",     // false positive: a call returning its out array
                         "a view or copy of one does not count",       // a view is a new array
                         "#pragma warning disable NDW012",             // the false-positive exit
                         "with a comment that gives the reason",
                         "silences every NDW012 in it",                // a mark is a whole-method decision
                         "a constructor cannot take them",             // CS0592: Method/Property-only attributes
                         "can run many times",                         // a scope held across a long loop
                         "here or in a caller",                        // a caller's loop holds a covered helper's arrays
                         "runs on another thread or after that scope closes", // arrays no scope tracks
                         "count calls through a delegate, interface, or base virtual", // callers a search misses
                         "on the calling thread",
                         "creates no array after an await",            // a covered async helper
                         "nothing outside this assembly can call it",  // [NDScopedCovered] needs every caller known
                         "mark it [NDScopedCovered], which costs nothing at run time, not {0}", // no nested scope
                         "static cache",                               // a callee's cache is tracked by the scope
                         "NDScope.Detach",                             // the legal way to keep an array
                         "an [NDScopedExit] parameter",
                         "an INDArrayCarrier struct, a tuple of those", // the return shapes a scope yields
                         "even inside a tuple",                        // the weaver's silent tuple gap
                         "no pointer into a temporary",
                         "returns, yields, or assigns to out parameters is an NDArray", // iterators yield, too
                         "callers [NDScoped] ([NDScopedAsync] for an async one) under these same conditions", // each caller its own kind
                         "'using var t = ...;'",                       // the fallback
                         "before the next pass",                       // per-pass arrays
                         "never an array the method returns, stores, or was given",
                         "np.asarray(x)",                              // never dispose a possible passthrough
                     })
                StringAssert.Contains(array, fact, $"the NDW012 NDArray text must keep: {fact}");
            Assert.IsFalse(array.Contains('\n') || array.Contains('\r'), "one build-log line");
            Assert.IsTrue(array.EndsWith("."), "a multi-sentence message ends with a period (RS1032)");

            string holder = NDArrayLeakAnalyzer.Nw012Holder.MessageFormat.ToString();
            foreach (var fact in new[]
                     {
                         "never calls this object's Dispose",
                         "neither [NDScoped] nor [NDScopedCovered] fixes this",
                         "act on the first case that holds",
                         "#pragma warning disable NDW012",
                         "'.Close()'",
                         "hold the collection in a 'using' first",
                     })
                StringAssert.Contains(holder, fact, $"the NDW012 holder text must keep: {fact}");
            Assert.IsFalse(holder.Contains('\n') || holder.Contains('\r'), "one build-log line");
            Assert.IsTrue(holder.EndsWith("."), "a multi-sentence message ends with a period (RS1032)");
        }

        /// <summary>
        ///     Each leak gets the text that fits it: a dropped NDArray in a synchronous method names
        ///     <c>[NDScoped]</c>, one in an async method names <c>[NDScopedAsync]</c> and never the synchronous
        ///     attribute (while the advice for its callers names each kind, because a caller's kind is its own),
        ///     and a dropped holder gets the holder text, which offers no scope attribute as a fix.
        /// </summary>
        [TestMethod]
        public async Task Ndw012_PicksTheTextThatFitsTheValue()
        {
            const string src = "using NumSharp;\nusing System.Threading.Tasks;\n" +
                               "public sealed class Batch : System.IDisposable {\n" +
                               "  private NDArray _a;\n" +
                               "  public Batch(NDArray a) { _a = a; }\n" +
                               "  public void Dispose() => _a.Dispose();\n}\n" +
                               "public static class T {\n" +
                               "  public static void Sync(NDArray a, NDArray b) { var t = a + b; }\n" +
                               "  public static async Task Async(NDArray a, NDArray b) { var t = a + b; await Task.Yield(); }\n" +
                               "  public static void Holder(NDArray a) { var batch = new Batch(a); }\n}";
            var r = await AnalyzerTestHarness.RunAsync(src, "ndw012_texts.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inline source must compile");

            var leaks = r.Ndw.Where(x => x.Id == "NDW012").OrderBy(AnalyzerTestHarness.LineOf).ToList();
            Assert.AreEqual(3, leaks.Count, "one leak per method");

            string sync = leaks[0].GetMessage(), async = leaks[1].GetMessage(), holder = leaks[2].GetMessage();
            Assert.AreSame(NDArrayLeakAnalyzer.Nw012, leaks[0].Descriptor);
            StringAssert.Contains(sync, "This NDArray is never disposed", "the value is named");
            StringAssert.Contains(sync, "mark it [NDScoped],", "a synchronous method is offered [NDScoped]");

            Assert.AreSame(NDArrayLeakAnalyzer.Nw012, leaks[1].Descriptor);
            StringAssert.Contains(async, "Do not add [NDScopedAsync] just to silence this warning", "the warning names the async attribute");
            StringAssert.Contains(async, "mark it [NDScopedAsync],", "an async method is offered [NDScopedAsync]");
            Assert.IsFalse(async.Contains("mark it [NDScoped],") || async.Contains("Do not add [NDScoped] "),
                "an async method is never offered the synchronous attribute");
            // The callers' attribute is named per caller, never copied from this method's slot: a synchronous
            // caller of an async method takes [NDScoped] (the weaver rejects the wrong one, NDW009/NDW010).
            StringAssert.Contains(async, "mark its unscoped callers [NDScoped] ([NDScopedAsync] for an async one)");
            StringAssert.Contains(sync, "mark its unscoped callers [NDScoped] ([NDScopedAsync] for an async one)");

            Assert.AreSame(NDArrayLeakAnalyzer.Nw012Holder, leaks[2].Descriptor);
            StringAssert.Contains(holder, "This 'Batch' instance (a disposable that stores NDArrays) is never disposed");
            StringAssert.Contains(holder, "neither [NDScoped] nor [NDScopedCovered] fixes this");
        }

        [TestMethod]
        public async Task Ndw012_OnARealLeak_IsReportedAsWarning_NotError()
        {
            const string src = "using NumSharp;\npublic static class T {\n" +
                               "  public static void M(NDArray a, NDArray b) { var t = a + b; }\n}";
            var r = await AnalyzerTestHarness.RunAsync(src, "severity_leak.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inline source must compile");

            var leak = r.Ndw.Single(x => x.Id == "NDW012");
            Assert.AreEqual(DiagnosticSeverity.Warning, leak.Severity,
                "a detected leak must surface as a Warning");
            Assert.AreEqual(0, r.Ndw.Count(x => x.Severity == DiagnosticSeverity.Error),
                "no diagnostic produced for a leak may be an Error");
        }

        [DataTestMethod]
        [DataRow("NDW016")]
        [DataRow("NDW017")]
        public void OwnershipDiagnostics_DefaultSeverity_IsWarning_And_EnabledByDefault(string id)
        {
            // The type-level ownership diagnostics are nudges of the same kind as NDW012: a stored
            // array that is never disposed still reaches the finalizer, so they must never break a build.
            var d = new NDArrayHolderAnalyzer().SupportedDiagnostics.Single(x => x.Id == id);
            Assert.AreEqual(DiagnosticSeverity.Warning, d.DefaultSeverity, $"{id} must be a WARNING, never an Error");
            Assert.IsTrue(d.IsEnabledByDefault, $"{id} must be enabled by default");
            Assert.IsTrue(d.HelpLinkUri.EndsWith("#" + id.ToLowerInvariant()), $"{id} links its own anchor on the build-compiler page");
        }

        [TestMethod]
        public async Task OwnershipDiagnostics_OnRealCases_AreReportedAsWarnings_NotErrors()
        {
            const string src = "using NumSharp;\n" +
                               "public class Stores { NDArray _a; }\n" +
                               "public class Forgets : System.IDisposable { NDArray _a; public void Dispose() { } }";
            var r = await AnalyzerTestHarness.RunAsync(src, "severity_ownership.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inline source must compile");
            Assert.AreEqual(DiagnosticSeverity.Warning, r.Ndw.Single(x => x.Id == "NDW016").Severity);
            Assert.AreEqual(DiagnosticSeverity.Warning, r.Ndw.Single(x => x.Id == "NDW017").Severity);
            Assert.AreEqual(0, r.Ndw.Count(x => x.Severity == DiagnosticSeverity.Error), "no ownership finding may be an Error");
        }

        [TestMethod]
        public void TheThreeAnalyzers_CoverDisjointDiagnosticIds()
        {
            // Distinct: one id may carry several descriptors (NDW012's two texts, NDW013's three).
            var leak = new NDArrayLeakAnalyzer().SupportedDiagnostics.Select(d => d.Id).Distinct().ToArray();
            var gate = new NDScopedTargetAnalyzer().SupportedDiagnostics.Select(d => d.Id).Distinct().ToArray();
            var holder = new NDArrayHolderAnalyzer().SupportedDiagnostics.Select(d => d.Id).Distinct().ToArray();
            CollectionAssert.AreEquivalent(new[] { "NDW012" }, leak);
            CollectionAssert.AreEquivalent(new[] { "NDW016", "NDW017" }, holder);
            Assert.AreEqual(0, leak.Intersect(gate).Count() + leak.Intersect(holder).Count() + gate.Intersect(holder).Count(),
                "no id is owned by two analyzers");
        }

        [DataTestMethod]
        [DataRow("NDW002")]
        [DataRow("NDW003")]
        [DataRow("NDW005")]
        [DataRow("NDW006")]
        [DataRow("NDW009")]
        [DataRow("NDW010")]
        [DataRow("NDW011")]
        public void GateDiagnostics_AreErrors_ByDesign(string id)
        {
            // The target gate is a different contract: a wrong/unsupported [NDScoped] target is a
            // hard error (the same one the weaver emits). This pins the leak/gate severity split.
            var d = new NDScopedTargetAnalyzer().SupportedDiagnostics.Single(x => x.Id == id);
            Assert.AreEqual(DiagnosticSeverity.Error, d.DefaultSeverity,
                $"{id} is a target-gate error, distinct from the NDW012 leak warning");
        }

        [TestMethod]
        public void Ndw013_WeaverMissing_IsAWarning_LikeItsMsBuildTwin()
        {
            // The analyzer's NDW013 — an own [NDScoped]/[NDScopedAsync], an inherited one, or an
            // [NDScopedExit] parameter, while the build says the weaver is inactive — shares its id and
            // severity with the fallback MSBuild warning in NumSharp's build/NumSharp.targets: a nudge
            // about inert attributes, never a build error. Three descriptors carry the one id (three
            // sentences for three shapes); every one of them is a default-on warning.
            var ds = new NDScopedTargetAnalyzer().SupportedDiagnostics.Where(x => x.Id == "NDW013").ToList();
            Assert.AreEqual(3, ds.Count, "own attribute / inherited attribute / [NDScopedExit] — one NDW013 descriptor each");
            foreach (var d in ds)
            {
                Assert.AreEqual(DiagnosticSeverity.Warning, d.DefaultSeverity, $"NDW013 '{d.Title}' is a warning");
                Assert.IsTrue(d.IsEnabledByDefault, $"NDW013 '{d.Title}' is on by default");
                StringAssert.Contains(d.MessageFormat.ToString(), "dotnet add package NumSharp.Build", $"NDW013 '{d.Title}' names the fix");
            }
        }
    }
}
