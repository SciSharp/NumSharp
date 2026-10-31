using System.Collections.Immutable;
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
        ///     All three NDW012 descriptors (a dropped NDArray with the weaver, the same without it, and a dropped
        ///     holder) are default-on warnings that link the NDW012 anchor: a leak is a nudge, never a build-breaker.
        /// </summary>
        [TestMethod]
        public void Ndw012_DefaultSeverity_IsWarning_And_EnabledByDefault()
        {
            var ds = new NDArrayLeakAnalyzer().SupportedDiagnostics.Where(x => x.Id == "NDW012").ToList();
            Assert.AreEqual(3, ds.Count, "NDW012 texts: a dropped NDArray with and without the weaver, and a dropped holder");
            foreach (var d in ds)
            {
                Assert.AreEqual(DiagnosticSeverity.Warning, d.DefaultSeverity,
                    $"NDW012 '{d.Title}' must be a WARNING, never an Error — a leak is a nudge, not a build-breaker");
                Assert.IsTrue(d.IsEnabledByDefault, $"NDW012 '{d.Title}' must be enabled by default");
                Assert.IsTrue(d.HelpLinkUri.EndsWith("#ndw012"), $"NDW012 '{d.Title}' links its anchor on the build-compiler page");
            }
        }

        /// <summary>
        ///     The longest a message format may be. A build prints the message once per warning site and again in
        ///     its summary, so a long text floods logs (NumSharp.Core has 171 NDW012 sites) and an agent's context;
        ///     the full conditions belong in the description and on the help page, whose link the compiler prints
        ///     after the message. The NDArray text sits just under this cap: it is the shortest that still names
        ///     every fix with the words that block a use-after-free and says that a caller's scope, which the
        ///     analysis cannot see, may already reclaim the array.
        /// </summary>
        private const int Ndw012MessageCap = 680;

        /// <summary>Cap for the holder text, whose fix has a single branch (dispose it, or suppress a hand-off).</summary>
        private const int Ndw012HolderMessageCap = 400;

        /// <summary>
        ///     Pins what each NDW012 tier must carry. The MESSAGE (the build-log line an agent acts on) must claim
        ///     only what a per-method analysis knows (nothing in THIS method reclaims the array; a caller's scope
        ///     may), forbid adding the scope attribute as a reflex and say why, then give the four fixes in order
        ///     with the words that block their misuse: the false-positive exit (a call handing back an array the
        ///     method already holds), <c>[NDScopedCovered]</c> when every caller already has a scope, none runs the
        ///     method in a long loop and it is not public, the scope attribute only when no new array is stored,
        ///     cached, captured or handed out in a holder and no long loop creates arrays, and disposal that spares a
        ///     possible input; and it must stay short. The DESCRIPTION must carry every condition the message
        ///     abbreviates (the per-method limit, detaching, delegate and base-virtual callers, threads and awaits,
        ///     nesting under a caller's scope, static caches, the return shapes a scope yields, the tuple gap,
        ///     pointers, constructors, loops, lambdas and tasks). The holder text must say that no scope attribute
        ///     fixes it. Wording may change; these facts and the caps may not.
        /// </summary>
        [TestMethod]
        public void Ndw012_Texts_CarryTheFixProcedure_AndStayShort()
        {
            string message = NDArrayLeakAnalyzer.Nw012.MessageFormat.ToString();
            foreach (var fact in new[]
                     {
                         "Nothing in this method disposes, returns, stores, or scopes this {1}", // only what it knows
                         "Do not add {0} just to silence this",        // no reflex attribute
                         "it disposes every array the call creates except what the method returns", // why
                         "Take the first that fits",                   // an ordered procedure
                         "a call hands back an array the method already holds (@out, inplace: true)", // false positive
                         "suppress only this line",                    // a line, never a whole file
                         "every caller already has a scope ([NDScoped*] or NDScope.Open())", // the scope it cannot see
                         "none runs it in a long loop",                // a nested [NDScoped] frees per call instead
                         "this method is not public: [NDScopedCovered]",
                         "stored, cached, captured, or handed out in a holder", // use-after-free shapes
                         "no long loop in it creates arrays: {0}",     // a scope frees only when it closes
                         "otherwise dispose it after its last use",    // the fallback
                         "unless it may be an input (np.asarray(x))",  // never dispose a possible passthrough
                     })
                StringAssert.Contains(message, fact, $"the NDW012 message must keep: {fact}");
            // A per-method analysis cannot know that no caller scoped the array: "never ... scoped" is false whenever
            // a caller's scope reclaims it, and sends an agent to add a scope that already exists.
            Assert.IsFalse(message.Contains("never disposed, returned, stored, or scoped"),
                "the message must not claim the array is never scoped: a caller's scope is invisible to this check");
            Assert.IsTrue(message.Length <= Ndw012MessageCap,
                $"the NDW012 message is a build-log line, not a document: {message.Length} > {Ndw012MessageCap} characters; " +
                "move detail to the description and the help page");
            Assert.IsFalse(message.Contains('\n') || message.Contains('\r'), "one build-log line");
            Assert.IsTrue(message.EndsWith("."), "a multi-sentence message ends with a period (RS1032)");

            string description = NDArrayLeakAnalyzer.Nw012.Description.ToString();
            foreach (var fact in new[]
                     {
                         "cannot see a scope a caller opened",         // why a covered helper is flagged
                         "except what it returns, yields, or assigns to out parameters", // what a scope keeps
                         "NDScope.Detach",                             // the legal way to keep an array
                         "not a view or copy",                         // a view is a new array
                         "the callee takes the array over",            // the other false-positive class
                         "with a comment",
                         "counting delegate, interface, and base-virtual calls", // callers a search misses
                         "runs on the calling thread",
                         "no caller runs it in a long loop",           // covered arrays wait for the caller's scope
                         "Scopes nest",                                // [NDScoped] under a caller's scope is safe
                         "frees the method's arrays when it returns and passes its result to the outer scope",
                         "no array after an await",                    // a covered async helper
                         "static cache",                               // a callee's cache is tracked by the scope
                         "an [NDScopedExit] parameter",
                         "INDArrayCarrier structs, tuples of those",   // the return shapes a scope yields
                         "comes back disposed, even inside a tuple",   // the weaver's silent tuple gap
                         "no pointer into a temporary",
                         "its unscoped callers",                       // scoping the callers instead
                         "a constructor",                              // CS0592: Method/Property-only attributes
                         "a long loop that creates arrays",
                         "on another thread or after the scope closed", // arrays no scope tracks
                         "never an array the method returns, stores, or was given",
                         "x.astype(t, copy: false)",                   // another possible passthrough
                     })
                StringAssert.Contains(description, fact, $"the NDW012 description must keep: {fact}");
            Assert.IsTrue(description.EndsWith("."), "a description ends with a period (RS1033)");

            // Without the weaver the scope attributes are inert, so that text never offers one as a fix, counts
            // only scopes opened by hand, makes explicit disposal the fix, and names installing the weaver.
            string noWeaver = NDArrayLeakAnalyzer.Nw012NoWeaver.MessageFormat.ToString();
            foreach (var fact in new[]
                     {
                         "Nothing in this method disposes, returns, stores, or scopes this {1}", // only what it knows
                         "{0} is inert here: this project does not use the NumSharp.Build package", // why no [NDScoped]
                         "Take the first that fits",
                         "a call hands back an array the method already holds (@out, inplace: true)",
                         "suppress only this line",
                         "every call already runs inside a scope opened with NDScope.Open()", // the only scope there
                         "[NDScoped] callers open none here",          // an inert attribute opens no scope
                         "none runs it in a long loop",
                         "this method is not public: [NDScopedCovered]",
                         "(3) otherwise dispose it after its last use", // the best fix that works without the weaver
                         "unless it may be an input (np.asarray(x))",
                         "installing NumSharp.Build lets {0} reclaim them all", // the route to the attribute
                     })
                StringAssert.Contains(noWeaver, fact, $"the NDW012 no-weaver message must keep: {fact}");
            Assert.IsFalse(noWeaver.Contains(": {0};") || noWeaver.Contains("Do not add {0}"),
                "without the weaver the attribute is never offered as a case's fix");
            Assert.IsTrue(noWeaver.Length <= Ndw012MessageCap,
                $"the NDW012 no-weaver message is a build-log line: {noWeaver.Length} > {Ndw012MessageCap} characters");
            Assert.IsFalse(noWeaver.Contains('\n') || noWeaver.Contains('\r'), "one build-log line");
            Assert.IsTrue(noWeaver.EndsWith("."), "a multi-sentence message ends with a period (RS1032)");

            string noWeaverDescription = NDArrayLeakAnalyzer.Nw012NoWeaver.Description.ToString();
            foreach (var fact in new[]
                     {
                         "declares the NumSharp.Build weaver inactive",
                         "inert (NDW013)",
                         "no scope opens in the method or in an attributed caller",
                         "dotnet add package NumSharp.Build",
                         "route every return through scope.Returns(...)", // the hand scope's one rule
                         "a return that skips it hands back a disposed array",
                         "x.astype(t, copy: false)",
                     })
                StringAssert.Contains(noWeaverDescription, fact, $"the NDW012 no-weaver description must keep: {fact}");
            Assert.IsTrue(noWeaverDescription.EndsWith("."), "a description ends with a period (RS1033)");

            string holder = NDArrayLeakAnalyzer.Nw012Holder.MessageFormat.ToString();
            foreach (var fact in new[]
                     {
                         "An NDScope never calls its Dispose",
                         "neither [NDScoped] nor [NDScopedCovered] fixes this",
                         "Close()",
                         "suppress only this line if the method it is passed to disposes or returns it",
                         "hold the collection in a 'using'",
                     })
                StringAssert.Contains(holder, fact, $"the NDW012 holder text must keep: {fact}");
            Assert.IsTrue(holder.Length <= Ndw012HolderMessageCap,
                $"the NDW012 holder message is a build-log line: {holder.Length} > {Ndw012HolderMessageCap} characters");
            Assert.IsFalse(holder.Contains('\n') || holder.Contains('\r'), "one build-log line");
            Assert.IsTrue(holder.EndsWith("."), "a multi-sentence message ends with a period (RS1032)");
        }

        /// <summary>
        ///     Each leak gets the text that fits it: a dropped NDArray in a synchronous method names
        ///     <c>[NDScoped]</c>, one in an async method names <c>[NDScopedAsync]</c> and never the synchronous
        ///     attribute, and a dropped holder gets the holder text, which offers no scope attribute as a fix.
        ///     Every rendered line stays under the cap, the longer async attribute name included.
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
            StringAssert.Contains(sync, "Nothing in this method disposes, returns, stores, or scopes this NDArray.", "the value is named");
            StringAssert.Contains(sync, "Do not add [NDScoped] just to silence this", "the warning names the sync attribute");
            StringAssert.Contains(sync, "creates arrays: [NDScoped];", "a synchronous method is offered [NDScoped]");

            Assert.AreSame(NDArrayLeakAnalyzer.Nw012, leaks[1].Descriptor);
            StringAssert.Contains(async, "Do not add [NDScopedAsync] just to silence this", "the warning names the async attribute");
            StringAssert.Contains(async, "creates arrays: [NDScopedAsync];", "an async method is offered [NDScopedAsync]");
            Assert.IsFalse(async.Contains("arrays: [NDScoped];") || async.Contains("Do not add [NDScoped] "),
                "an async method is never offered the synchronous attribute");

            Assert.AreSame(NDArrayLeakAnalyzer.Nw012Holder, leaks[2].Descriptor);
            StringAssert.Contains(holder, "This 'Batch' instance (a disposable that stores NDArrays) is never disposed");
            StringAssert.Contains(holder, "neither [NDScoped] nor [NDScopedCovered] fixes this");

            // The rendered lines, not just the formats, stay short: [NDScopedAsync] fills two slots.
            foreach (var line in new[] { sync, async })
                Assert.IsTrue(line.Length <= Ndw012MessageCap + 40, $"rendered NDW012 line too long: {line.Length}");
            Assert.IsTrue(holder.Length <= Ndw012HolderMessageCap + 60, $"rendered holder line too long: {holder.Length}");
        }

        /// <summary>
        ///     The NDArray text follows the weaver state the build declares, through the property NDW013 reads:
        ///     <c>NumSharpBuildActive</c> present and not <c>true</c> (NumSharp's targets default it to
        ///     <c>false</c>) selects the text that never offers the inert scope attribute; <c>true</c> (set by the
        ///     weaver's targets) or an absent property (unknown) keeps the text that offers it. A property declared
        ///     but empty is not <c>true</c>, so it reads as inactive, as NDW013 reads it; case and surrounding
        ///     spaces do not matter. A dropped holder keeps its own text in every state, since no scope attribute
        ///     reclaims it either way.
        /// </summary>
        /// <param name="active">The declared <c>NumSharpBuildActive</c> value, or null for an absent property.</param>
        /// <param name="expectNoWeaver">Whether the no-weaver text must be chosen for the dropped NDArrays.</param>
        /// <returns>The asynchronous test run.</returns>
        [DataTestMethod]
        [DataRow("false", true)]
        [DataRow("False", true)]
        [DataRow("", true)]
        [DataRow("true", false)]
        [DataRow(" TRUE ", false)]
        [DataRow(null, false)]
        public async Task Ndw012_FollowsTheDeclaredWeaverState(string active, bool expectNoWeaver)
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
            var options = active is null ? null : AnalyzerTestHarness.BuildProperties(("NumSharpBuildActive", active));
            var r = await AnalyzerTestHarness.RunAsync(src, "ndw012_weaver_state.cs", ImmutableArray<MetadataReference>.Empty, options);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "inline source must compile");

            var leaks = r.Ndw.Where(x => x.Id == "NDW012").OrderBy(AnalyzerTestHarness.LineOf).ToList();
            Assert.AreEqual(3, leaks.Count, "one leak per method");
            var expected = expectNoWeaver ? NDArrayLeakAnalyzer.Nw012NoWeaver : NDArrayLeakAnalyzer.Nw012;
            Assert.AreSame(expected, leaks[0].Descriptor, $"sync leak, NumSharpBuildActive='{active ?? "<absent>"}'");
            Assert.AreSame(expected, leaks[1].Descriptor, $"async leak, NumSharpBuildActive='{active ?? "<absent>"}'");
            Assert.AreSame(NDArrayLeakAnalyzer.Nw012Holder, leaks[2].Descriptor, "a holder keeps the holder text");

            if (expectNoWeaver)
            {
                // Each method still names the attribute that WOULD fit it, as the inert one.
                StringAssert.Contains(leaks[0].GetMessage(), "[NDScoped] is inert here");
                StringAssert.Contains(leaks[1].GetMessage(), "[NDScopedAsync] is inert here");
                Assert.IsFalse(leaks[0].GetMessage().Contains("creates arrays: [NDScoped];"),
                    "without the weaver no case offers the attribute as its fix");
            }
            else
            {
                StringAssert.Contains(leaks[0].GetMessage(), "creates arrays: [NDScoped];");
                StringAssert.Contains(leaks[1].GetMessage(), "creates arrays: [NDScopedAsync];");
            }
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
            // The analyzer's NDW013 — an own [NDScoped]/[NDScopedAsync], an inherited one, an
            // [NDScopedExit] parameter, or an [NDAnchored] method, while the build says the weaver is
            // inactive — shares its id and severity with the fallback MSBuild warning in NumSharp's
            // build/NumSharp.targets: a nudge about inert attributes, never a build error. Four descriptors
            // carry the one id (four sentences for four shapes); every one of them is a default-on warning.
            var ds = new NDScopedTargetAnalyzer().SupportedDiagnostics.Where(x => x.Id == "NDW013").ToList();
            Assert.AreEqual(4, ds.Count, "own attribute / inherited attribute / [NDScopedExit] / [NDAnchored] — one NDW013 descriptor each");
            foreach (var d in ds)
            {
                Assert.AreEqual(DiagnosticSeverity.Warning, d.DefaultSeverity, $"NDW013 '{d.Title}' is a warning");
                Assert.IsTrue(d.IsEnabledByDefault, $"NDW013 '{d.Title}' is on by default");
                StringAssert.Contains(d.MessageFormat.ToString(), "dotnet add package NumSharp.Build", $"NDW013 '{d.Title}' names the fix");
            }
        }
    }
}
