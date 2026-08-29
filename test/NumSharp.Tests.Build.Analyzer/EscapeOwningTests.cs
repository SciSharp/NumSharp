using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN §3.2 (escape / consume nuances) and §3.3 (owning-expression detection). Every
    ///     legitimate egress stays clean; view reads (property/indexer) are excluded; fresh constructions
    ///     and operator chains leak.
    /// </summary>
    [TestClass]
    public class EscapeOwningTests
    {
        [TestMethod]
        public Task EscapeScenarios_MatchTagsExactly()
            => AnalyzerTestHarness.AssertExactAsync("EscapeScenarios.cs");

        [TestMethod]
        public Task OwningScenarios_MatchTagsExactly()
            => AnalyzerTestHarness.AssertExactAsync("OwningScenarios.cs");

        [TestMethod]
        public async Task EveryEgress_IsClean()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("EscapeScenarios.cs");
            FixtureFacts.NoneContains(flagged, "Foo(ref t)", "a ref-argument write-back");
            FixtureFacts.NoneContains(flagged, "FooOut(out t)", "an out-argument write-back");
            FixtureFacts.NoneContains(flagged, "FooIn(in t)", "an in-argument hand-off");
            FixtureFacts.NoneContains(flagged, "Bar(a + b, c - d)", "params-array elements");
            FixtureFacts.NoneContains(flagged, "list.Add(a + b)", "a collection add");
            FixtureFacts.NoneContains(flagged, "new List<NDArray> { a + b }", "a collection initializer");
            FixtureFacts.NoneContains(flagged, "map[k] = a + b", "an indexer store");
            FixtureFacts.NoneContains(flagged, "Action f = () => Use(t)", "a lambda capture");
            FixtureFacts.NoneContains(flagged, "NDArray L() => a + b", "a local function that returns");
            FixtureFacts.NoneContains(flagged, "yield return a + b", "an iterator yield");
            FixtureFacts.NoneContains(flagged, "await ComputeAsync(a + b)", "an awaited argument");
            FixtureFacts.NoneContains(flagged, "Console.WriteLine(a + b)", "an observed value");
            FixtureFacts.NoneContains(flagged, "return (a + b, c - d)", "a returned tuple");
            FixtureFacts.NoneContains(flagged, "r = a + b;", "a ref-local store");
        }

        [TestMethod]
        public async Task EveryStoreEgress_IsClean()
        {
            // The store family: property setters (static/instance/initializer), fields on other
            // objects, array elements, NumSharp's own indexer-set, base-ctor args, void NumSharp ops.
            var flagged = await FixtureFacts.FlaggedLineTexts("EscapeScenarios.cs");
            FixtureFacts.NoneContains(flagged, "StaticProp = a + b", "a static property store");
            FixtureFacts.NoneContains(flagged, "h.Prop = a + b", "an instance property store");
            FixtureFacts.NoneContains(flagged, "h.Field = a + b", "an instance field store");
            FixtureFacts.NoneContains(flagged, "new Holder { Prop = a + b }", "an object-initializer store");
            FixtureFacts.NoneContains(flagged, "arr[0] = a + b", "an array-element store");
            FixtureFacts.NoneContains(flagged, "a[\"1:3\"] = b + c", "a NumSharp indexer-set");
            FixtureFacts.NoneContains(flagged, "_refField += a", "a compound assignment into a field");
            FixtureFacts.NoneContains(flagged, "np.copyto(dst, a + b)", "an argument to a void NumSharp op");
            FixtureFacts.NoneContains(flagged, "Take(t)", "an argument to a local function");
            FixtureFacts.NoneContains(flagged, "$\"{a + b}\"", "a string interpolation");
            FixtureFacts.NoneContains(flagged, "(a + b)?.reshape", "a conditional-access receiver");
            FixtureFacts.NoneContains(flagged, ": base(a + b)", "a base-constructor argument");
        }

        [TestMethod]
        public async Task EscapeAnchor_Warns()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("EscapeScenarios.cs");
            FixtureFacts.AnyStartsWith(flagged, "np.add(a, b);", "the deliberate leak anchor");
        }

        [TestMethod]
        public async Task OwningExpressions_LeakButViewsDoNot()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("OwningScenarios.cs");
            // producers leak
            FixtureFacts.AnyStartsWith(flagged, "return a + b + c;", "the inner temp of a chained operator");
            FixtureFacts.AnyStartsWith(flagged, "var t = NDArray.Scalar(5.0);", "a dropped constructed scalar");
            FixtureFacts.AnyStartsWith(flagged, "var t = (a & b).MakeGeneric<bool>();", "a dropped MakeGeneric result");
            // views / write-backs do not
            FixtureFacts.NoneContains(flagged, "var t = a.T;", "a `.T` property view");
            FixtureFacts.NoneContains(flagged, "var t = a[\"1:3\"];", "an indexer view");
            FixtureFacts.NoneContains(flagged, "a += b;", "a compound assignment onto the input");
            FixtureFacts.NoneContains(flagged, "NDArray t = 5;", "an implicit scalar conversion");
        }
    }
}
