using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     NDW016 — a type that STORES NDArrays but is not disposable. The exact-match gate pins every
    ///     tagged type declaration; the content facts pin the holder vocabulary (fields, auto-properties,
    ///     arrays, tuples, collections, generics, carriers, contagious types) and every exemption
    ///     (static, computed, delegates, comparers, weak refs, unconstrained generics, <c>[NDBorrowed]</c>,
    ///     carriers, disposables, NumSharp's own borrowed iterator types).
    /// </summary>
    [TestClass]
    public class HolderTypeTests
    {
        [TestMethod]
        public Task HolderTypeScenarios_MatchTagsExactly() => AnalyzerTestHarness.AssertExactAsync("HolderTypeScenarios.cs");

        [TestMethod]
        public async Task StoringTypes_AreFlagged()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("HolderTypeScenarios.cs", "NDW016");
            FixtureFacts.AnyStartsWith(flagged, "public class FieldHolder", "a plain NDArray field");
            FixtureFacts.AnyStartsWith(flagged, "public class AutoPropertyHolder", "an auto-property");
            FixtureFacts.AnyStartsWith(flagged, "public class InitOnlyPropertyHolder", "an init-only auto-property");
            FixtureFacts.AnyStartsWith(flagged, "public class ArrayHolder", "an NDArray[]");
            FixtureFacts.AnyStartsWith(flagged, "public class RankTwoArrayHolder", "a rank-2 array");
            FixtureFacts.AnyStartsWith(flagged, "public class ListHolder", "a List<NDArray>");
            FixtureFacts.AnyStartsWith(flagged, "public class DictionaryHolder", "a Dictionary<string, NDArray>");
            FixtureFacts.AnyStartsWith(flagged, "public class NestedGenericHolder", "a List<(string, NDArray)>");
            FixtureFacts.AnyStartsWith(flagged, "public class TupleHolder", "a tuple with an NDArray component");
            FixtureFacts.AnyStartsWith(flagged, "public class CarrierFieldHolder", "an INDArrayCarrier field");
            FixtureFacts.AnyStartsWith(flagged, "public class LazyHolder", "a Lazy<NDArray>");
            FixtureFacts.AnyStartsWith(flagged, "public class TaskHolder", "a Task<NDArray>");
            FixtureFacts.AnyStartsWith(flagged, "public class TypedSubclassHolder", "an NDArray<double>");
            FixtureFacts.AnyStartsWith(flagged, "public class ConstrainedGenericHolder<T>", "a T : NDArray");
            FixtureFacts.AnyStartsWith(flagged, "public class ConsumerBoxHolder", "a consumer's Box<NDArray>");
            FixtureFacts.AnyStartsWith(flagged, "public record PositionalRecord", "a positional record");
            FixtureFacts.AnyStartsWith(flagged, "public struct PlainStruct", "a plain struct");
            FixtureFacts.AnyStartsWith(flagged, "public ref struct RefStructWithoutDispose", "a ref struct without Dispose()");
            FixtureFacts.AnyStartsWith(flagged, "public abstract class AbstractHolder", "an abstract class");
            FixtureFacts.AnyStartsWith(flagged, "public class HoldsDisposableOwner", "a contagious disposable member");
            FixtureFacts.AnyStartsWith(flagged, "public class HoldsOwnerList", "a List of contagious disposables");
            FixtureFacts.AnyStartsWith(flagged, "public class HoldsNonDisposableHolder", "a member whose type stores NDArrays");
            FixtureFacts.AnyStartsWith(flagged, "public class NumSharpIteratorHolder", "NumSharp's np.nditer iterator");
            FixtureFacts.AnyStartsWith(flagged, "public class NpzArchiveHolder", "NumSharp's NpzFile");
            FixtureFacts.AnyStartsWith(flagged, "public class CycleWithArrayA", "a cyclic type holding through its partner");
            FixtureFacts.AnyStartsWith(flagged, "public class CycleWithArrayB", "the partner that stores the array");
        }

        [TestMethod]
        public async Task ExemptTypes_AreNotFlagged()
        {
            var flagged = await FixtureFacts.FlaggedLineTexts("HolderTypeScenarios.cs", "NDW016");
            FixtureFacts.NoneContains(flagged, "StaticOnly", "static-only storage");
            FixtureFacts.NoneContains(flagged, "InstanceStaticMix", "a static member beside scalar instance members");
            FixtureFacts.NoneContains(flagged, "ComputedPropertyOnly", "a computed property");
            FixtureFacts.NoneContains(flagged, "class BorrowedMember", "an [NDBorrowed] member");
            FixtureFacts.NoneContains(flagged, "class BorrowedType", "an [NDBorrowed] type");
            FixtureFacts.NoneContains(flagged, "DelegateHolder", "delegate members");
            FixtureFacts.NoneContains(flagged, "ComparerHolder", "comparer members");
            FixtureFacts.NoneContains(flagged, "WeakHolder", "a weak reference");
            FixtureFacts.NoneContains(flagged, "ObserverHolder", "observer / progress members");
            FixtureFacts.NoneContains(flagged, "UnconstrainedGeneric", "an unconstrained type parameter");
            FixtureFacts.NoneContains(flagged, "ObjectHolder", "object / IDisposable members");
            FixtureFacts.NoneContains(flagged, "ResultCarrier", "an INDArrayCarrier struct");
            FixtureFacts.NoneContains(flagged, "class DisposableOwner", "a disposable that disposes");
            FixtureFacts.NoneContains(flagged, "AsyncDisposableOwner", "an IAsyncDisposable");
            FixtureFacts.NoneContains(flagged, "RefStructWithDispose", "a ref struct with the Dispose() pattern");
            FixtureFacts.NoneContains(flagged, "HoldsBorrowedDisposable", "a member of an [NDBorrowed] disposable type");
            FixtureFacts.NoneContains(flagged, "HoldsFlatIterator", "NumSharp's [NDBorrowed] FlatIterator");
            FixtureFacts.NoneContains(flagged, "HoldsScalarsOnly", "scalar members");
            FixtureFacts.NoneContains(flagged, "DerivedNoOwnHolders", "a derived type adding no holders");
            FixtureFacts.NoneContains(flagged, "class Node", "a self-referential type without arrays");
            FixtureFacts.NoneContains(flagged, "class CycleA", "a cycle without arrays");
            FixtureFacts.NoneContains(flagged, "class CycleB", "a cycle without arrays");
            FixtureFacts.NoneContains(flagged, "IHasArray", "an interface");
            FixtureFacts.NoneContains(flagged, "class Box<T>", "a generic container definition");
        }

        [TestMethod]
        public async Task Message_NamesTheStoringMembers_AndCountsTheRest()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("HolderTypeScenarios.cs");
            var many = r.Ndw.Single(d => d.Id == "NDW016" && d.GetMessage().StartsWith("'ManyHolders'"));
            StringAssert.Contains(many.GetMessage(), "'_one', '_two', '_three' and 2 more", "five holders list three and count two");

            var one = r.Ndw.Single(d => d.Id == "NDW016" && d.GetMessage().StartsWith("'FieldHolder'"));
            StringAssert.Contains(one.GetMessage(), "stores NDArrays in '_a'", "a single holder is named plainly");
            StringAssert.Contains(one.GetMessage(), "[NDBorrowed]", "the message points at the opt-out");

            var record = r.Ndw.Single(d => d.Id == "NDW016" && d.GetMessage().StartsWith("'PositionalRecord'"));
            StringAssert.Contains(record.GetMessage(), "'Data'", "a positional record names its property, not the backing field");
        }

        [TestMethod]
        public async Task Message_GivesTheStructAndRefStructHints()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("HolderTypeScenarios.cs");
            var s = r.Ndw.Single(d => d.Id == "NDW016" && d.GetMessage().StartsWith("'PlainStruct'"));
            StringAssert.Contains(s.GetMessage(), "INDArrayCarrier", "a struct is offered the carrier vocabulary");

            var rs = r.Ndw.Single(d => d.Id == "NDW016" && d.GetMessage().StartsWith("'RefStructWithoutDispose'"));
            StringAssert.Contains(rs.GetMessage(), "public void Dispose()", "a ref struct is offered the Dispose() pattern");

            var c = r.Ndw.Single(d => d.Id == "NDW016" && d.GetMessage().StartsWith("'FieldHolder'"));
            Assert.IsFalse(c.GetMessage().Contains("INDArrayCarrier"), "a class gets no struct hint");
        }

        [TestMethod]
        public async Task HolderTypeFixture_DrawsNoLeakOrGateDiagnostics()
        {
            // The fixture is about TYPES: its bodies must not leak (NDW012) nor trip the gate.
            var r = await AnalyzerTestHarness.RunFileAsync("HolderTypeScenarios.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "fixture must compile");
            Assert.AreEqual(0, r.CountOf("NDW012"), "no method body in the holder fixture leaks");
            Assert.AreEqual(0, r.CountOf("NDW017"), "every disposable in the holder fixture disposes");
            Assert.IsTrue(r.Ndw.All(d => d.Id == "NDW016"), "only NDW016 is expected here");
        }
    }
}
