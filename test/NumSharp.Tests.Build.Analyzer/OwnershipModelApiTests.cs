using System;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Build.Analyzer;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     <see cref="OwnershipModel"/>'s own surface, asked directly rather than through diagnostics: the
    ///     null and degenerate inputs, every kind of type the walk distinguishes (arrays of every rank,
    ///     tuples, pointers, dynamic, delegates, enums, the non-holding generic families, BCL and source
    ///     generics, carriers, borrowed types, source interfaces, type parameters by constraint), constructed
    ///     generics against their definitions, and the shape of <see cref="OwnershipModel.HolderMembersOf"/>'s
    ///     answer (a fresh list, declaration order, field versus property). The expectations are the
    ///     classification rules of the model's type summary.
    ///     <para>
    ///     Master's model gives the same answers on the type kinds, the type parameters, the constructed
    ///     generics and the member order, so those tests pin rules, not a new behavior. It fails three tests
    ///     here, each for a defect the fix removed: null inputs throw (a NullReferenceException; PR #632's
    ///     commit throws an ArgumentNullException), and two tests ask whether a type holds BEFORE its members
    ///     are classified — master then reuses that whole-graph answer and reports a back-reference as a
    ///     holder (DerivedOnly's self-reference, B2's link back to A2).
    ///     </para>
    /// </summary>
    [TestClass]
    public class OwnershipModelApiTests
    {
        /// <summary>
        ///     The declarations every test queries. <c>Probe</c> holds one field per type kind (named by the
        ///     <see cref="TypeKinds"/> table), <c>Gen</c> one field per constrained type parameter; the rest are
        ///     the source types those fields name.
        /// </summary>
        private const string Source =
            "using System;\nusing System.Collections.Generic;\nusing System.Threading.Tasks;\nusing NumSharp;\n" +
            "public class Box<T> { public T Value; }\n" +
            "public struct Carrier : INDArrayCarrier { public NDArray A; public void YieldTo(NDScope scope) { } }\n" +
            "[NDBorrowed] public class Borrowed { NDArray _a; }\n" +
            "public interface ISourceIface { NDArray Value { get; } }\n" +
            "public enum Color { Red }\n" +
            "public delegate void Handler(NDArray a);\n" +
            "public unsafe class Probe {\n" +
            "  NDArray f01; NDArray[] f02; NDArray[,] f03; NDArray[][] f04; List<NDArray> f05; Dictionary<string, NDArray> f06;\n" +
            "  (int, NDArray) f07; (int, string) f08; int* f09; dynamic f10; object f11; string f12; Func<NDArray> f13; Action<NDArray> f14;\n" +
            "  IComparer<NDArray> f15; IEqualityComparer<NDArray> f16; WeakReference<NDArray> f17; IObservable<NDArray> f18; Lazy<NDArray> f19;\n" +
            "  Task<NDArray> f20; KeyValuePair<string, NDArray> f21; int? f22; (NDArray, int)? f23; IEnumerable<NDArray> f24;\n" +
            "  NumSharp.Generic.NDArray<double> f25; Box<NDArray> f26; Box<int> f27; Carrier f28; Borrowed f29; ISourceIface f30; Color f31;\n" +
            "  Handler f32; Box<Box<NDArray>> f33; ArraySegment<NDArray> f34; IProgress<NDArray> f35; Comparer<NDArray> f36;\n" +
            "  EqualityComparer<NDArray> f37; IComparable<NDArray> f38; IEquatable<NDArray> f39; IObserver<NDArray> f40; Progress<NDArray> f41;\n" +
            "#nullable enable\n  NDArray? f42;\n#nullable restore\n" +
            "}\n" +
            "public class Gen<TFree, TArr, TClass, TDisp, TList, TStruct> where TArr : NDArray where TClass : class where TDisp : IDisposable where TList : List<NDArray> where TStruct : struct\n" +
            "  { TFree g1; TArr g2; TClass g3; TDisp g4; TList g5; TStruct g6; }\n" +
            "public class Holder { NDArray _c; Holder _self; NDArray[] _a; public NDArray Prop { get; set; } public NDArray Computed => _c; List<NDArray> _b; [NDBorrowed] NDArray _borrowed; static NDArray s_static; }\n" +
            "public class BaseHolder { NDArray _x; }\npublic class DerivedOnly : BaseHolder { DerivedOnly _self; int _n; }\n" +
            "public class A2 : IDisposable { B2 _b; public void Dispose() { } }\npublic class B2 { A2 _a; NDArray _x; }\n" +
            "public class C2 : IDisposable { D2 _d; public void Dispose() { } }\npublic class D2 { C2 _c; }\n";

        /// <summary>
        ///     Every field of <c>Probe</c> with the answer the rules give for its type: arrays of any rank,
        ///     tuples, nullable value types and BCL/source generics hold when an element or a type argument
        ///     does; pointers, dynamic, object, string, delegates, enums, borrowed types, source interfaces
        ///     and the non-holding generic families (comparers, equality, observers, progress, weak
        ///     references) never do; NDArray, its generic subclass and a carrier hold by themselves.
        /// </summary>
        private static readonly (string Field, string Type, bool Holds)[] TypeKinds =
        {
            ("f01", "NDArray", true), ("f02", "NDArray[]", true), ("f03", "NDArray[,]", true), ("f04", "NDArray[][]", true),
            ("f05", "List<NDArray>", true), ("f06", "Dictionary<string, NDArray>", true), ("f07", "(int, NDArray)", true),
            ("f08", "(int, string)", false), ("f09", "int*", false), ("f10", "dynamic", false), ("f11", "object", false),
            ("f12", "string", false), ("f13", "Func<NDArray>", false), ("f14", "Action<NDArray>", false),
            ("f15", "IComparer<NDArray>", false), ("f16", "IEqualityComparer<NDArray>", false), ("f17", "WeakReference<NDArray>", false),
            ("f18", "IObservable<NDArray>", false), ("f19", "Lazy<NDArray>", true), ("f20", "Task<NDArray>", true),
            ("f21", "KeyValuePair<string, NDArray>", true), ("f22", "int?", false), ("f23", "(NDArray, int)?", true),
            ("f24", "IEnumerable<NDArray>", true), ("f25", "NDArray<double>", true), ("f26", "Box<NDArray>", true),
            ("f27", "Box<int>", false), ("f28", "Carrier", true), ("f29", "Borrowed", false), ("f30", "ISourceIface", false),
            ("f31", "Color", false), ("f32", "Handler", false), ("f33", "Box<Box<NDArray>>", true), ("f34", "ArraySegment<NDArray>", true),
            ("f35", "IProgress<NDArray>", false), ("f36", "Comparer<NDArray>", false), ("f37", "EqualityComparer<NDArray>", false),
            ("f38", "IComparable<NDArray>", false), ("f39", "IEquatable<NDArray>", false), ("f40", "IObserver<NDArray>", false),
            ("f41", "Progress<NDArray>", false), ("f42", "NDArray? (annotated)", true),
        };

        /// <summary>Compiles <see cref="Source"/> (unsafe allowed, for the pointer field) and builds a fresh model over it.</summary>
        /// <returns>The compilation and the model.</returns>
        /// <exception cref="AssertFailedException">The source does not compile, or NumSharp's types cannot be resolved.</exception>
        private static (CSharpCompilation comp, OwnershipModel model) Compile()
        {
            var tree = CSharpSyntaxTree.ParseText(SourceText.From(Source), path: "model_api.cs");
            var comp = CSharpCompilation.Create("ModelApi", new[] { tree }, AnalyzerTestHarness.References,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true));
            var errors = comp.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).ToList();
            Assert.AreEqual(0, errors.Count, "the probe source must compile:\n" + string.Join("\n", errors));
            var known = KnownTypes.Resolve(comp);
            Assert.IsNotNull(known, "NumSharp must resolve");
            return (comp, new OwnershipModel(known, comp));
        }

        /// <summary>The type of field <paramref name="field"/> of source type <paramref name="type"/>.</summary>
        /// <param name="comp">The compilation.</param>
        /// <param name="type">The declaring type's metadata name.</param>
        /// <param name="field">The field's name.</param>
        /// <returns>The field's type.</returns>
        /// <exception cref="InvalidOperationException">The type or the field does not exist.</exception>
        private static ITypeSymbol FieldType(CSharpCompilation comp, string type, string field)
            => comp.GetTypeByMetadataName(type).GetMembers(field).OfType<IFieldSymbol>().Single().Type;

        /// <summary>Null inputs answer "holds nothing" and an empty list (never null, never a throw), and a type that stores no array has no holder.</summary>
        [TestMethod]
        public void NullAndArrayFreeQueries_AnswerFalseAndEmpty()
        {
            var (comp, model) = Compile();
            Assert.IsFalse(model.HoldsNDArrays(null));
            var none = model.HolderMembersOf(null);
            Assert.IsNotNull(none);
            Assert.AreEqual(0, none.Count);
            Assert.AreEqual(0, model.HolderMembersOf(comp.GetTypeByMetadataName("D2")).Count, "D2's only member is a C2, which holds nothing");
            Assert.IsFalse(model.HoldsNDArrays(comp.GetTypeByMetadataName("C2")));
        }

        /// <summary>Every row of <see cref="TypeKinds"/>, asked in table order and then in reverse on a second model: the answers are the rules' and do not depend on what was asked first.</summary>
        [TestMethod]
        public void TypeKinds_HoldAsTheRulesSay_InAnyQueryOrder()
        {
            var (comp, model) = Compile();
            foreach (var (field, type, holds) in TypeKinds)
                Assert.AreEqual(holds, model.HoldsNDArrays(FieldType(comp, "Probe", field)), $"{field}: {type}");

            var (comp2, reversed) = Compile();
            // Enumerable.Reverse, not the array's Reverse(): newer C# binds an array's .Reverse() to the
            // in-place span extension (void), which would also reorder the shared table.
            foreach (var (field, type, holds) in Enumerable.Reverse(TypeKinds))
                Assert.AreEqual(holds, reversed.HoldsNDArrays(FieldType(comp2, "Probe", field)), $"{field}: {type} (reverse order)");
        }

        /// <summary>A type parameter holds exactly when a constraint does: NDArray or List&lt;NDArray&gt; yes; none, class, struct or IDisposable no.</summary>
        [TestMethod]
        public void TypeParameters_HoldThroughTheirConstraints()
        {
            var (comp, model) = Compile();
            var expected = new[] { ("g1", false), ("g2", true), ("g3", false), ("g4", false), ("g5", true), ("g6", false) };
            foreach (var (field, holds) in expected)
                Assert.AreEqual(holds, model.HoldsNDArrays(FieldType(comp, "Gen`6", field)), field);
        }

        /// <summary>
        ///     A constructed generic answers for its arguments and lists its substituted members — Box&lt;NDArray&gt;
        ///     holds through Value, Box&lt;int&gt; holds nothing — while the definition Box&lt;T&gt; holds nothing.
        /// </summary>
        [TestMethod]
        public void ConstructedGenerics_AnswerForTheirArguments_TheDefinitionForNothing()
        {
            var (comp, model) = Compile();
            var box = comp.GetTypeByMetadataName("Box`1");
            var ndarray = comp.GetTypeByMetadataName("NumSharp.NDArray");
            var boxOfArray = box.Construct(ndarray);
            var boxOfInt = box.Construct(comp.GetSpecialType(SpecialType.System_Int32));
            Assert.IsTrue(model.HoldsNDArrays(boxOfArray));
            CollectionAssert.AreEqual(new[] { "Value" }, model.HolderMembersOf(boxOfArray).Select(h => h.Name).ToList());
            Assert.IsFalse(model.HoldsNDArrays(boxOfInt));
            Assert.AreEqual(0, model.HolderMembersOf(boxOfInt).Count);
            Assert.IsFalse(model.HoldsNDArrays(box), "an unconstrained T cannot know");
            Assert.AreEqual(0, model.HolderMembersOf(box).Count);
        }

        /// <summary>
        ///     <see cref="OwnershipModel.HolderMembersOf"/> returns a FRESH list in declaration order: the
        ///     fields in the order written (not alphabetical), the auto-property reported as the property, and
        ///     nothing for the self-reference, the computed property, the borrowed field and the static field.
        ///     Clearing a returned list does not touch the next answer.
        /// </summary>
        [TestMethod]
        public void HolderMembers_AreFreshListsInDeclarationOrder_WithFieldAndPropertyKinds()
        {
            var (comp, model) = Compile();
            var holder = comp.GetTypeByMetadataName("Holder");
            var members = model.HolderMembersOf(holder);
            var names = members.Select(h => h.Name).ToList();
            Assert.AreEqual(4, names.Count, string.Join(",", names));
            CollectionAssert.AreEqual(new[] { "_c", "_a", "_b" }, names.Where(n => n.StartsWith("_")).ToList(), "fields in declaration order");
            var prop = members.Single(h => h.Name == "Prop");
            Assert.AreEqual("property", prop.Kind);
            Assert.IsInstanceOfType(prop.Symbol, typeof(IPropertySymbol));
            Assert.AreEqual("field", members.Single(h => h.Name == "_c").Kind);
            Assert.AreEqual("NDArray[]", members.Single(h => h.Name == "_a").Type.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat));

            members.Clear();
            Assert.AreEqual(4, model.HolderMembersOf(holder).Count, "the caller's list is its own");
        }

        /// <summary>A type that holds only through its base class holds, but lists no holder of its own (its own members are a self-reference and a scalar).</summary>
        [TestMethod]
        public void HoldingOnlyThroughTheBase_HoldsButListsNothing()
        {
            var (comp, model) = Compile();
            var derived = comp.GetTypeByMetadataName("DerivedOnly");
            Assert.IsTrue(model.HoldsNDArrays(derived));
            Assert.AreEqual(0, model.HolderMembersOf(derived).Count);
        }

        /// <summary>
        ///     A disposable on a cycle owns arrays exactly when the cycle holds: A2 (↔ B2, which stores one) is an
        ///     NDArray-owning disposable, C2 (↔ D2, no array) is not; and B2's link back to A2 is not a holder.
        /// </summary>
        [TestMethod]
        public void OwningDisposable_OnACycle_OnlyWhenTheCycleHolds()
        {
            var (comp, model) = Compile();
            Assert.IsTrue(model.IsNDOwningDisposable(comp.GetTypeByMetadataName("A2")));
            Assert.IsFalse(model.IsNDOwningDisposable(comp.GetTypeByMetadataName("C2")));
            CollectionAssert.AreEqual(new[] { "_x" }, model.HolderMembersOf(comp.GetTypeByMetadataName("B2")).Select(h => h.Name).ToList());
            CollectionAssert.AreEqual(new[] { "_b" }, model.HolderMembersOf(comp.GetTypeByMetadataName("A2")).Select(h => h.Name).ToList());
        }

        /// <summary>Asking the same questions twice on one model, and once on a fresh model, gives the same answers: the memo never changes an answer.</summary>
        [TestMethod]
        public void RepeatedQueries_GiveTheSameAnswers()
        {
            var (comp, model) = Compile();
            string Snapshot(OwnershipModel m) => string.Join(" ", new[] { "Holder", "DerivedOnly", "A2", "B2", "C2", "D2", "Probe" }
                .Select(n => comp.GetTypeByMetadataName(n))
                .Select(t => t.Name + (m.HoldsNDArrays(t) ? "+" : "-") + ":" + string.Join(",", m.HolderMembersOf(t).Select(h => h.Name))));
            string first = Snapshot(model);
            Assert.AreEqual(first, Snapshot(model), "a second round on the same model");
            Assert.AreEqual(first, Snapshot(new OwnershipModel(KnownTypes.Resolve(comp), comp)), "a fresh model");
        }
    }
}
