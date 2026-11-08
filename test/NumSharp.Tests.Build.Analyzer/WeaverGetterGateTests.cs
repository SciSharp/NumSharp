using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The weaver's getter gate (<c>ScopeInheritance.PropertyOfGetter</c>): a property-level scope
    ///     attribute resolves to the property's getter, and finding that property is skipped for every type
    ///     with no scope-attributed property — the case of nearly every type, where the old per-getter scan
    ///     and Mono.Cecil's accessor-semantics read were both quadratic in the type's property count. These
    ///     tests pin that the gate changes cost only: a scoped getter among thousands of plain properties,
    ///     indexers, static properties, generic types, a derived type without scoped properties of its own
    ///     whose getter inherits from its base, a property-level [NDScopedCovered] that must still stop the
    ///     inheritance (the gate treats it as "scoped", not as absent), a method named like a getter, and
    ///     three assemblies declaring the same-named types (the gate's per-type map is keyed by the type
    ///     object, so one weave's answer cannot leak into the next).
    ///     <para>
    ///     Only a getter that INHERITS its scope reads the gate's answer: a getter's own property attribute
    ///     is found by <c>ScopeWeaver.CollectTargets</c>, which scans the properties itself. A test meant
    ///     for the gate therefore needs an override (or an interface implementation) whose declaration's
    ///     property carries the attribute.
    ///     </para>
    /// </summary>
    [TestClass]
    public class WeaverGetterGateTests
    {
        /// <summary>The usings every fixture of this class starts with.</summary>
        private const string Prelude = "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n";

        /// <summary>
        ///     A class of 3,000 plain auto-properties and one property-level [NDScoped] property: exactly that
        ///     getter is woven, it computes its value, and its temporary is reclaimed by the woven scope.
        /// </summary>
        [TestMethod]
        public void ManyPlainProperties_OneScopedProperty_OnlyItsGetterIsWoven()
        {
            const int count = 3000;
            var sb = new StringBuilder(Prelude + "public class Big {\n");
            for (int i = 0; i < count; i++)
                sb.Append("  public int P").Append(i).AppendLine(" { get; set; }");
            sb.AppendLine("  public static NDArray Last;");
            sb.AppendLine("  [NDScoped] public NDArray Scoped { get { var t = np.arange(3) + 1.0; Last = t; return t.copy(); } }");
            sb.AppendLine("}");
            var run = WeaverTestHarness.CompileAndWeave(sb.ToString(), "ManyProps");
            run.AssertWoven(1);
            Assert.IsTrue(run.HasScopeLocal("Big", "get_Scoped"));
            Assert.IsFalse(run.HasScopeLocal("Big", "get_P0"));
            Assert.IsFalse(run.HasScopeLocal("Big", "get_P" + (count - 1)));

            var value = (NDArray)run.GetProperty("Big", "Scoped", run.New("Big"));
            CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, value.ToArray<double>());
            Assert.IsTrue(((NDArray)run.GetStatic("Big", "Last")).IsDisposed, "the getter's temporary is reclaimed by the woven scope");
        }

        /// <summary>A property-level [NDScoped] on a virtual indexer weaves its getter, and an overriding indexer inherits it (its declaring type has no scoped property of its own).</summary>
        [TestMethod]
        public void ScopedIndexer_AndItsOverride_AreWoven()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public class Grid { [NDScoped] public virtual NDArray this[int i] { get { var t = np.arange(3) + i; return t.copy(); } } public int Width { get; set; } }\n" +
                "public class SubGrid : Grid { public override NDArray this[int i] { get { var t = np.arange(3) * i; return t.copy(); } } public int Height { get; set; } }\n",
                "ScopedIndexer");
            run.AssertWoven(2);
            Assert.IsTrue(run.HasScopeLocal("Grid", "get_Item"));
            Assert.IsTrue(run.HasScopeLocal("SubGrid", "get_Item"), "the override inherits through the base getter");
            Assert.IsFalse(run.HasScopeLocal("SubGrid", "get_Height"));
        }

        /// <summary>A property-level [NDScoped] on a static property weaves the static getter; the getter runs and its temporary is reclaimed.</summary>
        [TestMethod]
        public void ScopedStaticProperty_IsWoven()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public static class Defaults { public static NDArray Last; [NDScoped] public static NDArray Ramp { get { var t = np.arange(3) + 1.0; Last = t; return t.copy(); } } public static int Count { get; set; } }\n",
                "ScopedStatic");
            run.AssertWoven(1);
            Assert.IsTrue(run.HasScopeLocal("Defaults", "get_Ramp"));
            var value = (NDArray)run.GetProperty("Defaults", "Ramp", null);
            CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, value.ToArray<double>());
            Assert.IsTrue(((NDArray)run.GetStatic("Defaults", "Last")).IsDisposed);
        }

        /// <summary>
        ///     A derived type with 500 plain properties and no scoped property of its own overrides a scoped
        ///     base property: its getter must still inherit — the gate answers "no property-level attribute"
        ///     for the derived TYPE, while the walk finds the attribute through the base getter.
        /// </summary>
        [TestMethod]
        public void DerivedTypeWithoutScopedProperties_GetterStillInheritsFromTheBaseProperty()
        {
            var sb = new StringBuilder(Prelude);
            sb.AppendLine("public class Base { [NDScoped] public virtual NDArray V { get { var t = np.arange(3) + 1.0; return t.copy(); } } }");
            sb.Append("public class Derived : Base { public static NDArray Last;");
            for (int i = 0; i < 500; i++)
                sb.Append(" public int P").Append(i).Append(" { get; set; }");
            sb.AppendLine(" public override NDArray V { get { var t = np.arange(3) + 2.0; Last = t; return t.copy(); } } }");
            var run = WeaverTestHarness.CompileAndWeave(sb.ToString(), "DerivedNoScopedProps");
            run.AssertWoven(2);
            Assert.IsTrue(run.HasScopeLocal("Derived", "get_V"));
            run.GetProperty("Derived", "V", run.New("Derived"));
            Assert.IsTrue(((NDArray)run.GetStatic("Derived", "Last")).IsDisposed, "the inherited scope reclaims the override's temporary");
        }

        /// <summary>
        ///     A property-level [NDScopedCovered] in the middle of a chain (IRoot.V scoped, Mid.V covered,
        ///     Leaf.V overriding) stops the inheritance: neither Mid's nor Leaf's getter is woven. The control —
        ///     the same chain without the covered attribute — weaves both. A gate that treated a covered-only
        ///     type as having no scoped property would walk past Mid to IRoot and weave Leaf.
        /// </summary>
        [TestMethod]
        public void CoveredPropertyInTheMiddle_StopsTheInheritance()
        {
            const string chain =
                "public interface IRoot { [NDScoped] NDArray V { get; } }\n" +
                "public class Mid : IRoot { {ATTR} public virtual NDArray V { get { var t = np.arange(3) + 1.0; return t.copy(); } } }\n" +
                "public class Leaf : Mid { public override NDArray V { get { var t = np.arange(3) + 2.0; return t.copy(); } } }\n";

            var covered = WeaverTestHarness.CompileAndWeave(Prelude + chain.Replace("{ATTR}", "[NDScopedCovered]"), "CoveredMiddle");
            covered.AssertWoven(0);
            Assert.IsFalse(covered.HasScopeLocal("Mid", "get_V"), "the covered declaration opts out");
            Assert.IsFalse(covered.HasScopeLocal("Leaf", "get_V"), "the nearest declaration is the covered one, so Leaf inherits the opt-out");

            var control = WeaverTestHarness.CompileAndWeave(Prelude + chain.Replace("{ATTR}", ""), "UncoveredMiddle");
            control.AssertWoven(2);
            Assert.IsTrue(control.HasScopeLocal("Mid", "get_V") && control.HasScopeLocal("Leaf", "get_V"), "without the opt-out both inherit IRoot.V's attribute");
        }

        /// <summary>
        ///     A method merely NAMED like a getter (<c>get_Thing()</c>, no property) is not a getter: in a type
        ///     whose real scoped property makes the gate build its map, only the real getter is woven.
        /// </summary>
        [TestMethod]
        public void MethodNamedLikeAGetter_IsNotAGetter()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public class Weird { public NDArray get_Thing() { var t = np.arange(3) + 1.0; return t.copy(); } [NDScoped] public NDArray Real { get { var t = np.arange(3) + 2.0; return t.copy(); } } }\n",
                "GetterLookalike");
            run.AssertWoven(1);
            Assert.IsTrue(run.HasScopeLocal("Weird", "get_Real"));
            Assert.IsFalse(run.HasScopeLocal("Weird", "get_Thing"));
        }

        /// <summary>A property-level [NDScoped] in a generic class weaves the definition's getter; a closed subclass adds nothing.</summary>
        [TestMethod]
        public void ScopedPropertyOfAGenericType_IsWoven()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public class Holder<T> { [NDScoped] public NDArray Ramp { get { var t = np.arange(3) + 1.0; return t.copy(); } } public T Item { get; set; } }\n" +
                "public class IntHolder : Holder<int> { }\n", "GenericScopedProperty");
            run.AssertWoven(1);
            Assert.IsTrue(run.HasScopeLocal("Holder`1", "get_Ramp"));
        }

        /// <summary>
        ///     Three weaves in one process of assemblies that declare the same two types, <c>Shape</c> and its
        ///     override <c>Square</c> — scoped, then plain, then scoped again: each weave answers from its own
        ///     attributes. The gate's per-type map is keyed by the type object, so it can neither carry
        ///     "scoped" into the plain assembly nor a stale answer into the next one.
        ///     <para>
        ///     Square's getter is the one that goes through the gate. A getter's OWN property attribute is
        ///     found by <c>ScopeWeaver.CollectTargets</c>, which scans the properties itself; only a getter that
        ///     inherits its scope asks the gate about the declaration it overrides. A version of this test with
        ///     Shape alone passed with the gate's map cached by type NAME across weaves (a mutation run,
        ///     2026-10-24): every getter was woven by CollectTargets, and the gate's answer was never read.
        ///     </para>
        /// </summary>
        [TestMethod]
        public void SameTypeNameInThreeAssemblies_EachWeaveReadsItsOwnAttributes()
        {
            const string square = "public class Square : Shape { public override NDArray Area { get { var t = np.arange(4) + 1.0; return t.copy(); } } }\n";
            const string scoped = Prelude + "public class Shape { [NDScoped] public virtual NDArray Area { get { var t = np.arange(3) + 1.0; return t.copy(); } } public int Sides { get; set; } }\n" + square;
            const string plain = Prelude + "public class Shape { public virtual NDArray Area { get { var t = np.arange(3) + 1.0; return t.copy(); } } public int Sides { get; set; } }\n" + square;
            var first = WeaverTestHarness.CompileAndWeave(scoped, "SameNameA").AssertWoven(2);
            Assert.IsTrue(first.HasScopeLocal("Square", "get_Area"), "the override inherits the base property's scope");
            WeaverTestHarness.CompileAndWeave(plain, "SameNameB").AssertWoven(0);
            var third = WeaverTestHarness.CompileAndWeave(scoped, "SameNameC").AssertWoven(2);
            Assert.IsTrue(third.HasScopeLocal("Shape", "get_Area"));
            Assert.IsTrue(third.HasScopeLocal("Square", "get_Area"), "the third weave reads its own Shape, not the first or second weave's");
        }
    }
}
