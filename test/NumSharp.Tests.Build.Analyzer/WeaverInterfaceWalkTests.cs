using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The weaver's inheritance walk over INTERFACE hierarchies: its cost and its generic
    ///     bookkeeping. The C# compiler lists an interface's whole base-interface closure on the
    ///     interface itself and on every type implementing it, so a walk that recursed into every listed
    ///     interface without remembering what it had searched reached each base interface once per path:
    ///     2^k searches for a chain of k interfaces, for every public virtual method of every
    ///     implementing type. A class implementing a 26-deep chain took a minute to weave, a 40-deep
    ///     one would never finish. The walk now searches each interface instantiation once per level of a
    ///     method's walk (a type and the bases between it and the slot it overrides) — keyed by the
    ///     instantiation as seen from the method's own type, so <c>IFoo&lt;int&gt;</c> and
    ///     <c>IFoo&lt;string&gt;</c> are still both searched. Which interface member a method implements is
    ///     the CLR's interface map, pinned in <see cref="WeaverInterfaceMappingTests"/>.
    /// </summary>
    [TestClass]
    public class WeaverInterfaceWalkTests
    {
        /// <summary>The usings every fixture of this class starts with.</summary>
        private const string Prelude = "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n";

        /// <summary>
        ///     A class implementing a 40-deep interface chain whose every level declares an unscoped member
        ///     weaves in bounded time, inherits [NDScoped] for the root member only, and weaves nothing else.
        ///     Each unscoped member's walk searches the whole chain and finds nothing, so nothing ends it
        ///     early: the worst case for the walk, which doubled per level before it remembered what it had
        ///     searched (56 s at 26 levels; 40 would never finish, which the timeout turns into a failure).
        /// </summary>
        [TestMethod]
        [Timeout(120000)]
        public void DeepInterfaceChain_UnscopedMembersAtEveryLevel_WeaveInLinearTime()
        {
            // Every level declares an unscoped member, so the walk for each of them searches the whole
            // chain and finds nothing — the case with no early exit (measured on the unbounded walk: 3.3 s
            // at 22 levels, 13.7 s at 24, 56 s at 26 — doubling per level).
            const int depth = 40;
            var sb = new StringBuilder(Prelude);
            sb.AppendLine("public interface I0 { [NDScoped] NDArray Apply(NDArray a); NDArray M0(NDArray a); }");
            for (int i = 1; i < depth; i++)
                sb.AppendLine($"public interface I{i} : I{i - 1} {{ NDArray M{i}(NDArray a); }}");
            sb.Append($"public class Impl : I{depth - 1} {{ public NDArray Apply(NDArray a) {{ var t = a + 1.0; return t.copy(); }}");
            for (int i = 0; i < depth; i++)
                sb.Append($" public NDArray M{i}(NDArray a) {{ var t = a + {i}.0; return t.copy(); }}");
            sb.AppendLine(" }");

            var fixture = WeaverTestHarness.Compile(sb.ToString(), "DeepUnscopedChain");
            var sw = Stopwatch.StartNew();
            var run = WeaverTestHarness.Weave(fixture);
            sw.Stop();
            run.AssertWoven(1);
            Assert.IsTrue(run.HasScopeLocal("Impl", "Apply"), "the scoped root member is inherited through 39 levels");
            for (int i = 0; i < depth; i++)
                Assert.IsFalse(run.HasScopeLocal("Impl", $"M{i}"), $"M{i} implements an unscoped member");
            Assert.IsTrue(sw.ElapsedMilliseconds < 30_000, $"weaving a {depth}-deep interface chain took {sw.ElapsedMilliseconds} ms");
        }

        /// <summary>
        ///     An 18-level lattice of interface diamonds weaves in bounded time, with the scoped root member
        ///     inherited and the unscoped members left alone. The number of paths to the root roughly triples
        ///     per level, so this is the shape on which a walk that does not remember its searches grows
        ///     fastest (about ten minutes at 18 levels); 12 levels, which the old walk managed in 1.4 s, could
        ///     not tell the two walks apart.
        /// </summary>
        [TestMethod]
        [Timeout(120000)]
        public void InterfaceDiamondLattice_WeavesEveryImplementation_InBoundedTime()
        {
            // Eighteen levels of diamonds: every interface of a level inherits both interfaces of the level
            // below, so the number of paths to the root grows about threefold per level on top of the
            // flattened lists (measured on the unbounded walk: 1.4 s at 12 levels, 3.4 s at 13, 9.4 s at
            // 14 — about ten minutes at 18; the deduplicated walk takes the same 0.3 s at every size).
            const int levels = 18;
            var sb = new StringBuilder(Prelude);
            sb.AppendLine("public interface IRoot { [NDScoped] NDArray Apply(NDArray a); NDArray Plain(NDArray a); }");
            sb.AppendLine("public interface IA1 : IRoot { NDArray A1(NDArray a); }");
            sb.AppendLine("public interface IB1 : IRoot { NDArray B1(NDArray a); }");
            for (int l = 2; l <= levels; l++)
            {
                sb.AppendLine($"public interface IA{l} : IA{l - 1}, IB{l - 1} {{ NDArray A{l}(NDArray a); }}");
                sb.AppendLine($"public interface IB{l} : IA{l - 1}, IB{l - 1} {{ NDArray B{l}(NDArray a); }}");
            }
            sb.Append($"public class Lattice : IA{levels}, IB{levels} {{ public NDArray Apply(NDArray a) {{ var t = a + 1.0; return t.copy(); }} public NDArray Plain(NDArray a) {{ var t = a + 1.0; return t.copy(); }}");
            for (int l = 1; l <= levels; l++)
                sb.Append($" public NDArray A{l}(NDArray a) {{ var t = a + 1.0; return t.copy(); }} public NDArray B{l}(NDArray a) {{ var t = a + 1.0; return t.copy(); }}");
            sb.AppendLine(" }");

            var fixture = WeaverTestHarness.Compile(sb.ToString(), "DiamondLattice");
            var sw = Stopwatch.StartNew();
            var run = WeaverTestHarness.Weave(fixture);
            sw.Stop();
            run.AssertWoven(1);
            Assert.IsTrue(run.HasScopeLocal("Lattice", "Apply"));
            Assert.IsFalse(run.HasScopeLocal("Lattice", "Plain"));
            Assert.IsFalse(run.HasScopeLocal("Lattice", "A1") || run.HasScopeLocal("Lattice", "B7"));
            Assert.IsTrue(sw.ElapsedMilliseconds < 30_000, $"weaving a {levels}-level diamond lattice took {sw.ElapsedMilliseconds} ms");
        }

        /// <summary>Whether the overload of <paramref name="name"/> whose first parameter has the given type got a scope local.</summary>
        /// <param name="run">The woven assembly.</param>
        /// <param name="type">The declaring type's full name (<c>Open`1</c> for a generic type).</param>
        /// <param name="name">The method name.</param>
        /// <param name="firstParameterType">The full name of the overload's first parameter type.</param>
        /// <returns>True when that overload was woven.</returns>
        /// <exception cref="System.InvalidOperationException">No overload, or more than one, matches.</exception>
        private static bool OverloadWoven(WeaveRun run, string type, string name, string firstParameterType)
        {
            using var asm = run.ReadCecil();
            var m = WeaveRun.AllMethods(asm.MainModule)
                .Single(x => x.DeclaringType.FullName == type && x.Name == name && x.Parameters[0].ParameterType.FullName == firstParameterType);
            return WeaveRun.HasScopeLocal(m);
        }

        /// <summary>
        ///     The walk remembers INSTANTIATIONS, not definitions: <c>IBase&lt;int&gt;</c> and
        ///     <c>IBase&lt;string&gt;</c> are both searched (a definition key leaves <c>M(string)</c> unwoven —
        ///     the mutant this test kills), one instantiation reached along two paths is still found, and an
        ///     implementer's own type parameter (<c>IBase&lt;U[]&gt;</c>) is read through its context.
        /// </summary>
        [TestMethod]
        public void OneGenericInterface_UnderTwoInstantiations_BothImplementationsInherit()
        {
            // IBase<int> and IBase<string> share one definition: remembering "IBase`1 was searched" would
            // skip the second instantiation and leave M(string) unwoven. Each instantiation is its own key.
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public interface IBase<T> { [NDScoped] NDArray M(T x, NDArray a); NDArray Plain(T x, NDArray a); }\n" +
                "public interface IA : IBase<int> { }\n" +
                "public interface IB : IBase<string> { }\n" +
                "public class Both : IA, IB {\n" +
                "  public NDArray M(int x, NDArray a) { var t = a + x; return t.copy(); }\n" +
                "  public NDArray M(string x, NDArray a) { var t = a + 1.0; return t.copy(); }\n" +
                "  public NDArray Plain(int x, NDArray a) { var t = a + x; return t.copy(); }\n" +
                "  public NDArray Plain(string x, NDArray a) { var t = a + 1.0; return t.copy(); }\n" +
                "}\n" +
                // The same instantiation reached along two paths is searched once and still found.
                "public interface IL : IBase<double> { }\n" +
                "public interface IR : IBase<double> { }\n" +
                "public class Same : IL, IR { public NDArray M(double x, NDArray a) { var t = a + x; return t.copy(); } public NDArray Plain(double x, NDArray a) { var t = a + x; return t.copy(); } }\n" +
                // A generic implementer: one instantiation over its own type parameter, one closed.
                "public class Open<U> : IA, IBase<U[]> {\n" +
                "  public NDArray M(int x, NDArray a) { var t = a + x; return t.copy(); }\n" +
                "  public NDArray M(U[] x, NDArray a) { var t = a + 1.0; return t.copy(); }\n" +
                "  public NDArray Plain(int x, NDArray a) { var t = a + x; return t.copy(); }\n" +
                "  public NDArray Plain(U[] x, NDArray a) { var t = a + 1.0; return t.copy(); }\n" +
                "}\n", "TwoInstantiations");
            run.AssertWoven(5);
            Assert.IsTrue(OverloadWoven(run, "Both", "M", "System.Int32"), "M(int) implements IBase<int>.M");
            Assert.IsTrue(OverloadWoven(run, "Both", "M", "System.String"), "M(string) implements IBase<string>.M — a second instantiation of the same definition");
            Assert.IsFalse(OverloadWoven(run, "Both", "Plain", "System.Int32"));
            Assert.IsFalse(OverloadWoven(run, "Both", "Plain", "System.String"));
            Assert.IsTrue(run.HasScopeLocal("Same", "M"), "one instantiation listed twice");
            Assert.IsTrue(OverloadWoven(run, "Open`1", "M", "System.Int32"));
            Assert.IsTrue(OverloadWoven(run, "Open`1", "M", "U[]"), "M(U[]) implements IBase<U[]>.M through the implementer's own type parameter");
        }

        // ---------------------------------------------------------------- the instantiation key, edge by edge

        /// <summary>A fresh three-element float64 input, as the weaver tests use.</summary>
        /// <returns>[0, 1, 2] as float64.</returns>
        private static NumSharp.NDArray Input() => NumSharp.np.arange(3).astype(NumSharp.np.float64);

        /// <summary>How many overloads named <paramref name="name"/> of type <paramref name="type"/> got a scope local, out of how many.</summary>
        /// <param name="run">The woven assembly.</param>
        /// <param name="type">The declaring type's full name.</param>
        /// <param name="name">The method name.</param>
        /// <returns>The woven count and the total count.</returns>
        private static (int woven, int total) OverloadsWoven(WeaveRun run, string type, string name)
        {
            using var asm = run.ReadCecil();
            var overloads = WeaveRun.AllMethods(asm.MainModule).Where(m => m.DeclaringType.FullName == type && m.Name == name).ToList();
            return (overloads.Count(WeaveRun.HasScopeLocal), overloads.Count);
        }

        /// <summary>
        ///     Instantiations that differ only INSIDE a type argument are different keys: IBase&lt;List&lt;int&gt;&gt;,
        ///     IBase&lt;List&lt;string&gt;&gt; and IBase&lt;Dictionary&lt;string, List&lt;int&gt;&gt;&gt; are each searched. A key
        ///     that kept only an argument's outer name would skip the second list and leave its M unwoven.
        /// </summary>
        [TestMethod]
        public void NestedGenericArguments_EachInstantiationIsSearched()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public interface IBase<T> { [NDScoped] NDArray M(T x, NDArray a); NDArray Plain(T x, NDArray a); }\n" +
                "public interface IA : IBase<List<int>> { }\npublic interface IB : IBase<List<string>> { }\npublic interface IC : IBase<Dictionary<string, List<int>>> { }\n" +
                "public class Both : IA, IB, IC {\n" +
                "  public NDArray M(List<int> x, NDArray a) { var t = a + 1.0; return t.copy(); }\n" +
                "  public NDArray M(List<string> x, NDArray a) { var t = a + 2.0; return t.copy(); }\n" +
                "  public NDArray M(Dictionary<string, List<int>> x, NDArray a) { var t = a + 3.0; return t.copy(); }\n" +
                "  public NDArray Plain(List<int> x, NDArray a) => a.copy(); public NDArray Plain(List<string> x, NDArray a) => a.copy(); public NDArray Plain(Dictionary<string, List<int>> x, NDArray a) => a.copy();\n" +
                "}\n", "NestedArgs");
            run.AssertWoven(3);
            Assert.AreEqual((3, 3), OverloadsWoven(run, "Both", "M"), "every M implements its own instantiation's scoped member");
            Assert.AreEqual((0, 3), OverloadsWoven(run, "Both", "Plain"));
        }

        /// <summary>
        ///     Array arguments are keyed with their rank and nesting: IBase&lt;int[]&gt;, IBase&lt;int[,]&gt; and
        ///     IBase&lt;int[][]&gt; are three instantiations, each searched.
        /// </summary>
        [TestMethod]
        public void ArrayRanksAndJaggedness_EachInstantiationIsSearched()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public interface IBase<T> { [NDScoped] NDArray M(T x, NDArray a); }\n" +
                "public interface IR1 : IBase<int[]> { }\npublic interface IR2 : IBase<int[,]> { }\npublic interface IR3 : IBase<int[][]> { }\n" +
                "public class Ranks : IR1, IR2, IR3 {\n" +
                "  public NDArray M(int[] x, NDArray a) { var t = a + 1.0; return t.copy(); }\n" +
                "  public NDArray M(int[,] x, NDArray a) { var t = a + 2.0; return t.copy(); }\n" +
                "  public NDArray M(int[][] x, NDArray a) { var t = a + 3.0; return t.copy(); }\n" +
                "}\n", "ArrayRanks");
            run.AssertWoven(3);
            Assert.AreEqual((3, 3), OverloadsWoven(run, "Ranks", "M"));
        }

        /// <summary>
        ///     Nullable and tuple arguments are instantiations of their own: IBase&lt;int?&gt; beside
        ///     IBase&lt;int&gt;, and IBase&lt;(int, string)&gt; beside IBase&lt;(string, int)&gt;, are four searches.
        /// </summary>
        [TestMethod]
        public void NullableAndTupleArguments_EachInstantiationIsSearched()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public interface IBase<T> { [NDScoped] NDArray M(T x, NDArray a); }\n" +
                "public interface IN1 : IBase<int?> { }\npublic interface IN2 : IBase<int> { }\npublic interface IT1 : IBase<(int, string)> { }\npublic interface IT2 : IBase<(string, int)> { }\n" +
                "public class Specials : IN1, IN2, IT1, IT2 {\n" +
                "  public NDArray M(int? x, NDArray a) { var t = a + 1.0; return t.copy(); }\n" +
                "  public NDArray M(int x, NDArray a) { var t = a + 2.0; return t.copy(); }\n" +
                "  public NDArray M((int, string) x, NDArray a) { var t = a + 3.0; return t.copy(); }\n" +
                "  public NDArray M((string, int) x, NDArray a) { var t = a + 4.0; return t.copy(); }\n" +
                "}\n", "NullableTupleArgs");
            run.AssertWoven(4);
            Assert.AreEqual((4, 4), OverloadsWoven(run, "Specials", "M"));
        }

        /// <summary>
        ///     An interface listed on a GENERIC BASE class is keyed in the base's context: <c>Base&lt;T&gt; : IBase&lt;T&gt;</c>
        ///     seen from <c>Derived : Base&lt;int&gt;</c> is IBase&lt;int&gt;, a different search from Derived's own
        ///     IBase&lt;string&gt;. The base's method, both of Derived's and a deeper override all inherit.
        /// </summary>
        [TestMethod]
        public void InstantiationThroughAGenericBaseClass_IsReadInTheBasesContext()
        {
            var run = WeaverTestHarness.CompileAndWeave(Prelude +
                "public interface IBase<T> { [NDScoped] NDArray M(T x, NDArray a); }\n" +
                "public class Base<T> : IBase<T> { public virtual NDArray M(T x, NDArray a) { var t = a + 1.0; return t.copy(); } }\n" +
                "public class Derived : Base<int>, IBase<string> {\n" +
                "  public override NDArray M(int x, NDArray a) { var t = a + x; return t.copy(); }\n" +
                "  public NDArray M(string x, NDArray a) { var t = a + 2.0; return t.copy(); }\n" +
                "}\n" +
                "public class Deeper : Derived { public override NDArray M(int x, NDArray a) { var t = a + 3.0; return t.copy(); } }\n", "GenericBaseContext");
            run.AssertWoven(4);
            Assert.IsTrue(run.HasScopeLocal("Base`1", "M"), "the generic base implements IBase<T> itself");
            Assert.IsTrue(OverloadWoven(run, "Derived", "M", "System.Int32"), "the override reaches IBase<int> through Base<int>");
            Assert.IsTrue(OverloadWoven(run, "Derived", "M", "System.String"), "Derived's own IBase<string>");
            Assert.IsTrue(run.HasScopeLocal("Deeper", "M"), "an override two levels down still reaches IBase<int>");
        }

        // ---------------------------------------------------------------- depth limit and other entry points

        /// <summary>
        ///     A chain of 100 interfaces, deeper than the walk's depth limit (64), with an unscoped member at
        ///     every level: the root member is still inherited — the compiler lists the root on every
        ///     interface, so the walk meets it within the limit — the unscoped ones are not, the weave stays
        ///     linear, and the analyzer agrees (its unscoped members leak, the inherited one is exempt).
        /// </summary>
        [TestMethod]
        [Timeout(120000)]
        public async System.Threading.Tasks.Task ChainDeeperThanTheWalksDepthLimit_TheRootMemberIsStillInherited()
        {
            const int depth = 100;
            var sb = new StringBuilder(Prelude);
            sb.AppendLine("public interface I0 { [NDScoped] NDArray Apply(NDArray a); NDArray M0(NDArray a); }");
            for (int i = 1; i < depth; i++)
                sb.AppendLine($"public interface I{i} : I{i - 1} {{ NDArray M{i}(NDArray a); }}");
            // Each body drops its temporary (it returns a copy of the INPUT), so the analyzer reports every
            // member it does not consider scoped — `return t.copy()` would count as consuming t.
            sb.AppendLine($"public class Impl : I{depth - 1} {{");
            sb.AppendLine("  public NDArray Apply(NDArray a) { var t = a + 1.0; return a.copy(); }");
            for (int i = 0; i < depth; i++)
                sb.AppendLine($"  public NDArray M{i}(NDArray a) {{ var t = a + {i}.0; return a.copy(); }}");
            sb.AppendLine("}");
            var source = sb.ToString();

            var sw = Stopwatch.StartNew();
            var run = WeaverTestHarness.Weave(WeaverTestHarness.Compile(source, "BeyondDepthLimit"));
            sw.Stop();
            run.AssertWoven(1);
            Assert.IsTrue(run.HasScopeLocal("Impl", "Apply"), "the root member is inherited through 99 levels");
            Assert.AreEqual((0, 1), OverloadsWoven(run, "Impl", "M57"));
            Assert.IsTrue(sw.ElapsedMilliseconds < 30_000, $"weaving a {depth}-deep chain took {sw.ElapsedMilliseconds} ms");

            var analysis = await AnalyzerTestHarness.RunAsync(source, "beyond_depth_limit.cs");
            Assert.IsTrue(analysis.CompileErrors.IsEmpty);
            var lines = source.Split('\n');
            var leakLines = analysis.Ndw.Where(d => d.Id == "NDW012").Select(d => lines[AnalyzerTestHarness.LineOf(d) - 1]).ToList();
            Assert.AreEqual(depth, leakLines.Count, "each unscoped member leaks its temporary");
            Assert.IsFalse(leakLines.Any(l => l.Contains(" Apply(")), "the analyzer, too, sees Apply as inheriting [NDScoped]");
        }

        /// <summary>
        ///     Explicit implementations of every member of a 40-deep chain: each one names its interface member
        ///     directly (the .override table) and the walk ends at that member — an interface member implements
        ///     nothing in a base interface, so the walk never climbs the chain from there. Only the root's
        ///     explicit implementation inherits. This path was linear before the interface-search fix too (the
        ///     old weaver weaves this fixture in under a second); pinned so it stays bounded and correct.
        /// </summary>
        [TestMethod]
        [Timeout(120000)]
        public void ExplicitImplementationsOfADeepChain_EndAtTheirOwnSlot()
        {
            const int depth = 40;
            var sb = new StringBuilder(Prelude);
            sb.AppendLine("public interface I0 { [NDScoped] NDArray Apply(NDArray a); NDArray M0(NDArray a); }");
            for (int i = 1; i < depth; i++)
                sb.AppendLine($"public interface I{i} : I{i - 1} {{ NDArray M{i}(NDArray a); }}");
            sb.Append($"public class ExplicitImpl : I{depth - 1} {{ NDArray I0.Apply(NDArray a) {{ var t = a + 1.0; return t.copy(); }}");
            for (int i = 0; i < depth; i++)
                sb.Append($" NDArray I{i}.M{i}(NDArray a) {{ var t = a + {i}.0; return t.copy(); }}");
            sb.AppendLine(" }");

            var fixture = WeaverTestHarness.Compile(sb.ToString(), "ExplicitDeepChain");
            var sw = Stopwatch.StartNew();
            var run = WeaverTestHarness.Weave(fixture);
            sw.Stop();
            run.AssertWoven(1);
            Assert.IsTrue(run.HasScopeLocal("ExplicitImpl", "I0.Apply"));
            Assert.IsFalse(run.HasScopeLocal("ExplicitImpl", "I39.M39"));
            Assert.IsTrue(sw.ElapsedMilliseconds < 30_000, $"weaving {depth + 1} explicit implementations took {sw.ElapsedMilliseconds} ms");
        }

        /// <summary>
        ///     The SECOND walk — [NDScopedExit] parameter inheritance, asked for every method with a body — goes
        ///     through the same interface search: a 40-deep chain with an unscoped member at every level weaves
        ///     in bounded time, the root's Keep detaches its retained argument (it survives the caller's
        ///     scope), and an unscoped member's argument is reclaimed with the scope.
        /// </summary>
        [TestMethod]
        [Timeout(120000)]
        public void ExitParameterInheritance_ThroughADeepChain_IsLinear_AndKeepsTheArgument()
        {
            const int depth = 40;
            var sb = new StringBuilder(Prelude);
            sb.AppendLine("public interface IK0 { void Keep([NDScopedExit] NDArray w); void N0(NDArray w); }");
            for (int i = 1; i < depth; i++)
                sb.AppendLine($"public interface IK{i} : IK{i - 1} {{ void N{i}(NDArray w); }}");
            sb.Append($"public class Keeper : IK{depth - 1} {{ public NDArray Kept; public void Keep(NDArray w) => Kept = w;");
            for (int i = 0; i < depth; i++)
                sb.Append($" public void N{i}(NDArray w) => Kept = w;");
            sb.AppendLine(" }");

            var fixture = WeaverTestHarness.Compile(sb.ToString(), "ExitDeepChain");
            var sw = Stopwatch.StartNew();
            var run = WeaverTestHarness.Weave(fixture);
            sw.Stop();
            run.AssertWoven(1);
            Assert.IsTrue(run.CallsDetach("Keeper", "Keep"), "Keep inherits the retained parameter through 39 levels");
            Assert.IsFalse(run.CallsDetach("Keeper", "N17"), "an unscoped member retains nothing");
            Assert.IsTrue(sw.ElapsedMilliseconds < 30_000, $"weaving a {depth}-deep chain of exit members took {sw.ElapsedMilliseconds} ms");

            var keeper = run.New("Keeper");
            NumSharp.NDArray kept, lost;
            using (NumSharp.NDScope.Open())
            {
                kept = Input() + 1.0;
                run.Invoke("Keeper", "Keep", keeper, kept);
                lost = Input() + 2.0;
                run.Invoke("Keeper", "N17", keeper, lost);
            }
            Assert.IsFalse(kept.IsDisposed, "the inherited [NDScopedExit] argument is detached from the caller's scope");
            Assert.IsTrue(lost.IsDisposed, "an argument nothing retains is reclaimed with the caller's scope");
        }

        /// <summary>
        ///     [NDScopedExit] through an 18-level diamond lattice (the shape that grew threefold per level):
        ///     bounded time, and the implementation of the root's Keep detaches.
        /// </summary>
        [TestMethod]
        [Timeout(120000)]
        public void ExitParameterInheritance_ThroughADiamondLattice_IsBounded()
        {
            const int levels = 18;
            var sb = new StringBuilder(Prelude);
            sb.AppendLine("public interface IRoot { void Keep([NDScopedExit] NDArray w); void Plain(NDArray w); }");
            sb.AppendLine("public interface IA1 : IRoot { void A1(NDArray w); }");
            sb.AppendLine("public interface IB1 : IRoot { void B1(NDArray w); }");
            for (int l = 2; l <= levels; l++)
            {
                sb.AppendLine($"public interface IA{l} : IA{l - 1}, IB{l - 1} {{ void A{l}(NDArray w); }}");
                sb.AppendLine($"public interface IB{l} : IA{l - 1}, IB{l - 1} {{ void B{l}(NDArray w); }}");
            }
            sb.Append($"public class LatticeKeeper : IA{levels}, IB{levels} {{ public NDArray Kept; public void Keep(NDArray w) => Kept = w; public void Plain(NDArray w) => Kept = w;");
            for (int l = 1; l <= levels; l++)
                sb.Append($" public void A{l}(NDArray w) => Kept = w; public void B{l}(NDArray w) => Kept = w;");
            sb.AppendLine(" }");

            var fixture = WeaverTestHarness.Compile(sb.ToString(), "ExitLattice");
            var sw = Stopwatch.StartNew();
            var run = WeaverTestHarness.Weave(fixture);
            sw.Stop();
            run.AssertWoven(1);
            Assert.IsTrue(run.CallsDetach("LatticeKeeper", "Keep"));
            Assert.IsFalse(run.CallsDetach("LatticeKeeper", "Plain"));
            Assert.IsTrue(sw.ElapsedMilliseconds < 30_000, $"weaving an {levels}-level lattice of exit members took {sw.ElapsedMilliseconds} ms");
        }

        /// <summary>
        ///     A property-level [NDScoped] on the root of an 18-level diamond lattice reaches the implementing
        ///     getter, in bounded time; the unscoped property beside it inherits nothing.
        /// </summary>
        [TestMethod]
        [Timeout(120000)]
        public void PropertyGetterThroughADiamondLattice_InheritsInBoundedTime()
        {
            const int levels = 18;
            var sb = new StringBuilder(Prelude);
            sb.AppendLine("public interface IRoot { [NDScoped] NDArray Value { get; } NDArray Plain { get; } }");
            sb.AppendLine("public interface IA1 : IRoot { NDArray A1(NDArray a); }");
            sb.AppendLine("public interface IB1 : IRoot { NDArray B1(NDArray a); }");
            for (int l = 2; l <= levels; l++)
            {
                sb.AppendLine($"public interface IA{l} : IA{l - 1}, IB{l - 1} {{ NDArray A{l}(NDArray a); }}");
                sb.AppendLine($"public interface IB{l} : IA{l - 1}, IB{l - 1} {{ NDArray B{l}(NDArray a); }}");
            }
            sb.Append($"public class PropertyLattice : IA{levels}, IB{levels} {{ public NDArray Value {{ get {{ var t = np.arange(3) + 1.0; return t.copy(); }} }} public NDArray Plain {{ get {{ var t = np.arange(3) + 2.0; return t.copy(); }} }}");
            for (int l = 1; l <= levels; l++)
                sb.Append($" public NDArray A{l}(NDArray a) {{ var t = a + 1.0; return t.copy(); }} public NDArray B{l}(NDArray a) {{ var t = a + 1.0; return t.copy(); }}");
            sb.AppendLine(" }");

            var fixture = WeaverTestHarness.Compile(sb.ToString(), "PropertyLattice");
            var sw = Stopwatch.StartNew();
            var run = WeaverTestHarness.Weave(fixture);
            sw.Stop();
            run.AssertWoven(1);
            Assert.IsTrue(run.HasScopeLocal("PropertyLattice", "get_Value"));
            Assert.IsFalse(run.HasScopeLocal("PropertyLattice", "get_Plain"));
            Assert.IsTrue(sw.ElapsedMilliseconds < 30_000, $"weaving an {levels}-level lattice under a scoped property took {sw.ElapsedMilliseconds} ms");
        }
    }
}
