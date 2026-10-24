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
    ///     one would never finish. The walk now searches each interface instantiation once per method —
    ///     keyed by the instantiation as seen from the method's own type, so <c>IFoo&lt;int&gt;</c> and
    ///     <c>IFoo&lt;string&gt;</c> are still both searched.
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
    }
}
