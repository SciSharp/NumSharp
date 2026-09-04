using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Robustness of the ownership analyzer (technique T6 applied to types): cyclic and
    ///     self-referential type graphs, deep contagion chains, generic definitions versus
    ///     instantiations, nested and partial types, very wide types, aliases and nullable annotations,
    ///     the NumSharp-not-referenced no-op, and malformed source — none may crash, and the verdicts
    ///     must hold.
    /// </summary>
    [TestClass]
    public class OwnershipRobustnessTests
    {
        private const string H = "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n";

        private static async Task<AnalyzerResult> Run(string body, string name = "ownership_robust.cs")
        {
            var r = await AnalyzerTestHarness.RunAsync(H + body, name);
            Assert.IsTrue(r.CompileErrors.IsEmpty, "snippet must compile:\n" + body + "\n" + string.Join("\n", r.CompileErrors));
            return r;
        }

        [TestMethod]
        public async Task CycleWithoutArrays_IsClean()
        {
            var r = await Run("public class A { B _b; A _self; }\npublic class B { A _a; C _c; }\npublic class C { A _a; B _b; }");
            Assert.AreEqual(0, r.Ndw.Length, "a cycle that stores no NDArray anywhere is clean");
        }

        [TestMethod]
        public async Task CycleWithAnArray_FlagsEveryTypeThatReachesIt()
        {
            var r = await Run("public class A { B _b; }\npublic class B { C _c; }\npublic class C { A _a; NDArray _x; }");
            Assert.AreEqual(3, r.CountOf("NDW016"), "every type on the cycle holds the array through the ring");
        }

        [TestMethod]
        public async Task SelfReferentialHolder_IsFlaggedOnce()
        {
            var r = await Run("public class Node { Node _next; NDArray _v; }");
            Assert.AreEqual(1, r.CountOf("NDW016"));
        }

        [TestMethod]
        public async Task DeepContagionChain_FlagsEveryLevel()
        {
            const int depth = 25;
            var sb = new StringBuilder();
            for (int i = 0; i < depth; i++)
                sb.Append("public class L").Append(i).Append(" { ")
                  .Append(i + 1 < depth ? "L" + (i + 1) + " _next;" : "NDArray _leaf;")
                  .AppendLine(" }");
            var r = await Run(sb.ToString());
            Assert.AreEqual(depth, r.CountOf("NDW016"), "each level of a chain that ends in an array stores it transitively");
        }

        [TestMethod]
        public async Task DeepChain_OfDisposables_DisposingEachLevel_IsClean()
        {
            const int depth = 12;
            var sb = new StringBuilder();
            for (int i = 0; i < depth; i++)
                sb.Append("public class L").Append(i).Append(" : IDisposable { ")
                  .Append(i + 1 < depth ? "L" + (i + 1) + " _next; public void Dispose() => _next?.Dispose();" : "NDArray _leaf; public void Dispose() => _leaf?.Dispose();")
                  .AppendLine(" }");
            var r = await Run(sb.ToString());
            Assert.AreEqual(0, r.Ndw.Length, "a chain that disposes at every level is clean");
        }

        [TestMethod]
        public async Task GenericDefinition_IsJudgedByItsConstraint_NotItsInstantiations()
        {
            var r = await Run("public class Box<T> { public T Value; }\npublic class Uses { Box<NDArray> _b; }\npublic class Other { Box<int> _i; }");
            Assert.AreEqual(1, r.CountOf("NDW016"), "Box<T> itself cannot know; Uses stores a Box<NDArray>; Other stores a Box<int>");
            Assert.IsTrue(r.Ndw.Single(d => d.Id == "NDW016").GetMessage().StartsWith("'Uses'"));
        }

        [TestMethod]
        public async Task GenericMemberOverAlias_ResolvesBySymbol()
        {
            var r = await AnalyzerTestHarness.RunAsync("using ND = NumSharp.NDArray;\nusing System.Collections.Generic;\npublic class H { List<ND> _l; }", "alias.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty);
            Assert.AreEqual(1, r.CountOf("NDW016"), "a using alias resolves to the same symbol");
        }

        [TestMethod]
        public async Task NullableAnnotation_StillHolds()
        {
            var r = await AnalyzerTestHarness.RunAsync("#nullable enable\nusing NumSharp;\npublic class H { NDArray? _a; }", "nullable_holder.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty);
            Assert.AreEqual(1, r.CountOf("NDW016"));
        }

        [TestMethod]
        public async Task NestedTypes_AreAnalyzedIndependently()
        {
            var r = await Run("public class Outer { NDArray _a; public class Inner : IDisposable { NDArray _b; public void Dispose() { } } public class Clean { int _n; } }");
            Assert.AreEqual(1, r.CountOf("NDW016"), "the outer type stores and is not disposable");
            Assert.AreEqual(1, r.CountOf("NDW017"), "the inner disposable forgets its own member");
        }

        [TestMethod]
        public async Task PartialType_IsJudgedAsAWhole()
        {
            var r = await Run("public partial class P : IDisposable { NDArray _a; }\npublic partial class P { NDArray _b; public void Dispose() { _a?.Dispose(); _b?.Dispose(); } }");
            Assert.AreEqual(0, r.Ndw.Length, "members and Dispose across partial parts are one type");
        }

        [TestMethod]
        [Timeout(60000)]
        public async Task VeryWideType_Completes_AndCountsEveryForgottenMember()
        {
            const int n = 300;
            var sb = new StringBuilder("public class W : IDisposable {\n");
            for (int i = 0; i < n; i++)
                sb.Append("  NDArray _f").Append(i).AppendLine(";");
            sb.AppendLine("  public void Dispose() { _f0?.Dispose(); }\n}");
            var r = await Run(sb.ToString());
            Assert.AreEqual(n - 1, r.CountOf("NDW017"), "every member except the one disposed is reported");
        }

        [TestMethod]
        public async Task ManyTypes_Complete()
        {
            const int n = 200;
            var sb = new StringBuilder();
            for (int i = 0; i < n; i++)
                sb.Append("public class C").Append(i).Append(" { NDArray _a; }").AppendLine();
            var r = await Run(sb.ToString());
            Assert.AreEqual(n, r.CountOf("NDW016"));
        }

        [TestMethod]
        public async Task StaticClass_InterfaceEnumDelegate_AreSkipped()
        {
            var r = await Run("public static class S { static NDArray _a; }\npublic interface I { NDArray A { get; } }\npublic enum E { X }\npublic delegate NDArray D(NDArray x);");
            Assert.AreEqual(0, r.Ndw.Length);
        }

        [TestMethod]
        public async Task NumSharpNotReferenced_NoOps()
        {
            var r = await AnalyzerTestHarness.RunWithoutNumSharpAsync(
                "public class NDArray { }\npublic class H { NDArray _a; }", "notref_holder.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "the NumSharp-free snippet must compile");
            Assert.AreEqual(0, r.Ndw.Length, "a look-alike NDArray from another assembly is not NumSharp's");
        }

        [TestMethod]
        public async Task MalformedType_DoesNotCrash()
        {
            var r = await AnalyzerTestHarness.RunAsync("using NumSharp;\npublic class H : IDisposable { NDArray _a; public void Dispose() { _a?.Dispose(", "broken_holder.cs");
            Assert.IsFalse(r.CompileErrors.IsEmpty, "a broken type must surface as compile errors, not throw");
        }

        [TestMethod]
        public async Task UnresolvedMemberType_IsNotAHolder()
        {
            var r = await AnalyzerTestHarness.RunAsync("using NumSharp;\npublic class H { Nonexistent _x; NDArray _a; }", "unresolved_holder.cs");
            Assert.AreEqual(1, r.CountOf("NDW016"), "the unresolved member contributes nothing; the NDArray one still counts");
        }

        [TestMethod]
        public async Task ArrayOfHolderTypes_IsContagious()
        {
            var r = await Run("public class O : IDisposable { NDArray _x; public void Dispose() => _x?.Dispose(); }\npublic class H { O[] _os; }\npublic class K : IDisposable { O[] _os; public void Dispose() { foreach (var o in _os) o.Dispose(); } }");
            Assert.AreEqual(1, r.CountOf("NDW016"), "an array of owning disposables is held");
            Assert.AreEqual(0, r.CountOf("NDW017"), "disposing each element clears it");
        }

        [TestMethod]
        public async Task ConstructedGenericHolder_InstantiatedOverNDArray_IsContagious()
        {
            var r = await Run("public class Cell<T> : IDisposable { public T V; public void Dispose() { (V as IDisposable)?.Dispose(); } }\npublic class H { Cell<NDArray> _c; }\npublic class J { Cell<int> _c; }");
            Assert.AreEqual(1, r.CountOf("NDW016"), "Cell<NDArray> stores an array; Cell<int> does not");
        }

        [TestMethod]
        public async Task DisposeThroughOtherInstanceOfSameType_CountsAsCallee()
        {
            var r = await Run("public class H : IDisposable { NDArray _a; H _other; public void Dispose() { Free(); } void Free() { _a?.Dispose(); _other?.Dispose(); } }");
            Assert.AreEqual(0, r.Ndw.Length, "a same-type helper reached through this or another instance is on the path");
        }
    }
}
