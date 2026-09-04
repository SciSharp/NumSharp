using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Metamorphic pairs for the ownership diagnostics (COVERAGE_PLAN technique T3, applied to
    ///     NDW016 / NDW017 and the contagion side of NDW012): the SAME type with one minimal edit must
    ///     flip the verdict — add IDisposable (NDW016 → NDW017), add the dispose call (NDW017 → clean),
    ///     mark the member or the type <c>[NDBorrowed]</c>, make a field static, turn storage into a
    ///     computed view, <c>using</c> a dropped holder instance, and so on. Proving the DELTA catches an
    ///     analyzer that warns for the wrong reason.
    /// </summary>
    [TestClass]
    public class OwnershipMetamorphicTests
    {
        private const string H = "using System;\nusing System.Collections.Generic;\nusing NumSharp;\n";

        private static async Task<(int n16, int n17, int n12)> Counts(string body)
        {
            var r = await AnalyzerTestHarness.RunAsync(H + body, "ownership_meta.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "metamorphic snippet must compile:\n" + body + "\n" + string.Join("\n", r.CompileErrors));
            return (r.CountOf("NDW016"), r.CountOf("NDW017"), r.CountOf("NDW012"));
        }

        [TestMethod]
        public async Task AddingIDisposable_TurnsNDW016_IntoNDW017()
        {
            Assert.AreEqual((1, 0, 0), await Counts("public class H { NDArray _a; }"));
            Assert.AreEqual((0, 1, 0), await Counts("public class H : IDisposable { NDArray _a; public void Dispose() { } }"));
        }

        [TestMethod]
        public async Task AddingTheDisposeCall_FlipsNDW017Clean()
        {
            Assert.AreEqual((0, 1, 0), await Counts("public class H : IDisposable { NDArray _a; public void Dispose() { } }"));
            Assert.AreEqual((0, 0, 0), await Counts("public class H : IDisposable { NDArray _a; public void Dispose() { _a?.Dispose(); } }"));
        }

        [TestMethod]
        public async Task AddingIAsyncDisposable_SatisfiesTheContractToo()
        {
            Assert.AreEqual((1, 0, 0), await Counts("public class H { NDArray _a; }"));
            Assert.AreEqual((0, 0, 0), await Counts("public class H : IAsyncDisposable { NDArray _a; public System.Threading.Tasks.ValueTask DisposeAsync() { _a?.Dispose(); return default; } }"));
        }

        [TestMethod]
        public async Task BorrowedMember_FlipsNDW016Clean()
        {
            Assert.AreEqual((1, 0, 0), await Counts("public class H { NDArray _a; }"));
            Assert.AreEqual((0, 0, 0), await Counts("public class H { [NDBorrowed] NDArray _a; }"));
        }

        [TestMethod]
        public async Task BorrowedMember_FlipsNDW017Clean()
        {
            Assert.AreEqual((0, 1, 0), await Counts("public class H : IDisposable { NDArray _a; public void Dispose() { } }"));
            Assert.AreEqual((0, 0, 0), await Counts("public class H : IDisposable { [NDBorrowed] NDArray _a; public void Dispose() { } }"));
        }

        [TestMethod]
        public async Task BorrowedType_FlipsEverythingClean()
        {
            Assert.AreEqual((1, 0, 0), await Counts("public class H { NDArray _a; List<NDArray> _b; }"));
            Assert.AreEqual((0, 0, 0), await Counts("[NDBorrowed] public class H { NDArray _a; List<NDArray> _b; }"));
        }

        [TestMethod]
        public async Task StaticField_FlipsNDW016Clean()
        {
            Assert.AreEqual((1, 0, 0), await Counts("public class H { NDArray _a; }"));
            Assert.AreEqual((0, 0, 0), await Counts("public class H { static NDArray _a; }"));
        }

        [TestMethod]
        public async Task ComputedProperty_FlipsNDW016Clean()
        {
            Assert.AreEqual((1, 0, 0), await Counts("public class H { public NDArray A { get; set; } }"));
            Assert.AreEqual((0, 0, 0), await Counts("public class H { int n; public NDArray A => NDArray.Scalar(n); }"));
        }

        [TestMethod]
        public async Task CarrierInterface_FlipsStructClean()
        {
            Assert.AreEqual((1, 0, 0), await Counts("public struct P { public NDArray A; }"));
            Assert.AreEqual((0, 0, 0), await Counts("public readonly struct P : INDArrayCarrier { public readonly NDArray A; public P(NDArray a) { A = a; } void INDArrayCarrier.YieldTo(NDScope s) { s.Returns(A); } }"));
        }

        [TestMethod]
        public async Task RefStructDisposePattern_FlipsNDW016Clean()
        {
            Assert.AreEqual((1, 0, 0), await Counts("public ref struct R { NDArray _a; }"));
            Assert.AreEqual((0, 0, 0), await Counts("public ref struct R { NDArray _a; public void Dispose() => _a?.Dispose(); }"));
        }

        [TestMethod]
        public async Task ContagionChain_ResolvesOneLevelAtATime()
        {
            // A owns and disposes; B stores A; C stores B. B and C each warn; fixing B leaves C warning.
            const string a = "public class A : IDisposable { NDArray _x; public void Dispose() => _x?.Dispose(); }\n";
            Assert.AreEqual((2, 0, 0), await Counts(a + "public class B { A _a; }\npublic class C { B _b; }"));
            Assert.AreEqual((1, 0, 0), await Counts(a + "public class B : IDisposable { A _a; public void Dispose() => _a?.Dispose(); }\npublic class C { B _b; }"));
            Assert.AreEqual((0, 0, 0), await Counts(a + "public class B : IDisposable { A _a; public void Dispose() => _a?.Dispose(); }\npublic class C : IDisposable { B _b; public void Dispose() => _b?.Dispose(); }"));
        }

        [TestMethod]
        public async Task ContagionStops_AtABorrowedDisposable()
        {
            Assert.AreEqual((1, 0, 0), await Counts("public class A : IDisposable { NDArray _x; public void Dispose() => _x?.Dispose(); }\npublic class B { A _a; }"));
            Assert.AreEqual((0, 0, 0), await Counts("[NDBorrowed] public class A : IDisposable { NDArray _x; public void Dispose() { } }\npublic class B { A _a; }"));
        }

        [TestMethod]
        public async Task Using_FlipsDroppedHolderInstanceClean()
        {
            const string h = "public class H : IDisposable { NDArray _x; public H(NDArray x) { _x = x; } public void Dispose() => _x?.Dispose(); }\n";
            Assert.AreEqual((0, 0, 1), await Counts(h + "public static class T { public static void M(NDArray a, NDArray b) { var h = new H(a + b); } }"));
            Assert.AreEqual((0, 0, 0), await Counts(h + "public static class T { public static void M(NDArray a, NDArray b) { using var h = new H(a + b); } }"));
        }

        [TestMethod]
        public async Task MakingTheHolderNonDisposable_MovesTheVerdictToTheType()
        {
            // The SAME dropped instance: a disposable holder leaks (NDW012); a non-disposable one is
            // nothing the method can reclaim — the TYPE is what warns (NDW016).
            Assert.AreEqual((0, 0, 1), await Counts("public class H : IDisposable { NDArray _x; public H(NDArray x) { _x = x; } public void Dispose() => _x?.Dispose(); }\npublic static class T { public static void M(NDArray a, NDArray b) { var h = new H(a + b); } }"));
            Assert.AreEqual((1, 0, 0), await Counts("public class H { NDArray _x; public H(NDArray x) { _x = x; } }\npublic static class T { public static void M(NDArray a, NDArray b) { var h = new H(a + b); } }"));
        }

        [TestMethod]
        public async Task Close_IsADisposeSpelling_ForNumSharpIterators()
        {
            Assert.AreEqual((0, 0, 1), await Counts("public static class T { public static void M(NDArray a) { var it = np.nditer(a); } }"));
            Assert.AreEqual((0, 0, 0), await Counts("public static class T { public static void M(NDArray a) { var it = np.nditer(a); it.close(); } }"));
            Assert.AreEqual((0, 0, 0), await Counts("public static class T { public static void M(NDArray a) { using var it = np.nditer(a); } }"));
        }

        [TestMethod]
        public async Task ForeachOverAProducedNDArray_LeaksUntilHeldByUsing()
        {
            Assert.AreEqual((0, 0, 1), await Counts("public static class T { public static void M(NDArray a, NDArray b) { foreach (var r in a + b) { } } }"));
            Assert.AreEqual((0, 0, 0), await Counts("public static class T { public static void M(NDArray a, NDArray b) { using var t = a + b; foreach (var r in t) { } } }"));
        }

        [TestMethod]
        public async Task HelperReachability_DecidesNDW017()
        {
            // The SAME helper that disposes; the only delta is whether Dispose calls it.
            Assert.AreEqual((0, 1, 0), await Counts("public class H : IDisposable { NDArray _a; void Free() => _a?.Dispose(); public void Dispose() { } }"));
            Assert.AreEqual((0, 0, 0), await Counts("public class H : IDisposable { NDArray _a; void Free() => _a?.Dispose(); public void Dispose() { Free(); } }"));
        }

        [TestMethod]
        public async Task OverridingTheInheritedDispose_FlipsNDW017Clean()
        {
            const string b = "public abstract class B : IDisposable { public void Dispose() => Dispose(true); protected virtual void Dispose(bool d) { } }\n";
            Assert.AreEqual((0, 1, 0), await Counts(b + "public class D : B { NDArray _a; }"));
            Assert.AreEqual((0, 0, 0), await Counts(b + "public class D : B { NDArray _a; protected override void Dispose(bool d) { _a?.Dispose(); base.Dispose(d); } }"));
        }

        [TestMethod]
        public async Task ForeachDispose_FlipsCollectionMemberClean()
        {
            Assert.AreEqual((0, 1, 0), await Counts("public class H : IDisposable { List<NDArray> _l = new List<NDArray>(); public void Dispose() { _l.Clear(); } }"));
            Assert.AreEqual((0, 0, 0), await Counts("public class H : IDisposable { List<NDArray> _l = new List<NDArray>(); public void Dispose() { foreach (var x in _l) x.Dispose(); _l.Clear(); } }"));
        }
    }
}
