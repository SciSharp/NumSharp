using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN §4 — the gate's negative half (supported carriers stay CLEAN) plus two more
    ///     rejections, and the T4 weaver-parity table: a curated set of return types asserted to be
    ///     gated (NDW003) exactly when the weaver would reject them, driven through the real gate
    ///     analyzer so the analyzer's carrier classification cannot drift from the weaver's contract.
    /// </summary>
    [TestClass]
    public class GateNegativeTests
    {
        [TestMethod]
        public Task GateNegativeScenarios_MatchTagsExactly()
            => AnalyzerTestHarness.AssertExactAsync("GateNegativeScenarios.cs");

        // GT-9: the gate is a symbol action over methods/properties, which does NOT visit LOCAL
        // functions — so a [NDScoped] local function draws no gate diagnostic (deferred to the weaver).
        // Pins that: a scoped-and-returning local function, called and returned, is entirely clean.
        [TestMethod]
        public async Task LocalFunctionAttribute_IsNotGated()
        {
            var src = "using NumSharp;\npublic static class T {\n" +
                      "  public static NDArray M(NDArray a) {\n" +
                      "    [NDScoped] NDArray L() { var t = a + 1.0; return t.copy(); }\n" +
                      "    return L();\n  }\n}";
            var r = await AnalyzerTestHarness.RunAsync(src, "localfn_attr.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "local-function-attribute snippet must compile");
            Assert.AreEqual(0, r.Ndw.Length, "a [NDScoped] local function draws no analyzer diagnostic (deferred to the weaver)");
        }

        [TestMethod]
        public async Task GateFixture_HasExactlyTheExpectedGateErrors()
        {
            var r = await AnalyzerTestHarness.RunFileAsync("GateNegativeScenarios.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "gate-negative fixture must compile");
            Assert.AreEqual(2, r.CountOf("NDW002"), "two hidden ref/in egresses");
            Assert.AreEqual(2, r.CountOf("NDW003"), "two unsupported carriers");
            Assert.AreEqual(0, r.CountOf("NDW012"), "every method here is scoped -> no leak warnings");
            Assert.AreEqual(4, r.Ndw.Length, "the supported-carrier methods must add NO diagnostics");
        }

        // ---- T4: the analyzer's carrier classification must match the weaver's Classify ----

        // Supported shapes: a [NDScoped]/[NDScopedAsync] method returning one of these is NOT gated.
        [DataTestMethod]
        [DataRow("NDArray", false)]
        [DataRow("NDArray[]", false)]
        [DataRow("(NDArray, NDArray)", false)]
        [DataRow("(NDArray, NDArray, NDArray)", false)]
        [DataRow("(NDArray, NDArray, NDArray, NDArray, NDArray)", false)]
        [DataRow("NumSharp.Generic.NDArray<double>", false)]
        [DataRow("NumSharp.Backends.Unmanaged.IArraySlice", false)]
        [DataRow("NumSharp.Backends.UnmanagedStorage", false)]
        [DataRow("int", false)]
        [DataRow("double", false)]
        [DataRow("bool", false)]
        [DataRow("void", false)]
        // Tuple carriers with component nuance (weaver's IsSupportedTupleCarrier): a component that
        // carries NO NDArray constrains nothing; every ND-carrying component must be a shape the
        // runtime Returns(ITuple) dispatch sees through (nested tuple, NDArray[], carrier, buffer).
        [DataRow("((NDArray, NDArray), NDArray)", false)]
        [DataRow("(NDArray[], NDArray)", false)]
        [DataRow("(NDArray, int)", false)]
        [DataRow("(NDArray, int, (NDArray, NDArray))", false)]
        [DataRow("(object, NDArray)", false)]
        [DataRow("(System.Collections.Generic.List<NDArray>, NDArray)", true)]
        [DataRow("(System.Threading.Tasks.Task<NDArray>, NDArray)", true)]
        // Unsupported shapes: gated with NDW003.
        [DataRow("System.Collections.Generic.List<NDArray>", true)]
        [DataRow("object", true)]
        public async Task SyncCarrierParity(string returnType, bool gated)
        {
            var src = "using NumSharp;\npublic static class T {\n" +
                      $"  [NDScoped] public static {returnType} M(NDArray a) => throw null;\n}}";
            var r = await AnalyzerTestHarness.RunAsync(src, "sync_carrier.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "carrier snippet must compile:\n" + src);
            Assert.AreEqual(gated ? 1 : 0, r.CountOf("NDW003"),
                $"return type '{returnType}' should {(gated ? "" : "NOT ")}be gated (NDW003)");
        }

        // Task/ValueTask-of-carrier shapes under [NDScopedAsync]: the carrier is unwrapped once.
        [DataTestMethod]
        [DataRow("System.Threading.Tasks.Task<NDArray>", false)]
        [DataRow("System.Threading.Tasks.ValueTask<NDArray>", false)]
        [DataRow("System.Threading.Tasks.Task<NDArray[]>", false)]
        [DataRow("System.Threading.Tasks.Task<(NDArray, NDArray)>", false)]
        [DataRow("System.Threading.Tasks.Task", false)]
        [DataRow("System.Threading.Tasks.ValueTask", false)]
        [DataRow("System.Threading.Tasks.Task<System.Collections.Generic.List<NDArray>>", true)]
        [DataRow("System.Threading.Tasks.Task<object>", true)]
        // Component nuance rides through the task unwrap; a NESTED task is refused outright (its
        // completion outlives the scope's deferral lifecycle — weaver parity).
        [DataRow("System.Threading.Tasks.Task<((NDArray, NDArray), NDArray)>", false)]
        [DataRow("System.Threading.Tasks.Task<(System.Collections.Generic.List<NDArray>, NDArray)>", true)]
        [DataRow("System.Threading.Tasks.Task<System.Threading.Tasks.Task<NDArray>>", true)]
        [DataRow("System.Threading.Tasks.ValueTask<System.Threading.Tasks.Task<NDArray>>", true)]
        public async Task AsyncCarrierParity(string returnType, bool gated)
        {
            var src = "using NumSharp;\npublic static class T {\n" +
                      $"  [NDScopedAsync] public static {returnType} M(NDArray a) => throw null;\n}}";
            var r = await AnalyzerTestHarness.RunAsync(src, "async_carrier.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "carrier snippet must compile:\n" + src);
            Assert.AreEqual(gated ? 1 : 0, r.CountOf("NDW003"),
                $"return type '{returnType}' should {(gated ? "" : "NOT ")}be gated (NDW003)");
        }

        // ---- ref/out parameter parity: NDW002 widens to ANY NDArray-carrying ref/in shape, and
        //      NDW015 gates an out shape the out-escape cannot yield — mirroring the weaver's
        //      CarriesNDArray/ClassifyOutParam exactly. A shape carrying NO NDArray is never gated
        //      (R2 conservatism), and every supported out shape must stay clean (they ARE woven:
        //      out tuple → Returns(ITuple), out carrier → YieldTo, out buffer → counted ref).

        [DataTestMethod]
        // ref/in over an NDArray-carrying shape → NDW002 (hidden egress).
        [DataRow("ref", "(NDArray, int)", "NDW002")]
        [DataRow("ref", "System.Collections.Generic.List<NDArray>", "NDW002")]
        [DataRow("in", "(NDArray, NDArray)", "NDW002")]
        [DataRow("ref", "NDArray", "NDW002")]
        // out over a supported carrier shape → clean (the FINAL value is yielded by the weave).
        [DataRow("out", "NDArray", null)]
        [DataRow("out", "NDArray[]", null)]
        [DataRow("out", "(NDArray, NDArray)", null)]
        [DataRow("out", "(NDArray, int, (NDArray, NDArray))", null)]
        [DataRow("out", "NumSharp.Backends.Unmanaged.IArraySlice", null)]
        // out over an ND-carrying shape the out-escape cannot yield → NDW015.
        [DataRow("out", "System.Collections.Generic.List<NDArray>", "NDW015")]
        [DataRow("out", "System.Threading.Tasks.Task<NDArray>", "NDW015")]
        [DataRow("out", "NDArray[,]", "NDW015")]
        // shapes carrying NO NDArray are never the scope's business, whatever the ref kind.
        [DataRow("ref", "(int, double)", null)]
        [DataRow("out", "(int, string)", null)]
        [DataRow("out", "object", null)]
        public async Task RefOutParamParity(string modifier, string paramType, string expected)
        {
            var src = "using NumSharp;\npublic static class T {\n" +
                      $"  [NDScoped] public static void M({modifier} {paramType} p) => throw null;\n}}";
            var r = await AnalyzerTestHarness.RunAsync(src, "refout_param.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "ref/out snippet must compile:\n" + src);
            if (expected == null)
                Assert.AreEqual(0, r.Ndw.Length, $"'{modifier} {paramType}' must stay clean");
            else
            {
                Assert.AreEqual(1, r.CountOf(expected), $"'{modifier} {paramType}' should be gated ({expected})");
                Assert.AreEqual(1, r.Ndw.Length, $"'{modifier} {paramType}' should produce ONLY {expected}");
            }
        }

        // A generic out parameter CONSTRAINED to NDArray carries one the rewrite cannot yield
        // (Returns<T> instantiation over an open method parameter is deliberately not emitted).
        [TestMethod]
        public async Task OutGenericConstrainedToNDArray_IsGated()
        {
            var src = "using NumSharp;\npublic static class T {\n" +
                      "  [NDScoped] public static bool TryGet<T>(out T r) where T : NDArray => throw null;\n}";
            var r = await AnalyzerTestHarness.RunAsync(src, "out_generic.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "generic-out snippet must compile");
            Assert.AreEqual(1, r.CountOf("NDW015"), "out T where T : NDArray is an NDArray-carrying out the weave cannot yield");
        }

        // GT-8 — a custom [AsyncMethodBuilder] task-like ASYNC method: the weaver weaves it (the SM
        // result type is validated at IL time from SetResult's argument), so the gate must NOT flag
        // the task-like TYPE itself. Pinned against the old false positive (NDW003 on 'MyTask<NDArray>').
        [TestMethod]
        public async Task CustomTaskLikeAsyncMethod_IsNotGated()
        {
            var src = @"using System;
using System.Runtime.CompilerServices;
using NumSharp;

[AsyncMethodBuilder(typeof(MyTaskBuilder<>))]
public struct MyTask<T>
{
    public Awaiter GetAwaiter() => default;
    public struct Awaiter : ICriticalNotifyCompletion
    {
        public bool IsCompleted => true;
        public T GetResult() => default;
        public void OnCompleted(Action c) { }
        public void UnsafeOnCompleted(Action c) { }
    }
}

public struct MyTaskBuilder<T>
{
    public static MyTaskBuilder<T> Create() => default;
    public void Start<TSM>(ref TSM sm) where TSM : IAsyncStateMachine => sm.MoveNext();
    public void SetStateMachine(IAsyncStateMachine sm) { }
    public void SetResult(T r) { }
    public void SetException(Exception e) { }
    public MyTask<T> Task => default;
    public void AwaitOnCompleted<TA, TSM>(ref TA a, ref TSM sm) where TA : INotifyCompletion where TSM : IAsyncStateMachine { }
    public void AwaitUnsafeOnCompleted<TA, TSM>(ref TA a, ref TSM sm) where TA : ICriticalNotifyCompletion where TSM : IAsyncStateMachine { }
}

public static class T
{
    [NDScopedAsync]
    public static async MyTask<NDArray> M(NDArray a)
    {
        await System.Threading.Tasks.Task.Yield();
        return a + 1.0;
    }
}";
            var r = await AnalyzerTestHarness.RunAsync(src, "custom_tasklike.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty, "custom task-like snippet must compile");
            Assert.AreEqual(0, r.Ndw.Length, "an async custom task-like is the weaver's to validate — no gate diagnostic");
        }
    }
}
