using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     The weaver's TRANSFORM, shape by shape, on Release AND Debug IL (the two IL layouts every real
    ///     build produces — Debug keeps extra locals, spills every return value and places its <c>ret</c>
    ///     differently, so a rewrite that only ever saw optimized IL is not proven): every return shape
    ///     the weaver documents (bare, tuple, ITuple, array, carrier struct, generic <c>NDArray&lt;T&gt;</c>,
    ///     scalar, void), every egress (return, <c>out</c>, <c>out</c> tuple, yield, await, deferred task),
    ///     every body layout (try/finally with a return inside, four nested finallies, a switch with a
    ///     return per arm, a return inside a loop, a catch-then-return, a throw), and the member kinds
    ///     (property, local function, operators, generic method, recursion). Each shape is EXECUTED after
    ///     weaving: the result values are asserted and the fixture records its dropped temporaries so
    ///     their reclamation — the whole point of the weave — is asserted on real arrays.
    /// </summary>
    [TestClass]
    public class WeaverShapeTests
    {
        private const string Source = @"using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.Generic;

public readonly struct Pair : INDArrayCarrier
{
    public readonly NDArray P;
    public readonly NDArray M;
    public Pair(NDArray p, NDArray m) { P = p; M = m; }
    void INDArrayCarrier.YieldTo(NDScope scope) { scope.Returns(P); scope.Returns(M); }
}

public sealed class Box
{
    public NDArray V;
    public Box(NDArray v) { V = v; }
    [NDScoped] public static NDArray operator +(Box x, NDArray y) { var t = x.V + y; S.T = t; return t * 1.0; }
    [NDScoped] public static implicit operator NDArray(Box b) { var t = b.V + 1.0; S.T = t; return t.copy(); }
}

public static class S
{
    public static NDArray T, U, R;

    [NDScoped] public static NDArray Bare(NDArray a) { var t = a + 1.0; T = t; var u = t * 2.0; U = u; return u - 1.0; }
    [NDScoped] public static (NDArray, NDArray) Tuple2(NDArray a) { var t = a + 2.0; T = t; return (t - 1.0, a * 3.0); }
    [NDScoped] public static (NDArray, NDArray, NDArray, NDArray, NDArray) Tuple5(NDArray a) { var t = a + 1.0; T = t; return (t, t + 1.0, t + 2.0, t + 3.0, t + 4.0); }
    [NDScoped] public static NDArray[] Arr(NDArray a) { var t = a * 2.0; T = t; return new NDArray[] { t + 1.0, t - 1.0 }; }
    [NDScoped] public static Pair Carrier(NDArray a) { var sq = a * a; T = sq; return new Pair(a + 1.0, sq - 1.0); }
    [NDScoped] public static NDArray<double> Typed(NDArray a) { var t = a + 1.0; T = t; return (t * 2.0).MakeGeneric<double>(); }
    [NDScoped] public static NDArray Prop { get { var b = np.arange(1.0, 4.0); T = b; var t = b + 1.0; return t * 1.0; } }
    [NDScoped] public static bool Out1(NDArray a, out NDArray r) { var t = a + 5.0; T = t; r = t - 4.0; return true; }
    [NDScoped] public static void Out2(NDArray a, out NDArray r) { var t = a * 4.0; T = t; r = t / 2.0; }
    [NDScoped] public static void OutTuple(NDArray a, out (NDArray, NDArray) r) { var t = a + 1.0; T = t; r = (t - 1.0, t + 1.0); }
    [NDScoped] public static NDArray TryFinally(NDArray a) { var t = a + 1.0; T = t; try { var u = t + 0.0; return u; } finally { GC.KeepAlive(t); } }
    [NDScoped] public static NDArray DeepTry(NDArray a)
    {
        var t = a + 1.0; T = t;
        try { try { try { try { var u = t + 1.0; U = u; return u + 1.0; } finally { GC.KeepAlive(a); } } finally { GC.KeepAlive(t); } } finally { GC.KeepAlive(a); } } finally { GC.KeepAlive(t); }
    }
    [NDScoped] public static double Scalar(NDArray a) { var t = a + 1.0; T = t; return t.GetDouble(0); }
    [NDScoped] public static NDArray Hand(NDArray a) { using var scope = NDScope.Open(); var t = a + 1.0; T = t; return scope.Returns(t * 1.0); }
    [NDScoped] public static NDArray Switch(NDArray a, int k)
    {
        var t = a + 1.0; T = t;
        switch (k % 4)
        {
            case 0: { var u = t * 1.0; return u; }
            case 1: { var v = t + 0.0; U = v; return v; }
            case 2: return t - 0.0;
            default: { var w = t * 2.0; U = w; return w - t; }
        }
    }
    [NDScoped] public static NDArray LoopReturn(NDArray a)
    {
        foreach (var k in new[] { 1, 2, 3 })
        {
            var t = a + k; T = t;
            if (k == 2) return t.copy();
        }
        return a.copy();
    }
    [NDScoped] public static NDArray CatchReturn(NDArray a)
    {
        try { var t = a + 1.0; T = t; throw new InvalidOperationException(""boom""); }
        catch (InvalidOperationException) { var u = a + 2.0; U = u; return u.copy(); }
    }
    [NDScoped] public static NDArray Throws(NDArray a) { var t = a + 1.0; T = t; throw new InvalidOperationException(""boom""); }
    [NDScoped] public static NDArray LongBody(NDArray a)
    {
        var t0 = a + 1.0; T = t0; var t = t0;
        t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0;
        t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0;
        t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0;
        t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0; t = t + 1.0;
        return t;
    }
    [NDScoped] public static NDArray Rec(NDArray a, int n)
    {
        var t = a + 1.0;
        if (n == 0) { T = t; return t.copy(); }
        var r = Rec(t, n - 1); R = r;
        return r + 0.0;
    }
    [NDScoped] public static NDArray Gen<TArg>(TArg x, NDArray a) { var t = a + 1.0; T = t; return t.copy(); }
    public static NDArray Outer(NDArray a)
    {
        [NDScoped] NDArray L(NDArray x) { var t = x + 1.0; T = t; return t.copy(); }
        return L(a);
    }
    [NDScopedAsync] public static async Task<NDArray> AsyncM(NDArray a) { var t = a + 1.0; T = t; await Task.Yield(); return t * 2.0; }
    [NDScopedAsync] public static async ValueTask<NDArray> AsyncVT(NDArray a) { var t = a + 1.0; T = t; await Task.Yield(); return t * 3.0; }
    [NDScopedAsync] public static async Task<NDArray> AsyncThrows(NDArray a) { var t = a + 1.0; T = t; await Task.Yield(); throw new InvalidOperationException(""boom""); }
    [NDScopedAsync] public static Task<NDArray> Deferred(NDArray a) { var t = a + 2.0; T = t; return Task.FromResult(t - 1.0); }
    [NDScoped] public static IEnumerable<NDArray> Iter(NDArray a) { var t = a + 1.0; T = t; yield return t + 1.0; yield return t * 3.0; }
    [NDScopedAsync] public static async IAsyncEnumerable<NDArray> AIter(NDArray a) { var t = a + 1.0; T = t; yield return t + 1.0; await Task.Yield(); yield return t * 3.0; }
    [NDScoped] public static void OutCarrier(NDArray a, out Pair r) { var t = a + 1.0; T = t; r = new Pair(t - 1.0, t + 1.0); }
    [NDScopedAsync] public static async Task AsyncFire(NDArray a) { var t = a + 1.0; T = t; await Task.Yield(); U = t * 2.0; }
    [NDScopedAsync] public static Task TaskPlain(NDArray a) { var t = a + 1.0; T = t; return Task.CompletedTask; }
    [NDScopedAsync] public static async ValueTask ValueTaskFire(NDArray a) { var t = a + 1.0; T = t; await Task.Yield(); }
}
";

        private static readonly Lazy<WeaveRun[]> Runs = new(() => WeaverTestHarness.CompileAndWeaveBoth(Source, "Shapes").ToArray());

        private static NDArray Input() => np.arange(3).astype(np.float64);

        private static NDArray Static(WeaveRun run, string field) => (NDArray)run.GetStatic("S", field);

        private static void Reset(WeaveRun run)
        {
            foreach (var f in new[] { "T", "U", "R" })
                run.LoadType("S").GetField(f).SetValue(null, null);
        }

        private static string Label(WeaveRun run) => run.Fixture.Optimized ? "Release IL" : "Debug IL";

        private static void AssertTemp(WeaveRun run, string field, string what)
        {
            var t = Static(run, field);
            Assert.IsNotNull(t, $"{Label(run)} — {what}: the fixture did not record its temporary '{field}'");
            Assert.IsTrue(t.IsDisposed, $"{Label(run)} — {what}: temporary '{field}' must be reclaimed by the woven scope");
        }

        private static void AssertAlive(WeaveRun run, NDArray nd, double[] expected, string what)
        {
            Assert.IsFalse(nd.IsDisposed, $"{Label(run)} — {what}: the yielded value must survive");
            CollectionAssert.AreEqual(expected, nd.ToArray<double>(), $"{Label(run)} — {what}: values");
        }

        [TestMethod]
        public void Fixture_WeavesEveryAttributedBody_AndSkipsTheHandScopedOne()
        {
            foreach (var run in Runs.Value)
            {
                Assert.AreEqual(0, run.Result.Errors, run.Report);
                using var asm = run.ReadCecil();
                int attributed = WeaveRun.AllMethods(asm.MainModule).Count(m =>
                    m.CustomAttributes.Any(a => a.AttributeType.Name is "NDScopedAttribute" or "NDScopedAsyncAttribute"))
                    + asm.MainModule.Types.SelectMany(t => t.Properties).Count(p => p.CustomAttributes.Any(a => a.AttributeType.Name == "NDScopedAttribute"));
                Assert.AreEqual(1, run.Result.Skipped, $"{Label(run)}: exactly the hand-scoped body is skipped");
                Assert.AreEqual(attributed, run.Result.Woven + run.Result.Skipped, $"{Label(run)}: every attributed member is either woven or skipped:\n{run.Report}");
                Assert.IsTrue(run.Result.Woven >= 28, $"{Label(run)}: the fixture should carry 28+ woven shapes (got {run.Result.Woven})");
                foreach (var m in WeaveRun.AllMethods(asm.MainModule))
                {
                    bool own = m.CustomAttributes.Any(a => a.AttributeType.Name is "NDScopedAttribute" or "NDScopedAsyncAttribute")
                               || (m.IsGetter && m.DeclaringType.Properties.Any(p => p.GetMethod == m && p.CustomAttributes.Any(a => a.AttributeType.Name == "NDScopedAttribute")));
                    if (own)
                        Assert.IsTrue(WeaveRun.HasScopeLocal(m), $"{Label(run)}: {m.DeclaringType.Name}.{m.Name} carries no scope local");
                }
            }
        }

        [TestMethod]
        public void Bare_Tuple_Array_Carrier_Typed()
        {
            foreach (var run in Runs.Value)
            {
                Reset(run);
                var bare = (NDArray)run.Invoke("S", "Bare", null, Input());
                AssertAlive(run, bare, new[] { 1.0, 3.0, 5.0 }, "Bare");
                AssertTemp(run, "T", "Bare"); AssertTemp(run, "U", "Bare");

                Reset(run);
                var tuple = ((NDArray, NDArray))run.Invoke("S", "Tuple2", null, Input());
                AssertAlive(run, tuple.Item1, new[] { 1.0, 2.0, 3.0 }, "Tuple2.Item1");
                AssertAlive(run, tuple.Item2, new[] { 0.0, 3.0, 6.0 }, "Tuple2.Item2");
                AssertTemp(run, "T", "Tuple2");

                Reset(run);
                var five = ((NDArray, NDArray, NDArray, NDArray, NDArray))run.Invoke("S", "Tuple5", null, Input());
                AssertAlive(run, five.Item1, new[] { 1.0, 2.0, 3.0 }, "Tuple5.Item1 (the temp itself, yielded)");
                AssertAlive(run, five.Item5, new[] { 5.0, 6.0, 7.0 }, "Tuple5.Item5");
                Assert.IsFalse(Static(run, "T").IsDisposed, $"{Label(run)}: a temp that is YIELDED as a tuple component survives");

                Reset(run);
                var arr = (NDArray[])run.Invoke("S", "Arr", null, Input());
                AssertAlive(run, arr[0], new[] { 1.0, 3.0, 5.0 }, "Arr[0]");
                AssertAlive(run, arr[1], new[] { -1.0, 1.0, 3.0 }, "Arr[1]");
                AssertTemp(run, "T", "Arr");

                Reset(run);
                var pair = run.Invoke("S", "Carrier", null, Input());
                var p = (NDArray)pair.GetType().GetField("P").GetValue(pair);
                var m = (NDArray)pair.GetType().GetField("M").GetValue(pair);
                AssertAlive(run, p, new[] { 1.0, 2.0, 3.0 }, "Carrier.P");
                AssertAlive(run, m, new[] { -1.0, 0.0, 3.0 }, "Carrier.M");
                AssertTemp(run, "T", "Carrier");

                Reset(run);
                var typed = (NDArray)run.Invoke("S", "Typed", null, Input());
                AssertAlive(run, typed, new[] { 2.0, 4.0, 6.0 }, "Typed (NDArray<double>)");
                AssertTemp(run, "T", "Typed");
            }
        }

        [TestMethod]
        public void Property_OutParams_OutTuple_Scalar()
        {
            foreach (var run in Runs.Value)
            {
                Reset(run);
                var prop = (NDArray)run.GetProperty("S", "Prop", null);
                AssertAlive(run, prop, new[] { 2.0, 3.0, 4.0 }, "Prop");
                AssertTemp(run, "T", "Prop");

                Reset(run);
                var args = new object[] { Input(), null };
                var ok = (bool)run.Invoke("S", "Out1", null, args);
                Assert.IsTrue(ok);
                AssertAlive(run, (NDArray)args[1], new[] { 1.0, 2.0, 3.0 }, "Out1 (bool + out)");
                AssertTemp(run, "T", "Out1");

                Reset(run);
                args = new object[] { Input(), null };
                run.Invoke("S", "Out2", null, args);
                AssertAlive(run, (NDArray)args[1], new[] { 0.0, 2.0, 4.0 }, "Out2 (void + out)");
                AssertTemp(run, "T", "Out2");

                Reset(run);
                args = new object[] { Input(), null };
                run.Invoke("S", "OutTuple", null, args);
                var tup = ((NDArray, NDArray))args[1];
                AssertAlive(run, tup.Item1, new[] { 0.0, 1.0, 2.0 }, "OutTuple.Item1");
                AssertAlive(run, tup.Item2, new[] { 2.0, 3.0, 4.0 }, "OutTuple.Item2");
                AssertTemp(run, "T", "OutTuple");

                Reset(run);
                var scalar = (double)run.Invoke("S", "Scalar", null, Input());
                Assert.AreEqual(1.0, scalar);
                AssertTemp(run, "T", "Scalar (scope-only weave)");
            }
        }

        [TestMethod]
        public void ControlFlow_TryFinally_DeepTry_Switch_Loop_Catch_Throw()
        {
            foreach (var run in Runs.Value)
            {
                Reset(run);
                AssertAlive(run, (NDArray)run.Invoke("S", "TryFinally", null, Input()), new[] { 1.0, 2.0, 3.0 }, "TryFinally");
                AssertTemp(run, "T", "TryFinally");

                Reset(run);
                AssertAlive(run, (NDArray)run.Invoke("S", "DeepTry", null, Input()), new[] { 3.0, 4.0, 5.0 }, "DeepTry (4 nested finallies)");
                AssertTemp(run, "T", "DeepTry"); AssertTemp(run, "U", "DeepTry");

                for (int k = 0; k < 4; k++)
                {
                    // Every arm evaluates to a + 1: arm 0 returns t*1, arm 1 returns t+0 (the recorded U
                    // itself — yielded, so it survives), arm 2 returns t-0, arm 3 returns w - t (w dropped).
                    Reset(run);
                    AssertAlive(run, (NDArray)run.Invoke("S", "Switch", null, Input(), k), new[] { 1.0, 2.0, 3.0 }, $"Switch arm {k}");
                    AssertTemp(run, "T", $"Switch arm {k}");
                    if (k == 3)
                        AssertTemp(run, "U", "Switch arm 3 (arm-local temp, yielded only via a derived value)");
                    if (k == 1)
                        Assert.IsFalse(Static(run, "U").IsDisposed, $"{Label(run)}: Switch arm 1 returns its arm-local temp itself — it survives");
                }

                Reset(run);
                AssertAlive(run, (NDArray)run.Invoke("S", "LoopReturn", null, Input()), new[] { 2.0, 3.0, 4.0 }, "LoopReturn");
                AssertTemp(run, "T", "LoopReturn");

                Reset(run);
                AssertAlive(run, (NDArray)run.Invoke("S", "CatchReturn", null, Input()), new[] { 2.0, 3.0, 4.0 }, "CatchReturn");
                AssertTemp(run, "T", "CatchReturn (the temp built before the throw)");
                AssertTemp(run, "U", "CatchReturn (the temp built in the handler)");

                Reset(run);
                var input = Input();
                Assert.ThrowsException<InvalidOperationException>(() => run.Invoke("S", "Throws", null, input), $"{Label(run)}: Throws must propagate");
                AssertTemp(run, "T", "Throws (reclaimed on the exception path)");
                Assert.IsFalse(input.IsDisposed, $"{Label(run)}: the input is never touched, even on a throw");
            }
        }

        [TestMethod]
        public void LongBody_Recursion_GenericMethod_LocalFunction_Operators()
        {
            foreach (var run in Runs.Value)
            {
                Reset(run);
                AssertAlive(run, (NDArray)run.Invoke("S", "LongBody", null, Input()), new[] { 40.0, 41.0, 42.0 }, "LongBody (40 rebinds)");
                AssertTemp(run, "T", "LongBody (the first intermediate)");

                Reset(run);
                AssertAlive(run, (NDArray)run.Invoke("S", "Rec", null, Input(), 3), new[] { 4.0, 5.0, 6.0 }, "Rec (nested scopes, depth 4)");
                AssertTemp(run, "T", "Rec (leaf temp)");
                AssertTemp(run, "R", "Rec (an inner result dropped by its caller — reparented into the caller's scope)");

                Reset(run);
                var gen = run.LoadType("S").GetMethod("Gen").MakeGenericMethod(typeof(int));
                AssertAlive(run, (NDArray)gen.Invoke(null, new object[] { 1, Input() }), new[] { 1.0, 2.0, 3.0 }, "Gen<int>");
                AssertTemp(run, "T", "Gen<int>");

                Reset(run);
                AssertAlive(run, (NDArray)run.Invoke("S", "Outer", null, Input()), new[] { 1.0, 2.0, 3.0 }, "Outer → [NDScoped] local function");
                AssertTemp(run, "T", "local function");
                using (var asm = run.ReadCecil())
                {
                    var local = WeaveRun.AllMethods(asm.MainModule).Single(m => m.DeclaringType.Name == "S" && m.Name.Contains("g__L"));
                    Assert.IsTrue(WeaveRun.HasScopeLocal(local), $"{Label(run)}: the compiler-generated local function is woven");
                }

                Reset(run);
                var box = run.New("Box", Input());
                AssertAlive(run, (NDArray)run.Invoke("Box", "op_Addition", null, box, Input()), new[] { 0.0, 2.0, 4.0 }, "operator +");
                AssertTemp(run, "T", "operator +");

                Reset(run);
                AssertAlive(run, (NDArray)run.Invoke("Box", "op_Implicit", null, box), new[] { 1.0, 2.0, 3.0 }, "implicit conversion operator");
                AssertTemp(run, "T", "op_Implicit");
            }
        }

        [TestMethod]
        public async Task Async_ValueTask_AsyncThrow_Deferred_Iterator_AsyncIterator()
        {
            foreach (var run in Runs.Value)
            {
                Reset(run);
                var asyncResult = await (Task<NDArray>)run.Invoke("S", "AsyncM", null, Input());
                AssertAlive(run, asyncResult, new[] { 2.0, 4.0, 6.0 }, "AsyncM");
                AssertTemp(run, "T", "AsyncM (reclaimed at SetResult)");

                Reset(run);
                var vt = await (ValueTask<NDArray>)run.Invoke("S", "AsyncVT", null, Input());
                AssertAlive(run, vt, new[] { 3.0, 6.0, 9.0 }, "AsyncVT");
                AssertTemp(run, "T", "AsyncVT");

                Reset(run);
                var throwing = (Task<NDArray>)run.Invoke("S", "AsyncThrows", null, Input());
                await Assert.ThrowsExceptionAsync<InvalidOperationException>(async () => await throwing);
                AssertTemp(run, "T", "AsyncThrows (reclaimed at SetException)");

                Reset(run);
                var deferred = await (Task<NDArray>)run.Invoke("S", "Deferred", null, Input());
                AssertAlive(run, deferred, new[] { 1.0, 2.0, 3.0 }, "Deferred (non-async Task egress)");
                AssertTemp(run, "T", "Deferred");

                Reset(run);
                var items = ((IEnumerable<NDArray>)run.Invoke("S", "Iter", null, Input())).ToList();
                AssertAlive(run, items[0], new[] { 2.0, 3.0, 4.0 }, "Iter[0]");
                AssertAlive(run, items[1], new[] { 3.0, 6.0, 9.0 }, "Iter[1]");
                AssertTemp(run, "T", "Iter (reclaimed at the final MoveNext)");

                // Early break: the enumerator's Dispose seam reclaims mid-iteration.
                Reset(run);
                NDArray first;
                using (var e = ((IEnumerable<NDArray>)run.Invoke("S", "Iter", null, Input())).GetEnumerator())
                {
                    Assert.IsTrue(e.MoveNext());
                    first = e.Current;
                    Assert.IsFalse(Static(run, "T").IsDisposed, $"{Label(run)}: mid-iteration the scope is suspended, its temps alive");
                }

                AssertTemp(run, "T", "Iter (abandoned after one element — Dispose reclaims)");
                AssertAlive(run, first, new[] { 2.0, 3.0, 4.0 }, "Iter — the yielded element survives the abandonment");

                Reset(run);
                var collected = new List<NDArray>();
                await foreach (var x in (IAsyncEnumerable<NDArray>)run.Invoke("S", "AIter", null, Input()))
                    collected.Add(x);
                Assert.AreEqual(2, collected.Count);
                AssertAlive(run, collected[0], new[] { 2.0, 3.0, 4.0 }, "AIter[0]");
                AssertAlive(run, collected[1], new[] { 3.0, 6.0, 9.0 }, "AIter[1]");
                AssertTemp(run, "T", "AIter (reclaimed at completion)");
            }
        }

        [TestMethod]
        public void HandScoped_IsSkipped_AndStillReclaims()
        {
            foreach (var run in Runs.Value)
            {
                Assert.AreEqual(1, run.CountScopeCalls("S", "Hand", "Open"), $"{Label(run)}: the hand-written Open is the only one — not double-wrapped");
                Reset(run);
                AssertAlive(run, (NDArray)run.Invoke("S", "Hand", null, Input()), new[] { 1.0, 2.0, 3.0 }, "Hand");
                AssertTemp(run, "T", "Hand (the author's own scope)");
            }
        }

        [TestMethod]
        public async Task ResultlessTasks_AndOutCarrier()
        {
            foreach (var run in Runs.Value)
            {
                Reset(run);
                await (Task)run.Invoke("S", "AsyncFire", null, Input());
                AssertTemp(run, "T", "AsyncFire (async Task, no result — reclaimed at completion)");
                AssertTemp(run, "U", "AsyncFire (a temp built AFTER the await — the resumed segment is under the same scope)");

                Reset(run);
                await (Task)run.Invoke("S", "TaskPlain", null, Input());
                AssertTemp(run, "T", "TaskPlain (a completed resultless Task — reclaimed in the finally, no deferral)");

                Reset(run);
                await (ValueTask)run.Invoke("S", "ValueTaskFire", null, Input());
                AssertTemp(run, "T", "ValueTaskFire (async ValueTask, no result)");

                Reset(run);
                var args = new object[] { Input(), null };
                run.Invoke("S", "OutCarrier", null, args);
                var pair = args[1];
                var p = (NDArray)pair.GetType().GetField("P").GetValue(pair);
                var m = (NDArray)pair.GetType().GetField("M").GetValue(pair);
                AssertAlive(run, p, new[] { 0.0, 1.0, 2.0 }, "OutCarrier.P (an INDArrayCarrier out parameter is yielded through YieldTo)");
                AssertAlive(run, m, new[] { 2.0, 3.0, 4.0 }, "OutCarrier.M");
                AssertTemp(run, "T", "OutCarrier");
            }
        }

        [TestMethod]
        public void WovenMethods_AreThreadSafe_ScopesArePerThread()
        {
            // NDScope is thread-static: 64 concurrent invocations each open, fill and reclaim their own
            // scope, and no thread ever sees another's temps or has its result reclaimed under it.
            foreach (var run in Runs.Value)
            {
                var results = new NDArray[64];
                var errors = new System.Collections.Concurrent.ConcurrentQueue<string>();
                System.Threading.Tasks.Parallel.For(0, results.Length, i =>
                {
                    try
                    {
                        var a = Input() + i;
                        var r = (NDArray)run.Invoke("S", "Bare", null, a);
                        var tuple = ((NDArray, NDArray))run.Invoke("S", "Tuple2", null, a);
                        if (r.IsDisposed || tuple.Item1.IsDisposed)
                            errors.Enqueue($"thread {i}: a result was reclaimed");
                        var expected = new[] { 2 * i + 1.0, 2 * i + 3.0, 2 * i + 5.0 };
                        if (!expected.SequenceEqual(r.ToArray<double>()))
                            errors.Enqueue($"thread {i}: Bare = [{string.Join(",", r.ToArray<double>())}]");
                        results[i] = r;
                    }
                    catch (Exception e)
                    {
                        errors.Enqueue($"thread {i}: {e.GetType().Name}: {e.Message}");
                    }
                });
                Assert.AreEqual(0, errors.Count, $"{Label(run)}:\n  " + string.Join("\n  ", errors));
                for (int i = 0; i < results.Length; i++)
                    CollectionAssert.AreEqual(new[] { 2 * i + 1.0, 2 * i + 3.0, 2 * i + 5.0 }, results[i].ToArray<double>(), $"{Label(run)}: result {i} intact after all threads finished");
            }
        }

        [TestMethod]
        public void HugeBody_ManyLocals_LongBranches_Weave()
        {
            // 300 distinct locals push local indices past the short-form (ldloc.s) range and a long body
            // pushes branch distances past the short-form range: the weave's SimplifyMacros/OptimizeMacros
            // round-trip and its in-place ret rewrite must survive both, on Release and Debug IL.
            var sb = new StringBuilder("using System;\nusing NumSharp;\npublic static class H {\n  public static NDArray T;\n");
            sb.Append("  [NDScoped] public static NDArray M(NDArray a) {\n");
            for (int i = 0; i < 300; i++)
                sb.Append($"    var t{i} = a + {i}.0;\n");
            sb.Append("    T = t1;\n");
            sb.Append("    try { if (a.size == 3) { return t0 + t299; } } finally { GC.KeepAlive(t150); }\n");
            sb.Append("    return a.copy();\n  }\n}\n");
            foreach (var run in WeaverTestHarness.CompileAndWeaveBoth(sb.ToString(), "Huge"))
            {
                run.AssertWoven(1);
                Assert.IsTrue(run.HasFinallyDispose("H", "M"), $"{Label(run)}: the outer finally wraps the whole body");
                // Release IL keeps a `ret` per return; Debug IL funnels both through one epilogue `ret`
                // (a spilled return local + branches) — either way every ret is routed through Returns.
                int returns = run.CountScopeCalls("H", "M", "Returns");
                Assert.IsTrue(returns >= 1 && returns <= 2, $"{Label(run)}: every ret is routed through Returns (saw {returns})");
                var result = (NDArray)run.Invoke("H", "M", null, Input());
                AssertAlive(run, result, new[] { 299.0, 301.0, 303.0 }, "Huge");
                Assert.IsTrue(((NDArray)run.GetStatic("H", "T")).IsDisposed, $"{Label(run)}: a dropped local among 300 is reclaimed");
            }
        }

        [TestMethod]
        public void ValuesStable_UnderRepeatedCallsAndForcedCollections()
        {
            // The stress harness's "values under GC pressure" check, in-process: a wrong scope (a double
            // dispose, a missed Returns) surfaces as a pooled buffer reused under a live result — i.e.
            // corrupted values on a later call — which the per-call expected values catch.
            foreach (var run in Runs.Value)
            {
                var keep = new List<NDArray>();
                for (int i = 0; i < 25; i++)
                {
                    var a = Input();
                    var bare = (NDArray)run.Invoke("S", "Bare", null, a);
                    var tuple = ((NDArray, NDArray))run.Invoke("S", "Tuple2", null, a);
                    var arr = (NDArray[])run.Invoke("S", "Arr", null, a);
                    keep.Add(bare); keep.Add(tuple.Item1); keep.Add(arr[1]);
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();
                    CollectionAssert.AreEqual(new[] { 1.0, 3.0, 5.0 }, bare.ToArray<double>(), $"{Label(run)}: Bare at iteration {i}");
                    CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, tuple.Item1.ToArray<double>(), $"{Label(run)}: Tuple2 at iteration {i}");
                    CollectionAssert.AreEqual(new[] { -1.0, 1.0, 3.0 }, arr[1].ToArray<double>(), $"{Label(run)}: Arr at iteration {i}");
                }

                foreach (var nd in keep)
                    Assert.IsFalse(nd.IsDisposed, $"{Label(run)}: a kept result must never be reclaimed behind the caller's back");
                CollectionAssert.AreEqual(new[] { 1.0, 3.0, 5.0 }, keep[0].ToArray<double>(), $"{Label(run)}: the first result is intact after 25 rounds");
            }
        }
    }
}
