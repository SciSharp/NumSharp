using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;
using NDScoping.Instrumentation;

namespace NDScoping
{
    /// <summary>
    ///     LEVEL C — <c>[NDScopedAsync]</c>. The shapes that suspend across <c>await</c> or defer to a
    ///     task's completion. The weaver gives each state machine ONE scope for the whole logical
    ///     invocation (in a weaver-added field), uninstalled before every continuation is scheduled and
    ///     reclaimed at completion — so temporaries survive the <c>await</c> (an in-flight callee may
    ///     still use them) and are reclaimed only when the INVOCATION finishes.
    /// </summary>
    internal static class Level3_ScopedAsync
    {
        private const int SpinMs = 5000;

        private static NDArray A() => np.array(new double[] { 1, 2, 3 });
        private static NDArray B() => np.array(new double[] { 10, 20, 30 });

        public static async Task<List<Report>> RunAll(bool stress)
        {
            var reports = new List<Report>();

            reports.Add(await ReclamationProbe.RunAsync("Async_Task", async () =>
            {
                var temps = new List<NDArray>();
                var r = await Async_Task(A(), B(), temps);
                return new DemoRun(temps, r);
            }, new double?[] { 22 }, stress));

            reports.Add(await ReclamationProbe.RunAsync("Async_ValueTask", async () =>
            {
                var temps = new List<NDArray>();
                var r = await Async_ValueTask(A(), B(), temps);
                return new DemoRun(temps, r);
            }, new double?[] { 22 }, stress));

            reports.Add(await Async_Void_Demo(stress));
            reports.Add(await Async_Iterator_Demo(stress));
            reports.Add(await Async_NonAsyncTask_Demo(stress));
            reports.Add(await Async_ValueTaskPreserve_Demo(stress));
            reports.Add(await Async_CrossAwaitSurvival_Demo(stress));

            return reports;
        }

        // ---- the woven async workers -----------------------------------------------------------

        [NDScopedAsync]
        private static async Task<NDArray> Async_Task(NDArray a, NDArray b, List<NDArray> temps)
        {
            var t = a + b;
            temps.Add(t);
            await Task.Yield();          // suspend — the temp survives the await
            return t * 2.0;
        }

        [NDScopedAsync]
        private static async ValueTask<NDArray> Async_ValueTask(NDArray a, NDArray b, List<NDArray> temps)
        {
            var t = a + b;
            temps.Add(t);
            await Task.Yield();
            return t * 2.0;
        }

        [NDScopedAsync]
        private static async void Async_Void(NDArray a, NDArray b, List<NDArray> temps, TaskCompletionSource done)
        {
            var t = a + b;
            temps.Add(t);
            await Task.Yield();
            done.SetResult();            // body finished; DisposeSlot runs before the void builder's SetResult
        }

        [NDScopedAsync]
        private static async IAsyncEnumerable<NDArray> Async_Iterator(NDArray a, NDArray b, List<NDArray> hoisted)
        {
            var scale = a + b;           // hoisted invocation state — reclaimed at the final MoveNext
            hoisted.Add(scale);
            for (int i = 1; i <= 3; i++)
            {
                await Task.Yield();
                yield return scale * (double)i;   // each element -> Returns -> the consumer owns it
            }
        }

        // A NON-async method returning Task: the weaver's ReturnsTask yields a COMPLETED task's result
        // now and DEFERS an INCOMPLETE one's reclamation to the task's completion — so the operand the
        // in-flight callee still holds is protected.
        [NDScopedAsync]
        private static Task<NDArray> Async_NonAsyncTask(NDArray a, NDArray b, List<NDArray> sink, SemaphoreSlim gate)
        {
            var t = a + b;               // tracked; kept alive until the returned task completes
            sink.Add(t);
            return SlowSquare(t, gate);  // an incomplete Task<NDArray> that reads t AFTER the gate
        }

        [NDScopedAsync]
        private static ValueTask<NDArray> Async_ValueTaskPreserve(NDArray a, NDArray b, List<NDArray> sink, SemaphoreSlim gate)
        {
            var t = a + b;
            sink.Add(t);
            return SlowSquareVt(t, gate);   // incomplete ValueTask -> Preserve()d; caller gets the multi-observable form
        }

        [NDScopedAsync]
        private static async Task<NDArray> Async_CrossAwaitSurvival(NDArray a, NDArray b, List<NDArray> temps, List<bool> observations)
        {
            var t = a + b;
            temps.Add(t);
            await SlowReader(t, observations);   // the callee reads t mid-flight; it MUST still be alive
            return t * 2.0;
        }

        // ---- plain (unscoped) async helpers the workers await -----------------------------------

        private static async Task<NDArray> SlowSquare(NDArray t, SemaphoreSlim gate)
        {
            await gate.WaitAsync();
            return t + t;                // uses the operand handed to it while "in flight"
        }

        private static async ValueTask<NDArray> SlowSquareVt(NDArray t, SemaphoreSlim gate)
        {
            await gate.WaitAsync();
            return t + t;
        }

        private static async Task SlowReader(NDArray t, List<bool> observations)
        {
            await Task.Delay(20);
            observations.Add(!t.IsDisposed);   // a scope-per-segment bug would have reclaimed t here
        }

        // ---- the bespoke async demos -----------------------------------------------------------

        private static async Task<Report> Async_Void_Demo(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                var temps = new List<NDArray>();
                // RunContinuationsAsynchronously: done.SetResult() must NOT inline our await-continuation
                // onto the async-void's own thread — that thread still has to reach DisposeSlot, and a
                // SpinUntil here would otherwise block it before it does.
                var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                Async_Void(A(), B(), temps, done);
                await done.Task;
                bool reclaimed = SpinWait.SpinUntil(
                    () => temps.Count > 0 && temps.TrueForAll(t => t.IsDisposed), SpinMs);
                ok &= reclaimed;
                detail = $"fire-and-forget; temp reclaimed at completion {ReclamationProbe.Mark(reclaimed)}";
                if (stress) { GC.Collect(); GC.WaitForPendingFinalizers(); }
            }

            return ReclamationProbe.Verdict("Async_Void", ok, detail);
        }

        private static async Task<Report> Async_Iterator_Demo(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                var hoisted = new List<NDArray>();
                var yielded = new List<NDArray>();
                await foreach (var x in Async_Iterator(A(), B(), hoisted))
                    yielded.Add(x);

                bool elementsAlive = yielded.Count == 3 && yielded.All(y => !y.IsDisposed);
                bool hoistedReclaimed = SpinWait.SpinUntil(() => hoisted.TrueForAll(t => t.IsDisposed), SpinMs);
                bool valueOk = elementsAlive &&
                               Math.Abs(yielded[0].GetDouble(0) - 11.0) < 1e-9 &&
                               Math.Abs(yielded[2].GetDouble(0) - 33.0) < 1e-9;
                ok &= elementsAlive && hoistedReclaimed && valueOk;
                detail = $"await-foreach elements owned {ReclamationProbe.Mark(elementsAlive)}   state reclaimed " +
                         $"{ReclamationProbe.Mark(hoistedReclaimed)}   value {ReclamationProbe.Mark(valueOk)}";
                if (stress)
                {
                    foreach (var y in yielded) y.Dispose();
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
            }

            return ReclamationProbe.Verdict("Async_Iterator", ok, detail);
        }

        private static async Task<Report> Async_NonAsyncTask_Demo(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                var sink = new List<NDArray>();
                var gate = new SemaphoreSlim(0);
                var task = Async_NonAsyncTask(A(), B(), sink, gate);

                bool aliveInFlight = !sink[0].IsDisposed;   // deferral kept the operand alive
                gate.Release();
                var result = await task;
                bool reclaimedAfter = SpinWait.SpinUntil(() => sink[0].IsDisposed, SpinMs);
                bool valueOk = !result.IsDisposed && Math.Abs(result.GetDouble(0) - 22.0) < 1e-9;

                ok &= aliveInFlight && reclaimedAfter && valueOk;
                detail = $"operand alive in-flight {ReclamationProbe.Mark(aliveInFlight)}   reclaimed after " +
                         $"{ReclamationProbe.Mark(reclaimedAfter)}   value {ReclamationProbe.Mark(valueOk)}";
                if (stress) { result.Dispose(); GC.Collect(); GC.WaitForPendingFinalizers(); }
            }

            return ReclamationProbe.Verdict("Async_NonAsyncTask", ok, detail);
        }

        private static async Task<Report> Async_ValueTaskPreserve_Demo(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                var sink = new List<NDArray>();
                var gate = new SemaphoreSlim(0);
                var vt = Async_ValueTaskPreserve(A(), B(), sink, gate);

                bool aliveInFlight = !sink[0].IsDisposed;
                gate.Release();
                var result = await vt;                       // the preserved (multi-observable) form
                bool reclaimedAfter = SpinWait.SpinUntil(() => sink[0].IsDisposed, SpinMs);
                bool valueOk = !result.IsDisposed && Math.Abs(result.GetDouble(0) - 22.0) < 1e-9;

                ok &= aliveInFlight && reclaimedAfter && valueOk;
                detail = $"incomplete ValueTask Preserve()d; operand alive in-flight " +
                         $"{ReclamationProbe.Mark(aliveInFlight)}   reclaimed after {ReclamationProbe.Mark(reclaimedAfter)}   " +
                         $"value {ReclamationProbe.Mark(valueOk)}";
                if (stress) { result.Dispose(); GC.Collect(); GC.WaitForPendingFinalizers(); }
            }

            return ReclamationProbe.Verdict("Async_ValueTaskPreserve", ok, detail);
        }

        private static async Task<Report> Async_CrossAwaitSurvival_Demo(bool stress)
        {
            bool ok = true;
            string detail = "";
            int rounds = stress ? ReclamationProbe.StressRounds : 1;
            for (int r = 0; r < rounds; r++)
            {
                var temps = new List<NDArray>();
                var obs = new List<bool>();
                var result = await Async_CrossAwaitSurvival(A(), B(), temps, obs);

                bool survivedMidAwait = obs.Count > 0 && obs.All(x => x);
                bool reclaimedAfter = SpinWait.SpinUntil(() => temps.TrueForAll(t => t.IsDisposed), SpinMs);
                bool valueOk = !result.IsDisposed && Math.Abs(result.GetDouble(0) - 22.0) < 1e-9;

                ok &= survivedMidAwait && reclaimedAfter && valueOk;
                detail = $"operand survived the suspension {ReclamationProbe.Mark(survivedMidAwait)}   reclaimed after " +
                         $"{ReclamationProbe.Mark(reclaimedAfter)}   value {ReclamationProbe.Mark(valueOk)}";
                if (stress) { result.Dispose(); GC.Collect(); GC.WaitForPendingFinalizers(); }
            }

            return ReclamationProbe.Verdict("Async_CrossAwaitSurvival", ok, detail);
        }
    }
}
