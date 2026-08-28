using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NumSharp;
using NDScoping.Instrumentation;

namespace NDScoping
{
    /// <summary>
    ///     LEVEL Z — the seam, hand-driven. <b>This is what the <c>[NDScopedAsync]</c> weaver emits.
    ///     You never write it.</b> It is here only to show the machinery underneath Level C — the exact
    ///     <c>OpenOrResume</c> / <c>Suspend</c> / <c>Returns</c> / <c>DisposeSlot</c> protocol the woven
    ///     <c>MoveNext</c> performs, mirroring <c>NDScopeAsyncTests</c>.
    /// </summary>
    internal static class LevelZ_Seam
    {
        public static List<Report> RunAll(bool stress) => new() { Seam_HandDriven() };

        private static Report Seam_HandDriven()
        {
            NDScope slot = null;   // the weaver-added <>ndscope state-machine field
            var a = np.array(new double[] { 1, 2, 3 });
            var b = np.array(new double[] { 10, 20, 30 });

            // --- segment 1 (MoveNext #1): open the invocation scope, build a temp, suspend ---
            var scope1 = NDScope.OpenOrResume(ref slot);
            var t = a + b;                     // tracked in the invocation scope [11,22,33]
            NDScope.Suspend(scope1);           // uninstall BEFORE the (simulated) await schedule

            // --- the continuation resumes on ANOTHER thread ---
            NDArray result = null;
            NDScope scope2 = null;
            bool aliveAcrossSuspension = false;
            Task.Run(() =>
            {
                scope2 = NDScope.OpenOrResume(ref slot);   // re-install the SAME scope on this thread
                aliveAcrossSuspension = !t.IsDisposed;      // the temp survived the suspension
                result = scope2.Returns(t * 2.0);           // yield the result out of the scope
                NDScope.DisposeSlot(ref slot);              // completion: reclaim the invocation scope
            }).Wait();

            bool tempReclaimed = t.IsDisposed;
            bool resultAlive = result is not null && !result.IsDisposed;
            bool sameScope = ReferenceEquals(scope1, scope2);
            bool valueOk = resultAlive && Math.Abs(result.GetDouble(0) - 22.0) < 1e-9;

            bool ok = aliveAcrossSuspension && tempReclaimed && resultAlive && sameScope && valueOk;
            return ReclamationProbe.Verdict("Seam_HandDriven", ok,
                $"survived suspension {ReclamationProbe.Mark(aliveAcrossSuspension)}   reclaimed at DisposeSlot " +
                $"{ReclamationProbe.Mark(tempReclaimed)}   same scope across threads {ReclamationProbe.Mark(sameScope)}   " +
                $"result alive {ReclamationProbe.Mark(resultAlive)}   value {ReclamationProbe.Mark(valueOk)}");
        }
    }
}
