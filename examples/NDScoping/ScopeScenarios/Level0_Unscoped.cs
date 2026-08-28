using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp;
using NDScoping.Instrumentation;

namespace NDScoping
{
    /// <summary>
    ///     LEVEL 0 — the problem. With NO scope, a dropped temporary's pooled buffer is reclaimed only
    ///     when its finalizer runs — non-deterministically, on a later GC. This demo makes that
    ///     concrete and motivates every level below it.
    /// </summary>
    internal static class Level0_Unscoped
    {
        public static List<Report> RunAll(bool stress) => new() { Unscoped_TempsLinger() };

        // We do NOT use NDArray.IsDisposed here — the finalizer path drops the buffer via Abandon and
        // never sets IsDisposed (only an explicit Dispose / NDScope does). So the honest unscoped
        // signal is GC reachability itself, observed with WeakReferences: right after the call the
        // temps are still ALIVE (garbage the finalizer has not reached), and only a forced GC +
        // finalizer drain reclaims them. That non-determinism is exactly what NDScope removes.
        private static Report Unscoped_TempsLinger()
        {
            var weaks = new List<WeakReference>();
            ProduceUnscoped(weaks);

            bool aliveRightAfter = weaks.All(w => w.IsAlive);

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            bool reclaimedAfterGc = weaks.All(w => !w.IsAlive);

            bool ok = aliveRightAfter && reclaimedAfterGc;
            return ReclamationProbe.Verdict("Unscoped_TempsLinger", ok,
                $"temps after return: {(aliveRightAfter ? "ALIVE" : "gone")}   after forced GC: " +
                $"{(reclaimedAfterGc ? "reclaimed" : "STILL ALIVE")}   => finalizer backstop (non-deterministic)");
        }

        // NoInlining and no kept reference: after this returns nothing roots the temps, so they become
        // collectible — the whole point (an unscoped composition leaks them to a future GC).
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ProduceUnscoped(List<WeakReference> weaks)
        {
            var a = np.array(new double[] { 1, 2, 3 });
            var b = np.array(new double[] { 10, 20, 30 });
            var c = np.array(new double[] { 100, 200, 300 });

            var ab = a * b;          // temp 1
            var abc = ab + c;        // temp 2
            var scaled = abc * 0.5;  // temp 3
            weaks.Add(new WeakReference(ab));
            weaks.Add(new WeakReference(abc));
            weaks.Add(new WeakReference(scaled));

            GC.KeepAlive(a);
            GC.KeepAlive(b);
            GC.KeepAlive(c);
        }
    }
}
