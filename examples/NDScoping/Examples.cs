using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NumSharp;

namespace NDScoping
{
    /// <summary>
    ///     <b>Examples.cs — a top-to-bottom tour of NDArray memory reclamation, for users.</b>
    ///     <para>
    ///     Read it in order. Every section runs the SAME little computation (<see cref="RowNorms_Leaky"/>
    ///     and its twins), so you can compare "leaks" vs "does not leak" on IDENTICAL code:
    ///     </para>
    ///     <list type="number">
    ///       <item>the leak — the natural way to write it;</item>
    ///       <item>the same code, leak PREVENTED BY HAND with <c>using</c>/<c>Dispose</c>;</item>
    ///       <item>the same code with <see cref="NDScope"/> — one scope reclaims every temporary;</item>
    ///       <item>the same code with <c>[NDScoped]</c> — the build injects the scope for you;</item>
    ///       <item>the same for async, with <c>[NDScopedAsync]</c>.</item>
    ///     </list>
    ///     <para>
    ///     Why this matters: an <see cref="NDArray"/> owns an UNMANAGED pooled buffer. A dropped
    ///     temporary's buffer is returned to the pool only when its finalizer runs — which in a hot
    ///     loop lags badly and forces slow cold allocations. Disposing (or scoping) returns it
    ///     immediately. NumPy is refcounted and frees instantly; these tools close that gap.
    ///     </para>
    ///     <para>Run it:  <c>dotnet run -- examples</c></para>
    /// </summary>
    public static class Examples
    {
        // =====================================================================================
        // THE WORKLOAD  —  one small, realistic computation reused by every section.
        //
        //   "Row norms after centering": subtract each column's mean, square, sum across columns,
        //   take the square root, and clamp. It deliberately exercises everything you use daily:
        //     • operators:      -   *              (element-wise, via NDArray operator overloads)
        //     • np.* functions: np.mean, np.sum, np.sqrt, np.maximum
        //     • broadcasting:   a (4,3) matrix combined with a (3,) row vector, and with scalars
        //     • TWO named temporaries in a chained math expression: `centered` and `squared`
        //       (plus the *hidden* temporaries that np.mean / np.sum / np.sqrt each allocate).
        //
        //   The `watch` parameter is DEMO INSTRUMENTATION ONLY: each variant appends its two
        //   temporaries so Run() can look at them afterwards (via the public NDArray.IsDisposed)
        //   and show you whether they were reclaimed. Real code never takes a `watch` list.
        // =====================================================================================

        private static NDArray MakeMatrix()
        {
            return (np.arange(12) * 1.0).reshape(4, 3); // (4,3) float64
        }

        // -------------------------------------------------------------------------------------
        // 1) LEAKS.  The natural way to write it. Every temporary is dropped on the floor.
        // -------------------------------------------------------------------------------------
        private static NDArray RowNorms_Leaky(NDArray x, List<NDArray> watch)
        {
            // Two temporaries in a chained math expression:
            var centered = x - np.mean(x, axis: 0);   // temp #1 — broadcasts (4,3) - (3,)
            var squared  = centered * centered;        // temp #2 — chained onto temp #1
            watch.Add(centered);
            watch.Add(squared);

            // The chain continues through more np.* calls, each of which makes ANOTHER temporary
            // (the outputs of np.sum and np.sqrt) that is likewise dropped:
            return np.maximum(np.sqrt(np.sum(squared, axis: 1)), 1.0);

            // When this returns, `centered`, `squared` and the sum/sqrt intermediates all become
            // garbage. Their pooled buffers come back only when the GC finalizer eventually runs.
        }

        // -------------------------------------------------------------------------------------
        // 2) PREVENTED BY HAND.  The SAME math, but every temporary is disposed with `using`.
        //    Correct — and tedious: you must name and dispose EACH intermediate (including the
        //    np.* outputs), must NOT dispose the input `x`, and must NOT dispose the value you
        //    return. Miss one and it leaks; dispose the wrong one and you corrupt a live buffer.
        // -------------------------------------------------------------------------------------
        private static NDArray RowNorms_Manual(NDArray x, List<NDArray> watch)
        {
            using var mean     = np.mean(x, axis: 0);   // dispose every intermediate by hand...
            using var centered = x - mean;
            using var squared  = centered * centered;
            using var summed   = np.sum(squared, axis: 1);
            using var root     = np.sqrt(summed);
            watch.Add(centered);
            watch.Add(squared);

            // np.maximum allocates the RESULT — it is the caller's to own, so we never `using` it.
            return np.maximum(root, 1.0);

            // Each `using` disposes at method exit, which is AFTER that value's last use, so this
            // is safe. Correct, but five `using`s for one expression — this is what NDScope removes.
        }

        // -------------------------------------------------------------------------------------
        // 3) PREVENTED WITH NDScope.  The body is byte-for-byte the LEAKY version again — you only
        //    add the scope and route the return through Returns. EVERYTHING built while the scope
        //    is open (centered, squared, AND the hidden np.* intermediates) is reclaimed when the
        //    scope closes; the value you Returns() survives for the caller.
        // -------------------------------------------------------------------------------------
        private static NDArray RowNorms_Scope(NDArray x, List<NDArray> watch)
        {
            using var scope = NDScope.Open();

            var centered = x - np.mean(x, axis: 0);   // identical to RowNorms_Leaky from here...
            var squared  = centered * centered;
            watch.Add(centered);
            watch.Add(squared);
            return scope.Returns(np.maximum(np.sqrt(np.sum(squared, axis: 1)), 1.0));

            // `x` was built BEFORE the scope opened, so it is never tracked — you cannot dispose an
            // input by accident. That is what makes "wrap every return in Returns" a safe blanket rule.
        }

        // -------------------------------------------------------------------------------------
        // 4) PREVENTED WITH [NDScoped].  Now the body is 100% the ORIGINAL leaky code — no scope
        //    anywhere in the source. The build-time weaver (the NumSharp.Weaver package) injects
        //    exactly what section 3 wrote: it opens the scope, wraps the body in try/finally, and
        //    routes each return through Returns. You write the attribute; the build writes the rest.
        // -------------------------------------------------------------------------------------
        [NDScoped]
        private static NDArray RowNorms_Scoped(NDArray x, List<NDArray> watch)
        {
            var centered = x - np.mean(x, axis: 0);
            var squared  = centered * centered;
            watch.Add(centered);
            watch.Add(squared);
            return np.maximum(np.sqrt(np.sum(squared, axis: 1)), 1.0);
        }

        // -------------------------------------------------------------------------------------
        // 5) PREVENTED WITH [NDScopedAsync].  The same, for methods that `await`. The weave holds
        //    ONE scope for the whole call across every suspension, so the temporaries survive the
        //    await (an in-flight callee might still be using them) and are reclaimed only when the
        //    method COMPLETES — never at the suspension point.
        // -------------------------------------------------------------------------------------
        [NDScopedAsync]
        private static async Task<NDArray> RowNorms_ScopedAsync(NDArray x, List<NDArray> watch)
        {
            var centered = x - np.mean(x, axis: 0);
            var squared  = centered * centered;
            watch.Add(centered);
            watch.Add(squared);
            await Task.Yield();                                        // temporaries survive this...
            return np.maximum(np.sqrt(np.sum(squared, axis: 1)), 1.0); // ...reclaimed at completion.
        }

        // =====================================================================================
        // The runnable tour.
        // =====================================================================================
        public static async Task Run()
        {
            Console.WriteLine("NDArray memory reclamation — from leak to [NDScoped], on identical code");
            Console.WriteLine(new string('=', 74));
            Console.WriteLine("Each section runs the SAME computation (center columns -> square -> per-row L2");
            Console.WriteLine("norm -> clamp): operators (- *), np.* (mean/sum/sqrt/maximum), broadcasting, and");
            Console.WriteLine("two named temporaries (`centered`, `squared`) in a chained expression.");

            // 1) LEAK -----------------------------------------------------------------------------
            Section("1) LEAK — the natural code; temporaries wait for the finalizer");
            {
                var x = MakeMatrix();
                var watch = new List<NDArray>();
                var result = RowNorms_Leaky(x, watch);      // result is dropped too (we never dispose it)
                Observe(watch, result);                     // -> reclaimed immediately: NO
                Console.WriteLine("    In a hot loop this is the slow path: dropped buffers miss the warm pool.");
                Console.WriteLine("    And it really is the finalizer (not deterministic disposal) that reclaims:");
                var weak = LeakOneWeakly();                 // a temporary kept only by a WeakReference
                Console.WriteLine($"      right after the call:  temporary still alive = {weak.IsAlive}");
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                Console.WriteLine($"      after a forced GC:     temporary still alive = {weak.IsAlive}   (gone — but only now)");
            }

            // 2) MANUAL ---------------------------------------------------------------------------
            Section("2) PREVENTED BY HAND — the same math, every temporary `using`-disposed");
            {
                var x = MakeMatrix();
                var watch = new List<NDArray>();
                using var result = RowNorms_Manual(x, watch);   // the caller owns the result -> dispose it too
                Observe(watch, result);                         // -> reclaimed immediately: YES
                Console.WriteLine("    Correct, but every intermediate had to be named and disposed by hand.");
            }

            // 3) NDScope --------------------------------------------------------------------------
            Section("3) NDScope — the original body + two lines; reclaims ALL temporaries");
            {
                var x = MakeMatrix();
                var watch = new List<NDArray>();
                using var result = RowNorms_Scope(x, watch);
                Observe(watch, result);
                Console.WriteLine("    One scope reclaimed centered, squared AND the hidden np.* intermediates.");
            }

            // 4) [NDScoped] -----------------------------------------------------------------------
            Section("4) [NDScoped] — 100% original body; the build injects the scope");
            {
                var x = MakeMatrix();
                var watch = new List<NDArray>();
                using var result = RowNorms_Scoped(x, watch);
                Observe(watch, result);
                Console.WriteLine("    Identical behaviour to section 3, with zero boilerplate in your source.");
            }

            // 5) [NDScopedAsync] ------------------------------------------------------------------
            Section("5) [NDScopedAsync] — the same, for async / Task-returning methods");
            {
                var x = MakeMatrix();
                var watch = new List<NDArray>();
                using var result = await RowNorms_ScopedAsync(x, watch);
                Observe(watch, result);
                Console.WriteLine("    The temporaries survived the await and were reclaimed at completion.");
            }

            Console.WriteLine();
            Console.WriteLine(new string('=', 74));
            Console.WriteLine("Takeaways:");
            Console.WriteLine("  • A METHOD reclaims the temporaries it OWNS (sections 2-5).");
            Console.WriteLine("  • A CALLER disposes the RESULTS it RECEIVES (the `using var result` above, or");
            Console.WriteLine("    its own NDScope around a hot loop).");
            Console.WriteLine("  • Prefer [NDScoped]/[NDScopedAsync]: same result, zero boilerplate. They need");
            Console.WriteLine("    the NumSharp.Weaver package; without it the attributes are inert and the");
            Console.WriteLine("    method just runs unscoped (the pre-weave finalizer backstop) — never wrong,");
            Console.WriteLine("    only slower.");
        }

        // ---- small helpers -------------------------------------------------------------------

        private static void Section(string title)
        {
            Console.WriteLine();
            Console.WriteLine(title);
            Console.WriteLine(new string('-', 74));
        }

        // Prints whether the internal temporaries were reclaimed IMMEDIATELY (deterministically),
        // plus the result value — identical across every approach, proving the math never changed.
        private static void Observe(List<NDArray> temps, NDArray result)
        {
            bool reclaimed = temps.Count > 0 && temps.TrueForAll(t => t.IsDisposed);
            Console.WriteLine($"    internal temporaries reclaimed immediately: {(reclaimed ? "YES" : "NO ")}" +
                              $"    (result[0] = {result.GetDouble(0):0.000}, same everywhere)");
        }

        // Builds a temporary that nothing keeps alive (only a WeakReference), so the GC is free to
        // finalize it — used to SHOW that, unscoped, the finalizer is the only thing that reclaims.
        private static WeakReference LeakOneWeakly()
        {
            var x = MakeMatrix();
            var centered = x - np.mean(x, axis: 0);
            var squared = centered * centered;
            GC.KeepAlive(x);
            return new WeakReference(squared);
        }
    }
}
