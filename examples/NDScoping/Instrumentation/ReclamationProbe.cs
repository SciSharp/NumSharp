using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NumSharp;

namespace NDScoping.Instrumentation
{
    /// <summary>The verdict of one demo: a name, PASS/FAIL, and a one-line evidence string.</summary>
    public readonly record struct Report(string Name, bool Ok, string Detail);

    /// <summary>
    ///     What a single scoped call produced: the INTERNAL temporaries it made (which must be
    ///     reclaimed at the method boundary) and the RESULT(s) it returned (which must survive).
    ///     A demo appends its temps to <see cref="Temps"/> as it makes them and hands back its
    ///     <see cref="Results"/>; the probe then asserts the contract.
    /// </summary>
    public readonly struct DemoRun
    {
        public readonly List<NDArray> Temps;
        public readonly NDArray[] Results;
        public readonly string Note;

        public DemoRun(List<NDArray> temps, NDArray[] results, string note = null)
        {
            Temps = temps ?? new List<NDArray>();
            Results = results ?? Array.Empty<NDArray>();
            Note = note;
        }

        public DemoRun(List<NDArray> temps, NDArray result, string note = null)
            : this(temps, new[] { result }, note) { }
    }

    /// <summary>
    ///     The falsifiable observable behind every demo. It uses one public, honest signal —
    ///     <see cref="NDArray.IsDisposed"/> — exactly what an external consumer could write (no
    ///     internals, no reflection into buffers). A demo hands the probe its temps and results;
    ///     the probe asserts: every temp reclaimed (<c>IsDisposed == true</c>), every result still
    ///     alive (<c>IsDisposed == false</c>), and — optionally — a correct value. Under
    ///     <c>--stress</c> the body is re-run 25× with forced GCs between, so a mis-scoped result
    ///     whose buffer was freed early surfaces as a wrong value from a reused pool buffer.
    /// </summary>
    public static class ReclamationProbe
    {
        public const int StressRounds = 25;

        // ---- the generic sync/async runners ----------------------------------------------------

        public static Report Run(string name, Func<DemoRun> body, double?[] expect = null, bool stress = false)
        {
            int rounds = stress ? StressRounds : 1;
            bool ok = true;
            string detail = "";
            try
            {
                for (int r = 0; r < rounds; r++)
                {
                    var run = body();
                    var (rok, rdetail) = Evaluate(run, expect);
                    ok &= rok;
                    detail = rdetail;
                    Recycle(run, stress);
                }
            }
            catch (Exception e)
            {
                return new Report(name, false, "threw " + e.GetType().Name + ": " + e.Message);
            }

            return new Report(name, ok, detail);
        }

        public static async Task<Report> RunAsync(string name, Func<Task<DemoRun>> body, double?[] expect = null, bool stress = false)
        {
            int rounds = stress ? StressRounds : 1;
            bool ok = true;
            string detail = "";
            try
            {
                for (int r = 0; r < rounds; r++)
                {
                    var run = await body().ConfigureAwait(false);
                    var (rok, rdetail) = Evaluate(run, expect);
                    ok &= rok;
                    detail = rdetail;
                    Recycle(run, stress);
                }
            }
            catch (Exception e)
            {
                return new Report(name, false, "threw " + e.GetType().Name + ": " + e.Message);
            }

            return new Report(name, ok, detail);
        }

        // ---- the shared verdict ----------------------------------------------------------------

        private static (bool ok, string detail) Evaluate(DemoRun run, double?[] expect)
        {
            // NDArray overloads == / != element-wise, so a null check MUST use the pattern form.
            bool tempsReclaimed = run.Temps.Count == 0 || run.Temps.TrueForAll(t => t is not null && t.IsDisposed);
            bool resultsAlive = run.Results.Length > 0 && run.Results.All(x => x is not null && !x.IsDisposed);

            bool valueOk = true;
            if (expect != null && resultsAlive)
                for (int i = 0; i < expect.Length && i < run.Results.Length; i++)
                    if (expect[i] is double e && Math.Abs(run.Results[i].GetDouble(0) - e) > 1e-9)
                        valueOk = false;

            string note = string.IsNullOrEmpty(run.Note) ? "" : "   " + run.Note;
            string detail =
                $"temps {run.Temps.Count} reclaimed {Mark(tempsReclaimed)}   " +
                $"result{(run.Results.Length == 1 ? "" : "s")} alive {Mark(resultsAlive)}" +
                (expect != null ? $"   value {Mark(valueOk)}" : "") + note;
            return (tempsReclaimed && resultsAlive && valueOk, detail);
        }

        // Return each result's buffer to the pool and (in stress) churn, so the next round can only
        // read correct data if nothing was freed while still needed. Results were already evaluated.
        private static void Recycle(DemoRun run, bool stress)
        {
            if (!stress)
                return;
            foreach (var res in run.Results)
                res?.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        // ---- helpers for bespoke demos (buffer / iterator / seam / cross-cutting) ---------------

        /// <summary>A demo that computes its own verdict hands it here.</summary>
        public static Report Verdict(string name, bool ok, string detail) => new Report(name, ok, detail);

        public static string Mark(bool ok) => ok ? "OK" : "FAIL";

        // ---- table rendering -------------------------------------------------------------------

        public static void RenderHeader(string title)
        {
            Console.WriteLine();
            Console.WriteLine(title);
        }

        public static void Render(Report r)
        {
            string tag = r.Ok ? "OK  " : "FAIL";
            Console.WriteLine($"  [{tag}] {r.Name,-26} {r.Detail}");
        }
    }
}
