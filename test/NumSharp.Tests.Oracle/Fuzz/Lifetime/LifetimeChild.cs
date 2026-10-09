using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>
    ///     The entry point of the Oracle assembly. A test run never calls it (MSTest's host loads the assembly as a
    ///     library); <see cref="LifetimeChildProcess"/> runs the assembly as a CHILD PROCESS with
    ///     <c>--numsharp-lifetime</c> to replay the lifetime oracle under optimized code (see <see cref="LifetimeChild"/>).
    /// </summary>
    /// <remarks>
    ///     The csproj sets <c>GenerateProgramFile=false</c>, so this replaces the empty entry point Microsoft.NET.Test.Sdk
    ///     generates for an Exe test project; any other invocation still does nothing and returns 0, as that one did.
    /// </remarks>
    internal static class LifetimeChildProgram
    {
        /// <summary>Dispatches <c>--numsharp-lifetime …</c> to <see cref="LifetimeChild.Run"/>; anything else is a no-op.</summary>
        /// <param name="args">The command line.</param>
        /// <returns>The child's exit code (0 on completion), or 0 for any other invocation.</returns>
        public static int Main(string[] args)
            => args.Length > 0 && args[0] == LifetimeChild.Flag ? LifetimeChild.Run(args) : 0;
    }

    /// <summary>
    ///     The child side of the lifetime oracle: replays lifetime cases with every operand marked <c>temp</c> built
    ///     afresh inside the call, while <see cref="LifetimeThief"/> collects and poisons, and writes one outcome line per
    ///     case to a report file.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why a child process.</b> Two reasons, both load-bearing. (1) The bug class exists only in OPTIMIZED code:
    ///     tier-0 code reports every local and parameter live for the whole method, so an operand stays reachable for
    ///     the whole call and no window opens. The parent starts the child with <c>DOTNET_TieredCompilation=0</c>, which
    ///     compiles every method fully optimized on first use — the code a long-running application ends up running.
    ///     (2) A lost anchor can free memory the kernel is still using; when the pool hands the buffer back to the OS the
    ///     kernel dies with a fatal <see cref="AccessViolationException"/>, which no handler can catch. In the child that
    ///     ends one replay; the parent records the case as crashed and resumes after it.
    ///     </para>
    ///     <para>
    ///     <b>Protocol.</b> <c>--numsharp-lifetime &lt;file&gt; &lt;start&gt; &lt;reps&gt; &lt;report&gt; [gap]</c>: replay the
    ///     cases of <c>Fuzz/Lifetime/corpus/&lt;file&gt;</c> from line <c>start</c> (only gap <c>gap</c>'s, when given),
    ///     each with <c>reps</c> stressed repetitions. Before each case the child prints <c>BEGIN &lt;index&gt;</c> on
    ///     stdout (flushed), so a crash names its case; after it, one JSON line goes to the report (appended and flushed,
    ///     so a later crash keeps it). The first report line is the CONTROL: <see cref="ControlCopy"/> — the bug pattern
    ///     written by hand — replayed under the same stress; zero corrupted repetitions means this run could not have
    ///     seen a lost anchor, and the parent reports the run inconclusive instead of passing it.
    ///     </para>
    /// </remarks>
    internal static class LifetimeChild
    {
        /// <summary>The command-line flag that selects the lifetime child.</summary>
        public const string Flag = "--numsharp-lifetime";

        /// <summary>The control's repetitions (each copies a 2 MB temporary four times).</summary>
        public const int ControlReps = 40;

        /// <summary>Runs the child.</summary>
        /// <param name="args"><c>--numsharp-lifetime file start reps report [gap]</c>.</param>
        /// <returns>0 when every requested case was replayed.</returns>
        /// <exception cref="ArgumentException">The command line is malformed.</exception>
        public static int Run(string[] args)
        {
            if (args.Length < 5)
                throw new ArgumentException("usage: --numsharp-lifetime <file> <start> <reps> <report> [gap]");
            string file = args[1];
            int start = int.Parse(args[2]);
            int reps = int.Parse(args[3]);
            string report = args[4];
            string gap = args.Length > 5 ? args[5] : null;

            var cases = LifetimeCase.ReadAll(LifetimeCase.CorpusPath(file));
            using var thief = new LifetimeThief();
            if (start == 0)
                Append(report, new { control = RunControl(thief), reps = ControlReps });
            for (int i = start; i < cases.Count; i++)
            {
                var c = cases[i];
                if (gap != null && c.Gap != gap)
                    continue;
                Console.Out.WriteLine($"BEGIN {c.Index} {c.Id}");
                Console.Out.Flush();
                Append(report, RunCase(c, reps, thief));
            }
            Append(report, new { done = true });
            return 0;
        }

        /// <summary>Replays one case: the reference run over held operands, then the stressed repetitions.</summary>
        /// <param name="c">The case.</param>
        /// <param name="reps">The stressed repetitions.</param>
        /// <param name="thief">The stress thread.</param>
        /// <returns>The outcome line (serialized by <see cref="Append"/>).</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static object RunCase(LifetimeCase c, int reps, LifetimeThief thief)
        {
            var held = new NDArray[c.Args.Length];
            try
            {
                for (int k = 0; k < held.Length; k++)
                    held[k] = c.Args[k].Make();

                // The reference run: every operand held (this frame's `held` array keeps them reachable).
                List<LifetimeItem> reference;
                string refStatus;
                try
                {
                    var makers = held.Select(h => (Func<NDArray>)(() => h)).ToArray();
                    var r0 = LifetimeOps.Invoke(c, makers);
                    reference = LifetimeResult.Record(r0);
                    LifetimeResult.Dispose(r0, held);
                    bool bytes = c.Parity && (!c.Host || IsPinnedHost);
                    refStatus = LifetimeResult.CompareWithNumPy(reference, c.Expected, bytes) is string diff
                        ? "mismatch: " + diff
                        : "ok";
                }
                catch (Exception e)
                {
                    return Outcome(c, "threw: " + Describe(e), 0, 0, null, thief.Collections);
                }

                // The stressed repetitions: operands marked temp are built inside the call expression.
                thief.SetCaseSizes(c.Args.Where(a => a.Temp).Select(a => a.ByteSize));
                int wrong = 0, threw = 0;
                string first = null;
                int collectionsBefore = thief.Collections;
                var stressMakers = new Func<NDArray>[c.Args.Length];
                for (int k = 0; k < c.Args.Length; k++)
                {
                    var operand = c.Args[k];
                    int slot = k;
                    stressMakers[k] = operand.Temp ? operand.Make : () => held[slot];
                }
                for (int r = 0; r < reps; r++)
                {
                    object res;
                    thief.Enter();
                    try
                    {
                        res = LifetimeOps.Invoke(c, stressMakers);
                    }
                    catch (Exception e)
                    {
                        thief.Exit();
                        threw++;
                        first ??= $"rep {r}: " + Describe(e);
                        continue;
                    }
                    thief.Exit();
                    var items = LifetimeResult.Record(res);
                    LifetimeResult.Dispose(res, held);
                    if (LifetimeResult.CompareWithReference(items, reference) is string diff)
                    {
                        wrong++;
                        first ??= $"rep {r}: " + diff;
                    }
                }
                return Outcome(c, refStatus, wrong, threw, first, thief.Collections - collectionsBefore);
            }
            finally
            {
                foreach (var h in held)
                    h?.Dispose();
                foreach (var operand in c.Args)
                    operand.ReleaseBytes();
            }
        }

        /// <summary>Builds an outcome line.</summary>
        /// <param name="c">The case.</param>
        /// <param name="refStatus">The reference run against NumPy: <c>ok</c>, <c>mismatch: ...</c> or <c>threw: ...</c>.</param>
        /// <param name="wrong">Stressed repetitions that differed from the reference.</param>
        /// <param name="threw">Stressed repetitions that threw.</param>
        /// <param name="first">The first divergence, or null.</param>
        /// <param name="collections">Collections forced during the case's repetitions.</param>
        /// <returns>An anonymous object serialized as one report line (read by <see cref="LifetimeChildProcess"/>).</returns>
        private static object Outcome(LifetimeCase c, string refStatus, int wrong, int threw, string first, int collections)
            => new { index = c.Index, id = c.Id, gap = c.Gap, @ref = refStatus, wrong, threw, first, collections };

        /// <summary>A one-line description of an exception (type and the first line of the message).</summary>
        /// <param name="e">The exception.</param>
        /// <returns>The description.</returns>
        private static string Describe(Exception e) => e.GetType().Name + ": " + e.Message.Split('\n')[0];

        /// <summary>True on the host the corpus's host-dependent values were recorded on (win-amd64).</summary>
        public static bool IsPinnedHost =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && RuntimeInformation.ProcessArchitecture == Architecture.X64;

        /// <summary>
        ///     Replays the control <see cref="ControlCopy"/> — the lost-anchor pattern written by hand — under the stress
        ///     thread.
        /// </summary>
        /// <param name="thief">The stress thread.</param>
        /// <returns>How many of <see cref="ControlReps"/> repetitions came back corrupted (0 = no detection power).</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static int RunControl(LifetimeThief thief)
        {
            var operand = new LifetimeOperand
            {
                Dtype = "float64", Shape = new long[] { 1 << 18 }, View = "", Temp = true,
                Recipe = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(
                    "{\"k\":\"mod\",\"start\":0,\"step\":1,\"mod\":997,\"scale\":1.0,\"off\":0.0}")
            };
            var expected = operand.BuildBytes();
            thief.SetCaseSizes(new[] { operand.ByteSize });
            int hits = 0;
            for (int r = 0; r < ControlReps; r++)
            {
                thief.Enter();
                NDArray res = ControlCopy(operand.Make());
                thief.Exit();
                if (!res.tobytes().AsSpan().SequenceEqual(expected))
                    hits++;
                res.Dispose();
            }
            return hits;
        }

        /// <summary>
        ///     The bug pattern, by hand: read the operand's pointer, then copy through it four times without touching the
        ///     operand again (no <see cref="GC.KeepAlive"/>). A collection during the copy finds the operand unreachable.
        /// </summary>
        /// <param name="a">A float64 operand the caller does not hold.</param>
        /// <returns>The copy (corrupted when the operand's buffer was freed and poisoned mid-copy).</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe NDArray ControlCopy(NDArray a)
        {
            var src = (double*)a.Address;
            long n = a.size;
            var r = np.empty(new Shape(n), np.float64);
            var dst = (double*)r.Address;
            for (int rep = 0; rep < 4; rep++)
                for (long i = 0; i < n; i++)
                    dst[i] = src[i];
            return r;
        }

        /// <summary>Appends one JSON line to the report and flushes it (a later crash keeps it).</summary>
        /// <param name="report">The report file the parent reads.</param>
        /// <param name="line">The object to serialize.</param>
        /// <exception cref="IOException">The report cannot be written.</exception>
        private static void Append(string report, object line)
        {
            using var w = new StreamWriter(report, append: true);
            w.WriteLine(JsonSerializer.Serialize(line));
        }
    }
}
