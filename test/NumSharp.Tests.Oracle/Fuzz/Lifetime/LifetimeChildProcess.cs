using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>One case's outcome in a lifetime replay.</summary>
    /// <param name="Index">The case's line index in its file.</param>
    /// <param name="Id">The case id.</param>
    /// <param name="Gap">The case's gap id (null for a gate case).</param>
    /// <param name="Reference">The reference run against NumPy: <c>ok</c>, <c>mismatch: …</c> or <c>threw: …</c>.</param>
    /// <param name="Wrong">Stressed repetitions whose bytes differed from the reference run.</param>
    /// <param name="Threw">Stressed repetitions that threw.</param>
    /// <param name="First">The first divergence or exception, for the report.</param>
    /// <param name="Crashed">True when the child process died inside this case (a fatal access violation).</param>
    /// <param name="Collections">The collections forced while the case's repetitions ran.</param>
    internal sealed record LifetimeOutcome(int Index, string Id, string Gap, string Reference, int Wrong, int Threw,
        string First, bool Crashed, int Collections)
    {
        /// <summary>True when the stressed repetitions diverged (a lost anchor), threw, or killed the process.</summary>
        public bool LifetimeBroken => Crashed || Wrong > 0 || Threw > 0;

        /// <summary>True when the reference run (all operands held) disagreed with NumPy or threw.</summary>
        public bool ReferenceBroken => Reference != "ok";
    }

    /// <summary>A whole lifetime replay: the control's hits and every case's outcome.</summary>
    /// <param name="ControlHits">How many control repetitions came back corrupted (0 = the run had no detection power).</param>
    /// <param name="ControlReps">The control's repetitions.</param>
    /// <param name="Outcomes">The cases' outcomes, in file order.</param>
    /// <param name="ChildRuns">How many child processes the replay took (one plus one per crash).</param>
    internal sealed record LifetimeReplay(int ControlHits, int ControlReps, List<LifetimeOutcome> Outcomes, int ChildRuns);

    /// <summary>
    ///     The parent side of the lifetime oracle: runs the Oracle assembly as a child process with tiered compilation
    ///     off (<see cref="LifetimeChild"/> explains why), and resumes after a case that crashed the child.
    /// </summary>
    internal static class LifetimeChildProcess
    {
        /// <summary>The longest one child process may run before it is killed and its current case reported as hung.</summary>
        public static readonly TimeSpan ChildTimeout = TimeSpan.FromMinutes(10);

        /// <summary>
        ///     Replays a lifetime corpus file in child processes.
        /// </summary>
        /// <param name="fileName">The file in <c>Fuzz/Lifetime/corpus/</c>.</param>
        /// <param name="reps">Stressed repetitions per case.</param>
        /// <param name="gap">Replay only this gap's cases (null: all).</param>
        /// <returns>The replay; a case that crashed a child appears with <see cref="LifetimeOutcome.Crashed"/> set.</returns>
        /// <exception cref="InvalidOperationException">A child could not be started, or died before its first case
        ///     (the harness itself is broken, not a case).</exception>
        public static LifetimeReplay Run(string fileName, int reps, string gap = null)
        {
            var cases = LifetimeCase.ReadAll(LifetimeCase.CorpusPath(fileName));
            var outcomes = new Dictionary<int, LifetimeOutcome>();
            int controlHits = -1, controlReps = 0, runs = 0;
            int start = 0;
            while (true)
            {
                runs++;
                string report = Path.Combine(Path.GetTempPath(), $"numsharp-lifetime-{Guid.NewGuid():N}.jsonl");
                try
                {
                    var (exit, stdout, stderr, timedOut) = Spawn(fileName, start, reps, report, gap);
                    bool done = false;
                    if (File.Exists(report))
                    {
                        foreach (var line in File.ReadLines(report))
                        {
                            // A line cut short by a crash mid-write cannot be parsed: the crash itself is reported below.
                            JsonDocument parsed;
                            try { parsed = JsonDocument.Parse(line); }
                            catch (JsonException) { continue; }
                            using var doc = parsed;
                            var root = doc.RootElement;
                            if (root.TryGetProperty("control", out var ctl))
                            {
                                controlHits = ctl.GetInt32();
                                controlReps = root.GetProperty("reps").GetInt32();
                            }
                            else if (root.TryGetProperty("done", out _))
                                done = true;
                            else
                            {
                                var o = new LifetimeOutcome(
                                    root.GetProperty("index").GetInt32(), root.GetProperty("id").GetString(),
                                    root.GetProperty("gap").ValueKind == JsonValueKind.String ? root.GetProperty("gap").GetString() : null,
                                    root.GetProperty("ref").GetString(), root.GetProperty("wrong").GetInt32(),
                                    root.GetProperty("threw").GetInt32(),
                                    root.GetProperty("first").ValueKind == JsonValueKind.String ? root.GetProperty("first").GetString() : null,
                                    false, root.GetProperty("collections").GetInt32());
                                outcomes[o.Index] = o;
                            }
                        }
                    }
                    if (done && exit == 0)
                        break;

                    // The child died (or was killed) inside a case: the last BEGIN it printed names that case.
                    int crashed = LastBegin(stdout);
                    if (crashed < 0 || crashed < start)
                        throw new InvalidOperationException(
                            $"the lifetime child died before its first case (exit {exit}{(timedOut ? ", timed out" : "")}):\n{Tail(stderr)}\n{Tail(stdout)}");
                    var c = cases[crashed];
                    string why = timedOut ? $"timed out after {ChildTimeout}" : $"process died (exit 0x{exit:X8}): {FirstCrashLine(stderr, stdout)}";
                    outcomes[crashed] = new LifetimeOutcome(c.Index, c.Id, c.Gap, "ok", 0, 0, why, true, 0);
                    start = crashed + 1;
                    if (start >= cases.Count)
                        break;
                }
                finally
                {
                    try { File.Delete(report); } catch (IOException) { }
                }
            }
            return new LifetimeReplay(controlHits, controlReps, outcomes.OrderBy(kv => kv.Key).Select(kv => kv.Value).ToList(), runs);
        }

        /// <summary>Starts one child and waits for it.</summary>
        /// <param name="fileName">The corpus file in <c>Fuzz/Lifetime/corpus/</c>.</param>
        /// <param name="start">The first case index to replay.</param>
        /// <param name="reps">Stressed repetitions per case.</param>
        /// <param name="report">The report file the child appends to.</param>
        /// <param name="gap">The gap filter, or null.</param>
        /// <returns>The exit code (-1 when the timeout killed it), the captured stdout / stderr, and whether it timed out.</returns>
        /// <exception cref="InvalidOperationException">The child could not be started.</exception>
        /// <exception cref="InvalidOperationException">The child could not be started.</exception>
        private static (int exit, string stdout, string stderr, bool timedOut) Spawn(string fileName, int start, int reps, string report, string gap)
        {
            string dll = typeof(LifetimeChild).Assembly.Location;
            var psi = new ProcessStartInfo(DotnetHost())
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = AppContext.BaseDirectory,
            };
            psi.ArgumentList.Add("exec");
            psi.ArgumentList.Add(dll);
            psi.ArgumentList.Add(LifetimeChild.Flag);
            psi.ArgumentList.Add(fileName);
            psi.ArgumentList.Add(start.ToString());
            psi.ArgumentList.Add(reps.ToString());
            psi.ArgumentList.Add(report);
            if (gap != null)
                psi.ArgumentList.Add(gap);
            // Fully optimized code from the first call: the bug class does not exist in tier-0 code.
            psi.Environment["DOTNET_TieredCompilation"] = "0";
            psi.Environment["DOTNET_TieredPGO"] = "0";
            // A dump of a deliberate access violation is not wanted.
            psi.Environment["DOTNET_DbgEnableMiniDump"] = "0";

            using var p = Process.Start(psi) ?? throw new InvalidOperationException("could not start the lifetime child");
            var stdout = new StringBuilder();
            var stderr = new StringBuilder();
            p.OutputDataReceived += (_, e) => { if (e.Data != null) lock (stdout) stdout.AppendLine(e.Data); };
            p.ErrorDataReceived += (_, e) => { if (e.Data != null) lock (stderr) stderr.AppendLine(e.Data); };
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            bool timedOut = !p.WaitForExit((int)ChildTimeout.TotalMilliseconds);
            if (timedOut)
            {
                try { p.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                p.WaitForExit();
            }
            p.WaitForExit();   // drains the async readers
            lock (stdout) lock (stderr)
                return (timedOut ? -1 : p.ExitCode, stdout.ToString(), stderr.ToString(), timedOut);
        }

        /// <summary>
        ///     The <c>dotnet</c> host that runs this test process: <c>DOTNET_HOST_PATH</c> (set by the CLI for the processes
        ///     it starts), else the host beside the running shared framework, else <c>dotnet</c> from PATH.
        /// </summary>
        /// <returns>The host executable's path (or the bare name, resolved through PATH).</returns>
        private static string DotnetHost()
        {
            var fromEnv = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH");
            if (!string.IsNullOrEmpty(fromEnv) && File.Exists(fromEnv))
                return fromEnv;
            string exe = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "dotnet.exe" : "dotnet";
            // <root>/shared/Microsoft.NETCore.App/<version>/ -> <root>/dotnet
            var fw = new DirectoryInfo(RuntimeEnvironment.GetRuntimeDirectory());
            var root = fw.Parent?.Parent?.Parent;
            if (root != null && File.Exists(Path.Combine(root.FullName, exe)))
                return Path.Combine(root.FullName, exe);
            return exe;
        }

        /// <summary>The index of the last <c>BEGIN</c> line a child printed, or -1.</summary>
        /// <param name="stdout">The child's captured stdout.</param>
        /// <returns>The case index the child was replaying when it stopped.</returns>
        private static int LastBegin(string stdout)
        {
            int last = -1;
            foreach (var line in stdout.Split('\n'))
            {
                if (line.StartsWith("BEGIN ", StringComparison.Ordinal))
                {
                    var parts = line.Split(' ');
                    if (parts.Length > 1 && int.TryParse(parts[1], out int idx))
                        last = idx;
                }
            }
            return last;
        }

        /// <summary>The first line of a crash report (the runtime's "Fatal error." block names the exception).</summary>
        /// <param name="stderr">The child's captured stderr.</param>
        /// <param name="stdout">The child's captured stdout.</param>
        /// <returns>The first line naming an exception, else the tail of stderr.</returns>
        private static string FirstCrashLine(string stderr, string stdout)
        {
            foreach (var text in new[] { stderr, stdout })
                foreach (var line in text.Split('\n'))
                    if (line.Contains("Exception", StringComparison.Ordinal))
                        return line.Trim();
            return Tail(stderr).Replace('\n', ' ');
        }

        /// <summary>The last few lines of a captured stream.</summary>
        /// <param name="text">The captured text.</param>
        /// <returns>At most its last six non-empty lines.</returns>
        private static string Tail(string text)
        {
            var lines = text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            return string.Join("\n", lines.Skip(Math.Max(0, lines.Length - 6)));
        }
    }
}
