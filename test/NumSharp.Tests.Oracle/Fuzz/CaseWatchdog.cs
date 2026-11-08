using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Threading;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     A per-case deadline for every corpus replay: when ONE case has been running longer than <see cref="Limit"/>, the
    ///     test host is terminated with a message naming the corpus and the case — instead of an infinite loop spinning, unseen,
    ///     for as long as nobody notices.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why it exists.</b> A corpus case takes milliseconds, yet some NumSharp code can never return on some inputs
    ///     BY DESIGN: NumPy's rejection samplers loop forever on the parameters their constraint checks reject (legacy
    ///     <c>logseries</c> at <c>p = 1</c>, <c>zipf</c> at <c>a &lt;= 1</c>), so for an error case those checks are all that
    ///     separates "raise NumPy's ValueError" from a hang — and a check that regresses does not FAIL its case, it HANGS the
    ///     run. On 2026-09-26 an in-progress build whose array check read <c>p &lt;= 1</c> replayed
    ///     <c>rnd/logseries/bcast:viol:p&gt;=1/857</c> (<c>p = [0.5, 1.0]</c>): its test host was still spinning five days later,
    ///     84 CPU-hours in, with no test failed, no timeout fired and nothing naming the case — finding it took a process dump.
    ///     CI's step timeouts bound such a hang there, but a local <c>dotnet test</c> has none.</para>
    ///     <para><b>How cases are watched.</b> Each <see cref="CorpusFile"/> enumeration owns one <see cref="Slot"/>
    ///     (<see cref="Watch"/>): the enumerator arms it with a case's id and start time as it hands the case out, and the
    ///     enumeration's disposal (the end of a <c>foreach</c>, an early <c>break</c> included) removes it — so every
    ///     streaming replay loop is covered without a change. A replay that iterates an already-parsed list
    ///     (<see cref="FuzzCorpus.Load"/>) takes a slot itself and calls <see cref="Slot.Enter"/> per case. A background timer
    ///     checks the live slots every <see cref="Period"/>; a case is reported once.</para>
    ///     <para><b>Why the process ends.</b> The looping code runs on the TEST thread, and nothing in .NET can stop a
    ///     running thread (there is no thread abort; <see cref="Thread.Interrupt"/> reaches only a blocked one), so the test
    ///     cannot be failed from here. Ending the process is the only way to stop the spin, and the message passed to
    ///     <see cref="Environment.FailFast(string)"/> is what <c>dotnet test</c> reports as the reason the run aborted
    ///     ("Test host process crashed : …"), so the case is named where the failure is read.</para>
    ///     <para><b>When it stays quiet.</b> Under an attached debugger a long case is a breakpoint, not a hang: the report
    ///     goes to the debugger's output and stderr and the process lives. <see cref="LimitVariable"/> sets the limit in
    ///     seconds (0 disables the watchdog), e.g. for a profiler run that slows every case down.</para>
    ///     <para><b>Contract.</b> A <see cref="CorpusFile"/> enumeration must be disposed (<c>foreach</c> and LINQ do it). An
    ///     enumerator abandoned mid-way keeps its slot armed, so its last case is reported as hung once the limit passes.</para>
    /// </remarks>
    internal static class CaseWatchdog
    {
        /// <summary>The environment variable that overrides <see cref="DefaultLimit"/>: seconds, <c>0</c> disables.</summary>
        internal const string LimitVariable = "NUMSHARP_ORACLE_CASE_TIMEOUT_SECONDS";

        /// <summary>
        ///     The per-case limit when <see cref="LimitVariable"/> is unset — two minutes: over 100x the slowest real case
        ///     (calibrated 2026-10-01: the whole Oracle suite, leak sweeps included, passed with a 1 s limit checked every
        ///     100 ms, so no committed case runs past ~1.1 s on the dev host), yet short enough that a hang is reported while
        ///     someone is still waiting for the run.
        /// </summary>
        internal static readonly TimeSpan DefaultLimit = TimeSpan.FromMinutes(2);

        /// <summary>How often the live slots are checked; a hang is reported within one period after the limit passes.</summary>
        internal static readonly TimeSpan Period = TimeSpan.FromSeconds(5);

        /// <summary>
        ///     Guards <see cref="Live"/>, every slot's reported-case marker, the limit/reaction pair (<see cref="Configure"/>)
        ///     and the lazy timer start.
        /// </summary>
        private static readonly object Gate = new();

        /// <summary>The slots of the enumerations currently open, in creation order.</summary>
        private static readonly List<Slot> Live = new();

        /// <summary>
        ///     The checking timer, started by the first <see cref="Watch"/> and never stopped: a dormant check costs one lock
        ///     per period, less than creating and disposing a timer per enumeration. Held in this field because an
        ///     unreferenced <see cref="Timer"/> can be collected, which would silently end the checks.
        /// </summary>
        private static Timer _timer;

        /// <summary>The per-case limit in force (see <see cref="Limit"/>); guarded by <see cref="Gate"/>.</summary>
        private static TimeSpan _limit = ParseLimit(Environment.GetEnvironmentVariable(LimitVariable));

        /// <summary>What a hang does with its report (see <see cref="Configure"/>); guarded by <see cref="Gate"/>.</summary>
        private static Action<string> _onHang = Terminate;

        /// <summary>
        ///     The per-case limit in force; <see cref="TimeSpan.Zero"/> (or less) disables reporting. Read from
        ///     <see cref="LimitVariable"/> once, at first use; changed only through <see cref="Configure"/>.
        /// </summary>
        internal static TimeSpan Limit
        {
            get
            {
                lock (Gate)
                    return _limit;
            }
        }

        /// <summary>
        ///     Replaces the limit and the reaction to a hang TOGETHER — for this class's own tests, which record reports
        ///     instead of ending their host, and restore the previous pair afterwards.
        /// </summary>
        /// <param name="limit">The per-case limit (zero or less disables reporting).</param>
        /// <param name="onHang">What a hang does with its report; the default ends the process (<see cref="Terminate"/>).</param>
        /// <returns>The pair in force before, to restore.</returns>
        /// <remarks>
        ///     One call under <see cref="Gate"/>, not two settable properties: <see cref="CheckNow"/> reads both in the same
        ///     lock, so a timer check can never pair a test's 1 ms limit with the process-ending default — which two separate
        ///     writes (or a restore landing between a check's two reads) would allow, killing the test host mid-run.
        /// </remarks>
        internal static (TimeSpan limit, Action<string> onHang) Configure(TimeSpan limit, Action<string> onHang)
        {
            ArgumentNullException.ThrowIfNull(onHang);
            lock (Gate)
            {
                var previous = (_limit, _onHang);
                _limit = limit;
                _onHang = onHang;
                return previous;
            }
        }

        /// <summary>
        ///     Reads <see cref="LimitVariable"/>'s value: seconds as an invariant-culture number.
        /// </summary>
        /// <param name="text">The variable's value, or null when unset.</param>
        /// <returns><see cref="DefaultLimit"/> for null, blank or unreadable text — a typo must not switch the watchdog off;
        ///     <see cref="TimeSpan.Zero"/> (disabled) for <c>0</c> or a negative number; otherwise that many seconds.</returns>
        internal static TimeSpan ParseLimit(string text)
        {
            if (string.IsNullOrWhiteSpace(text)
                || !double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double seconds)
                || double.IsNaN(seconds))
                return DefaultLimit;
            if (seconds <= 0)
                return TimeSpan.Zero;
            // TimeSpan.FromSeconds throws past ~29 million years; anything that long is "never" anyway.
            return seconds >= TimeSpan.MaxValue.TotalSeconds ? TimeSpan.MaxValue : TimeSpan.FromSeconds(seconds);
        }

        /// <summary>
        ///     Opens a slot for one replay of <paramref name="corpus"/>: disarmed until <see cref="Slot.Enter"/>, removed by
        ///     <see cref="Slot.Dispose"/>. Starts the checking timer on first use.
        /// </summary>
        /// <param name="corpus">The corpus file name the replay reads, for the report.</param>
        /// <returns>The slot; dispose it when the replay ends (a <c>using</c>, or the enumerator's own disposal).</returns>
        internal static Slot Watch(string corpus)
        {
            var slot = new Slot(corpus);
            lock (Gate)
            {
                Live.Add(slot);
                _timer ??= new Timer(_ => CheckNow(), null, Period, Period);
            }
            return slot;
        }

        /// <summary>
        ///     Checks every live slot once — the timer's callback; callable directly by this class's own tests — and hands the
        ///     first case found past <see cref="Limit"/> to the reaction in force (<see cref="Configure"/>), once per case.
        /// </summary>
        /// <remarks>
        ///     The limit and the reaction are read in the same lock as the slots (a consistent pair, see
        ///     <see cref="Configure"/>), and the reaction runs OUTSIDE it: <see cref="Terminate"/> under a debugger writes to
        ///     the console, and a test's recorder may block, neither of which may hold up the enumerations opening and closing
        ///     slots.
        /// </remarks>
        internal static void CheckNow()
        {
            Action<string> react;
            string report = null;
            lock (Gate)
            {
                if (_limit <= TimeSpan.Zero)
                    return;
                react = _onHang;
                foreach (var slot in Live)
                    if (slot.TryReport(_limit, out report))
                        break;
            }
            if (report != null)
                react(report);
        }

        /// <summary>
        ///     The cases the live slots are currently running, oldest slot first — a snapshot for this class's own tests.
        /// </summary>
        /// <returns>(corpus, case id) per armed slot; disarmed slots (no case handed out yet) are left out.</returns>
        internal static List<(string corpus, string caseId)> LiveCases()
        {
            var cases = new List<(string, string)>();
            lock (Gate)
                foreach (var slot in Live)
                    if (slot.CurrentCase is string id)
                        cases.Add((slot.Corpus, id));
            return cases;
        }

        /// <summary>
        ///     The default reaction to a hang (<see cref="Configure"/>): writes the report to stderr and ends the process with
        ///     <see cref="Environment.FailFast(string)"/> — unless a debugger is attached, where the report goes to the
        ///     debugger's output as well and the process lives (a breakpoint inside a case is not a hang).
        /// </summary>
        /// <param name="report">The hang report (corpus, case, elapsed time, limit).</param>
        private static void Terminate(string report)
        {
            Console.Error.WriteLine(report);
            if (Debugger.IsAttached)
            {
                Debugger.Log(0, nameof(CaseWatchdog), report + Environment.NewLine);
                return;
            }
            Environment.FailFast(report);
        }

        /// <summary>
        ///     One replay's watch: the case it is running and since when.
        /// </summary>
        /// <remarks>
        ///     The replay thread writes (<see cref="Enter"/>) while the timer thread reads (<see cref="TryReport"/>), lock-free:
        ///     the start time is stored BEFORE the case id and read AFTER it (release stores, acquire loads), so a check that
        ///     sees a case's id also sees that case's start — never the previous case's, which would report a fresh case as
        ///     overdue. A check that still sees the previous id may see either start time; at worst it measures the previous
        ///     case, which is the one that was running.
        /// </remarks>
        internal sealed class Slot : IDisposable
        {
            /// <summary>The <see cref="Stopwatch"/> timestamp at which the current case was handed out.</summary>
            private long _since;

            /// <summary>The current case's id; null until the first <see cref="Enter"/>.</summary>
            private string _caseId;

            /// <summary>The case already reported (compared by reference); touched only under <see cref="Gate"/>.</summary>
            private string _reported;

            /// <summary>Creates a disarmed slot.</summary>
            /// <param name="corpus">The corpus file name, for the report.</param>
            internal Slot(string corpus) => Corpus = corpus;

            /// <summary>The corpus file name this slot watches.</summary>
            internal string Corpus { get; }

            /// <summary>The case currently running, or null before the first <see cref="Enter"/>.</summary>
            internal string CurrentCase => Volatile.Read(ref _caseId);

            /// <summary>Marks the start of a case; the previous case (if any) is over.</summary>
            /// <param name="caseId">The case's id (<see cref="FuzzCorpus.Case.Id"/>).</param>
            internal void Enter(string caseId)
            {
                // Start time first, id second: see the class remarks.
                Volatile.Write(ref _since, Stopwatch.GetTimestamp());
                Volatile.Write(ref _caseId, caseId);
            }

            /// <summary>
            ///     Builds this slot's report when its current case is past <paramref name="limit"/> and has not been reported;
            ///     called under <see cref="Gate"/>.
            /// </summary>
            /// <param name="limit">The limit in force (positive).</param>
            /// <param name="report">Receives the report, or null.</param>
            /// <returns>True when a report was produced (the case is then marked reported).</returns>
            internal bool TryReport(TimeSpan limit, out string report)
            {
                report = null;
                string id = Volatile.Read(ref _caseId);
                if (id == null || ReferenceEquals(id, _reported))
                    return false;
                TimeSpan elapsed = Stopwatch.GetElapsedTime(Volatile.Read(ref _since));
                if (elapsed < limit)
                    return false;
                _reported = id;
                report = string.Format(CultureInfo.InvariantCulture,
                    // ASCII only: the text crosses stderr into vstest's crash report, whose console may not be UTF-8.
                    "[NumSharp oracle watchdog] corpus '{0}', case '{1}': still running after {2:F0} s (limit {3:F0} s). " +
                    "A corpus case takes milliseconds, so this is an infinite loop: typically a parameter the constraint " +
                    "checks should have rejected reaching a rejection loop (logseries p = 1, zipf a <= 1). The looping test " +
                    "thread cannot be stopped, so the test host is terminated. {4} sets the limit in seconds (0 disables).",
                    Corpus, id, elapsed.TotalSeconds, limit.TotalSeconds, LimitVariable);
                return true;
            }

            /// <summary>Removes the slot: the replay is over. Idempotent.</summary>
            public void Dispose()
            {
                lock (Gate)
                    Live.Remove(this);
            }
        }
    }
}
