using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The per-case hang deadline of the corpus replays (<see cref="CaseWatchdog"/>): which cases it arms, when it
    ///     reports, that it reports once, and that every <see cref="CorpusFile"/> enumeration is watched without the replay
    ///     loop doing anything.
    /// </summary>
    /// <remarks>
    ///     The real reaction to a hang ends the process, so every test here swaps it for a recorder — together with a tiny
    ///     limit, in ONE <see cref="CaseWatchdog.Configure"/> call — and restores the previous pair afterwards. The recorder is
    ///     thread-safe because the watchdog's own timer may report a case between two explicit
    ///     <see cref="CaseWatchdog.CheckNow"/> calls — which is fine: a case is reported ONCE whoever finds it, so the counts
    ///     asserted here hold either way. The terminating path itself is proven end to end by planting the inclusive bound
    ///     that once hung the oracle (see the commit that added this class), not here.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class CaseWatchdogTests
    {
        /// <summary>
        ///     A case handed out and still running past the limit is reported once, naming its corpus and its id; the next
        ///     case is a new report. A slot with no case handed out yet reports nothing.
        /// </summary>
        [TestMethod]
        public void ArmedCase_PastTheLimit_IsReportedOnce_NamingItsCorpusAndCase()
        {
            var reports = new ConcurrentQueue<string>();
            WithWatchdog(TimeSpan.FromMilliseconds(1), reports.Enqueue, () =>
            {
                using var slot = CaseWatchdog.Watch("synthetic.jsonl");
                Thread.Sleep(20);
                CaseWatchdog.CheckNow();
                Assert.AreEqual(0, reports.Count, "a slot with no case handed out is disarmed");

                slot.Enter("case/1");
                Thread.Sleep(20);
                CaseWatchdog.CheckNow();
                CaseWatchdog.CheckNow();
                Assert.AreEqual(1, reports.Count, "a hung case is reported once, however many checks find it");
                string first = reports.First();
                StringAssert.Contains(first, "corpus 'synthetic.jsonl', case 'case/1'");
                StringAssert.Contains(first, CaseWatchdog.LimitVariable);

                slot.Enter("case/2");
                Thread.Sleep(20);
                CaseWatchdog.CheckNow();
                Assert.AreEqual(2, reports.Count, "the next case is a new report");
                StringAssert.Contains(reports.Last(), "case 'case/2'");
            });
        }

        /// <summary>
        ///     Nothing is reported for a case within the limit, for a slot already disposed, or with the watchdog disabled
        ///     (a zero limit).
        /// </summary>
        [TestMethod]
        public void CaseWithinTheLimit_DisposedSlot_OrDisabledWatchdog_IsNotReported()
        {
            var reports = new ConcurrentQueue<string>();
            WithWatchdog(TimeSpan.FromMinutes(10), reports.Enqueue, () =>
            {
                using var slot = CaseWatchdog.Watch("synthetic.jsonl");
                slot.Enter("case/within");
                CaseWatchdog.CheckNow();
            });
            WithWatchdog(TimeSpan.FromMilliseconds(1), reports.Enqueue, () =>
            {
                var slot = CaseWatchdog.Watch("synthetic.jsonl");
                slot.Enter("case/disposed");
                slot.Dispose();
                slot.Dispose();   // idempotent
                Thread.Sleep(20);
                CaseWatchdog.CheckNow();
            });
            WithWatchdog(TimeSpan.Zero, reports.Enqueue, () =>
            {
                using var slot = CaseWatchdog.Watch("synthetic.jsonl");
                slot.Enter("case/disabled");
                Thread.Sleep(20);
                CaseWatchdog.CheckNow();
            });
            Assert.AreEqual(0, reports.Count, string.Join("\n", reports));
        }

        /// <summary>
        ///     A <see cref="CorpusFile"/> enumeration arms its slot with the case it has just handed out — the one the replay
        ///     loop is running — and an early exit from the loop, or its completion, removes the slot.
        /// </summary>
        [TestMethod]
        public void CorpusFileEnumeration_ArmsTheCaseItHandsOut_AndDisarmsWhenTheLoopEnds()
        {
            const string tier = "where.jsonl";
            using var file = FuzzCorpus.Open(tier);
            string first = null;
            foreach (var c in file)
            {
                var running = CaseWatchdog.LiveCases().Where(x => x.corpus == tier).Select(x => x.caseId).ToList();
                CollectionAssert.AreEqual(new[] { c.Id }, running, "exactly the case just handed out is armed");
                if (first == null)
                {
                    first = c.Id;
                    continue;
                }
                Assert.AreNotEqual(first, c.Id);
                break;   // the foreach disposes the enumerator here
            }
            Assert.IsNotNull(first);
            Assert.IsFalse(CaseWatchdog.LiveCases().Any(x => x.corpus == tier), "an early break removes the slot");

            int seen = 0;
            foreach (var _ in file)
                seen++;
            Assert.AreEqual(file.Count, seen);
            Assert.IsFalse(CaseWatchdog.LiveCases().Any(x => x.corpus == tier), "a completed enumeration removes the slot");
        }

        /// <summary>
        ///     The limit variable reads seconds; zero or a negative number disables the watchdog, while unset, blank or
        ///     unreadable text keeps the default (a typo must not switch it off), and an enormous value means "never".
        /// </summary>
        [TestMethod]
        public void ParseLimit_ReadsSeconds_ZeroDisables_UnreadableKeepsTheDefault()
        {
            Assert.AreEqual(CaseWatchdog.DefaultLimit, CaseWatchdog.ParseLimit(null));
            Assert.AreEqual(CaseWatchdog.DefaultLimit, CaseWatchdog.ParseLimit(""));
            Assert.AreEqual(CaseWatchdog.DefaultLimit, CaseWatchdog.ParseLimit("   "));
            Assert.AreEqual(CaseWatchdog.DefaultLimit, CaseWatchdog.ParseLimit("2m"));
            Assert.AreEqual(CaseWatchdog.DefaultLimit, CaseWatchdog.ParseLimit("NaN"));
            Assert.AreEqual(TimeSpan.Zero, CaseWatchdog.ParseLimit("0"));
            Assert.AreEqual(TimeSpan.Zero, CaseWatchdog.ParseLimit("-5"));
            Assert.AreEqual(TimeSpan.FromSeconds(15), CaseWatchdog.ParseLimit("15"));
            Assert.AreEqual(TimeSpan.FromSeconds(2.5), CaseWatchdog.ParseLimit("2.5"));
            Assert.AreEqual(TimeSpan.MaxValue, CaseWatchdog.ParseLimit("1e300"));
            Assert.AreEqual(TimeSpan.MaxValue, CaseWatchdog.ParseLimit("Infinity"));
        }

        /// <summary>Runs <paramref name="body"/> with the watchdog's limit and reaction swapped, restoring both after.</summary>
        /// <param name="limit">The limit to apply.</param>
        /// <param name="onHang">The reaction to apply — a recorder, never the process-ending default.</param>
        /// <param name="body">The test body.</param>
        /// <remarks>Both are swapped and restored as one pair (<see cref="CaseWatchdog.Configure"/>), so the watchdog's timer
        ///     never sees the tiny limit paired with the reaction that ends the process.</remarks>
        private static void WithWatchdog(TimeSpan limit, Action<string> onHang, Action body)
        {
            var saved = CaseWatchdog.Configure(limit, onHang);
            try
            {
                body();
            }
            finally
            {
                CaseWatchdog.Configure(saved.limit, saved.onHang);
            }
        }
    }
}
