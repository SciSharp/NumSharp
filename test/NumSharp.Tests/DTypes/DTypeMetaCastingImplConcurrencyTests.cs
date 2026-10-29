using System;
using System.Collections.Generic;
using System.Threading;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     The per-class cast table behind <see cref="DTypeMeta.GetCastingImpl"/> and
    ///     <see cref="DTypeRegistry.AddCastingImpl"/>, which holds no lock: a <c>ConcurrentDictionary</c> for the casts
    ///     to other classes and a compare-exchange slot for the within-dtype cast. These tests pin what the lock used to
    ///     guarantee: one registration per class pair (the rest raise NumSharp's duplicate error), lookups that see
    ///     either nothing or the registered cast while registrations run, and the argument checks.
    /// </summary>
    /// <remarks>
    ///     Every class here is a private probe that is never added to <see cref="DTypeRegistry"/>'s lookup tables, so
    ///     the casts registered on it leave the builtin classes' tables, which every other test shares, untouched.
    /// </remarks>
    [TestClass]
    [DoNotParallelize] // a race test calibrated to run alone: its threads must own the cores to overlap
    public class DTypeMetaCastingImplConcurrencyTests
    {
        /// <summary>
        ///     The single-threaded contract: an empty table answers null, a registration is found on its source class
        ///     only, a second registration of the pair (or of the within-dtype cast) raises with the exact message and
        ///     leaves the first in place, and null or misfiled arguments raise.
        /// </summary>
        [TestMethod]
        public void Registration_Contract()
        {
            var from = new ProbeDTypeMeta("ContractFrom");
            var to = new ProbeDTypeMeta("ContractTo");

            from.GetCastingImpl(to).Should().BeNull("nothing is registered yet");
            from.GetCastingImpl(from).Should().BeNull("no within-dtype cast is registered yet");
            from.within_dtype_castingimpl.Should().BeNull();

            var cast = new ProbeCastingImpl(from, to, "first");
            DTypeRegistry.AddCastingImpl(cast);
            from.GetCastingImpl(to).Should().BeSameAs(cast);
            to.GetCastingImpl(from).Should().BeNull("a cast is stored on its source class only");

            Action again = () => DTypeRegistry.AddCastingImpl(new ProbeCastingImpl(from, to, "second"));
            again.Should().Throw<InvalidOperationException>()
                .WithMessage("A cast from <class 'numsharp.tests.ContractFrom'> to <class 'numsharp.tests.ContractTo'> is already registered.");
            from.GetCastingImpl(to).Should().BeSameAs(cast, "a refused registration leaves the first one in place");

            var within = new ProbeCastingImpl(from, from, "within");
            DTypeRegistry.AddCastingImpl(within);
            from.GetCastingImpl(from).Should().BeSameAs(within);
            from.within_dtype_castingimpl.Should().BeSameAs(within);
            from.GetCastingImpl(to).Should().BeSameAs(cast, "the within-dtype slot and the table are separate");

            Action withinAgain = () => DTypeRegistry.AddCastingImpl(new ProbeCastingImpl(from, from, "within2"));
            withinAgain.Should().Throw<InvalidOperationException>()
                .WithMessage("A cast within <class 'numsharp.tests.ContractFrom'> is already registered.");
            from.within_dtype_castingimpl.Should().BeSameAs(within, "a refused registration leaves the first one in place");

            Action nullTo = () => from.GetCastingImpl(null);
            nullTo.Should().Throw<ArgumentNullException>().WithParameterName("to");
            Action nullImpl = () => DTypeRegistry.AddCastingImpl(null);
            nullImpl.Should().Throw<ArgumentNullException>().WithParameterName("impl");
            Action misfiled = () => to.AddCastingImpl(new ProbeCastingImpl(from, to, "misfiled"));
            misfiled.Should().Throw<ArgumentException>().WithParameterName("impl")
                .WithMessage("casting implementation 'probe_cast_misfiled' casts from <class 'numsharp.tests.ContractFrom'>, not from <class 'numsharp.tests.ContractTo'>*");
        }

        /// <summary>
        ///     Several threads register their own cast for the same pair of two classes at the same moment, over many
        ///     rounds: in each round exactly one registration returns, every other one raises the duplicate error, and
        ///     the pair resolves to the one that returned.
        /// </summary>
        [TestMethod]
        public void SamePair_RegisteredByRacingThreads_ExactlyOneWins() => RaceRegistrations(withinDType: false);

        /// <summary>
        ///     The same race on the within-dtype slot (a cast from a class to itself), which is claimed by a
        ///     compare-exchange: a null check followed by a plain store would let two racing registrations both
        ///     return, the later one replacing the earlier.
        /// </summary>
        [TestMethod]
        public void WithinDType_RegisteredByRacingThreads_ExactlyOneWins() => RaceRegistrations(withinDType: true);

        /// <summary>
        ///     Reader threads look up casts while a writer registers them, and the writer registers 160 casts per class,
        ///     more than the table's initial 37 buckets (its capacity of 31 rounded up to a prime) hold, so lookups also run
        ///     while the table grows (at the 38th and the 90th registration). Every lookup returns
        ///     null or exactly the cast registered for that pair, a cast once seen never disappears, an unregistered
        ///     pair stays null, and no lookup throws. Afterwards every registered pair resolves.
        /// </summary>
        [TestMethod]
        public void Lookups_DuringRegistrationAndGrowth_SeeNullOrTheRegisteredCast()
        {
            const int sources = 48;
            const int destinations = 160;
            int readers = System.Math.Clamp(Environment.ProcessorCount - 1, 1, 4);

            var froms = new ProbeDTypeMeta[sources];
            var dests = new ProbeDTypeMeta[destinations];
            var casts = new ProbeCastingImpl[sources, destinations];
            for (int d = 0; d < destinations; d++)
                dests[d] = new ProbeDTypeMeta($"GrowTo{d}");
            for (int s = 0; s < sources; s++)
            {
                froms[s] = new ProbeDTypeMeta($"GrowFrom{s}");
                for (int d = 0; d < destinations; d++)
                    casts[s, d] = new ProbeCastingImpl(froms[s], dests[d], $"{s}_{d}");
            }
            var unregistered = new ProbeDTypeMeta("GrowNeverRegistered");

            // The writer publishes the source class it is filling; readers sweep that class's whole row, so most
            // lookups land on a table the writer is inserting into (and growing) at that moment.
            int filling = -1;
            int finished = 0;
            var problems = new List<string>();
            using var go = new ManualResetEventSlim(false);

            var writer = new Thread(() =>
            {
                go.Wait();
                try
                {
                    for (int s = 0; s < sources; s++)
                    {
                        Volatile.Write(ref filling, s);
                        for (int d = 0; d < destinations; d++)
                            DTypeRegistry.AddCastingImpl(casts[s, d]);
                    }
                }
                catch (Exception e)
                {
                    lock (problems)
                        problems.Add($"writer: {e}");
                }
                finally
                {
                    Volatile.Write(ref finished, 1);
                }
            }) { IsBackground = true, Name = "cast-table writer" };

            var readerThreads = new Thread[readers];
            for (int r = 0; r < readers; r++)
            {
                readerThreads[r] = new Thread(() =>
                {
                    var seen = new bool[sources, destinations];
                    go.Wait();
                    try
                    {
                        // One more sweep after the writer finishes, so the last row is read against a quiet table too.
                        bool lastSweep = false;
                        while (!lastSweep)
                        {
                            lastSweep = Volatile.Read(ref finished) == 1;
                            int s = Volatile.Read(ref filling);
                            if (s < 0)
                            {
                                // The writer has not published its first class yet; yield rather than burn the core it
                                // may need.
                                Thread.Yield();
                                continue;
                            }
                            for (int d = 0; d < destinations; d++)
                            {
                                var got = froms[s].GetCastingImpl(dests[d]);
                                if (got is null)
                                {
                                    if (seen[s, d])
                                        Report($"cast {s}->{d} disappeared after it was seen");
                                }
                                else if (!ReferenceEquals(got, casts[s, d]))
                                    Report($"cast {s}->{d} resolved to {got.name}");
                                else
                                    seen[s, d] = true;
                            }
                            if (froms[s].GetCastingImpl(unregistered) is not null)
                                Report($"an unregistered pair resolved on source {s}");
                        }
                    }
                    catch (Exception e)
                    {
                        Report($"reader: {e}");
                    }
                }) { IsBackground = true, Name = $"cast-table reader {r}" };
            }

            writer.Start();
            foreach (var t in readerThreads)
                t.Start();
            go.Set();
            writer.Join(TimeSpan.FromMinutes(1)).Should().BeTrue("the writer must finish instead of hanging the run");
            foreach (var t in readerThreads)
                t.Join(TimeSpan.FromMinutes(1)).Should().BeTrue("a reader must finish instead of hanging the run");

            problems.Should().BeEmpty();
            for (int s = 0; s < sources; s++)
                for (int d = 0; d < destinations; d++)
                    froms[s].GetCastingImpl(dests[d]).Should().BeSameAs(casts[s, d]);

            void Report(string problem)
            {
                lock (problems)
                    if (problems.Count < 20)
                        problems.Add(problem);
            }
        }

        /// <summary>
        ///     Releases one thread per racer at the start of every round; each registers its own cast for that round's
        ///     fresh class pair. Checks that each round had exactly one registration return, that every other one raised
        ///     the duplicate error with its exact message, and that the pair resolves to the registration that returned.
        /// </summary>
        /// <param name="withinDType">Race casts from a class to itself (the compare-exchange slot) instead of casts to
        ///     another class (the dictionary).</param>
        private static void RaceRegistrations(bool withinDType)
        {
            const int rounds = 1500;
            int racers = System.Math.Clamp(Environment.ProcessorCount, 2, 8);

            var froms = new ProbeDTypeMeta[rounds];
            var tos = new ProbeDTypeMeta[rounds];
            var attempts = new ProbeCastingImpl[rounds, racers];
            for (int r = 0; r < rounds; r++)
            {
                froms[r] = new ProbeDTypeMeta($"RaceFrom{r}");
                tos[r] = withinDType ? froms[r] : new ProbeDTypeMeta($"RaceTo{r}");
                for (int t = 0; t < racers; t++)
                    attempts[r, t] = new ProbeCastingImpl(froms[r], tos[r], $"{r}_{t}");
            }

            var returned = new bool[rounds, racers];
            var raised = new Exception[rounds, racers];
            var start = new Barrier(racers);
            var threads = new Thread[racers];
            for (int t = 0; t < racers; t++)
            {
                int racer = t;
                threads[t] = new Thread(() =>
                {
                    for (int r = 0; r < rounds; r++)
                    {
                        // Every racer enters a round together, so the registrations of one pair overlap as closely as
                        // the scheduler allows.
                        start.SignalAndWait();
                        try
                        {
                            DTypeRegistry.AddCastingImpl(attempts[r, racer]);
                            returned[r, racer] = true;
                        }
                        catch (Exception e)
                        {
                            raised[r, racer] = e;
                        }
                    }
                }) { IsBackground = true, Name = $"cast-table racer {racer}" };
            }

            foreach (var t in threads)
                t.Start();
            foreach (var t in threads)
                t.Join(TimeSpan.FromMinutes(1)).Should().BeTrue("a racer must finish its rounds instead of hanging the run");
            // Disposed only after every racer left: a racer still waiting in SignalAndWait would get an
            // ObjectDisposedException on its own thread, where nothing catches it, and that would end the test host.
            start.Dispose();

            var problems = new List<string>();
            for (int r = 0; r < rounds; r++)
            {
                string expected = withinDType
                    ? $"A cast within {froms[r]} is already registered."
                    : $"A cast from {froms[r]} to {tos[r]} is already registered.";
                ProbeCastingImpl winner = null;
                int winners = 0;
                for (int t = 0; t < racers; t++)
                {
                    if (returned[r, t])
                    {
                        winners++;
                        winner = attempts[r, t];
                    }
                    else if (raised[r, t] is not InvalidOperationException e || e.Message != expected)
                        problems.Add($"round {r}, racer {t}: {raised[r, t]}");
                }
                if (winners != 1)
                    problems.Add($"round {r}: {winners} registrations returned");
                else if (!ReferenceEquals(froms[r].GetCastingImpl(tos[r]), winner))
                    problems.Add($"round {r}: the pair resolves to {froms[r].GetCastingImpl(tos[r])?.name ?? "null"}, not to the registration that returned ({winner.name})");
                else if (withinDType && !ReferenceEquals(froms[r].within_dtype_castingimpl, winner))
                    problems.Add($"round {r}: within_dtype_castingimpl is not the registration that returned");
            }
            problems.Should().BeEmpty();
        }

        /// <summary>
        ///     <see cref="BuiltinCastingImpl.casting"/> is computed on its first read, and two threads may make that first
        ///     read together. It used to be a <c>Nullable&lt;NPY_CASTING&gt;</c> read with <c>??=</c>, which is two loads
        ///     (the value, then <c>HasValue</c>): a reader whose loads straddled the other thread's store returned
        ///     <see cref="NPY_CASTING.NPY_NO_CASTING"/>, and <see cref="DTypeCasting.CheckCastSafety"/>'s shortcut then
        ///     answered True for every rule (<c>np.can_cast(float64, int32, 'no')</c>). Two threads here read the casting
        ///     of a fresh float64 → int32 implementation in lockstep, round after round; every read must be
        ///     <see cref="NPY_CASTING.NPY_UNSAFE_CASTING"/>.
        /// </summary>
        /// <remarks>
        ///     Calibrated against the old pattern (a private class with the same <c>??=</c> getter, scratch probe): 843 and
        ///     1,241 torn reads per 2,000,000 racing first reads on an i9-13900K, so 200,000 rounds catch it many times
        ///     over; the fixed getter read 0 of 4,000,000. The threads wait for each other by spinning, yielding after a few
        ///     spins so a host with fewer free cores than threads still makes progress, and a time budget ends the run
        ///     early on a slow host rather than hanging it.
        /// </remarks>
        [TestMethod]
        public void BuiltinCasting_FirstReadRacedByTwoThreads_NeverTears()
        {
            const int rounds = 200_000;
            var impls = new BuiltinCastingImpl[rounds];
            for (int r = 0; r < rounds; r++)
                impls[r] = new BuiltinCastingImpl(DTypeRegistry.Double, DTypeRegistry.Int32);

            long wrong = 0, reads = 0;
            int stop = 0;
            // One slot per thread, a cache line apart, holding the round it has reached; a thread reads round r only once
            // the other has reached it too, so the two first reads of each fresh object overlap as closely as possible.
            var reached = new long[2 * 16];
            var budget = System.Diagnostics.Stopwatch.StartNew();
            void Racer(object state)
            {
                int me = (int)state, other = 1 - me;
                for (int r = 0; r < rounds; r++)
                {
                    Volatile.Write(ref reached[me * 16], r + 1);
                    int spins = 0;
                    while (Volatile.Read(ref reached[other * 16]) < r + 1)
                    {
                        if (Volatile.Read(ref stop) != 0)
                            return;
                        if (++spins > 64)
                            Thread.Yield();
                    }
                    if (impls[r].casting != NPY_CASTING.NPY_UNSAFE_CASTING)
                        Interlocked.Increment(ref wrong);
                    Interlocked.Increment(ref reads);
                    // The budget check runs rarely; once it fires, the other thread leaves its wait through `stop`.
                    if ((r & 1023) == 0 && budget.Elapsed > TimeSpan.FromSeconds(5))
                    {
                        Volatile.Write(ref stop, 1);
                        return;
                    }
                }
            }

            var a = new Thread(Racer) { IsBackground = true, Name = "casting racer 0" };
            var b = new Thread(Racer) { IsBackground = true, Name = "casting racer 1" };
            a.Start(0);
            b.Start(1);
            a.Join(TimeSpan.FromMinutes(1)).Should().BeTrue("a racer must finish instead of hanging the run");
            b.Join(TimeSpan.FromMinutes(1)).Should().BeTrue("a racer must finish instead of hanging the run");

            Interlocked.Read(ref wrong).Should().Be(0, $"no racing first read may see a torn casting ({Interlocked.Read(ref reads)} reads)");
            Interlocked.Read(ref reads).Should().BeGreaterThan(0);
            // Every implementation, read or not, still reports the cast's real safety.
            impls[0].casting.Should().Be(NPY_CASTING.NPY_UNSAFE_CASTING);
            np.can_cast(DType.Double, DType.Int32, "no").Should().BeFalse();
        }

        /// <summary>
        ///     A DType class that exists only to own a cast table: abstract, without storage, and never added to
        ///     <see cref="DTypeRegistry"/>, so its registrations are invisible to every other test.
        /// </summary>
        private sealed class ProbeDTypeMeta : DTypeMeta
        {
            /// <summary>Builds a probe class in the <c>numsharp.tests</c> module.</summary>
            /// <param name="name">The class name, which the duplicate-registration messages quote.</param>
            public ProbeDTypeMeta(string name)
                : base(name, "numsharp.tests", typeNum: -1, scalarType: null, scalarName: null, NPTypeCode.Empty,
                    kind: 'V', typeChar: 'V', itemSize: -1, alignment: 1, DTypeFlags.NPY_DT_ABSTRACT)
            {
            }

            /// <summary>Not used by these tests: a probe class has no descriptors.</summary>
            /// <returns>Always null.</returns>
            public override DType default_descr() => null;

            /// <summary>Not used by these tests: a probe class takes part in no promotion.</summary>
            /// <param name="other">Ignored.</param>
            /// <returns>Always null (NumPy's <c>NotImplemented</c>).</returns>
            public override DTypeMeta common_dtype(DTypeMeta other) => null;
        }

        /// <summary>A cast whose identity is all the tests read: it is registered and looked up, never resolved.</summary>
        private sealed class ProbeCastingImpl : CastingImpl
        {
            /// <summary>Builds a probe cast named <c>probe_cast_&lt;tag&gt;</c>.</summary>
            /// <param name="from">The source class.</param>
            /// <param name="to">The destination class (the source class itself for a within-dtype cast).</param>
            /// <param name="tag">Tells the competing registrations of one pair apart in failure messages.</param>
            public ProbeCastingImpl(DTypeMeta from, DTypeMeta to, string tag)
                : base("probe_cast_" + tag, NPY_CASTING.NPY_UNSAFE_CASTING, NDArrayMethodFlags.None, from, to)
            {
            }

            /// <summary>Never called: the tests only register and look up.</summary>
            /// <param name="givenDescrs">Ignored.</param>
            /// <param name="loopDescrs">Ignored.</param>
            /// <param name="viewOffset">Never assigned.</param>
            /// <returns>Never returns.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public override NPY_CASTING resolve_descriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
                => throw new NotSupportedException("a probe cast is only registered and looked up");
        }
    }
}
