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
            from.WithinDTypeCastingImpl.Should().BeNull();

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
            from.WithinDTypeCastingImpl.Should().BeSameAs(within);
            from.GetCastingImpl(to).Should().BeSameAs(cast, "the within-dtype slot and the table are separate");

            Action withinAgain = () => DTypeRegistry.AddCastingImpl(new ProbeCastingImpl(from, from, "within2"));
            withinAgain.Should().Throw<InvalidOperationException>()
                .WithMessage("A cast within <class 'numsharp.tests.ContractFrom'> is already registered.");
            from.WithinDTypeCastingImpl.Should().BeSameAs(within, "a refused registration leaves the first one in place");

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
                                    Report($"cast {s}->{d} resolved to {got.Name}");
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
            // Disposed only after every racer left: disposing a Barrier with a waiter inside throws, which would mask
            // the join failure above.
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
                    problems.Add($"round {r}: the pair resolves to {froms[r].GetCastingImpl(tos[r])?.Name ?? "null"}, not to the registration that returned ({winner.Name})");
                else if (withinDType && !ReferenceEquals(froms[r].WithinDTypeCastingImpl, winner))
                    problems.Add($"round {r}: WithinDTypeCastingImpl is not the registration that returned");
            }
            problems.Should().BeEmpty();
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
                    kind: 'V', typeChar: 'V', itemSize: -1, alignment: 1, DTypeFlags.Abstract)
            {
            }

            /// <summary>Not used by these tests: a probe class has no descriptors.</summary>
            /// <returns>Always null.</returns>
            public override DType DefaultDescr() => null;

            /// <summary>Not used by these tests: a probe class takes part in no promotion.</summary>
            /// <param name="other">Ignored.</param>
            /// <returns>Always null (NumPy's <c>NotImplemented</c>).</returns>
            public override DTypeMeta CommonDType(DTypeMeta other) => null;
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
            public override NPY_CASTING ResolveDescriptors(DType[] givenDescrs, DType[] loopDescrs, out long viewOffset)
                => throw new NotSupportedException("a probe cast is only registered and looked up");
        }
    }
}
