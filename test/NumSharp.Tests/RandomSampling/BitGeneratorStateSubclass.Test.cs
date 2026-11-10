using System;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     The five engine state snapshots are unsealed (the owner's decision of 2026-11-10, "Unseal"): a subclass engine that
    ///     adds state of its own carries it through <c>state</c> in a derived snapshot. NumPy's state is a dict, and a subclass
    ///     adds its own keys to it.
    /// </summary>
    /// <remarks>
    ///     Each derived snapshot below compiles only because its engine's <c>State</c> is unsealed; each test then checks
    ///     what a derived snapshot must pass: the engine's typed <c>state</c> getter (a cast to <c>State</c>) and its
    ///     <c>SetStateCore</c> (an <c>is State</c> test).
    /// </remarks>
    [TestClass]
    public class BitGeneratorStateSubclassTest
    {
        /// <summary>A consumer's PCG64 that counts its 64-bit draws and carries the count in its state.</summary>
        private sealed class CountingPCG64 : PCG64
        {
            /// <summary>Seeds the engine as <c>PCG64(seed)</c> does.</summary>
            /// <param name="seed">The seed.</param>
            public CountingPCG64(long seed) : base(seed) { }

            /// <summary>The number of <see cref="NextUInt64"/> draws since construction or the last restore.</summary>
            public long draws;

            /// <summary>The PCG64 snapshot plus the draw count.</summary>
            public sealed class CountingState : PCG64.State
            {
                /// <summary>The draw count at the snapshot.</summary>
                public long draws { get; set; }
            }

            /// <inheritdoc/>
            public override ulong NextUInt64()
            {
                draws++;
                return base.NextUInt64();
            }

            /// <inheritdoc/>
            public override BitGeneratorState GetStateCore()
            {
                var s = (PCG64.State)base.GetStateCore();
                return new CountingState { state = s.state, inc = s.inc, has_uint32 = s.has_uint32, uinteger = s.uinteger, draws = draws };
            }

            /// <inheritdoc/>
            public override void SetStateCore(BitGeneratorState value)
            {
                // The base validates the snapshot (is PCG64.State) and restores the engine words; the count is ours.
                base.SetStateCore(value);
                draws = value is CountingState c ? c.draws : 0;
            }
        }

        /// <summary>
        ///     A subclass's derived snapshot round-trips through the typed <c>state</c>: reading it returns the derived class
        ///     with the subclass's field, and assigning it back restores both the engine's words and that field.
        /// </summary>
        [TestMethod]
        public void SubclassState_RoundTripsThroughTheTypedState()
        {
            var g = new CountingPCG64(7);
            g.NextUInt64();
            g.NextUInt64();
            PCG64.State snapshot = g.state;
            snapshot.Should().BeOfType<CountingPCG64.CountingState>().Which.draws.Should().Be(2);

            ulong next = g.NextUInt64();
            g.draws.Should().Be(3);
            g.state = snapshot;
            g.draws.Should().Be(2);
            g.NextUInt64().Should().Be(next, "the stream resumes where the snapshot was taken");
        }

        /// <summary>A derived MT19937 snapshot (no members of its own).</summary>
        private sealed class DerivedMT19937State : MT19937.State { }

        /// <summary>A derived PCG64 snapshot.</summary>
        private sealed class DerivedPCG64State : PCG64.State { }

        /// <summary>A derived PCG64DXSM snapshot.</summary>
        private sealed class DerivedPCG64DXSMState : PCG64DXSM.State { }

        /// <summary>A derived Philox snapshot.</summary>
        private sealed class DerivedPhiloxState : Philox.State { }

        /// <summary>A derived SFC64 snapshot.</summary>
        private sealed class DerivedSFC64State : SFC64.State { }

        /// <summary>
        ///     Every engine takes a derived snapshot of its own state: copied from one engine into another, it moves the second
        ///     onto the first's stream.
        /// </summary>
        [TestMethod]
        public void EveryEngine_TakesADerivedSnapshot()
        {
            var mtA = new MT19937(5);
            mtA.NextUInt64();
            var mt = mtA.state;
            var mtB = new MT19937(9) { state = new DerivedMT19937State { key = (uint[])mt.key.Clone(), pos = mt.pos } };
            mtB.NextUInt64().Should().Be(mtA.NextUInt64(), "MT19937");

            var pA = new PCG64(5);
            pA.NextUInt32();
            var p = pA.state;
            var pB = new PCG64(9) { state = new DerivedPCG64State { state = p.state, inc = p.inc, has_uint32 = p.has_uint32, uinteger = p.uinteger } };
            pB.NextUInt32().Should().Be(pA.NextUInt32(), "PCG64, the buffered half included");

            var dA = new PCG64DXSM(5);
            dA.NextUInt64();
            var d = dA.state;
            var dB = new PCG64DXSM(9) { state = new DerivedPCG64DXSMState { state = d.state, inc = d.inc, has_uint32 = d.has_uint32, uinteger = d.uinteger } };
            dB.NextUInt64().Should().Be(dA.NextUInt64(), "PCG64DXSM");

            var xA = new Philox(5);
            xA.NextUInt64();
            var x = xA.state;
            var xB = new Philox(9)
            {
                state = new DerivedPhiloxState
                {
                    counter = (ulong[])x.counter.Clone(), key = (ulong[])x.key.Clone(), buffer = (ulong[])x.buffer.Clone(),
                    buffer_pos = x.buffer_pos, has_uint32 = x.has_uint32, uinteger = x.uinteger,
                },
            };
            xB.NextUInt64().Should().Be(xA.NextUInt64(), "Philox");

            var sA = new SFC64(5);
            sA.NextUInt64();
            var s = sA.state;
            var sB = new SFC64(9) { state = new DerivedSFC64State { state = (ulong[])s.state.Clone(), has_uint32 = s.has_uint32, uinteger = s.uinteger } };
            sB.NextUInt64().Should().Be(sA.NextUInt64(), "SFC64");
        }
    }
}
