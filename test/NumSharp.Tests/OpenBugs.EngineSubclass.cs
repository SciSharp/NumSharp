using System;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open bugs of a consumer's subclass of a built-in bit generator (the engines are unsealed since 0.80.0), found on
    ///     2026-11-10 while unsealing their state snapshots. NumPy 2.4.2 names the SUBCLASS wherever it names the class - its
    ///     state label and refusals read <c>self.__class__.__name__</c>, <c>spawn</c> builds <c>type(self)(seed=...)</c> and
    ///     <c>jumped</c> builds <c>self.__class__()</c> - where NumSharp names, and builds, the built-in class. Pinned and
    ///     deferred (found bugs become <c>[OpenBugs]</c> reproductions); every test asserts NumPy's answer for the minimal
    ///     subclass <c>type("MyPCG64", (np.random.PCG64,), {})</c>.
    /// </summary>
    /// <remarks>
    ///     A NumSharp subclass can work around three of them today: override <c>Name</c> for the repr, return a derived
    ///     snapshot (the <c>State</c> classes are unsealed) whose <c>bit_generator</c> names the subclass for the label, and
    ///     override <c>CreateFromSeed</c> for <c>spawn</c>; the refusals and <c>jumped</c> have no hook. A fix would read the
    ///     runtime class where NumPy reads <c>self.__class__</c> (its name for the label and the refusal texts, a construction
    ///     of the receiver's class for <c>spawn</c> and <c>jumped</c>).
    /// </remarks>
    [TestClass]
    public class OpenBugsEngineSubclassTests : TestClass
    {
        /// <summary>The minimal MT19937 subclass (NumPy: <c>class MyMT19937(np.random.MT19937): pass</c>).</summary>
        private sealed class MyMT19937 : MT19937
        {
            /// <summary>Seeds as <c>MT19937(seed)</c>.</summary>
            /// <param name="seed">The seed.</param>
            public MyMT19937(long seed) : base(seed) { }
        }

        /// <summary>The minimal PCG64 subclass.</summary>
        private sealed class MyPCG64 : PCG64
        {
            /// <summary>Seeds as <c>PCG64(seed)</c>.</summary>
            /// <param name="seed">The seed.</param>
            public MyPCG64(long seed) : base(seed) { }
        }

        /// <summary>The minimal PCG64DXSM subclass.</summary>
        private sealed class MyPCG64DXSM : PCG64DXSM
        {
            /// <summary>Seeds as <c>PCG64DXSM(seed)</c>.</summary>
            /// <param name="seed">The seed.</param>
            public MyPCG64DXSM(long seed) : base(seed) { }
        }

        /// <summary>The minimal Philox subclass.</summary>
        private sealed class MyPhilox : Philox
        {
            /// <summary>Seeds as <c>Philox(seed)</c>.</summary>
            /// <param name="seed">The seed.</param>
            public MyPhilox(long seed) : base(seed) { }
        }

        /// <summary>The minimal SFC64 subclass.</summary>
        private sealed class MySFC64 : SFC64
        {
            /// <summary>Seeds as <c>SFC64(seed)</c>.</summary>
            /// <param name="seed">The seed.</param>
            public MySFC64(long seed) : base(seed) { }
        }

        // ============================================================================
        //  BUG: the state label names the built-in class.
        //
        //  NumPy: g.state['bit_generator'] is self.__class__.__name__, 'MyPCG64'. NumSharp: each State class returns a
        //  constant ("PCG64"), whatever engine built it.
        // ============================================================================

        /// <summary>A subclass's state is labelled with the subclass's name, for each of the five engines.</summary>
        [TestMethod, OpenBugs(IssueUrl = "engine-subclass-state-label")]
        public void StateLabel_NamesTheSubclass()
        {
            new MyMT19937(1).state.bit_generator.Should().Be("MyMT19937");
            new MyPCG64(1).state.bit_generator.Should().Be("MyPCG64");
            new MyPCG64DXSM(1).state.bit_generator.Should().Be("MyPCG64DXSM");
            new MyPhilox(1).state.bit_generator.Should().Be("MyPhilox");
            new MySFC64(1).state.bit_generator.Should().Be("MySFC64");
        }

        // ============================================================================
        //  BUG: a subclass and its built-in class take each other's state.
        //
        //  NumPy's setter compares the dict's label with self.__class__.__name__ and raises ValueError when they differ, in
        //  both directions. NumSharp's SetStateCore tests `is State`, which both snapshots pass. SFC64's text is NumPy 2.4.2's
        //  own, an f-string missing its f prefix.
        // ============================================================================

        /// <summary>A subclass refuses a plain engine's state with NumPy's text.</summary>
        [TestMethod, OpenBugs(IssueUrl = "engine-subclass-state-refusal")]
        public void Subclass_RefusesAPlainEnginesState()
        {
            new Action(() => new MyMT19937(1).state = new MT19937(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a MyMT19937 PRNG");
            new Action(() => new MyPCG64(1).state = new PCG64(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a MyPCG64 RNG");
            new Action(() => new MyPCG64DXSM(1).state = new PCG64DXSM(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a MyPCG64DXSM RNG");
            new Action(() => new MyPhilox(1).state = new Philox(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a MyPhilox PRNG");
            new Action(() => new MySFC64(1).state = new SFC64(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a {self.__class__.__name__} RNG");
        }

        /// <summary>A plain engine refuses a subclass's state with NumPy's text.</summary>
        [TestMethod, OpenBugs(IssueUrl = "engine-subclass-state-refusal")]
        public void PlainEngine_RefusesASubclassState()
        {
            new Action(() => new MT19937(1).state = new MyMT19937(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a MT19937 PRNG");
            new Action(() => new PCG64(1).state = new MyPCG64(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a PCG64 RNG");
            new Action(() => new PCG64DXSM(1).state = new MyPCG64DXSM(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a PCG64DXSM RNG");
            new Action(() => new Philox(1).state = new MyPhilox(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a Philox PRNG");
            new Action(() => new SFC64(1).state = new MySFC64(1).state).Should().Throw<ValueError>()
                .WithMessage("state must be for a {self.__class__.__name__} RNG");
        }

        // ============================================================================
        //  BUG: Generator's repr names the built-in class.
        //
        //  NumPy: repr(Generator(g)) reads g.__class__.__name__ ("Generator(MyPCG64)"). NumSharp prints Name, which each
        //  engine overrides with its constant.
        // ============================================================================

        /// <summary>Generator's repr names the subclass.</summary>
        [TestMethod, OpenBugs(IssueUrl = "engine-subclass-generator-repr")]
        public void GeneratorRepr_NamesTheSubclass()
        {
            new Generator(new MyPCG64(1)).ToString().Should().Be("Generator(MyPCG64)");
            new Generator(new MyMT19937(1)).ToString().Should().Be("Generator(MyMT19937)");
            new Generator(new MyPCG64DXSM(1)).ToString().Should().Be("Generator(MyPCG64DXSM)");
            new Generator(new MyPhilox(1)).ToString().Should().Be("Generator(MyPhilox)");
            new Generator(new MySFC64(1)).ToString().Should().Be("Generator(MySFC64)");
        }

        // ============================================================================
        //  BUG: spawn and jumped build the built-in class.
        //
        //  NumPy: BitGenerator.spawn builds type(self)(seed=child) and jumped builds self.__class__(), so both return the
        //  subclass. NumSharp: spawn builds through CreateFromSeed (each engine returns its own built-in type unless a
        //  subclass overrides it) and jumped through `new PCG64()` and its siblings.
        // ============================================================================

        /// <summary>spawn returns children of the subclass.</summary>
        [TestMethod, OpenBugs(IssueUrl = "engine-subclass-spawn-jumped")]
        public void Spawn_ReturnsTheSubclass()
        {
            new MyMT19937(1).spawn(1)[0].Should().BeOfType<MyMT19937>();
            new MyPCG64(1).spawn(1)[0].Should().BeOfType<MyPCG64>();
            new MyPCG64DXSM(1).spawn(1)[0].Should().BeOfType<MyPCG64DXSM>();
            new MyPhilox(1).spawn(1)[0].Should().BeOfType<MyPhilox>();
            new MySFC64(1).spawn(1)[0].Should().BeOfType<MySFC64>();
        }

        /// <summary>jumped returns the subclass (SFC64 has no jumped).</summary>
        [TestMethod, OpenBugs(IssueUrl = "engine-subclass-spawn-jumped")]
        public void Jumped_ReturnsTheSubclass()
        {
            new MyMT19937(1).jumped().Should().BeOfType<MyMT19937>();
            new MyPCG64(1).jumped().Should().BeOfType<MyPCG64>();
            new MyPCG64DXSM(1).jumped().Should().BeOfType<MyPCG64DXSM>();
            new MyPhilox(1).jumped().Should().BeOfType<MyPhilox>();
        }
    }
}
