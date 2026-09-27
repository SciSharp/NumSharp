using System;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Regression pins for the divergences the random-API oracle (<c>docs/plans/random-oracle-coverage.md</c>) surfaced
    ///     and NumSharp fixed: NumPy's handling of <c>None</c> arguments, a default <see cref="Shape"/> for <c>rand</c>, null
    ///     seed arrays, the integer <c>has_gauss</c> flag, and the 0-d cast wording of <c>permuted</c>.
    /// </summary>
    /// <remarks>
    ///     Every expected value and message was produced by NumPy 2.4.2. The oracle gates the same behaviors across engines
    ///     and seeds (<c>random_api*.jsonl</c>); these tests name each fix where a reader will look for it.
    /// </remarks>
    [TestClass]
    public class RandomOracleFindingsTests
    {
        /// <summary>A PCG64-backed Generator for the Generator pins (the seed is irrelevant to every error below).</summary>
        /// <returns>A fresh Generator.</returns>
        private static Generator Gen() => new Generator(new PCG64(1));

        /// <summary>
        ///     <c>rand(default(Shape))</c> is NumPy's <c>rand()</c> — one draw, returned 0-d — where it used to throw
        ///     <see cref="ArgumentNullException"/> allocating from a shape without dimensions.
        /// </summary>
        [TestMethod]
        public void Rand_DefaultShape_IsOneDraw()
        {
            using var r = np.random.RandomState(42).rand(default(Shape));
            r.ndim.Should().Be(0);
            r.GetDouble().Should().Be(0.3745401188473625); // np.random.RandomState(42).rand()
        }

        /// <summary>A null seed array is NumPy's <c>seed(None)</c> / <c>RandomState(None)</c>: fresh OS entropy, not an error.</summary>
        [TestMethod]
        public void NullSeedArray_IsOsEntropy()
        {
            var rs = np.random.RandomState(7);
            rs.standard_normal().Dispose(); // fill the Gaussian cache
            rs.seed((int[])null);
            ((NumPyRandom.State)rs.get_state(false)).has_gauss.Should().Be(0, "a re-seed clears the Gaussian cache");
            rs.seed((long[])null);
            rs.seed((uint[])null);

            // Two entropy seedings produce different keys (a 624-word collision is not a real outcome).
            var a = np.random.RandomState((uint[])null).get_state().Key;
            var b = np.random.RandomState((long[])null).get_state().Key;
            var c = np.random.RandomState((int[])null).get_state().Key;
            a.Should().NotEqual(b);
            b.Should().NotEqual(c);

            // An EMPTY array is still NumPy's "Seed must be non-empty".
            ((Action)(() => np.random.RandomState(1).seed(new int[0]))).Should().Throw<ValueError>().WithMessage("Seed must be non-empty");
        }

        /// <summary>
        ///     <c>set_state</c> keeps the C int NumPy stores for <c>has_gauss</c> (2 reads back 2) until the cached Gaussian
        ///     is consumed, which returns the tuple's value and clears the flag to 0.
        /// </summary>
        [TestMethod]
        public void SetState_HasGaussInt_RoundTrips()
        {
            var rs = np.random.RandomState(42);
            var st = rs.get_state();
            rs.set_state(new NativeRandomState(st.Key, st.Pos, 2, -1.5));
            rs.get_state().HasGauss.Should().Be(2);
            ((NumPyRandom.State)rs.get_state(false)).has_gauss.Should().Be(2);
            using (var z = rs.standard_normal())
                z.GetDouble().Should().Be(-1.5);
            rs.get_state().HasGauss.Should().Be(0);
        }

        /// <summary>The legacy members' <c>None</c> errors (NumPy: <c>np.asarray(None)</c> is a 0-d object array).</summary>
        [TestMethod]
        public void Legacy_NullArguments_RaiseNumPysErrors()
        {
            var rs = np.random.RandomState(1);
            ((Action)(() => rs.choice((NDArray)null))).Should().Throw<ValueError>().WithMessage("a must be 1-dimensional or an integer");
            ((Action)(() => rs.permutation((NDArray)null))).Should().Throw<IndexError>().WithMessage("x must be an integer or at least 1-dimensional");
            ((Action)(() => rs.shuffle(null))).Should().Throw<TypeError>().WithMessage("object of type 'NoneType' has no len()");
        }

        /// <summary>The Generator members' <c>None</c> errors, in NumPy's check order.</summary>
        [TestMethod]
        public void Generator_NullArguments_RaiseNumPysErrors()
        {
            var g = Gen();
            ((Action)(() => g.choice((NDArray)null))).Should().Throw<ValueError>()
                .WithMessage("a must be a sequence or an integer, not <class 'NoneType'>");
            ((Action)(() => g.permutation((NDArray)null))).Should().Throw<AxisError>()
                .WithMessage("axis 0 is out of bounds for array of dimension 0*");
            ((Action)(() => g.shuffle(null))).Should().Throw<TypeError>().WithMessage("object of type 'NoneType' has no len()");
            ((Action)(() => g.permuted(null))).Should().Throw<TypeError>().WithMessage("len() of unsized object");
            ((Action)(() => g.permuted(null, axis: 1))).Should().Throw<AxisError>()
                .WithMessage("axis 1 is out of bounds for array of dimension 0*");
            ((Action)(() => g.permuted(null, @out: np.zeros(3)))).Should().Throw<ValueError>().WithMessage("out must have the same shape as x");
            ((Action)(() => g.permuted(null, @out: np.zeros(Shape.Scalar)))).Should().Throw<TypeError>()
                .WithMessage("Cannot cast scalar from dtype('O') to dtype('float64') according to the rule 'safe'");
        }

        /// <summary>
        ///     <c>vonmises(mu=NaN)</c> returns NumPy's POSITIVE quiet NaN on every runtime: the wrap into [-pi, pi) is C's
        ///     <c>fmod</c>, which propagates the input NaN, while .NET 8's double <c>%</c> returned the default NaN (sign bit
        ///     set) — a divergence only the net8.0 replay showed.
        /// </summary>
        [TestMethod]
        public void VonMises_NaNMu_KeepsNumPysNaNBits()
        {
            const long positiveQuietNaN = 0x7ff8000000000000;
            using (var legacy = np.random.RandomState(0).vonmises(double.NaN, 2.0, new Shape(3)))
                for (long i = 0; i < 3; i++)
                    BitConverter.DoubleToInt64Bits(legacy.GetAtIndex<double>(i)).Should().Be(positiveQuietNaN);
            using (var modern = Gen().vonmises(double.NaN, 2.0, new Shape(3)))
                for (long i = 0; i < 3; i++)
                    BitConverter.DoubleToInt64Bits(modern.GetAtIndex<double>(i)).Should().Be(positiveQuietNaN);
        }

        /// <summary><c>permuted</c>'s copy into <c>out</c> words a 0-d source as a "scalar", as NumPy's <c>copyto</c> does.</summary>
        [TestMethod]
        public void Permuted_ZeroDimensionalCastError_SaysScalar()
        {
            using var x = NDArray.Scalar(5L);
            using var o = np.zeros(Shape.Scalar, np.int32);
            ((Action)(() => Gen().permuted(x, @out: o))).Should().Throw<TypeError>()
                .WithMessage("Cannot cast scalar from dtype('int64') to dtype('int32') according to the rule 'safe'");
        }
    }
}
