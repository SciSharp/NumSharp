using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Guards the `using` on `x = np.log(1 - uniform(...))` inside
    /// np.random.exponential. The log buffer is dead once np.negative(x)
    /// has been multiplied by scale.
    /// </summary>
    /// <remarks>
    /// <c>Exponential_TightLoop_DoesNotLeakWorkingSet</c> used to close this class and was removed;
    /// it sampled process RSS rather than the generator. See <see cref="LeakGuards"/>.
    /// </remarks>
    [TestClass]
    public class np_random_exponential_using_test : TestClass
    {
        // --------------------------- correctness ---------------------------

        [TestMethod]
        public void Exponential_Shape_MatchesRequested()
        {
            var samples = np.random.exponential(1.0, new Shape(1000));
            samples.shape.Should().ContainInOrder(1000L);
            samples.dtype.Should().Be(typeof(double));
        }

        [TestMethod]
        public void Exponential_AllSamples_Nonnegative()
        {
            // log(1 - U) where U ∈ [0, 1) is always ≤ 0; negated is ≥ 0.
            var samples = np.random.exponential(2.0, new Shape(5000));
            double min = (double)np.amin(samples);
            min.Should().BeGreaterThanOrEqualTo(0.0);
        }

        [TestMethod]
        public void Exponential_MeanCloseToScale()
        {
            // E[Exp(scale)] = scale. With 10K samples, sample mean should land
            // close. Use a wide tolerance so we don't fail on RNG variance.
            var samples = np.random.exponential(3.0, new Shape(10_000));
            double mean = (double)np.mean(samples);
            mean.Should().BeApproximately(3.0, 0.15);
        }

        // --------------------------- lifetime ---------------------------

        /// <summary>
        /// The released <c>log</c> buffer must not be the one handed back.
        /// </summary>
        /// <remarks>
        /// <c>-x * scale</c> is what the caller receives, and elementwise ops in NumSharp can
        /// return a buffer they were given rather than a fresh one. If the returned samples ever
        /// aliased the <c>using</c>-bound <c>x</c>, every caller would get an array over freed
        /// pages — values would read back as garbage rather than as a valid exponential draw.
        /// Asserting the refcount AND the distribution catches both halves of that.
        /// </remarks>
        [TestMethod]
        public void Exponential_ResultIsNotTheReleasedLogBuffer()
        {
            var samples = np.random.exponential(2.0, new Shape(5_000));

            LeakGuards.StillUsable(samples, " — the returned samples must not alias the released log buffer");

            // Live memory holding real values, not freed pages: every draw is finite and >= 0,
            // which the log buffer's own contents (all <= 0) could not satisfy.
            ((double)np.amin(samples)).Should().BeGreaterThanOrEqualTo(0.0);
            np.all(np.isfinite(samples)).Should().BeTrue();
            ((double)np.mean(samples)).Should().BeApproximately(2.0, 0.25);
        }
    }
}
