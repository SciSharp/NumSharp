using System;
using System.Diagnostics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.RandomSampling
{
    /// <summary>
    /// Guards the `using` on `x = np.log(1 - uniform(...))` inside
    /// np.random.exponential. The log buffer is dead once np.negative(x)
    /// has been multiplied by scale.
    /// </summary>
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

        // --------------------------- leak guard ---------------------------

        /// <summary>
        /// Tight loop. Each call allocated three transient buffers
        /// (uniform, 1-uniform, log) before the using; one of them (`x`)
        /// is now atomically released. Working set should not grow.
        /// </summary>
        [TestMethod]
        public void Exponential_TightLoop_DoesNotLeakWorkingSet()
        {
            for (int i = 0; i < 20; i++)
                _ = np.random.exponential(1.0, new Shape(50_000));
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var p = Process.GetCurrentProcess();
            p.Refresh();
            long start = p.WorkingSet64;

            for (int i = 0; i < 500; i++)
            {
                using var samples = np.random.exponential(1.0, new Shape(50_000));
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            p.Refresh();
            long deltaMB = (p.WorkingSet64 - start) / (1024 * 1024);

            deltaMB.Should().BeLessThan(30);
        }
    }
}
