using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling;

/// <summary>
/// Tests for np.random.pareto following NumPy 2.4.2 behavior.
/// NumPy's pareto returns samples from Pareto II (Lomax) distribution.
/// </summary>

    public class RandomParetoTests : TestClass
{
    [TestMethod]
    public void Pareto_ScalarCall_Returns0dArray()
    {
        var rng = np.random.RandomState(42);
        NDArray result = rng.pareto(2.0);
        Assert.AreEqual(0, result.ndim, "Scalar call should return 0-d array");
        Assert.IsTrue(result.GetDouble() >= 0, "Pareto samples should be non-negative");
    }

    [TestMethod]
    public void Pareto_ArraySize_ReturnsCorrectShape()
    {
        var rng = np.random.RandomState(42);
        var result = rng.pareto(2.0, 5L);
        result.Should().BeShaped(5);
        Assert.AreEqual(typeof(double), result.dtype);
    }

    [TestMethod]
    public void Pareto_MultiDimensionalSize_ReturnsCorrectShape()
    {
        var rng = np.random.RandomState(42);
        var result = rng.pareto(2.5, new Shape(2, 3));
        result.Should().BeShaped(2, 3);
    }

    [TestMethod]
    public void Pareto_ShapeSize_ReturnsCorrectShape()
    {
        var rng = np.random.RandomState(42);
        var result = rng.pareto(1.5, new Shape(3, 4));
        result.Should().BeShaped(3, 4);
    }

    [TestMethod]
    public void Pareto_AllValuesNonNegative()
    {
        var rng = np.random.RandomState(12345);
        var samples = rng.pareto(2.0, 1000L);

        for (int i = 0; i < samples.size; i++)
        {
            Assert.IsTrue((double)samples.GetAtIndex(i) >= 0,
                $"Sample at index {i} should be non-negative");
        }
    }

    [TestMethod]
    public void Pareto_MeanConvergesToExpected()
    {
        // For Pareto II (Lomax), mean = 1/(a-1) for a > 1
        // With a=3, expected mean = 1/(3-1) = 0.5
        var rng = np.random.RandomState(12345);
        var samples = rng.pareto(3.0, 100000L);
        double mean = (double)np.mean(samples);

        // Allow 5% tolerance for statistical test
        Assert.IsTrue(Math.Abs(mean - 0.5) < 0.05,
            $"Mean {mean} should be close to 0.5 for a=3");
    }

    [TestMethod]
    public void Pareto_DifferentShapeParameters()
    {
        // Higher 'a' means heavier tail
        var rng = np.random.RandomState(42);
        var samples_low_a = rng.pareto(0.5, 10000L);
        var samples_high_a = rng.pareto(5.0, 10000L);

        double mean_low = (double)np.mean(samples_low_a);
        double mean_high = (double)np.mean(samples_high_a);

        // Lower a should produce larger values on average
        Assert.IsTrue(mean_low > mean_high,
            $"Mean with a=0.5 ({mean_low}) should be > mean with a=5 ({mean_high})");
    }

    [TestMethod]
    public void Pareto_ZeroParameter_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.pareto(0.0, 5L));
    }

    [TestMethod]
    public void Pareto_NegativeParameter_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.pareto(-1.0, 5L));
    }

    [TestMethod]
    public void Pareto_ScalarZeroParameter_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.pareto(0.0));
    }

    [TestMethod]
    public void Pareto_ScalarNegativeParameter_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.pareto(-2.0));
    }

    [TestMethod]
    public void Pareto_SmallA_ProducesLargerValues()
    {
        // Small 'a' produces heavier tails (more extreme values)
        var rng = np.random.RandomState(42);
        var samples = rng.pareto(0.5, 1000L);
        double max_val = (double)np.max(samples);

        // With a=0.5, should see some large values
        Assert.IsTrue(max_val > 10,
            $"With a=0.5, max value {max_val} should be > 10");
    }

    [TestMethod]
    public void Pareto_LargeA_ProducesSmallValues()
    {
        // Large 'a' concentrates values near zero
        var rng = np.random.RandomState(42);
        var samples = rng.pareto(10.0, 1000L);
        double max_val = (double)np.max(samples);

        // With a=10, values should be relatively small
        Assert.IsTrue(max_val < 2,
            $"With a=10, max value {max_val} should be < 2");
    }

    [TestMethod]
    public void Pareto_Reproducibility()
    {
        var rng1 = np.random.RandomState(42);
        var result1 = rng1.pareto(2.0, 5L);

        var rng2 = np.random.RandomState(42);
        var result2 = rng2.pareto(2.0, 5L);

        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual((double)result1.GetAtIndex(i), (double)result2.GetAtIndex(i),
                $"Values at index {i} should be identical with same seed");
        }
    }
}
