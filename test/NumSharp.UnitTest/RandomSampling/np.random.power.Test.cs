using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling;

/// <summary>
/// Tests for np.random.power following NumPy 2.4.2 behavior.
/// Power distribution with PDF: P(x; a) = a * x^(a-1), 0 <= x <= 1, a > 0
/// </summary>
[NotInParallel]
    public class RandomPowerTests : TestClass
{
    [Test]
    public void Power_ScalarCall_ReturnsDouble()
    {
        np.random.seed(42);
        double result = (double)np.random.power(5.0);
        Assert.IsTrue(result >= 0 && result <= 1, $"Result {result} should be in [0, 1]");
    }

    [Test]
    public void Power_ArraySize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.power(5.0, 5L);
        result.Should().BeShaped(5);
        Assert.AreEqual(typeof(double), result.dtype);
    }

    [Test]
    public void Power_MultiDimensionalSize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.power(2.0, new Shape(2, 3));
        result.Should().BeShaped(2, 3);
    }

    [Test]
    public void Power_ShapeSize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.power(3.0, new Shape(3, 4));
        result.Should().BeShaped(3, 4);
    }

    [Test]
    public void Power_AllValuesInRange()
    {
        // All values must be in [0, 1]
        np.random.seed(12345);
        var samples = np.random.power(5.0, 10000L);

        double min = (double)np.min(samples);
        double max = (double)np.max(samples);

        Assert.IsTrue(min >= 0, $"Min {min} should be >= 0");
        Assert.IsTrue(max <= 1, $"Max {max} should be <= 1");
    }

    [Test]
    public void Power_HigherA_SkewsTowardOne()
    {
        // Higher 'a' skews distribution toward 1
        np.random.seed(42);
        var samples_low = np.random.power(0.5, 10000L);
        var samples_high = np.random.power(5.0, 10000L);

        double mean_low = (double)np.mean(samples_low);
        double mean_high = (double)np.mean(samples_high);

        Assert.IsTrue(mean_high > mean_low,
            $"Mean with a=5 ({mean_high}) should be > mean with a=0.5 ({mean_low})");
    }

    [Test]
    public void Power_A1_IsUniform()
    {
        // When a=1, PDF = 1 for 0<=x<=1, which is uniform
        // Mean of uniform(0,1) = 0.5
        np.random.seed(12345);
        var samples = np.random.power(1.0, 100000L);
        double mean = (double)np.mean(samples);

        Assert.IsTrue(Math.Abs(mean - 0.5) < 0.01,
            $"Mean {mean} should be close to 0.5 for a=1 (uniform)");
    }

    [Test]
    public void Power_MeanConvergesToExpected()
    {
        // For power distribution, mean = a / (a + 1)
        // With a=5, expected mean = 5/6 ≈ 0.833
        double a = 5.0;
        double expectedMean = a / (a + 1);

        np.random.seed(12345);
        var samples = np.random.power(a, 100000L);
        double mean = (double)np.mean(samples);

        Assert.IsTrue(Math.Abs(mean - expectedMean) < 0.01,
            $"Mean {mean} should be close to {expectedMean}");
    }

    [Test]
    public void Power_ZeroParameter_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.power(0.0, 5L));
    }

    [Test]
    public void Power_NegativeParameter_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.power(-1.0, 5L));
    }

    [Test]
    public void Power_ScalarZeroParameter_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.power(0.0));
    }

    [Test]
    public void Power_ScalarNegativeParameter_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.power(-2.0));
    }

    [Test]
    public void Power_Reproducibility()
    {
        np.random.seed(42);
        var result1 = np.random.power(5.0, 5L);

        np.random.seed(42);
        var result2 = np.random.power(5.0, 5L);

        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual((double)result1.GetAtIndex(i), (double)result2.GetAtIndex(i),
                $"Values at index {i} should be identical with same seed");
        }
    }

    [Test]
    public void Power_SmallA_SkewsTowardZero()
    {
        // Small 'a' (< 1) skews distribution toward 0
        np.random.seed(42);
        var samples = np.random.power(0.2, 10000L);
        double mean = (double)np.mean(samples);

        // For a=0.2, mean = 0.2/1.2 ≈ 0.167
        Assert.IsTrue(mean < 0.25,
            $"Mean {mean} should be < 0.25 for a=0.2");
    }
}
