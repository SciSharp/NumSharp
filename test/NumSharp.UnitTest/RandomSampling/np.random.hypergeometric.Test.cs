using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling;

/// <summary>
/// Tests for np.random.hypergeometric following NumPy 2.4.2 behavior.
/// Samples from hypergeometric distribution without replacement.
/// Mean = nsample * ngood / (ngood + nbad)
/// </summary>
[NotInParallel]
    public class RandomHypergeometricTests : TestClass
{
    [Test]
    public void Hypergeometric_ScalarCall_Returns0dArray()
    {
        np.random.seed(42);
        var result = np.random.hypergeometric(15, 15, 10);
        Assert.AreEqual(0, result.ndim, "Scalar call should return 0-d array");
        long value = result.GetInt64();
        Assert.IsTrue(value >= 0 && value <= 10, $"Result {value} should be in [0, 10]");
    }

    [Test]
    public void Hypergeometric_ArraySize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.hypergeometric(15, 15, 10, 5L);
        result.Should().BeShaped(5);
        Assert.AreEqual(typeof(long), result.dtype);
    }

    [Test]
    public void Hypergeometric_MultiDimensionalSize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.hypergeometric(15, 15, 10, new Shape(2, 3));
        result.Should().BeShaped(2, 3);
    }

    [Test]
    public void Hypergeometric_ShapeSize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.hypergeometric(15, 15, 10, new Shape(3, 4));
        result.Should().BeShaped(3, 4);
    }

    [Test]
    public void Hypergeometric_AllValuesInRange()
    {
        // Result must be in [max(0, nsample-nbad), min(nsample, ngood)]
        // For (15, 15, 10): [0, 10]
        np.random.seed(12345);
        var samples = np.random.hypergeometric(15, 15, 10, 10000L);

        for (int i = 0; i < samples.size; i++)
        {
            long val = (long)samples.GetAtIndex(i);
            Assert.IsTrue(val >= 0 && val <= 10,
                $"Sample {val} at index {i} should be in [0, 10]");
        }
    }

    [Test]
    public void Hypergeometric_MeanConvergesToExpected()
    {
        // Mean = nsample * ngood / (ngood + nbad)
        // For (15, 15, 10): mean = 10 * 15/30 = 5
        np.random.seed(12345);
        var samples = np.random.hypergeometric(15, 15, 10, 100000L);

        // Convert to double for mean calculation
        double sum = 0;
        for (int i = 0; i < samples.size; i++)
        {
            sum += (long)samples.GetAtIndex(i);
        }
        double mean = sum / samples.size;

        Assert.IsTrue(Math.Abs(mean - 5.0) < 0.05,
            $"Mean {mean} should be close to 5");
    }

    [Test]
    public void Hypergeometric_AllGood_ReturnsNsample()
    {
        // When nbad=0, result is always nsample (all are good)
        np.random.seed(42);
        var samples = np.random.hypergeometric(20, 0, 10, 10L);

        for (int i = 0; i < samples.size; i++)
        {
            Assert.AreEqual(10L, (long)samples.GetAtIndex(i),
                "When nbad=0, result should always be nsample");
        }
    }

    [Test]
    public void Hypergeometric_AllBad_ReturnsZero()
    {
        // When ngood=0, result is always 0 (none are good)
        np.random.seed(42);
        var samples = np.random.hypergeometric(0, 20, 10, 10L);

        for (int i = 0; i < samples.size; i++)
        {
            Assert.AreEqual(0L, (long)samples.GetAtIndex(i),
                "When ngood=0, result should always be 0");
        }
    }

    [Test]
    public void Hypergeometric_TakeAll_ReturnsNgood()
    {
        // When nsample = ngood + nbad, result is always ngood
        np.random.seed(42);
        var samples = np.random.hypergeometric(5, 5, 10, 10L);

        for (int i = 0; i < samples.size; i++)
        {
            Assert.AreEqual(5L, (long)samples.GetAtIndex(i),
                "When taking all, result should be ngood");
        }
    }

    [Test]
    public void Hypergeometric_MostlyGood()
    {
        // (100, 2, 10) - should get mostly ~10 good
        np.random.seed(42);
        var samples = np.random.hypergeometric(100, 2, 10, 1000L);

        double sum = 0;
        for (int i = 0; i < samples.size; i++)
        {
            sum += (long)samples.GetAtIndex(i);
        }
        double mean = sum / samples.size;

        // Mean should be close to 10 * 100/102 ≈ 9.8
        Assert.IsTrue(mean > 9.5, $"Mean {mean} should be close to 10 for mostly good");
    }

    [Test]
    public void Hypergeometric_MostlyBad()
    {
        // (2, 100, 10) - should get mostly ~0 good
        np.random.seed(42);
        var samples = np.random.hypergeometric(2, 100, 10, 1000L);

        double sum = 0;
        for (int i = 0; i < samples.size; i++)
        {
            sum += (long)samples.GetAtIndex(i);
        }
        double mean = sum / samples.size;

        // Mean should be close to 10 * 2/102 ≈ 0.2
        Assert.IsTrue(mean < 0.5, $"Mean {mean} should be close to 0 for mostly bad");
    }

    [Test]
    public void Hypergeometric_NegativeNgood_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.hypergeometric(-1, 15, 10, 5L));
    }

    [Test]
    public void Hypergeometric_NegativeNbad_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.hypergeometric(15, -1, 10, 5L));
    }

    [Test]
    public void Hypergeometric_ZeroNsample_ThrowsArgumentException()
    {
        // NumPy requires nsample >= 1
        Assert.ThrowsException<ArgumentException>(() => np.random.hypergeometric(15, 15, 0, 5L));
    }

    [Test]
    public void Hypergeometric_NegativeNsample_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.hypergeometric(15, 15, -1, 5L));
    }

    [Test]
    public void Hypergeometric_NsampleTooLarge_ThrowsArgumentException()
    {
        // nsample > ngood + nbad should throw
        Assert.ThrowsException<ArgumentException>(() => np.random.hypergeometric(15, 15, 40, 5L));
    }

    [Test]
    public void Hypergeometric_Reproducibility()
    {
        np.random.seed(42);
        var result1 = np.random.hypergeometric(15, 15, 10, 5L);

        np.random.seed(42);
        var result2 = np.random.hypergeometric(15, 15, 10, 5L);

        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual((long)result1.GetAtIndex(i), (long)result2.GetAtIndex(i),
                $"Values at index {i} should be identical with same seed");
        }
    }

    [Test]
    public void Hypergeometric_NsampleOne()
    {
        // With nsample=1, result is either 0 or 1
        np.random.seed(42);
        var samples = np.random.hypergeometric(10, 10, 1, 100L);

        for (int i = 0; i < samples.size; i++)
        {
            long val = (long)samples.GetAtIndex(i);
            Assert.IsTrue(val == 0 || val == 1,
                $"With nsample=1, result should be 0 or 1, got {val}");
        }
    }
}
