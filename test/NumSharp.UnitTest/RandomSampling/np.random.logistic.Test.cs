using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling;

/// <summary>
/// Tests for np.random.logistic following NumPy 2.4.2 behavior.
/// Mean = loc, Variance = scale^2 * pi^2 / 3
/// </summary>
[NotInParallel]
    public class RandomLogisticTests : TestClass
{
    [Test]
    public void Logistic_DefaultParameters_ReturnsScalar()
    {
        np.random.seed(42);
        var result = np.random.logistic();
        Assert.AreEqual(1, result.size);
        Assert.AreEqual(typeof(double), result.dtype);
    }

    [Test]
    public void Logistic_ArraySize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.logistic(0, 1, 5);
        result.Should().BeShaped(5);
        Assert.AreEqual(typeof(double), result.dtype);
    }

    [Test]
    public void Logistic_MultiDimensionalSize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.logistic(0, 1, new Shape(2, 3));
        result.Should().BeShaped(2, 3);
    }

    [Test]
    public void Logistic_ShapeSize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.logistic(0, 1, new Shape(3, 4));
        result.Should().BeShaped(3, 4);
    }

    [Test]
    public void Logistic_MeanConvergesToLoc()
    {
        // Mean of logistic distribution = loc
        np.random.seed(12345);
        var samples = np.random.logistic(0, 1, 100000);
        double mean = (double)np.mean(samples);

        // Allow tolerance for statistical test
        Assert.IsTrue(Math.Abs(mean) < 0.05,
            $"Mean {mean} should be close to 0 (loc=0)");
    }

    [Test]
    public void Logistic_StdConvergesToExpected()
    {
        // Standard deviation = scale * pi / sqrt(3) ≈ 1.814 for scale=1
        double expectedStd = Math.PI / Math.Sqrt(3);

        np.random.seed(12345);
        var samples = np.random.logistic(0, 1, 100000);
        double std = (double)np.std(samples);

        // Allow 5% tolerance
        Assert.IsTrue(Math.Abs(std - expectedStd) < 0.1,
            $"Std {std} should be close to {expectedStd}");
    }

    [Test]
    public void Logistic_WithLocAndScale()
    {
        // Mean = loc = 5
        np.random.seed(12345);
        var samples = np.random.logistic(5.0, 2.0, 100000);
        double mean = (double)np.mean(samples);

        Assert.IsTrue(Math.Abs(mean - 5.0) < 0.1,
            $"Mean {mean} should be close to 5 (loc=5)");
    }

    [Test]
    public void Logistic_ScaleZero_ReturnsLoc()
    {
        // When scale=0, all values should equal loc
        np.random.seed(42);
        var samples = np.random.logistic(5.0, 0.0, 10);

        for (int i = 0; i < samples.size; i++)
        {
            Assert.AreEqual(5.0, (double)samples.GetAtIndex(i),
                $"With scale=0, all values should equal loc=5");
        }
    }

    [Test]
    public void Logistic_NegativeScale_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.logistic(0, -1, 5));
    }

    [Test]
    public void Logistic_DefaultScalar_NegativeScale_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.logistic(0, -1));
    }

    [Test]
    public void Logistic_Reproducibility()
    {
        np.random.seed(42);
        var result1 = np.random.logistic(0, 1, 5);

        np.random.seed(42);
        var result2 = np.random.logistic(0, 1, 5);

        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual((double)result1.GetAtIndex(i), (double)result2.GetAtIndex(i),
                $"Values at index {i} should be identical with same seed");
        }
    }

    [Test]
    public void Logistic_CanProduceNegativeValues()
    {
        np.random.seed(42);
        var samples = np.random.logistic(0, 1, 1000);

        bool hasNegative = false;
        bool hasPositive = false;
        for (int i = 0; i < samples.size; i++)
        {
            double val = (double)samples.GetAtIndex(i);
            if (val < 0) hasNegative = true;
            if (val > 0) hasPositive = true;
        }

        Assert.IsTrue(hasNegative, "Logistic distribution should produce negative values");
        Assert.IsTrue(hasPositive, "Logistic distribution should produce positive values");
    }

    [Test]
    public void Logistic_LargerScaleProducesLargerVariance()
    {
        np.random.seed(42);
        var samples1 = np.random.logistic(0, 1, 10000);
        var samples2 = np.random.logistic(0, 3, 10000);

        double std1 = (double)np.std(samples1);
        double std2 = (double)np.std(samples2);

        Assert.IsTrue(std2 > std1 * 2,
            $"Std with scale=3 ({std2}) should be ~3x std with scale=1 ({std1})");
    }
}
