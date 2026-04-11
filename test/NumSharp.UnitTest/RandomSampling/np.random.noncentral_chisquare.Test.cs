using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling;

/// <summary>
/// Tests for np.random.noncentral_chisquare following NumPy 2.4.2 behavior.
/// Mean = df + nonc
/// </summary>
[NotInParallel]
    public class RandomNoncentralChisquareTests : TestClass
{
    [Test]
    public void NoncentralChisquare_ScalarCall_ReturnsDouble()
    {
        np.random.seed(42);
        double result = (double)np.random.noncentral_chisquare(3, 2);
        Assert.IsTrue(result >= 0, $"Result {result} should be non-negative");
    }

    [Test]
    public void NoncentralChisquare_ArraySize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.noncentral_chisquare(3, 2, 5L);
        result.Should().BeShaped(5);
        Assert.AreEqual(typeof(double), result.dtype);
    }

    [Test]
    public void NoncentralChisquare_MultiDimensionalSize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.noncentral_chisquare(3, 2, new Shape(2, 3));
        result.Should().BeShaped(2, 3);
    }

    [Test]
    public void NoncentralChisquare_ShapeSize_ReturnsCorrectShape()
    {
        np.random.seed(42);
        var result = np.random.noncentral_chisquare(3, 2, new Shape(3, 4));
        result.Should().BeShaped(3, 4);
    }

    [Test]
    public void NoncentralChisquare_AllValuesNonNegative()
    {
        np.random.seed(12345);
        var samples = np.random.noncentral_chisquare(3, 2, 10000L);

        for (int i = 0; i < samples.size; i++)
        {
            Assert.IsTrue((double)samples.GetAtIndex(i) >= 0,
                $"Sample at index {i} should be non-negative");
        }
    }

    [Test]
    public void NoncentralChisquare_MeanConvergesToExpected()
    {
        // Mean = df + nonc = 3 + 2 = 5
        np.random.seed(12345);
        var samples = np.random.noncentral_chisquare(3, 2, 100000L);
        double mean = (double)np.mean(samples);

        Assert.IsTrue(Math.Abs(mean - 5.0) < 0.1,
            $"Mean {mean} should be close to 5 (df + nonc)");
    }

    [Test]
    public void NoncentralChisquare_ZeroNonc_IsCentralChisquare()
    {
        // When nonc=0, it's central chi-square with mean = df
        np.random.seed(12345);
        var samples = np.random.noncentral_chisquare(3, 0, 100000L);
        double mean = (double)np.mean(samples);

        Assert.IsTrue(Math.Abs(mean - 3.0) < 0.1,
            $"Mean {mean} should be close to 3 (df) when nonc=0");
    }

    [Test]
    public void NoncentralChisquare_SmallDf()
    {
        // df <= 1 uses the Poisson method
        np.random.seed(42);
        var samples = np.random.noncentral_chisquare(0.5, 2, 10000L);

        // All should be non-negative
        for (int i = 0; i < samples.size; i++)
        {
            Assert.IsTrue((double)samples.GetAtIndex(i) >= 0,
                $"Sample at index {i} should be non-negative");
        }

        // Mean should be close to df + nonc = 0.5 + 2 = 2.5
        double mean = (double)np.mean(samples);
        Assert.IsTrue(Math.Abs(mean - 2.5) < 0.2,
            $"Mean {mean} should be close to 2.5");
    }

    [Test]
    public void NoncentralChisquare_LargeDf()
    {
        // Large df should work correctly
        np.random.seed(42);
        var samples = np.random.noncentral_chisquare(10, 5, 10000L);
        double mean = (double)np.mean(samples);

        // Mean = df + nonc = 10 + 5 = 15
        Assert.IsTrue(Math.Abs(mean - 15.0) < 0.5,
            $"Mean {mean} should be close to 15");
    }

    [Test]
    public void NoncentralChisquare_LargeNonc()
    {
        // Large non-centrality
        np.random.seed(42);
        var samples = np.random.noncentral_chisquare(3, 20, 10000L);
        double mean = (double)np.mean(samples);

        // Mean = df + nonc = 3 + 20 = 23
        Assert.IsTrue(Math.Abs(mean - 23.0) < 0.5,
            $"Mean {mean} should be close to 23");
    }

    [Test]
    public void NoncentralChisquare_ZeroDf_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.noncentral_chisquare(0, 2, 5L));
    }

    [Test]
    public void NoncentralChisquare_NegativeDf_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.noncentral_chisquare(-1, 2, 5L));
    }

    [Test]
    public void NoncentralChisquare_NegativeNonc_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.noncentral_chisquare(3, -1, 5L));
    }

    [Test]
    public void NoncentralChisquare_ScalarZeroDf_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.noncentral_chisquare(0, 2));
    }

    [Test]
    public void NoncentralChisquare_ScalarNegativeNonc_ThrowsArgumentException()
    {
        Assert.ThrowsException<ArgumentException>(() => np.random.noncentral_chisquare(3, -1));
    }

    [Test]
    public void NoncentralChisquare_Reproducibility()
    {
        np.random.seed(42);
        var result1 = np.random.noncentral_chisquare(3, 2, 5L);

        np.random.seed(42);
        var result2 = np.random.noncentral_chisquare(3, 2, 5L);

        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual((double)result1.GetAtIndex(i), (double)result2.GetAtIndex(i),
                $"Values at index {i} should be identical with same seed");
        }
    }
}
