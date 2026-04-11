using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Random number generator alignment bugs with NumPy.
    ///
    ///     CRITICAL FINDING: NumSharp's Randomizer uses .NET's Subtractive Generator
    ///     (Knuth's algorithm), NOT NumPy's Mersenne Twister (MT19937).
    ///     This means same seed produces completely different sequences.
    ///
    ///     NumPy 2.4.2 expected values generated with:
    ///     <code>
    ///     import numpy as np
    ///     np.random.seed(42)
    ///     print(np.random.rand())  # etc.
    ///     </code>
    /// </summary>
    [OpenBugs]
    [NotInParallel]
    public class OpenBugsRandom : TestClass
    {
        // ===== CRITICAL: RNG Algorithm Mismatch =====
        // NumPy uses Mersenne Twister (MT19937)
        // NumSharp uses .NET Subtractive Generator (Knuth)
        // These tests document the expected NumPy values that NumSharp should produce

        /// <summary>
        ///     BUG: rand() produces different values than NumPy with same seed.
        ///
        ///     NumPy seed=42:    0.3745401188473625
        ///     NumSharp seed=42: 0.668106465911542 (WRONG - different algorithm)
        /// </summary>
        [Test]
        public void Rand_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.rand(0L);

            // NumPy expected value
            const double expected = 0.3745401188473625;
            var actual = result.GetDouble(0);

            actual.Should().BeApproximately(expected, 1e-10,
                $"rand() with seed=42 should match NumPy. NumSharp uses different RNG algorithm.");
        }

        /// <summary>
        ///     BUG: rand(5) produces different sequence than NumPy.
        ///
        ///     NumPy seed=42: [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864]
        /// </summary>
        [Test]
        public void Rand5_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.rand(5L);

            // NumPy expected values
            var expected = new double[] {
                0.3745401188473625,
                0.9507143064099162,
                0.7319939418114051,
                0.5986584841970366,
                0.15601864044243652
            };

            for (int i = 0; i < 5; i++)
            {
                var actual = result.GetDouble(i);
                actual.Should().BeApproximately(expected[i], 1e-10,
                    $"rand(5)[{i}] should match NumPy");
            }
        }

        /// <summary>
        ///     BUG: randn() produces different values than NumPy with same seed.
        ///
        ///     NumPy seed=42:    0.4967141530112327
        ///     NumSharp seed=42: Different value (wrong RNG + Box-Muller may differ)
        /// </summary>
        [Test]
        public void Randn_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.randn(0L);

            // NumPy expected value
            const double expected = 0.4967141530112327;
            var actual = result.GetDouble(0);

            actual.Should().BeApproximately(expected, 1e-10,
                "randn() with seed=42 should match NumPy");
        }

        /// <summary>
        ///     BUG: randn(5) produces different sequence than NumPy.
        ///
        ///     NumPy seed=42: [0.4967141530112327, -0.13826430117118466, 0.6476885381006925,
        ///                    1.5230298564080254, -0.23415337472333597]
        /// </summary>
        [Test]
        public void Randn5_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.randn(5L);

            var expected = new double[] {
                0.4967141530112327,
                -0.13826430117118466,
                0.6476885381006925,
                1.5230298564080254,
                -0.23415337472333597
            };

            for (int i = 0; i < 5; i++)
            {
                var actual = result.GetDouble(i);
                actual.Should().BeApproximately(expected[i], 1e-10,
                    $"randn(5)[{i}] should match NumPy");
            }
        }

        /// <summary>
        ///     BUG: randint produces different values than NumPy with same seed.
        ///
        ///     NumPy seed=42, randint(0,10): 6
        /// </summary>
        [Test]
        public void Randint_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.randint(0, 10);

            const int expected = 6;
            var actual = (int)result;

            actual.Should().Be(expected,
                "randint(0,10) with seed=42 should match NumPy");
        }

        /// <summary>
        ///     BUG: randint(0,10,5) produces different sequence than NumPy.
        ///
        ///     NumPy seed=42: [6, 3, 7, 4, 6]
        /// </summary>
        [Test]
        public void Randint5_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.randint(0, 10, new Shape(5));

            var expected = new int[] { 6, 3, 7, 4, 6 };

            for (int i = 0; i < 5; i++)
            {
                var actual = result.GetInt32(i);
                actual.Should().Be(expected[i],
                    $"randint(0,10,5)[{i}] should match NumPy");
            }
        }

        /// <summary>
        ///     BUG: normal(0,1) produces different values than NumPy.
        ///
        ///     NumPy seed=42: 0.4967141530112327
        /// </summary>
        [Test]
        public void Normal_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.normal(0, 1);

            const double expected = 0.4967141530112327;
            var actual = result.GetDouble(0);

            actual.Should().BeApproximately(expected, 1e-10,
                "normal(0,1) with seed=42 should match NumPy");
        }

        /// <summary>
        ///     BUG: uniform(0,1) should match rand() and NumPy.
        ///
        ///     NumPy seed=42: 0.3745401188473625
        /// </summary>
        [Test]
        public void Uniform_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.uniform(0.0, 1.0, 1);

            const double expected = 0.3745401188473625;
            var actual = result.GetDouble(0);

            actual.Should().BeApproximately(expected, 1e-10,
                "uniform(0,1) with seed=42 should match NumPy");
        }

        /// <summary>
        ///     BUG: choice(10) produces different value than NumPy.
        ///
        ///     NumPy seed=42: 6
        /// </summary>
        [Test]
        public void Choice_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.choice(10);

            const int expected = 6;
            var actual = result.GetInt32(0);

            actual.Should().Be(expected,
                "choice(10) with seed=42 should match NumPy");
        }

        /// <summary>
        ///     BUG: permutation(5) produces different sequence than NumPy.
        ///
        ///     NumPy seed=42: [1, 4, 2, 0, 3]
        /// </summary>
        [Test]
        public void Permutation_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.permutation(5);

            var expected = new int[] { 1, 4, 2, 0, 3 };

            for (int i = 0; i < 5; i++)
            {
                var actual = result.GetInt32(i);
                actual.Should().Be(expected[i],
                    $"permutation(5)[{i}] should match NumPy");
            }
        }

        // ===== Distribution-specific tests =====
        // These test that distributions produce NumPy-compatible values
        // (depends on fixing the base RNG first)

        /// <summary>
        ///     BUG: exponential(1) produces different value than NumPy.
        ///
        ///     NumPy seed=42: 0.4692680899768591
        /// </summary>
        [Test]
        public void Exponential_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.exponential(1);

            const double expected = 0.4692680899768591;
            var actual = result.GetDouble(0);

            actual.Should().BeApproximately(expected, 1e-10,
                "exponential(1) with seed=42 should match NumPy");
        }

        /// <summary>
        ///     BUG: poisson(5) produces different value than NumPy.
        ///
        ///     NumPy seed=42: 5
        /// </summary>
        [Test]
        public void Poisson_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.poisson(5.0, 1);

            const long expected = 5;
            var actual = result.GetInt64(0);

            actual.Should().Be(expected,
                "poisson(5) with seed=42 should match NumPy");
        }

        /// <summary>
        ///     BUG: binomial(10,0.5) produces different value than NumPy.
        ///
        ///     NumPy seed=42: 4
        /// </summary>
        [Test]
        public void Binomial_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.binomial(10, 0.5, 1);

            const long expected = 4;
            var actual = result.GetInt64(0);

            actual.Should().Be(expected,
                "binomial(10,0.5) with seed=42 should match NumPy");
        }

        /// <summary>
        ///     BUG: beta(0.5,0.5) produces different value than NumPy.
        ///
        ///     NumPy seed=42: 0.5992069666276891
        /// </summary>
        [Test]
        public void Beta_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.beta(0.5, 0.5);

            const double expected = 0.5992069666276891;
            var actual = result.GetDouble(0);

            actual.Should().BeApproximately(expected, 1e-10,
                "beta(0.5,0.5) with seed=42 should match NumPy");
        }

        /// <summary>
        ///     BUG: gamma(2,1) produces different value than NumPy.
        ///
        ///     NumPy seed=42: 2.3936793898692366
        /// </summary>
        [Test]
        public void Gamma_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var result = np.random.gamma(2, 1);

            const double expected = 2.3936793898692366;
            var actual = result.GetDouble(0);

            actual.Should().BeApproximately(expected, 1e-10,
                "gamma(2,1) with seed=42 should match NumPy");
        }
    }
}
