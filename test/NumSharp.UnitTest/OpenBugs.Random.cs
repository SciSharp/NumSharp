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
    [TestClass]
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
        [TestMethod]
        [OpenBugs]
        public void Rand_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.rand(0L);

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
        [TestMethod]
        public void Rand5_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.rand(5L);

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
        [TestMethod]
        [OpenBugs]
        public void Randn_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.randn(0L);

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
        [TestMethod]
        public void Randn5_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.randn(5L);

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
        [TestMethod]
        public void Randint_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.randint(0, 10);

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
        [TestMethod]
        public void Randint5_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.randint(0, 10, new Shape(5));

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
        [TestMethod]
        public void Normal_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.normal(0, 1);

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
        [TestMethod]
        public void Uniform_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.uniform(0.0, 1.0, 1);

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
        [TestMethod]
        public void Choice_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.choice(10);

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
        [TestMethod]
        [OpenBugs]
        public void Permutation_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.permutation(5);

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
        [TestMethod]
        public void Exponential_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.exponential(1);

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
        [TestMethod]
        public void Poisson_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.poisson(5.0, 1);

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
        [TestMethod]
        public void Binomial_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.binomial(10, 0.5, 1);

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
        [TestMethod]
        public void Beta_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.beta(0.5, 0.5);

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
        [TestMethod]
        public void Gamma_Seed42_ShouldMatchNumPy()
        {
            var rng = np.random.RandomState(42);
            var result = rng.gamma(2, 1);

            const double expected = 2.3936793898692366;
            var actual = result.GetDouble(0);

            actual.Should().BeApproximately(expected, 1e-10,
                "gamma(2,1) with seed=42 should match NumPy");
        }

        // ===== random_parity tier carve-outs (2026-08-14) ==================================
        // Six samplers whose whole STREAM diverges from NumPy 2.4.2 (different algorithm /
        // draw order, not rounding) — found by the random_parity differential tiers and
        // CARVED from the green corpus (gen_oracle.gen_random_parity documents each). The
        // underlying MT19937 stream and most transforms are byte-identical (uniform/randint/
        // normal/standard_gamma/... all match), so these are per-sampler composition bugs.
        // Expected values are real NumPy 2.4.2 output. Remove a pin AND re-add the spec to
        // gen_random_parity when its sampler is fixed.

        /// <summary>gamma(shape&lt;1) via the two-arg API gross-diverges while NumSharp's own
        /// standard_gamma(shape&lt;1) and gamma(shape&gt;=1, any scale) match byte-for-byte —
        /// the two-arg route handles shape&lt;1 with a different sampler than standard_gamma.</summary>
        [TestMethod]
        [OpenBugs]
        public void RandomParity_GammaShapeBelowOne_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var r = np.random.gamma(0.5, 1.0, new Shape(3));
            r.GetDouble(0).Should().Be(0.14028030062619642, "np.random.gamma(0.5, 1) first draw, seed 42");
        }

        /// <summary>binomial counts drift on BOTH internal algorithms (inversion at small n*p,
        /// BTPE at large) — accept/reject boundaries land differently from NumPy's.</summary>
        [TestMethod]
        [OpenBugs]
        public void RandomParity_Binomial_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var r = np.random.binomial(10, 0.35, new Shape(7));
            var expected = new long[] { 3, 6, 4, 4, 2, 2, 1 };
            for (int i = 0; i < expected.Length; i++)
                Convert.ToInt64(r.GetAtIndex(i)).Should().Be(expected[i],
                    $"np.random.binomial(10, 0.35)[{i}], seed 42");
        }

        /// <summary>negative_binomial = poisson(gamma(n, (1-p)/p)): occasional count flips
        /// downstream of ~ULP lambda differences, incl. a gross 21-vs-29 at index 6.</summary>
        [TestMethod]
        [OpenBugs]
        public void RandomParity_NegativeBinomial_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var r = np.random.negative_binomial(5.0, 0.4, new Shape(7));
            var expected = new long[] { 7, 5, 1, 5, 5, 6, 29 };
            for (int i = 0; i < expected.Length; i++)
                Convert.ToInt64(r.GetAtIndex(i)).Should().Be(expected[i],
                    $"np.random.negative_binomial(5, 0.4)[{i}], seed 42");
        }

        /// <summary>f() is not NumPy's (chisquare(dfnum)/dfnum)/(chisquare(dfden)/dfden) composition.</summary>
        [TestMethod]
        [OpenBugs]
        public void RandomParity_F_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var r = np.random.f(5.0, 7.0, new Shape(2));
            r.GetDouble(0).Should().Be(1.426878677609774, "np.random.f(5, 7) first draw, seed 42");
        }

        /// <summary>pareto() is not NumPy's expm1(standard_exponential()/a).</summary>
        [TestMethod]
        [OpenBugs]
        public void RandomParity_Pareto_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var r = np.random.pareto(3.0, new Shape(2));
            r.GetDouble(0).Should().Be(0.16932036645405568, "np.random.pareto(3) first draw, seed 42");
        }

        /// <summary>standard_cauchy() is not NumPy's gauss/gauss ratio.</summary>
        [TestMethod]
        [OpenBugs]
        public void RandomParity_StandardCauchy_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var r = np.random.standard_cauchy(new Shape(2));
            r.GetDouble(0).Should().Be(-3.5924974762375737, "np.random.standard_cauchy() first draw, seed 42");
        }

        /// <summary>multinomial's per-category binomial loop consumes the stream differently
        /// (dtype matches; the COUNTS diverge).</summary>
        [TestMethod]
        [OpenBugs]
        public void RandomParity_Multinomial_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var r = np.random.multinomial(20, new[] { 0.2, 0.3, 0.5 });
            Convert.ToInt64(r.GetAtIndex(0)).Should().Be(3, "np.random.multinomial(20, [.2,.3,.5]) counts, seed 42");
            Convert.ToInt64(r.GetAtIndex(1)).Should().Be(10);
            Convert.ToInt64(r.GetAtIndex(2)).Should().Be(7);
        }

        /// <summary>multivariate_normal uses a different factorization/transform (sign flips + values).</summary>
        [TestMethod]
        [OpenBugs]
        public void RandomParity_MultivariateNormal_Seed42_ShouldMatchNumPy()
        {
            np.random.seed(42);
            var cov = new double[2, 2] { { 1.0, 0.5 }, { 0.5, 2.0 } };
            var r = np.random.multivariate_normal(new[] { 0.0, 1.0 }, cov, new[] { 2 });
            r.GetDouble(0, 0).Should().Be(0.16865044563071147, "mvn([0,1], [[1,.5],[.5,2]]) first row, seed 42");
            r.GetDouble(0, 1).Should().Be(1.728877966541595);
        }
    }
}
