using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Regression pins for the view-<see cref="Shape"/> overrun reached through <c>np.random</c>'s <c>size</c> parameter
    ///     (found by the 2026-10-02 random-commit review): <c>np.random.default_rng(42).integers(0, 100, x.Shape)</c> with
    ///     <c>x = np.arange(20)["::4"]</c> returned heap pointers (<c>8589934592</c>, <c>140710211227288</c>) instead of
    ///     NumPy's <c>[8 77 65 43 43]</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumPy's <c>size</c> is a tuple of dimensions and every sampler allocates <c>np.empty(size)</c> — a fresh C-ordered
    ///     array — then fills it in memory order. NumSharp's <c>size</c> is a <see cref="Shape"/>, which also carries a view's
    ///     strides and offset; several samplers handed it to the allocator as given, so the fill wrote <c>size</c> values in
    ///     a row while the result read them through the view's stride (past the buffer for a stepped view, transposed for an
    ///     F-ordered one).
    ///     </para>
    ///     <para>
    ///     The contract every check below enforces, for every sampler family that allocates from <c>size</c> and every
    ///     hostile layout: a view's Shape gives exactly what a fresh Shape of the same dimensions gives (same dtype, shape and
    ///     values from the same seed, same flags, same engine state afterwards), and the result is C-contiguous at offset 0
    ///     over a buffer of exactly its size — NumPy's <c>np.empty(size)</c>. The oracle replay <c>RandomApiSizeLayouts</c>
    ///     extends the same contract to every public overload against NumPy itself.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class RandomSizeViewShapeTests
    {
        // =====================================================================================================================
        // The reported case and NumPy-verified values
        // =====================================================================================================================

        [TestMethod]
        public void Integers_StridedViewShape_MatchesNumPy()
        {
            // NumPy 2.4.2: x = np.arange(20)[::4]; np.random.default_rng(42).integers(0, 100, x.shape) -> [ 8 77 65 43 43]
            using var x = np.arange(20)["::4"];
            x.Shape.strides.Should().Equal(new long[] { 4 }, "the reproduction needs a stepped view");
            using var r = np.random.default_rng(42).integers(0, 100, x.Shape);
            r.ToArray<long>().Should().Equal(8L, 77L, 65L, 43L, 43L);
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.strides.Should().Equal(new long[] { 1 });
            r.Shape.offset.Should().Be(0);
        }

        [TestMethod]
        public void Random_TransposedViewShape_FillsInCOrderLikeNumPy()
        {
            // NumPy 2.4.2: np.random.default_rng(42).random(np.arange(12).reshape(3, 4).T.shape)
            //   -> [[0.77395605 0.43887844 0.85859792] [0.69736803 0.09417735 0.97562235] ...], C-contiguous.
            // The transpose's Shape is dense (memory-safe) but F-ordered: allocating under it put the C-order stream in
            // column order, so every off-diagonal value was misplaced.
            using var t = np.arange(12).reshape(3, 4).T;
            using var r = np.random.default_rng(42).random(t.Shape);
            r.shape.Should().Equal(new long[] { 4, 3 });
            r.Shape.IsContiguous.Should().BeTrue("NumPy's np.empty(size) is C-ordered");
            r.GetDouble(0, 0).Should().Be(0.7739560485559633);
            r.GetDouble(0, 1).Should().Be(0.4388784397520523);
            r.GetDouble(1, 0).Should().Be(0.6973680290593639);
            using var clean = np.random.default_rng(42).random(new Shape(4, 3));
            np.array_equal(r, clean).Should().BeTrue();
        }

        [TestMethod]
        public void LegacyRandint_StridedViewShape_MatchesCleanShape()
        {
            using var x = np.arange(40)["3::4"];
            using var r = np.random.RandomState(42).randint(0, 100, x.Shape);
            using var clean = np.random.RandomState(42).randint(0, 100, new Shape(10));
            np.array_equal(r, clean).Should().BeTrue();
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.offset.Should().Be(0);
        }

        // =====================================================================================================================
        // Every allocating sampler family × every hostile layout
        // =====================================================================================================================

        /// <summary>
        ///     One sampler family: draws with a fresh receiver (seed 42) for a given size, then draws one more uniform from the
        ///     same receiver — equal next draws mean the call consumed the stream identically.
        /// </summary>
        /// <param name="Name">The call, for messages.</param>
        /// <param name="Run">Draws for a size; returns the result and the next uniform.</param>
        private sealed record Family(string Name, Func<Shape, (NDArray result, double next)> Run);

        /// <summary>A Generator family (PCG64 seeded 42), probing the state with <c>random()</c>.</summary>
        /// <param name="name">The call.</param>
        /// <param name="draw">The draw.</param>
        /// <returns>The family.</returns>
        private static Family G(string name, Func<Generator, Shape, NDArray> draw) => new(name, s =>
        {
            var g = np.random.default_rng(42);
            var r = draw(g, s);
            using var next = g.random();
            return (r, next.GetDouble());
        });

        /// <summary>A legacy RandomState family (MT19937 seeded 42), probing the state with <c>random_sample()</c>.</summary>
        /// <param name="name">The call.</param>
        /// <param name="draw">The draw.</param>
        /// <returns>The family.</returns>
        private static Family L(string name, Func<NumPyRandom, Shape, NDArray> draw) => new(name, s =>
        {
            var rs = np.random.RandomState(42);
            var r = draw(rs, s);
            using var next = rs.random_sample();
            return (r, next.GetDouble());
        });

        /// <summary>A bare bit-generator family, probing the state with one more raw word.</summary>
        /// <param name="name">The call.</param>
        /// <param name="make">Builds the engine.</param>
        /// <returns>The family.</returns>
        private static Family B(string name, Func<BitGenerator> make) => new(name, s =>
        {
            var bg = make();
            var r = bg.random_raw(s);
            using var next = bg.random_raw();
            return (r, (double)next.GetValue<ulong>());
        });

        /// <summary>
        ///     The families: at least one per allocation site that takes <c>size</c> (each Generator sampler group, the bounded
        ///     integers in both strategies and the array-bounds path, the broadcast-parameter path, choice, the multivariate
        ///     outputs, the legacy samplers and <c>random_raw</c> on every engine).
        /// </summary>
        private static readonly Family[] Families =
        {
            G("Generator.random", (g, s) => g.random(s)),
            G("Generator.random(float32)", (g, s) => g.random(s, np.float32)),
            G("Generator.integers", (g, s) => g.integers(0, 100, s)),
            G("Generator.integers(uint8)", (g, s) => g.integers(0, 100, s, np.uint8)),
            G("Generator.integers(bool)", (g, s) => g.integers(0, 2, s, np.@bool)),
            G("Generator.integers(array bounds)", (g, s) => g.integers(np.array(new long[] { 0 }), np.array(new long[] { 100 }), s)),
            G("Generator.integers(one value)", (g, s) => g.integers(5, 6, s)),
            G("Generator.standard_normal", (g, s) => g.standard_normal(s)),
            G("Generator.standard_normal(float32)", (g, s) => g.standard_normal(s, np.float32)),
            G("Generator.normal", (g, s) => g.normal(1.0, 2.0, s)),
            G("Generator.normal(array params)", (g, s) => g.normal(np.array(new[] { 0.5 }), np.array(new[] { 2.0 }), s)),
            G("Generator.standard_exponential", (g, s) => g.standard_exponential(s)),
            G("Generator.standard_exponential(float32, inv)", (g, s) => g.standard_exponential(s, np.float32, "inv")),
            G("Generator.exponential", (g, s) => g.exponential(2.0, s)),
            G("Generator.uniform", (g, s) => g.uniform(-1.0, 3.0, s)),
            G("Generator.standard_gamma", (g, s) => g.standard_gamma(2.5, s)),
            G("Generator.standard_gamma(float32)", (g, s) => g.standard_gamma(2.5, s, np.float32)),
            G("Generator.gamma", (g, s) => g.gamma(2.5, 1.5, s)),
            G("Generator.beta", (g, s) => g.beta(2.0, 3.0, s)),
            G("Generator.weibull", (g, s) => g.weibull(1.5, s)),
            G("Generator.weibull(0)", (g, s) => g.weibull(0.0, s)),
            G("Generator.standard_cauchy", (g, s) => g.standard_cauchy(s)),
            G("Generator.binomial", (g, s) => g.binomial(10, 0.3, s)),
            G("Generator.poisson", (g, s) => g.poisson(3.0, s)),
            G("Generator.hypergeometric", (g, s) => g.hypergeometric(5, 7, 4, s)),
            G("Generator.choice", (g, s) => g.choice(10L, s)),
            G("Generator.choice(no replace)", (g, s) => g.choice(1000L, s, replace: false)),
            G("Generator.multinomial", (g, s) => g.multinomial(10, new[] { 0.2, 0.3, 0.5 }, s)),
            G("Generator.dirichlet", (g, s) => g.dirichlet(new[] { 1.0, 2.0 }, s)),
            L("RandomState.rand", (rs, s) => rs.rand(s)),
            L("RandomState.random_sample", (rs, s) => rs.random_sample(s)),
            L("RandomState.randint", (rs, s) => rs.randint(0, 100, s)),
            L("RandomState.randint(uint16)", (rs, s) => rs.randint(0, 100, s, np.uint16)),
            L("RandomState.randint(array bounds)", (rs, s) => rs.randint(np.array(new long[] { 0 }), np.array(new long[] { 100 }), s)),
            L("RandomState.random_integers", (rs, s) => rs.random_integers(1, 6, s)),
            L("RandomState.standard_normal", (rs, s) => rs.standard_normal(s)),
            L("RandomState.normal", (rs, s) => rs.normal(1.0, 2.0, s)),
            L("RandomState.uniform", (rs, s) => rs.uniform(-1.0, 3.0, s)),
            L("RandomState.exponential", (rs, s) => rs.exponential(2.0, s)),
            L("RandomState.gamma", (rs, s) => rs.gamma(2.5, 1.5, s)),
            L("RandomState.weibull", (rs, s) => rs.weibull(1.5, s)),
            L("RandomState.weibull(0)", (rs, s) => rs.weibull(0.0, s)),
            L("RandomState.poisson", (rs, s) => rs.poisson(3.0, s)),
            L("RandomState.binomial", (rs, s) => rs.binomial(10, 0.3, s)),
            L("RandomState.tomaxint", (rs, s) => rs.tomaxint(s)),
            L("RandomState.choice", (rs, s) => rs.choice(10L, s)),
            L("RandomState.multinomial", (rs, s) => rs.multinomial(10, new[] { 0.2, 0.3, 0.5 }, s)),
            L("RandomState.dirichlet", (rs, s) => rs.dirichlet(new[] { 1.0, 2.0 }, s)),
            B("PCG64.random_raw", () => new PCG64(42)),
            B("PCG64DXSM.random_raw", () => new PCG64DXSM(42)),
            B("MT19937.random_raw", () => new MT19937(42)),
            B("Philox.random_raw", () => new Philox(42)),
            B("SFC64.random_raw", () => new SFC64(42)),
        };

        /// <summary>
        ///     The hostile layouts of <paramref name="dims"/>: each a real view's shape with exactly these dimensions (asserted),
        ///     so the size NumPy would see is unchanged while the strides, offset and buffer size are a view's.
        /// </summary>
        /// <param name="dims">The size's dimensions (all non-negative).</param>
        /// <returns>(layout name, shape) pairs; layouts that need more axes than <paramref name="dims"/> has are skipped.</returns>
        private static IEnumerable<(string name, Shape shape)> HostileLayouts(long[] dims)
        {
            static Shape Of(Func<NDArray> make, long[] expect)
            {
                using var view = make();
                view.shape.Should().Equal(expect, "a hostile layout must keep the requested dimensions");
                return view.Shape;
            }

            int k = dims.Length;
            string Steps(string each) => string.Join(", ", Enumerable.Repeat(each, k));
            if (k == 0)
                yield break;

            // Every axis doubled and stepped by 2 from 0, then from 1 (an offset on every axis).
            var doubled = dims.Select(d => 2 * d).ToArray();
            var doubledPlus = dims.Select(d => 2 * d + 1).ToArray();
            yield return ("stepped", Of(() => np.zeros(new Shape(doubled))[Steps("::2")], dims));
            yield return ("offset stepped", Of(() => np.zeros(new Shape(doubledPlus))[Steps("1::2")], dims));
            yield return ("reversed", Of(() => np.zeros(new Shape(dims))[Steps("::-1")], dims));
            yield return ("broadcast", Of(() => np.broadcast_to(np.zeros(new Shape(1)), new Shape(dims)), dims));
            long size = dims.Aggregate(1L, (a, b) => a * b);
            yield return ("contiguous slice", Of(() => np.zeros(new Shape(size + 3))[$":{size}"].reshape(dims), dims));
            if (k >= 2)
                yield return ("transposed", Of(() => np.zeros(new Shape(Enumerable.Reverse(dims).ToArray())).T, dims));
            if (k >= 3)
            {
                // Swap the first two axes: dense, neither C nor F.
                var swapped = (long[])dims.Clone();
                (swapped[0], swapped[1]) = (swapped[1], swapped[0]);
                var perm = Enumerable.Range(0, k).ToArray();
                (perm[0], perm[1]) = (perm[1], perm[0]);
                yield return ("permuted", Of(() => np.transpose(np.zeros(new Shape(swapped)), perm), dims));
            }
        }

        /// <summary>The sizes every family is drawn at: 1-D, 2-D, 3-D, and an empty extent.</summary>
        private static readonly long[][] Sizes = { new long[] { 6 }, new long[] { 4, 3 }, new long[] { 2, 3, 4 }, new long[] { 3, 0 } };

        [TestMethod]
        public void EverySamplerFamily_ViewShapeSize_MatchesCleanShape()
        {
            var failures = new List<string>();
            foreach (var family in Families)
            foreach (var dims in Sizes)
            {
                var (cleanResult, cleanNext) = family.Run(new Shape(dims));
                using var clean = cleanResult;
                foreach (var (layout, shape) in HostileLayouts(dims))
                {
                    string what = $"{family.Name} size=({string.Join(",", dims)}) as {layout}";
                    NDArray got = null;
                    try
                    {
                        double next;
                        (got, next) = family.Run(shape);
                        if (got.typecode != clean.typecode || !got.shape.SequenceEqual(clean.shape))
                            failures.Add($"{what}: got {got.dtype} {Fmt(got.shape)}, clean {clean.dtype} {Fmt(clean.shape)}");
                        else if (!np.array_equal(got, clean))
                            failures.Add($"{what}: values differ from the clean size's draws");
                        else if (!got.Shape.IsContiguous || got.Shape.offset != 0 || got.Shape.BufferSize != got.size)
                            failures.Add($"{what}: not a C-contiguous array over its own buffer (strides {Fmt(got.Shape.strides)}, " +
                                         $"offset {got.Shape.offset}, buffer {got.Shape.BufferSize})");
                        else if (Layout(got) != Layout(clean))
                            // Ownership is compared with the clean size's result, not asserted: choice without replacement
                            // returns a reshape VIEW of its index array whatever the size (NumPy assigns idx.shape in place
                            // and owns it) — a separate, layout-independent flags divergence.
                            failures.Add($"{what}: layout {Layout(got)} differs from the clean size's {Layout(clean)}");
                        else if (BitConverter.DoubleToInt64Bits(next) != BitConverter.DoubleToInt64Bits(cleanNext))
                            failures.Add($"{what}: the engine state after the call differs (next draw {next} vs {cleanNext})");
                    }
                    catch (Exception e)
                    {
                        failures.Add($"{what}: threw {e.GetType().Name}: {e.Message}");
                    }
                    finally
                    {
                        got?.Dispose();
                    }
                }
            }

            failures.Should().BeEmpty();
        }

        /// <summary>Formats dimensions or strides as a Python tuple for messages.</summary>
        /// <param name="v">The values.</param>
        /// <returns>The text.</returns>
        private static string Fmt(long[] v) => "(" + string.Join(", ", v) + (v.Length == 1 ? ",)" : ")");

        /// <summary>The flags a caller can observe on a result: contiguity, offset, strides, writeability and ownership.</summary>
        /// <param name="a">The result.</param>
        /// <returns>The layout as text.</returns>
        private static string Layout(NDArray a)
            => $"C={a.Shape.IsContiguous} F={a.Shape.IsFContiguous} offset={a.Shape.offset} strides={Fmt(a.Shape.strides)} " +
               $"writeable={a.Shape.IsWriteable} owndata={a.Storage.OwnsData}";
    }
}
