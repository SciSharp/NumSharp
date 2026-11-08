using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     What KIND of array a random function returns: NumPy's random functions return a fresh buffer that the result owns
    ///     (<c>flags.owndata</c> True), C-ordered and writeable, with two exceptions NumPy itself makes — the legacy
    ///     <c>RandomState.choice(int, size, replace=False)</c> without <c>p</c> (a slice of <c>permutation(pop_size)</c>) and
    ///     <c>Generator.permutation(x, axis)</c> along a non-leading axis (its fancy index returns a transposed view of a fresh
    ///     buffer). Every expected value, stride and flag here is NumPy 2.4.2's answer for the same call (seed 42), probed on
    ///     2026-10-05.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The values were right before the fix (plan <c>docs/plans/view-shape-open-bugs.md</c>, bug 3); the kind of array was
    ///     not, and it is observable: <c>flags.owndata</c>, <c>base</c>, the contiguity flags, <c>strides</c>, and a
    ///     <c>resize</c> that changes the size (which a view refuses). Found by a 780-case NumPy/NumSharp probe of every
    ///     <c>choice</c> path, every sampler at five sizes, <c>permutation</c> and <c>multivariate_normal</c>:
    ///     </para>
    ///     <list type="bullet">
    ///         <item><c>Generator.choice</c> without replacement (Floyd, tail shuffle, weighted) and the legacy weighted path
    ///         returned <c>reshape(shape)</c> views; NumPy assigns <c>idx.shape = shape</c> in place. Three
    ///         <c>[OpenBugs]</c> pins (<c>2f92d91d</c>) moved here.</item>
    ///         <item><c>multivariate_normal</c> (both APIs) returned a reshape view; NumPy assigns
    ///         <c>x.shape = final_shape</c> in place.</item>
    ///         <item><c>Generator.permutation(x, axis &gt; 0)</c> returned <c>np.take</c>'s C-ordered owning array; NumPy's
    ///         <c>arr[tuple(slices)]</c> returns the MapIter buffer (indexed axis first) transposed back.</item>
    ///     </list>
    ///     <para>
    ///     NumPy 2.x reports all-zero strides for any EMPTY array; NumSharp's empty strides differ library-wide, so the empty
    ///     cases check ownership and flags, not raw strides.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class RandomResultOwnershipTests
    {
        /// <summary>The weights of the weighted <c>choice</c> pins: NumPy's <c>p=[.1, .2, .3, .2, .2]</c>.</summary>
        /// <returns>A fresh float64 array.</returns>
        private static NDArray Weights5() => np.array(new[] { .1, .2, .3, .2, .2 });

        /// <summary>
        ///     Asserts a result is what NumPy's random functions return by default: a fresh, writeable, C-contiguous array
        ///     that owns its data, of the given dimensions.
        /// </summary>
        /// <param name="r">The result.</param>
        /// <param name="dims">The expected dimensions.</param>
        /// <param name="what">The NumPy call, named in the failure message.</param>
        private static void ShouldOwnCLayout(NDArray r, long[] dims, string what)
        {
            r.shape.Should().Equal(dims, what);
            r.flags.owndata.Should().BeTrue($"NumPy's {what} owns its data");
            r.flags.c_contiguous.Should().BeTrue($"NumPy's {what} is C-contiguous");
            r.flags.writeable.Should().BeTrue($"NumPy's {what} is writeable");
        }

        // ================================================================================================================
        //  choice — the three former [OpenBugs] pins and the passing control (moved from OpenBugs.Random.cs)
        // ================================================================================================================

        /// <summary>
        ///     <c>Generator.choice(1000, 6, replace=False)</c> owns its data (NumPy's Floyd path fills <c>np.empty(size)</c>
        ///     and assigns <c>idx.shape = shape</c> in place). Was an <c>[OpenBugs]</c> pin: a reshape view.
        /// </summary>
        [TestMethod]
        public void Choice_GeneratorWithoutReplacement_OwnsItsData()
        {
            using var r = np.random.default_rng(42).choice(1000L, new Shape(6L), replace: false);
            r.ToArray<long>().Should().Equal(652L, 770L, 432L, 858L, 438L, 88L);
            r.flags.owndata.Should().BeTrue("NumPy assigns idx.shape in place, so the result owns its buffer");
        }

        /// <summary>
        ///     <c>Generator.choice(5, 3, replace=False, p=...)</c> owns its data (NumPy's weighted path fills
        ///     <c>found = np.zeros(shape)</c>). Was an <c>[OpenBugs]</c> pin: a reshape view.
        /// </summary>
        [TestMethod]
        public void Choice_GeneratorWithoutReplacementWeighted_OwnsItsData()
        {
            using var p = Weights5();
            using var r = np.random.default_rng(42).choice(5L, new Shape(3L), replace: false, p: p);
            r.ToArray<long>().Should().Equal(3L, 2L, 4L);
            r.flags.owndata.Should().BeTrue("NumPy's found = np.zeros(shape) is the result itself");
        }

        /// <summary>
        ///     <c>RandomState.choice(5, 3, replace=False, p=...)</c> owns its data (its weighted path builds a fresh
        ///     <c>found</c> array). Was an <c>[OpenBugs]</c> pin: a reshape view.
        /// </summary>
        [TestMethod]
        public void Choice_RandomStateWithoutReplacementWeighted_OwnsItsData()
        {
            using var p = Weights5();
            using var r = np.random.RandomState(42).choice(5L, new Shape(3L), replace: false, p: p);
            r.ToArray<long>().Should().Equal(2L, 4L, 3L);
            r.flags.owndata.Should().BeTrue("NumPy's weighted path assigns idx.shape in place on a fresh array");
        }

        /// <summary>
        ///     Control: <c>RandomState.choice(10, 6, replace=False)</c> is a VIEW in NumPy too (a slice of
        ///     <c>permutation(10)</c>, its shape assigned in place), for a 1-D and a 2-D size alike — the fix above must not
        ///     make this path own its data.
        /// </summary>
        [TestMethod]
        public void Choice_RandomStateWithoutReplacement_IsAViewLikeNumPy()
        {
            using var r = np.random.RandomState(42).choice(10L, new Shape(6L), replace: false);
            r.ToArray<long>().Should().Equal(8L, 1L, 5L, 0L, 7L, 2L);
            r.flags.owndata.Should().BeFalse("NumPy returns permutation(pop_size)[:size], a view");

            using var r2 = np.random.RandomState(42).choice(10L, new Shape(2L, 3L), replace: false);
            r2.ToArray<long>().Should().Equal(8L, 1L, 5L, 0L, 7L, 2L);
            r2.flags.owndata.Should().BeFalse("the in-place idx.shape = shape keeps the slice a view");
            r2.flags.c_contiguous.Should().BeTrue();
        }

        // ================================================================================================================
        //  choice — every size and path without replacement
        // ================================================================================================================

        /// <summary>
        ///     Floyd's path owns its data at every size NumPy accepts: N-D (laid out C from the size's dimensions), <c>()</c>
        ///     and <c>None</c> (0-d; NumPy returns a Python int for <c>None</c>, NumSharp the 0-d array), empty, and with
        ///     <c>shuffle=False</c> (Floyd's own order).
        /// </summary>
        [TestMethod]
        public void Choice_GeneratorFloyd_EverySizeOwnsItsData()
        {
            using var r2d = np.random.default_rng(42).choice(1000L, new Shape(2L, 3L), replace: false);
            ShouldOwnCLayout(r2d, new long[] { 2, 3 }, "choice(1000, (2,3), replace=False)");
            r2d.strides.Should().Equal(new long[] { 24, 8 });
            r2d.ToArray<long>().Should().Equal(652L, 770L, 432L, 858L, 438L, 88L);

            using var r0 = np.random.default_rng(42).choice(1000L, Shape.Scalar, replace: false);
            ShouldOwnCLayout(r0, Array.Empty<long>(), "choice(1000, (), replace=False)");
            r0.GetAtIndex<long>(0).Should().Be(89L);

            using var rNone = np.random.default_rng(42).choice(1000L, null, replace: false);
            ShouldOwnCLayout(rNone, Array.Empty<long>(), "choice(1000, replace=False) (NumPy: the Python int 89)");
            rNone.GetAtIndex<long>(0).Should().Be(89L);

            using var rEmpty = np.random.default_rng(42).choice(1000L, new Shape(0L), replace: false);
            ShouldOwnCLayout(rEmpty, new long[] { 0 }, "choice(1000, (0,), replace=False)");

            using var rNoShuffle = np.random.default_rng(42).choice(1000L, new Shape(6L), replace: false, shuffle: false);
            ShouldOwnCLayout(rNoShuffle, new long[] { 6 }, "choice(1000, 6, replace=False, shuffle=False)");
            rNoShuffle.ToArray<long>().Should().Equal(88L, 770L, 652L, 438L, 432L, 858L);
        }

        /// <summary>
        ///     The tail-shuffle path (a population over 10000 and more than 1/50 of it drawn; 1/20 with
        ///     <c>shuffle=False</c>) owns its data: NumPy copies the tail of the shuffled range (<c>.copy()</c>, "to allow
        ///     potentially large array backing idx to be gc") and assigns its shape in place.
        /// </summary>
        [TestMethod]
        public void Choice_GeneratorTailShuffle_EverySizeOwnsItsData()
        {
            using var r = np.random.default_rng(42).choice(20000L, new Shape(500L), replace: false);
            ShouldOwnCLayout(r, new long[] { 500 }, "choice(20000, 500, replace=False)");
            var v = r.ToArray<long>();
            v.Take(6).Should().Equal(8305L, 18068L, 7297L, 2318L, 8047L, 5653L);
            v[^1].Should().Be(1785L);
            v.Sum().Should().Be(4998559L);

            using var r2d = np.random.default_rng(42).choice(20000L, new Shape(25L, 20L), replace: false);
            ShouldOwnCLayout(r2d, new long[] { 25, 20 }, "choice(20000, (25,20), replace=False)");
            r2d.strides.Should().Equal(new long[] { 160, 8 });
            r2d.ToArray<long>().Should().Equal(v, "the same draws as the 1-D size, laid out C");

            using var rNoShuffle = np.random.default_rng(42).choice(20000L, new Shape(1500L), replace: false, shuffle: false);
            ShouldOwnCLayout(rNoShuffle, new long[] { 1500 }, "choice(20000, 1500, replace=False, shuffle=False)");
            var w = rNoShuffle.ToArray<long>();
            w.Take(6).Should().Equal(19404L, 8644L, 14443L, 10198L, 7981L, 7589L);
            w[^1].Should().Be(1785L);
            w.Sum().Should().Be(15208518L);
        }

        /// <summary>
        ///     The weighted paths without replacement own their data at an N-D size and at <c>()</c>, on both APIs, including
        ///     a large population (the legacy rounds over 20000 weights).
        /// </summary>
        [TestMethod]
        public void Choice_Weighted_EverySizeOwnsItsData()
        {
            using var p = Weights5();
            using var g2d = np.random.default_rng(42).choice(5L, new Shape(1L, 2L), replace: false, p: p);
            ShouldOwnCLayout(g2d, new long[] { 1, 2 }, "Generator.choice(5, (1,2), replace=False, p=p)");
            g2d.ToArray<long>().Should().Equal(3L, 2L);

            using var g0 = np.random.default_rng(42).choice(5L, Shape.Scalar, replace: false, p: p);
            ShouldOwnCLayout(g0, Array.Empty<long>(), "Generator.choice(5, (), replace=False, p=p)");
            g0.GetAtIndex<long>(0).Should().Be(3L);

            using var l2d = np.random.RandomState(42).choice(5L, new Shape(1L, 2L), replace: false, p: p);
            ShouldOwnCLayout(l2d, new long[] { 1, 2 }, "RandomState.choice(5, (1,2), replace=False, p=p)");
            l2d.ToArray<long>().Should().Equal(2L, 4L);

            using var pBig = np.full(new Shape(20000L), 5e-05);
            using var lBig = np.random.RandomState(42).choice(20000L, new Shape(500L), replace: false, p: pBig);
            ShouldOwnCLayout(lBig, new long[] { 500 }, "RandomState.choice(20000, 500, replace=False, p=p)");
            var v = lBig.ToArray<long>();
            v.Take(6).Should().Equal(7490L, 19014L, 14639L, 11973L, 3120L, 3119L);
            v[^1].Should().Be(10716L);
            v.Sum().Should().Be(4990261L);
        }

        /// <summary>
        ///     Guard for the array-population path, which was already right: <c>a.take(idx, axis)</c> is a fresh C-ordered
        ///     array in NumPy and NumSharp alike.
        /// </summary>
        [TestMethod]
        public void Choice_GeneratorArrayPopulationAlongAxis_OwnsItsData()
        {
            using var a = np.arange(12).reshape(4, 3);
            using var r = np.random.default_rng(42).choice(a, new Shape(2L), replace: false, axis: 1);
            ShouldOwnCLayout(r, new long[] { 4, 2 }, "choice(arange(12).reshape(4,3), 2, replace=False, axis=1)");
            r.ToArray<long>().Should().Equal(0L, 2L, 3L, 5L, 6L, 8L, 9L, 11L);
        }

        // ================================================================================================================
        //  multivariate_normal
        // ================================================================================================================

        /// <summary>
        ///     <c>multivariate_normal</c> owns its data at every size on both APIs: NumPy computes the product into a fresh
        ///     array and assigns <c>x.shape = final_shape</c> in place. (The identity covariance makes the managed SVD's
        ///     factors exact, so the values are NumPy's without a LAPACK backend.)
        /// </summary>
        [TestMethod]
        public void MultivariateNormal_EverySizeOwnsItsData()
        {
            var mean = new[] { 0.0, 0.0 };
            var cov = new double[,] { { 1, 0 }, { 0, 1 } };
            var gen = new[] { 0.30471707975443135, -1.0399841062404955, 0.7504511958064572, 0.9405647163912139,
                -1.9510351886538364, -1.302179506862318, 0.12784040316728537, -0.3162425923435822, -0.016801157504288795,
                -0.85304392757358, 0.8793979748628286, 0.7777919354289483 };
            var leg = new[] { 0.4967141530112327, -0.13826430117118466, 0.6476885381006925, 1.5230298564080254,
                -0.23415337472333597, -0.23413695694918055, 1.5792128155073915, 0.7674347291529088, -0.4694743859349521,
                0.5425600435859647, -0.46341769281246226, -0.46572975357025687 };

            var sizes = new (Shape GenSize, Shape? LegSize, long[] Dims)[]
            {
                (default, null, new long[] { 2 }),
                (Shape.Scalar, Shape.Scalar, new long[] { 2 }),
                (new Shape(3L), new Shape(3L), new long[] { 3, 2 }),
                (new Shape(2L, 3L), new Shape(2L, 3L), new long[] { 2, 3, 2 }),
            };
            foreach (var (genSize, legSize, dims) in sizes)
            {
                long n = dims.Aggregate(1L, (x, y) => x * y);
                using var g = np.random.default_rng(42).multivariate_normal(mean, cov, genSize);
                ShouldOwnCLayout(g, dims, $"Generator.multivariate_normal(size={string.Join(",", dims)})");
                g.ToArray<double>().Should().Equal(gen.Take((int)n));

                using var l = np.random.RandomState(42).multivariate_normal(mean, cov, legSize);
                ShouldOwnCLayout(l, dims, $"RandomState.multivariate_normal(size={string.Join(",", dims)})");
                l.ToArray<double>().Should().Equal(leg.Take((int)n));
            }
        }

        // ================================================================================================================
        //  Generator.permutation — NumPy's fancy-index layout
        // ================================================================================================================

        /// <summary>
        ///     Along axis 0, NumPy's <c>arr[idx]</c> is a fresh C-ordered array that owns its data.
        /// </summary>
        [TestMethod]
        public void Permutation_LeadingAxis_IsAFreshCArray()
        {
            using var a = np.arange(12).reshape(4, 3);
            using var r = np.random.default_rng(42).permutation(a, 0);
            ShouldOwnCLayout(r, new long[] { 4, 3 }, "permutation(a, axis=0)");
            r.strides.Should().Equal(new long[] { 24, 8 });
            r.ToArray<long>().Should().Equal(9L, 10L, 11L, 6L, 7L, 8L, 3L, 4L, 5L, 0L, 1L, 2L);
        }

        /// <summary>
        ///     Along a non-leading axis, NumPy's <c>arr[tuple(slices)]</c> returns its MapIter buffer (the permuted axis
        ///     first) transposed back: a writeable VIEW, F-contiguous for 2-D input, strides <c>(8, 32)</c> for a
        ///     <c>(4, 3)</c> int64 array. A negative axis is the same call.
        /// </summary>
        [TestMethod]
        public void Permutation_NonLeadingAxis_IsNumPysTransposedView()
        {
            using var a = np.arange(12).reshape(4, 3);
            foreach (int axis in new[] { 1, -1 })
            {
                using var r = np.random.default_rng(42).permutation(a, axis);
                r.shape.Should().Equal(new long[] { 4, 3 });
                r.strides.Should().Equal(new long[] { 8, 32 }, "the (3, 4) buffer transposed back");
                r.flags.owndata.Should().BeFalse("NumPy returns the buffer's transpose, a view");
                r.flags.c_contiguous.Should().BeFalse();
                r.flags.f_contiguous.Should().BeTrue();
                r.flags.writeable.Should().BeTrue();
                r.ToArray<long>().Should().Equal(2L, 1L, 0L, 5L, 4L, 3L, 8L, 7L, 6L, 11L, 10L, 9L);
            }
        }

        /// <summary>
        ///     In 3-D the buffer is <c>(n_axis,) + the other dims</c> in C order, so the returned view's strides are permuted:
        ///     <c>(32, 64, 8)</c> along axis 1 and <c>(24, 8, 48)</c> along axis 2 of a <c>(2, 3, 4)</c> int64 array —
        ///     neither C- nor F-contiguous.
        /// </summary>
        [TestMethod]
        public void Permutation_NonLeadingAxis_3D_HasNumPysStrides()
        {
            using var a = np.arange(24).reshape(2, 3, 4);

            using var r1 = np.random.default_rng(42).permutation(a, 1);
            r1.strides.Should().Equal(new long[] { 32, 64, 8 });
            r1.flags.owndata.Should().BeFalse();
            r1.flags.c_contiguous.Should().BeFalse();
            r1.flags.f_contiguous.Should().BeFalse();
            r1.ToArray<long>().Should().Equal(8L, 9L, 10L, 11L, 4L, 5L, 6L, 7L, 0L, 1L, 2L, 3L,
                20L, 21L, 22L, 23L, 16L, 17L, 18L, 19L, 12L, 13L, 14L, 15L);

            using var r2 = np.random.default_rng(42).permutation(a, 2);
            r2.strides.Should().Equal(new long[] { 24, 8, 48 });
            r2.flags.owndata.Should().BeFalse();
            r2.flags.c_contiguous.Should().BeFalse();
            r2.flags.f_contiguous.Should().BeFalse();
            r2.ToArray<long>().Should().Equal(3L, 2L, 1L, 0L, 7L, 6L, 5L, 4L, 11L, 10L, 9L, 8L,
                15L, 14L, 13L, 12L, 19L, 18L, 17L, 16L, 23L, 22L, 21L, 20L);
        }

        /// <summary>
        ///     The source's own layout does not leak into the result: an F-ordered and a column-stepped source give NumPy's
        ///     same transposed view (strides <c>(8, 32)</c>), with the source's values.
        /// </summary>
        [TestMethod]
        public void Permutation_NonLeadingAxis_AnySourceLayout()
        {
            using var c = np.arange(12).reshape(4, 3);
            using var f = np.asfortranarray(c);
            using var rf = np.random.default_rng(42).permutation(f, 1);
            rf.strides.Should().Equal(new long[] { 8, 32 });
            rf.flags.owndata.Should().BeFalse();
            rf.ToArray<long>().Should().Equal(2L, 1L, 0L, 5L, 4L, 3L, 8L, 7L, 6L, 11L, 10L, 9L);

            using var stepped = np.arange(24).reshape(4, 6)[":, ::2"];
            using var rs = np.random.default_rng(42).permutation(stepped, 1);
            rs.strides.Should().Equal(new long[] { 8, 32 });
            rs.flags.owndata.Should().BeFalse();
            rs.ToArray<long>().Should().Equal(4L, 2L, 0L, 10L, 8L, 6L, 16L, 14L, 12L, 22L, 20L, 18L);
        }

        /// <summary>
        ///     Degenerate axes keep NumPy's view: an empty result (either dimension 0) does not own its data, and a length-1
        ///     axis gives the <c>(4, 1)</c> view with strides <c>(8, 32)</c>, both C- and F-contiguous.
        /// </summary>
        [TestMethod]
        public void Permutation_NonLeadingAxis_EmptyAndLengthOneAxes()
        {
            using var e1 = np.zeros(new Shape(2L, 0L));
            using var r1 = np.random.default_rng(42).permutation(e1, 1);
            r1.shape.Should().Equal(new long[] { 2, 0 });
            r1.flags.owndata.Should().BeFalse("NumPy's empty fancy-index result is still the buffer's transpose");

            using var e2 = np.zeros(new Shape(0L, 3L));
            using var r2 = np.random.default_rng(42).permutation(e2, 1);
            r2.shape.Should().Equal(new long[] { 0, 3 });
            r2.flags.owndata.Should().BeFalse();

            using var col = np.arange(4.0).reshape(4, 1);
            using var r3 = np.random.default_rng(42).permutation(col, 1);
            r3.strides.Should().Equal(new long[] { 8, 32 });
            r3.flags.owndata.Should().BeFalse();
            r3.flags.c_contiguous.Should().BeTrue();
            r3.flags.f_contiguous.Should().BeTrue();
            r3.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0);
        }

        /// <summary>
        ///     Every slab width the gather copies: 1-byte (int8), 16-byte (complex128, two MOVs) and a 24-byte slab (three
        ///     float64 per index, the runtime-sized copy) — NumPy's values and strides for each.
        /// </summary>
        [TestMethod]
        public void Permutation_NonLeadingAxis_EverySlabWidth()
        {
            using var i8 = np.arange(15).astype(np.int8).reshape(3, 5);
            using var r8 = np.random.default_rng(42).permutation(i8, 1);
            r8.strides.Should().Equal(new long[] { 1, 3 });
            r8.flags.owndata.Should().BeFalse();
            r8.ToArray<sbyte>().Should().Equal(new sbyte[] { 4, 2, 3, 1, 0, 9, 7, 8, 6, 5, 14, 12, 13, 11, 10 });

            var cv = Enumerable.Range(0, 6).Select(k => new Complex(k, k)).ToArray();
            using var c16 = np.array(cv).reshape(2, 3);
            using var rc = np.random.default_rng(42).permutation(c16, 1);
            rc.strides.Should().Equal(new long[] { 16, 32 });
            rc.flags.owndata.Should().BeFalse();
            rc.ToArray<Complex>().Should().Equal(new Complex(2, 2), new Complex(1, 1), new Complex(0, 0),
                new Complex(5, 5), new Complex(4, 4), new Complex(3, 3));

            using var f8 = np.arange(12.0).reshape(2, 3, 2);
            using var rf = np.random.default_rng(42).permutation(f8, 1);
            rf.strides.Should().Equal(new long[] { 16, 32, 8 });
            rf.flags.owndata.Should().BeFalse();
            rf.ToArray<double>().Should().Equal(4.0, 5.0, 2.0, 3.0, 0.0, 1.0, 10.0, 11.0, 8.0, 9.0, 6.0, 7.0);
        }

        /// <summary>
        ///     A shape large enough for several tiles of source rows (3000 rows of 4000 bytes against the 1 MB read window),
        ///     with a partial last tile and a row count that is not a multiple of the 4x unroll: NumPy's values (checked at
        ///     both ends and by their sum) and strides <c>(8, 24000)</c>.
        /// </summary>
        [TestMethod]
        public void Permutation_NonLeadingAxis_ManyTiles()
        {
            using var x = np.arange(3000L * 500).astype(np.float64).reshape(3000, 500);
            using var r = np.random.default_rng(7).permutation(x, 1);
            r.strides.Should().Equal(new long[] { 8, 24000 });
            r.flags.owndata.Should().BeFalse();
            r.flags.f_contiguous.Should().BeTrue();
            var v = r.ToArray<double>();
            v.Take(5).Should().Equal(394.0, 63.0, 466.0, 471.0, 254.0);
            v.Skip(v.Length - 3).Should().Equal(1499649.0, 1499925.0, 1499639.0);
            v.Sum().Should().Be(1124999250000.0);
        }

        // ================================================================================================================
        //  Every sampler, every size: the default kind
        // ================================================================================================================

        /// <summary>
        ///     Every sampler of both APIs, at sizes <c>()</c>, <c>(5,)</c>, <c>(2,3)</c>, <c>(0,)</c> and <c>(2,1,3)</c>,
        ///     returns a fresh writeable C-contiguous array that owns its data, as NumPy's does (the probe found all of them
        ///     already right except <c>multivariate_normal</c>; this pins the whole surface against a reshape-view return).
        /// </summary>
        [TestMethod]
        public void EverySampler_EverySize_ReturnsAFreshCArray()
        {
            var cov = new double[,] { { 1, 0 }, { 0, 1 } };
            var generator = new (string Name, Func<Generator, Shape, NDArray> Call, long Trailing)[]
            {
                ("random", (g, s) => g.random(s), 0), ("random_f32", (g, s) => g.random(s, np.float32), 0),
                ("standard_normal", (g, s) => g.standard_normal(s), 0), ("standard_exponential", (g, s) => g.standard_exponential(s), 0),
                ("standard_gamma", (g, s) => g.standard_gamma(2.0, s), 0), ("standard_cauchy", (g, s) => g.standard_cauchy(s), 0),
                ("integers", (g, s) => g.integers(0L, 10L, s), 0), ("integers_u8", (g, s) => g.integers(0L, 10L, s, np.uint8), 0),
                ("uniform", (g, s) => g.uniform(0.0, 1.0, s), 0), ("normal", (g, s) => g.normal(0.0, 1.0, s), 0),
                ("exponential", (g, s) => g.exponential(1.0, s), 0), ("gamma", (g, s) => g.gamma(2.0, 1.0, s), 0),
                ("beta", (g, s) => g.beta(2.0, 3.0, s), 0), ("binomial", (g, s) => g.binomial(10L, 0.5, s), 0),
                ("chisquare", (g, s) => g.chisquare(3.0, s), 0), ("f", (g, s) => g.f(2.0, 3.0, s), 0),
                ("geometric", (g, s) => g.geometric(0.5, s), 0), ("gumbel", (g, s) => g.gumbel(0.0, 1.0, s), 0),
                ("hypergeometric", (g, s) => g.hypergeometric(5L, 5L, 3L, s), 0), ("laplace", (g, s) => g.laplace(0.0, 1.0, s), 0),
                ("logistic", (g, s) => g.logistic(0.0, 1.0, s), 0), ("lognormal", (g, s) => g.lognormal(0.0, 1.0, s), 0),
                ("logseries", (g, s) => g.logseries(0.5, s), 0), ("negative_binomial", (g, s) => g.negative_binomial(5.0, 0.5, s), 0),
                ("noncentral_chisquare", (g, s) => g.noncentral_chisquare(3.0, 1.0, s), 0),
                ("noncentral_f", (g, s) => g.noncentral_f(2.0, 3.0, 1.0, s), 0), ("pareto", (g, s) => g.pareto(2.0, s), 0),
                ("poisson", (g, s) => g.poisson(3.0, s), 0), ("power", (g, s) => g.power(2.0, s), 0),
                ("rayleigh", (g, s) => g.rayleigh(1.0, s), 0), ("standard_t", (g, s) => g.standard_t(3.0, s), 0),
                ("triangular", (g, s) => g.triangular(0.0, 0.5, 1.0, s), 0), ("vonmises", (g, s) => g.vonmises(0.0, 1.0, s), 0),
                ("wald", (g, s) => g.wald(1.0, 1.0, s), 0), ("weibull", (g, s) => g.weibull(2.0, s), 0),
                ("zipf", (g, s) => g.zipf(2.0, s), 0),
                ("multinomial", (g, s) => g.multinomial(10L, new[] { 0.2, 0.3, 0.5 }, s), 3),
                ("dirichlet", (g, s) => g.dirichlet(new[] { 1.0, 2.0, 3.0 }, s), 3),
                ("multivariate_hypergeometric", (g, s) => g.multivariate_hypergeometric(new long[] { 5, 5, 5 }, 3L, s), 3),
                ("multivariate_normal", (g, s) => g.multivariate_normal(new[] { 0.0, 0.0 }, cov, s), 2),
            };
            var legacy = new (string Name, Func<NumPyRandom, Shape, NDArray> Call, long Trailing)[]
            {
                ("random_sample", (r, s) => r.random_sample(s), 0), ("randint", (r, s) => r.randint(0L, 10L, s), 0),
                ("random_integers", (r, s) => r.random_integers(1L, 10L, s), 0), ("tomaxint", (r, s) => r.tomaxint(s), 0),
                ("standard_normal", (r, s) => r.standard_normal(s), 0), ("standard_exponential", (r, s) => r.standard_exponential(s), 0),
                ("standard_gamma", (r, s) => r.standard_gamma(2.0, s), 0), ("standard_cauchy", (r, s) => r.standard_cauchy(s), 0),
                ("uniform", (r, s) => r.uniform(0.0, 1.0, s), 0), ("normal", (r, s) => r.normal(0.0, 1.0, s), 0),
                ("exponential", (r, s) => r.exponential(1.0, s), 0), ("gamma", (r, s) => r.gamma(2.0, 1.0, s), 0),
                ("beta", (r, s) => r.beta(2.0, 3.0, s), 0), ("binomial", (r, s) => r.binomial(10L, 0.5, s), 0),
                ("chisquare", (r, s) => r.chisquare(3.0, s), 0), ("f", (r, s) => r.f(2.0, 3.0, s), 0),
                ("geometric", (r, s) => r.geometric(0.5, s), 0), ("gumbel", (r, s) => r.gumbel(0.0, 1.0, s), 0),
                ("hypergeometric", (r, s) => r.hypergeometric(5L, 5L, 3L, s), 0), ("laplace", (r, s) => r.laplace(0.0, 1.0, s), 0),
                ("logistic", (r, s) => r.logistic(0.0, 1.0, s), 0), ("lognormal", (r, s) => r.lognormal(0.0, 1.0, s), 0),
                ("logseries", (r, s) => r.logseries(0.5, s), 0), ("negative_binomial", (r, s) => r.negative_binomial(5.0, 0.5, s), 0),
                ("noncentral_chisquare", (r, s) => r.noncentral_chisquare(3.0, 1.0, s), 0),
                ("noncentral_f", (r, s) => r.noncentral_f(2.0, 3.0, 1.0, s), 0), ("pareto", (r, s) => r.pareto(2.0, s), 0),
                ("poisson", (r, s) => r.poisson(3.0, s), 0), ("power", (r, s) => r.power(2.0, s), 0),
                ("rayleigh", (r, s) => r.rayleigh(1.0, s), 0), ("standard_t", (r, s) => r.standard_t(3.0, s), 0),
                ("triangular", (r, s) => r.triangular(0.0, 0.5, 1.0, s), 0), ("vonmises", (r, s) => r.vonmises(0.0, 1.0, s), 0),
                ("wald", (r, s) => r.wald(1.0, 1.0, s), 0), ("weibull", (r, s) => r.weibull(2.0, s), 0),
                ("zipf", (r, s) => r.zipf(2.0, s), 0),
                ("multinomial", (r, s) => r.multinomial(10L, new[] { 0.2, 0.3, 0.5 }, (Shape?)s), 3),
                ("dirichlet", (r, s) => r.dirichlet(new[] { 1.0, 2.0, 3.0 }, (Shape?)s), 3),
                ("multivariate_normal", (r, s) => r.multivariate_normal(new[] { 0.0, 0.0 }, cov, (Shape?)s), 2),
            };
            var sizes = new[] { Array.Empty<long>(), new long[] { 5 }, new long[] { 2, 3 }, new long[] { 0 }, new long[] { 2, 1, 3 } };

            var failures = new List<string>();
            foreach (var dims in sizes)
            {
                var expected = dims.ToList();
                foreach (var (name, call, trailing) in generator)
                    Check($"Generator.{name}", () => call(np.random.default_rng(42), new Shape(dims)), trailing);
                foreach (var (name, call, trailing) in legacy)
                    Check($"RandomState.{name}", () => call(np.random.RandomState(42), new Shape(dims)), trailing);

                void Check(string what, Func<NDArray> run, long trailing)
                {
                    var want = trailing == 0 ? expected : expected.Append(trailing).ToList();
                    using var r = run();
                    if (!r.shape.SequenceEqual(want) || !r.flags.owndata || !r.flags.c_contiguous || !r.flags.writeable)
                        failures.Add($"{what}(size=({string.Join(",", dims)})): shape ({string.Join(",", r.shape)}) " +
                                     $"owndata={r.flags.owndata} c={r.flags.c_contiguous} writeable={r.flags.writeable}");
                }
            }

            failures.Should().BeEmpty("NumPy's samplers return a fresh writeable C-ordered array that owns its data");
        }
    }
}
