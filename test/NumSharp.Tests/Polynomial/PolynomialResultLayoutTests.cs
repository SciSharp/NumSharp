using System;
using System.Linq;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     WHERE the <c>numpy.polynomial</c> evaluation family and <c>polyutils.mapdomain</c> store their results — the
    ///     strides NumPy 2.4.2's chain of ufunc calls leaves the final value with (2026-10-01 review). Values are pinned
    ///     elsewhere; these tests pin the layout, which the N-D-series path used to allocate in C order whatever its
    ///     operands were. Every expected stride below was probed with NumPy 2.4.2 (the probe is spelled out per test) and
    ///     is in ELEMENTS; the broad matrix is the <c>polyeval.jsonl</c> section L / <c>polyseries.jsonl</c> section R
    ///     "strides" facet.
    /// </summary>
    [TestClass]
    public class PolynomialResultLayoutTests
    {
        /// <summary>The result's element strides, every extent-1 axis zeroed (NumPy assigns those per loop path).</summary>
        /// <param name="r">The result (disposed here).</param>
        /// <returns>The normalized strides.</returns>
        private static long[] Strides(NDArray r)
        {
            try
            {
                return r.Shape.dimensions.Select((d, i) => d <= 1 ? 0L : r.Shape.strides[i]).ToArray();
            }
            finally
            {
                r.Dispose();
            }
        }

        /// <summary><c>np.arange(n, dtype=float64).reshape(dims) * k</c>, C-contiguous.</summary>
        private static NDArray Ramp(double k, params long[] dims)
        {
            long n = dims.Aggregate(1L, (a, b) => a * b);
            return (np.arange((double)n) * k).reshape(dims);
        }

        /// <summary>
        ///     The N-D-series path's result follows x's memory order on x's axes (NumPy's NpyIter layout), not C — the
        ///     defect the 2026-10-01 review fixed: an F-ordered x came back C-ordered, invisible to every value
        ///     comparison.
        /// </summary>
        [TestMethod]
        public void Val_NdSeries_FOrderedX_KeepsXAxesInXOrder()
        {
            // xF = (np.arange(12.).reshape(4,3).T * 0.25); c = np.arange(6.).reshape(3,2) * 0.5 - 1
            // chebval(xF, c).strides // 8 == (12, 1, 3): the series axis outermost, x's axes F — not C (12, 4, 1).
            var xF = np.transpose(Ramp(0.25, 4, 3), new[] { 1, 0 });
            var c = Ramp(0.5, 3, 2) - 1;
            Strides(np.polynomial.chebyshev.chebval(xF, c)).Should().Equal(12L, 1L, 3L);
        }

        /// <summary>
        ///     polyval reads its series without copying (<c>copy=None</c>), so a permuted series' axes keep the series'
        ///     memory order in the result — the series block of the tensor layout is NpyIter's layout of <c>c[0]</c>.
        /// </summary>
        [TestMethod]
        public void Val_PermutedSeries_KeepsSeriesAxesInSeriesOrder()
        {
            // cT = (np.arange(12.).reshape(3,2,2) * 0.5).transpose(0,2,1); x = np.arange(12.).reshape(3,4) * 0.1
            // polyval(x, cT).strides // 8 == (12, 24, 4, 1) — polyval reads c without copying (copy=None).
            var cT = np.transpose(Ramp(0.5, 3, 2, 2), new[] { 0, 2, 1 });
            var x = Ramp(0.1, 3, 4);
            Strides(np.polynomial.polynomial.polyval(x, cT)).Should().Equal(12L, 24L, 4L, 1L);
        }

        /// <summary>
        ///     With a scalar x only <c>c[k]</c> ever votes, so an F-ordered series gives an F-ordered result (NumPy's
        ///     <c>hermval(0.5, cF)</c>), where the old path always allocated C.
        /// </summary>
        [TestMethod]
        public void Val_ScalarX_FOrderedSeries_IsFOrdered()
        {
            // cF = np.asfortranarray(np.arange(24.).reshape(4,2,3) * 0.125); hermval(0.5, cF).strides // 8 == (1, 2)
            var cF = np.asfortranarray(Ramp(0.125, 4, 2, 3));
            var r = np.polynomial.hermite.hermval(0.5, cF);
            r.flags.f_contiguous.Should().BeTrue();
            Strides(r).Should().Equal(1L, 2L);
        }

        /// <summary>
        ///     A broadcast x's stride-0 axis ABSTAINS in NpyIter's sort, so the result is C — the old 'K'-copy rule
        ///     (<see cref="Shape.KeepOrder"/>) sorted it innermost and gave F. The case that separates the two rules.
        /// </summary>
        [TestMethod]
        public void Val_OneDSeries_BroadcastX_IsC()
        {
            // legval(np.broadcast_to(np.arange(4.), (3,4)), [1., 2., 3.]).strides // 8 == (4, 1): NpyIter lets the
            // stride-0 axis abstain (the old order='K' allocation sorted it innermost, giving F).
            var xb = np.broadcast_to(np.arange(4.0), new Shape(3, 4));
            var r = np.polynomial.legendre.legval(xb, new[] { 1.0, 2.0, 3.0 });
            r.flags.c_contiguous.Should().BeTrue();
            Strides(r).Should().Equal(4L, 1L);
        }

        /// <summary>
        ///     tensor=False with an F series against a C x on the same axes: NpyIter's conflict rule makes C win, which
        ///     the replay (or the C-operand shortcut) must reproduce.
        /// </summary>
        [TestMethod]
        public void Val_TensorFalse_ConflictingOrders_ResolveToC()
        {
            // cF2 = np.asfortranarray(np.arange(36.).reshape(3,3,4) * 0.01); x = np.arange(12.).reshape(3,4) * 0.1
            // hermeval(x, cF2, tensor=False).strides // 8 == (4, 1): series F, x C on the same axes — C wins.
            var cF2 = np.asfortranarray(Ramp(0.01, 3, 3, 4));
            var x = Ramp(0.1, 3, 4);
            Strides(np.polynomial.hermite_e.hermeval(x, cF2, tensor: false)).Should().Equal(4L, 1L);
        }

        /// <summary>
        ///     Each basis copies its series its own way — polyval's int conversion is a ufunc (<c>c + 0.0</c>, NpyIter:
        ///     a broadcast axis abstains), legval's is <c>astype</c> (order 'K': it sorts innermost), and chebval
        ///     copies even a float series — so one broadcast series gives different result layouts per basis.
        /// </summary>
        [TestMethod]
        public void Val_BroadcastSeries_FollowsEachBasisCopy()
        {
            // ci = np.broadcast_to(np.arange(3).reshape(1,1,3), (4,2,3))   (int64)
            // polyval(0.5, ci): (3, 1) — `c + 0.0` is a ufunc, the stride-0 axes abstain;
            // legval(0.5, ci): (1, 2) — `astype(np.double)` copies order 'K', the stride-0 axes sort innermost;
            // chebval(0.5, ci.astype(float)): (1, 2) — chebval copies (copy=True, order 'K') even a float series.
            var ci = np.broadcast_to(np.arange(3L).reshape(1, 1, 3), new Shape(4, 2, 3));
            Strides(np.polynomial.polynomial.polyval(0.5, ci)).Should().Equal(3L, 1L);
            Strides(np.polynomial.legendre.legval(0.5, ci)).Should().Equal(1L, 2L);
            var cf = np.broadcast_to(np.arange(3.0).reshape(1, 1, 3), new Shape(4, 2, 3));
            Strides(np.polynomial.chebyshev.chebval(0.5, cf)).Should().Equal(1L, 2L);
        }

        /// <summary>
        ///     The two-pass <c>_valnd</c> composition keeps an F-ordered ordinate's order through both passes (the
        ///     second pass's x is the first pass's result).
        /// </summary>
        [TestMethod]
        public void Val2d_FOrderedOrdinates_IsFOrdered()
        {
            // xF as above; c = np.arange(6.).reshape(3,2) * 0.5 - 1; lagval2d(xF, xF, c).strides // 8 == (1, 3)
            var xF = np.transpose(Ramp(0.25, 4, 3), new[] { 1, 0 });
            var c = Ramp(0.5, 3, 2) - 1;
            var r = np.polynomial.laguerre.lagval2d(xF, xF, c);
            r.flags.f_contiguous.Should().BeTrue();
            Strides(r).Should().Equal(1L, 3L);
        }

        /// <summary>
        ///     The layout decides only WHERE each value lands: the same evaluation over an F-ordered x and its C copy
        ///     reads back identical — guards against a relabelled result being read in the wrong order.
        /// </summary>
        [TestMethod]
        public void Val_ValuesDoNotDependOnTheLayout()
        {
            // The layout decides only where each value lands: the same evaluation over the F-ordered x and over its
            // C copy reads back identical, element by element.
            var xF = np.transpose(Ramp(0.25, 4, 3), new[] { 1, 0 });
            var c = Ramp(0.5, 3, 2) - 1;
            using var a = np.polynomial.chebyshev.chebval(xF, c);
            using var b = np.polynomial.chebyshev.chebval(np.ascontiguousarray(xF), c);
            a.Shape.IsContiguous.Should().BeFalse();
            b.Shape.IsContiguous.Should().BeTrue();
            np.array_equal(a, b).Should().BeTrue();
        }

        /// <summary>
        ///     mapdomain's result keeps a permuted x's memory order (NpyIter over x alone), where it used to be C for
        ///     every N-D x.
        /// </summary>
        [TestMethod]
        public void MapDomain_PermutedX_KeepsXOrder()
        {
            // xp = (np.arange(24.).reshape(2,3,4) * 0.1).transpose(2,0,1)
            // polyutils.mapdomain(xp, (-1, 1), (0, 2)).strides // 8 == (1, 12, 4)
            var xp = np.transpose(Ramp(0.1, 2, 3, 4), new[] { 2, 0, 1 });
            Strides(np.polynomial.polyutils.mapdomain(xp, (-1, 1), (0, 2))).Should().Equal(1L, 12L, 4L);
        }

        /// <summary>
        ///     The complex64 route (float32 kernels that write C order) also comes back in NumPy's layout — through the
        ///     transposed view and the relabel, like every other mapdomain route.
        /// </summary>
        [TestMethod]
        public void MapDomain_Complex64Loop_FOrderedX_IsFOrdered()
        {
            // A float32 x with a Python complex domain is NEP 50's complex64 loop (the float32-kernel route, which
            // writes C order and is now copied into NumPy's layout):
            // xF = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3,4))
            // polyutils.mapdomain(xF, (-1, 1), (0, 2+1j)) -> complex64 (3, 4), strides // itemsize == (1, 3), F.
            var xF = np.asfortranarray(np.arange(12f).reshape(3, 4));
            var r = np.polynomial.polyutils.mapdomain(xF, (-1, 1), (0, new System.Numerics.Complex(2, 1)));
            r.flags.f_contiguous.Should().BeTrue();
            Strides(r).Should().Equal(1L, 3L);
        }

        /// <summary>
        ///     The closed forms and the tensor=False C-operand shortcut of <see cref="NDPolyEval.ResultShape"/> agree
        ///     with the brute-force replay of NumPy's statements over thousands of seeded random layouts — the C#
        ///     cross-check of what was validated against NumPy offline.
        /// </summary>
        [TestMethod]
        public void ResultShape_ClosedForms_AgreeWithTheReplay()
        {
            // ResultShape answers every case in closed form but tensor=False with an N-D series, where it shortcuts a
            // C-contiguous operand spanning the result and replays otherwise; ReplayResultShape always walks NumPy's
            // statements over layouts. Over seeded random layouts of x and c (C, F, permuted, strided, reversed,
            // broadcast, unit and empty axes) and every basis the two must agree on every non-unit axis — the shortcut
            // included, which is what the broadcast-compatible tensor=False cases below are for.
            var rng = new System.Random(20261001);   // System.Random: NumSharp.Tests.Random is a namespace
            int compared = 0, shortcut = 0;
            for (int t = 0; t < 6000; t++)
            {
                var basis = (NumSharp.Backends.Kernels.PolyBasis)rng.Next(6);
                bool tensor = rng.Next(3) != 0, scalarX = rng.Next(5) == 0;
                var cDims = new[] { (long)rng.Next(1, 5) }.Concat(Enumerable.Range(0, rng.Next(0, 3)).Select(_ => (long)rng.Next(1, 4))).ToArray();
                bool replayCase = !scalarX && !tensor && cDims.Length > 1;
                long[] xDims;
                if (replayCase)
                {
                    // x must broadcast against c[k] = c.shape[1:]: a suffix of it with some extents turned to 1 — or,
                    // half the time, all of it, with the unit extents moved to the series side instead.
                    var tail = cDims.Skip(1).ToArray();
                    if (rng.Next(2) == 0)
                        xDims = tail.Skip(rng.Next(tail.Length + 1)).Select(d => rng.Next(3) == 0 ? 1L : d).ToArray();
                    else
                    {
                        xDims = tail;
                        cDims = cDims.Select((d, i) => i > 0 && rng.Next(3) == 0 ? 1L : d).ToArray();
                    }
                }
                else
                    xDims = Enumerable.Range(0, rng.Next(0, 4)).Select(_ => (long)rng.Next(0, 4)).ToArray();
                using var c = RandomView(rng, cDims);
                using var x = RandomView(rng, xDims);
                long[] outDims = scalarX ? cDims.Skip(1).ToArray()
                    : cDims.Length == 1 ? xDims
                    : tensor ? cDims.Skip(1).Concat(xDims).ToArray()
                    : Broadcast(cDims.Skip(1).ToArray(), xDims);
                var cl = NDPolyEval.SeriesLayout(basis, c.Shape, intLike: false);
                // ResultShape adopts its dims array: each call gets its own copy.
                var closed = NDPolyEval.ResultShape(basis, cl, scalarX ? null : x, tensor, (long[])outDims.Clone());
                var replay = NDPolyEval.ReplayResultShape(basis, cl, scalarX ? null : x, tensor, (long[])outDims.Clone());
                Normalized(closed).Should().Equal(Normalized(replay),
                    $"{basis} tensor={tensor} scalarX={scalarX} c=({string.Join(",", c.shape)})/({string.Join(",", c.Shape.strides)}) " +
                    $"x=({string.Join(",", x.shape)})/({string.Join(",", x.Shape.strides)})");
                if (replayCase && ((x.Shape.IsContiguous && x.Shape.dimensions.SequenceEqual(outDims))
                                   || (cl.IsContiguous && cl.dimensions.Skip(1).SequenceEqual(outDims))))
                    shortcut++;
                compared++;
            }
            compared.Should().BeGreaterThan(4000);
            shortcut.Should().BeGreaterThan(100, "the C-operand shortcut must actually be exercised against the replay");

            static long[] Normalized(Shape s)
            {
                bool empty = s.dimensions.Any(d => d == 0);
                return s.dimensions.Select((d, i) => empty || d <= 1 ? 0L : s.strides[i]).ToArray();
            }

            static long[] Broadcast(long[] a, long[] b)
            {
                int nd = System.Math.Max(a.Length, b.Length);   // NumSharp.Tests.Math is a namespace
                var r = new long[nd];
                for (int d = 0; d < nd; d++)
                {
                    long da = d - (nd - a.Length) < 0 ? 1 : a[d - (nd - a.Length)];
                    long db = d - (nd - b.Length) < 0 ? 1 : b[d - (nd - b.Length)];
                    r[d] = da == 1 ? db : da;
                }
                return r;
            }
        }

        /// <summary>
        ///     The relabel route (run on the transposed view, then <c>SetShapeUnsafe</c>) yields NumPy's strides,
        ///     values equal to the C-ordered computation, and a result that OWNS its data and is writeable — as NumPy's
        ///     ufunc output is — across the affine, fused, complex128 and complex64 routes.
        /// </summary>
        [TestMethod]
        public void MapDomain_NonCLayout_IsRelabelledNotCopied_AndOwnsItsData()
        {
            // mapdomain over an N-D non-C x runs on np.transpose(x, perm) and relabels the C result into NumPy's layout.
            // NumPy's result is a fresh ufunc output: OWNDATA and WRITEABLE, in x's memory order (probed 2.4.2):
            //   xp = (np.arange(24.).reshape(2,3,4) * 0.1).transpose(2,0,1); r = polyutils.mapdomain(xp, (-1,1), (0,2))
            //   r.strides // 8 == (1, 12, 4); r.flags.owndata, r.flags.writeable == True, True
            // and the values are those of the same map over the C copy, element for element — float64 (the affine
            // route), float16 (the fused pass), a complex128 loop, and NEP 50's complex64 loop (the float32 kernels).
            var xp = np.transpose(Ramp(0.1, 2, 3, 4), new[] { 2, 0, 1 });
            foreach (var (dtype, @new) in new (NPTypeCode, object)[]
                     {
                         (NPTypeCode.Double, (0, 2)), (NPTypeCode.Half, (0, 2)),
                         (NPTypeCode.Double, (0, new System.Numerics.Complex(2, 1))),
                         (NPTypeCode.Single, (0, new System.Numerics.Complex(2, 1))),
                     })
            {
                using var x = xp.astype(dtype);
                using var xc = np.ascontiguousarray(x);
                using var r = np.polynomial.polyutils.mapdomain(x, (-1, 1), @new);
                using var expected = np.polynomial.polyutils.mapdomain(xc, (-1, 1), @new);
                r.flags.owndata.Should().BeTrue($"{dtype} {@new}");
                r.flags.writeable.Should().BeTrue($"{dtype} {@new}");
                r.Shape.dimensions.Select((d, i) => d <= 1 ? 0L : r.Shape.strides[i]).Should().Equal(new[] { 1L, 12L, 4L }, $"{dtype} {@new}");
                np.array_equal(r, expected).Should().BeTrue($"{dtype} {@new}");
            }
        }

        /// <summary>A view of shape <paramref name="dims"/> over a fresh buffer, in a random layout: C, F, a random axis
        ///     permutation, every axis stepped, one axis reversed, or broadcast from a random sub-shape.</summary>
        /// <param name="rng">The seeded source.</param>
        /// <param name="dims">The view's shape.</param>
        /// <returns>The view (its base is owned by it through the view's reference).</returns>
        private static NDArray RandomView(System.Random rng, long[] dims)
        {
            int nd = dims.Length;
            long n = dims.Aggregate(1L, (a, b) => a * b);
            int kind = nd == 0 ? 0 : rng.Next(6);
            switch (kind)
            {
                case 1 when nd >= 2:   // F
                    return np.transpose(Ramp(1, Enumerable.Reverse(dims).ToArray()), Enumerable.Range(0, nd).Reverse().ToArray());
                case 2 when nd >= 2:   // a random permutation
                {
                    var perm = Enumerable.Range(0, nd).OrderBy(_ => rng.Next()).ToArray();
                    var inv = new int[nd];
                    for (int i = 0; i < nd; i++) inv[perm[i]] = i;
                    return np.transpose(Ramp(1, perm.Select(p => dims[p]).ToArray()), inv);
                }
                case 3:                // every axis stepped by 2
                    return Ramp(1, dims.Select(d => d * 2).ToArray())[string.Join(",", dims.Select(_ => "::2"))];
                case 4:                // one axis reversed
                {
                    int ax = rng.Next(nd);
                    return Ramp(1, dims)[string.Join(",", Enumerable.Range(0, nd).Select(i => i == ax ? "::-1" : ":"))];
                }
                case 5:                // broadcast from a random sub-shape
                {
                    var sub = dims.Select(d => rng.Next(2) == 0 ? 1L : d).ToArray();
                    return np.broadcast_to(Ramp(1, sub), new Shape(dims));
                }
                default:
                    if (nd == 0)
                        return NDArray.Scalar(0.5);   // a 0-d x
                    return n == 0 ? np.zeros(new Shape(dims)) : Ramp(1, dims);
            }
        }

        /// <summary>
        ///     <see cref="Shape.NpyIterOutputShape"/> against NumPy's own <c>np.add(…)</c> allocations: a stride-0 axis
        ///     abstains, a permutation keeps its memory order, and two disagreeing operands interleave by the conflict
        ///     rule.
        /// </summary>
        [TestMethod]
        public void NpyIterOutputShape_MatchesNumPysUfuncAllocation()
        {
            // np.add(a, 0.0).strides for single operands, and the two-operand interleave:
            //   broadcast_to(arange(4.), (3,4)) + 0           -> (4, 1)   (stride-0 axis abstains)
            //   arange(24.).reshape(2,3,4).transpose(2,0,1)+0 -> (1, 12, 4)
            //   reshape(2,1,1) + F(3,4)                       -> (12, 1, 3)
            var bcast = np.broadcast_to(np.arange(4.0), new Shape(3, 4));
            Shape.NpyIterOutputShape(new long[] { 3, 4 }, new[] { bcast.Shape }).strides.Should().Equal(4L, 1L);
            var perm = np.transpose(Ramp(1, 2, 3, 4), new[] { 2, 0, 1 });
            Shape.NpyIterOutputShape(new long[] { 4, 2, 3 }, new[] { perm.Shape }).strides.Should().Equal(1L, 12L, 4L);
            var c0 = Ramp(1, 2).reshape(2, 1, 1);
            var f = np.transpose(Ramp(1, 4, 3), new[] { 1, 0 });
            Shape.NpyIterOutputShape(new long[] { 2, 3, 4 }, new[] { c0.Shape, f.Shape }).strides.Should().Equal(12L, 1L, 3L);
        }
    }
}
