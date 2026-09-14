using System;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in
    ///     <c>docs/website-src/docs/fundamentals/ufuncs.md</c> (the "Universal functions (ufunc)
    ///     basics" fundamentals article). Each test mirrors one documented snippet and asserts the
    ///     behaviour the page claims. Values were captured by running the snippets against this branch.
    /// </summary>
    [TestClass]
    public class FundamentalsUfuncsDocTests
    {
        // ── Elementwise operation ──────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Elementwise_OperatorAndNamedForm()
        {
            (np.array(new[] { 0, 2, 3, 4 }) + np.array(new[] { 1, 1, -1, 2 })).ToArray<int>()
              .Should().Equal(1, 3, 2, 6);
            np.add(np.array(new[] { 0, 2, 3, 4 }), np.array(new[] { 1, 1, -1, 2 })).ToArray<int>()
              .Should().Equal(1, 3, 2, 6);
        }

        // ── Broadcasting ───────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Broadcasting_RowAcrossRows()
        {
            var a = np.ones((3, 4));
            var b = np.array(new[] { 1, 2, 3, 4 });
            (a + b).shape.Should().Equal(new long[] { 3, 4 });
        }

        // ── Type casting (NEP 50) ──────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Casting_PromotionAndWeakScalars()
        {
            var i = np.array(new[] { 1, 2, 3 });                 // int32
            var f = np.array(new[] { 1.5, 2.5, 3.5 });           // float64
            (i + f).typecode.Should().Be(NPTypeCode.Double);

            var x = np.array(new[] { 1, 2, 3 }, np.int8);
            (x + 1).typecode.Should().Be(NPTypeCode.SByte, "weak scalar does not upcast");
            (x + np.array(new[] { 1 })).typecode.Should().Be(NPTypeCode.Int32, "the int32 array forces promotion");
        }

        // ── out= / where= / dtype= ─────────────────────────────────────────────────────────────────

        [TestMethod]
        public void OutWhereDtype()
        {
            var a = np.array(new[] { 1.0, 4.0, 9.0 });
            var dst = np.zeros(3);
            var ret = np.sqrt(a, @out: dst);
            dst.ToArray<double>().Should().Equal(1.0, 2.0, 3.0);
            ReferenceEquals(ret, dst).Should().BeTrue("out is returned");

            var a2 = np.array(new[] { 1.0, 2.0, 3.0, 4.0 });
            var dst2 = np.zeros(4);
            np.add(a2, 10, @out: dst2, where: a2 > 2);
            dst2.ToArray<double>().Should().Equal(new[] { 0.0, 0.0, 13.0, 14.0 }, "false slots keep prior contents");

            np.add(0.1, 0.2, dtype: np.float32).typecode.Should().Be(NPTypeCode.Single, "dtype selects the loop precision");
        }

        // ── Reductions ─────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Reductions_AxisAndAllAxes()
        {
            var x = np.arange(9).reshape(3, 3);
            np.sum(x, axis: 1).ToArray<long>().Should().Equal(3L, 12, 21);
            ((long)np.sum(x)).Should().Be(36L, "no axis reduces all axes");
            np.max(x, axis: 0, keepdims: true).shape.Should().Equal(new long[] { 1, 3 });
        }

        [TestMethod]
        public void Reductions_UpcastRule()
        {
            var x = np.array(new[] { 1, 2, 3 }, np.int32);
            np.sum(x).typecode.Should().Be(NPTypeCode.Int64, "accumulating reductions widen");
            np.mean(x).typecode.Should().Be(NPTypeCode.Double);
            np.max(x).typecode.Should().Be(NPTypeCode.Int32, "min/max/amax preserve the input dtype");

            var r = np.sum(np.array(new byte[] { 200, 200, 200 }));
            r.typecode.Should().Be(NPTypeCode.UInt64, "unsigned inputs widen to uint64");
            ((ulong)r).Should().Be(600UL);
        }

        // ── Pairwise (distinct from reductions) ────────────────────────────────────────────────────

        [TestMethod]
        public void Pairwise_MaximumIsElementwise()
        {
            np.maximum(np.array(new[] { 1, 5, 3 }), np.array(new[] { 4, 2, 6 })).ToArray<int>()
              .Should().Equal(4, 5, 6);
        }

        // ── Fused np.evaluate ──────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Fused_Evaluate()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            var b = np.array(new[] { 4.0, 5.0, 6.0 });
            np.evaluate((NDExpr)a * b + 2).ToArray<double>().Should().Equal(6.0, 12.0, 20.0);
            ((double)np.evaluate(NDExpr.Sum((NDExpr)a * b))).Should().Be(32.0);
        }

        // ── Common patterns ────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Patterns_InPlaceAndMaskedAndOverflowSafe()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            np.multiply(a, 2.0, @out: a);
            a.ToArray<double>().Should().Equal(2.0, 4.0, 6.0);

            var m = np.array(new[] { -1.0, 2.0, -3.0 });
            np.add(m, 100, @out: m, where: m < 0);
            m.ToArray<double>().Should().Equal(99.0, 2.0, 97.0);

            // outer product via newaxis (newaxis is an index object, not a slice-string token)
            var v = np.arange(4);
            var b = np.arange(3);
            var outer = v[Slice.All, np.newaxis] + b;
            outer.shape.Should().Equal(new long[] { 4, 3 });
            ((long)outer[2, 1]).Should().Be(3L);
        }
    }
}
