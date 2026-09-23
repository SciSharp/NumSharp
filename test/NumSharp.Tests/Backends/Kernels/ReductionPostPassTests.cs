using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
/// The axis-reduction post-passes <see cref="ILKernelGenerator.SeedReduceIdentity"/> and
/// <see cref="ILKernelGenerator.MeanDivideByCount"/> (shared by the engine's np.sum/np.mean/np.min/np.max
/// along an axis and by np.evaluate's fused axis reductions). Both take a typed dense-block fast path for a
/// writeable C- or F-contiguous output and keep the boxed per-element loop for every other layout. The
/// contract pinned here: every logical element receives exactly the value the boxed loop would write (the
/// identity through the boxed setter's own conversion; <c>x / (T)count</c> bit-for-bit, NaN payloads and signed
/// zeros included), nothing OUTSIDE the output view is touched (a dense view at an offset, a strided view), and
/// a read-only output still raises the read-only error and stays unchanged.
/// </summary>
[TestClass]
public class ReductionPostPassTests
{
    /// <summary>
    /// The output layouts every case runs over, each with the base array it views and the slice that made it
    /// (so writes outside the view can be detected): a fresh C array; an F-contiguous array; a C-contiguous row
    /// block (NumSharp re-seats a contiguous slice's storage address and keeps offset 0); an F-contiguous column
    /// block of an F array, which keeps a NON-ZERO view offset (offset 10 here) — the layout that proves the dense
    /// fast path adds the offset and does not spill into the neighbouring columns; and a column-strided view (boxed
    /// fallback, must not touch the skipped columns).
    /// </summary>
    /// <param name="tc">The dtype of every layout.</param>
    /// <returns>(name, view, base, slice) tuples; the slice is null where the view is its own base.</returns>
    private static (string name, NDArray view, NDArray @base, string slice)[] Layouts(NPTypeCode tc)
    {
        var c = np.arange(24).astype(tc).reshape(4, 6);
        var f = np.asfortranarray(np.arange(24).astype(tc).reshape(4, 6));
        var big = np.arange(60).astype(tc).reshape(10, 6);
        var fbig = np.asfortranarray(np.arange(60).astype(tc).reshape(10, 6));
        var wide = np.arange(48).astype(tc).reshape(4, 12);
        var fView = fbig[":, 1:5"];
        Assert.IsTrue(fView.Shape.IsFContiguous && fView.Shape.offset != 0, "the F-offset layout must be a dense block at a non-zero offset");
        return new[]
        {
            ("C", c, c, (string)null),
            ("F", f, f, null),
            ("C-rows", big["3:7"], big, "3:7"),
            ("F-offset", fView, fbig, ":, 1:5"),
            ("strided", wide[":, ::2"], wide, ":, ::2"),
        };
    }

    /// <summary>Raw bytes of <paramref name="a"/> in logical C order (a snapshot for untouched-memory checks).</summary>
    /// <param name="a">The array to snapshot.</param>
    /// <returns>The C-ordered raw bytes.</returns>
    private static byte[] Bytes(NDArray a) => np.ascontiguousarray(a).Unsafe.ReadOnlyBytes().ToArray();

    /// <summary>
    /// Assert every element of <paramref name="baseArr"/> OUTSIDE <paramref name="view"/> is byte-identical to the
    /// <paramref name="before"/> snapshot: the post-passes may only write the output's own elements.
    /// </summary>
    /// <param name="name">The layout name, for the message.</param>
    /// <param name="view">The output view the post-pass wrote.</param>
    /// <param name="baseArr">The array the view aliases.</param>
    /// <param name="slice">The slice of <paramref name="baseArr"/> that is <paramref name="view"/>; null when the view is the whole base.</param>
    /// <param name="before">A snapshot of <paramref name="baseArr"/> taken before the post-pass.</param>
    private static void AssertOutsideUntouched(string name, NDArray view, NDArray baseArr, string slice, byte[] before)
    {
        if (slice is null)
            return;
        // Re-write the view's elements with the snapshot's values (the snapshot is in logical C order, so the
        // same slice of it names the same logical elements), then the whole base must equal the snapshot.
        var restore = np.array(before).view(baseArr.dtype).reshape(baseArr.shape);
        np.copyto(view, restore[slice]);
        CollectionAssert.AreEqual(before, Bytes(baseArr), $"{name}: memory outside the output view was modified");
    }

    [TestMethod]
    public void SeedReduceIdentity_EveryLayoutAndDtype_WritesTheIdentityOnlyIntoTheOutput()
    {
        foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Half, NPTypeCode.Int64,
                     NPTypeCode.Byte, NPTypeCode.Decimal, NPTypeCode.Complex })
        foreach (var op in new[] { ReductionOp.Sum, ReductionOp.Prod, ReductionOp.Min, ReductionOp.Max })
        foreach (var (name, view, baseArr, slice) in Layouts(tc))
        {
            var before = Bytes(baseArr);
            ILKernelGenerator.SeedReduceIdentity(view, op);

            object expected = tc == NPTypeCode.Complex
                ? op switch
                {
                    ReductionOp.Sum => Complex.Zero,
                    ReductionOp.Prod => Complex.One,
                    ReductionOp.Min => new Complex(double.PositiveInfinity, double.PositiveInfinity),
                    _ => new Complex(double.NegativeInfinity, double.NegativeInfinity),
                }
                : op.GetIdentity(tc);
            for (long i = 0; i < view.size; i++)
                Assert.AreEqual(expected, view.GetAtIndex(i), $"{tc} {op} {name}: element {i}");

            AssertOutsideUntouched(name, view, baseArr, slice, before);
        }
    }

    [TestMethod]
    public void SeedReduceIdentity_BoolAnyAll_WritesFalseTrue()
    {
        foreach (var (op, expected) in new[] { (ReductionOp.Any, false), (ReductionOp.All, true) })
        foreach (var (name, view, baseArr, slice) in Layouts(NPTypeCode.Boolean))
        {
            var before = Bytes(baseArr);
            ILKernelGenerator.SeedReduceIdentity(view, op);
            for (long i = 0; i < view.size; i++)
                Assert.AreEqual(expected, view.GetAtIndex(i), $"{op} {name}: element {i}");
            AssertOutsideUntouched(name, view, baseArr, slice, before);
        }
    }

    [TestMethod]
    public void MeanDivideByCount_EveryLayout_IsBitExactPerElementDivision()
    {
        // Specials ride through the division: a NaN with a payload (and its sign), signed zeros, infinities,
        // a subnormal, and values whose quotient rounds — every one must match x / (T)count bit for bit.
        double[] pool = { 1.0, -0.0, 0.0, double.PositiveInfinity, double.NegativeInfinity,
                          BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8_0000_0000_1234UL)), 5e-324, 1e308,
                          -7.25, 3.0, 0.1, 2.0 / 3.0 };
        foreach (long count in new long[] { 1, 3, 7, 1_000_003 })
        foreach (var tc in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Complex, NPTypeCode.Decimal })
        foreach (var (name, view, baseArr, slice) in Layouts(tc))
        {
            // Fill the view with the pool (decimal skips the non-finite values it cannot hold).
            for (long i = 0; i < view.size; i++)
            {
                double v = pool[i % pool.Length];
                object boxed = tc switch
                {
                    NPTypeCode.Double => v,
                    NPTypeCode.Single => (float)v,
                    NPTypeCode.Complex => new Complex(v, pool[(i + 5) % pool.Length]),
                    _ => double.IsFinite(v) && System.Math.Abs(v) < 1e20 ? (object)(decimal)v : (object)(decimal)(i + 1),
                };
                view.SetAtIndex(boxed, i);
            }

            var input = view.copy();
            var before = Bytes(baseArr);
            ILKernelGenerator.MeanDivideByCount(view, count);

            for (long i = 0; i < view.size; i++)
            {
                switch (tc)
                {
                    case NPTypeCode.Double:
                        Assert.AreEqual(BitConverter.DoubleToInt64Bits((double)input.GetAtIndex(i) / count),
                            BitConverter.DoubleToInt64Bits((double)view.GetAtIndex(i)), $"f64 {name} n={count} [{i}]");
                        break;
                    case NPTypeCode.Single:
                    {
                        float d = count;
                        Assert.AreEqual(BitConverter.SingleToInt32Bits((float)input.GetAtIndex(i) / d),
                            BitConverter.SingleToInt32Bits((float)view.GetAtIndex(i)), $"f32 {name} n={count} [{i}]");
                        break;
                    }
                    case NPTypeCode.Complex:
                    {
                        var x = (Complex)input.GetAtIndex(i);
                        var y = (Complex)view.GetAtIndex(i);
                        Assert.AreEqual(BitConverter.DoubleToInt64Bits(x.Real / count), BitConverter.DoubleToInt64Bits(y.Real), $"c128.re {name} [{i}]");
                        Assert.AreEqual(BitConverter.DoubleToInt64Bits(x.Imaginary / count), BitConverter.DoubleToInt64Bits(y.Imaginary), $"c128.im {name} [{i}]");
                        break;
                    }
                    default:
                        Assert.AreEqual((decimal)input.GetAtIndex(i) / count, (decimal)view.GetAtIndex(i), $"decimal {name} [{i}]");
                        break;
                }
            }

            AssertOutsideUntouched(name, view, baseArr, slice, before);
        }
    }

    [TestMethod]
    public void ReadOnlyOutput_StillRaisesAndStaysUnchanged()
    {
        var a = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
        a.setflags(write: false);
        var before = Bytes(a);

        Assert.ThrowsException<NumSharpException>(() => ILKernelGenerator.SeedReduceIdentity(a, ReductionOp.Sum));
        Assert.ThrowsException<NumSharpException>(() => ILKernelGenerator.MeanDivideByCount(a, 3));
        CollectionAssert.AreEqual(before, Bytes(a), "a read-only output must not be written");
    }

    [TestMethod]
    public void AxisMean_ThroughTheEngineAndEvaluate_MatchesAnExplicitSumThenDivide()
    {
        // End to end: the engine's np.mean(axis) and np.evaluate(Mean(axis)) — both finish with
        // MeanDivideByCount (evaluate's complex mean with its own typed loop) — equal sum/n bit for bit.
        var a = (np.arange(3 * 4001).astype(NPTypeCode.Double) * 0.37 - 11.0).reshape(3, 4001);
        var sum0 = np.sum(a, 0);
        var mean0 = np.mean(a, 0);
        for (long i = 0; i < sum0.size; i++)
            Assert.AreEqual(BitConverter.DoubleToInt64Bits((double)sum0.GetAtIndex(i) / 3),
                BitConverter.DoubleToInt64Bits((double)mean0.GetAtIndex(i)), $"np.mean f64 [{i}]");

        var af = a.astype(NPTypeCode.Single);
        var sumf = np.sum(af, 0);
        var meanf = np.mean(af, 0);
        for (long i = 0; i < sumf.size; i++)
            Assert.AreEqual(BitConverter.SingleToInt32Bits((float)sumf.GetAtIndex(i) / 3f),
                BitConverter.SingleToInt32Bits((float)meanf.GetAtIndex(i)), $"np.mean f32 [{i}]");

        var ev = np.evaluate(NDExpr.Mean((NDExpr)a * 1.0, 0));
        for (long i = 0; i < sum0.size; i++)
            Assert.AreEqual(BitConverter.DoubleToInt64Bits((double)mean0.GetAtIndex(i)),
                BitConverter.DoubleToInt64Bits((double)ev.GetAtIndex(i)), $"evaluate Mean f64 [{i}]");

        var ca = a.astype(NPTypeCode.Complex) * new Complex(1.0, -0.5);
        var evc = np.evaluate(NDExpr.Mean((NDExpr)ca * 1.0, 0));
        var sumc = np.evaluate(NDExpr.Sum((NDExpr)ca * 1.0, 0));
        for (long i = 0; i < sumc.size; i++)
        {
            // np.mean's complex true_divide: Smith's form with a real divisor (see ComplexDivideByCountLikeNumPy).
            var s = (Complex)sumc.GetAtIndex(i);
            double scl = 1.0 / 3;
            var want = new Complex((s.Real + s.Imaginary * 0.0) * scl, (s.Imaginary - s.Real * 0.0) * scl);
            var got = (Complex)evc.GetAtIndex(i);
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(want.Real), BitConverter.DoubleToInt64Bits(got.Real), $"evaluate Mean c128.re [{i}]");
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(want.Imaginary), BitConverter.DoubleToInt64Bits(got.Imaginary), $"evaluate Mean c128.im [{i}]");
        }
    }
}
