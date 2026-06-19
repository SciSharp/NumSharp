using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins two reduction correctness fixes against NumPy 2.4.2:
///
/// 1. Boolean <c>amax</c>/<c>amin</c> along an axis (FUZZ_FINDINGS #13). The axis
///    scalar reducer seeded the Boolean Max accumulator from a double bridge
///    (double.NegativeInfinity → ConvertFromDouble&lt;bool&gt; → <c>!= 0</c> → True),
///    so any all-False group reduced to True. Fixed by an explicit bool identity.
///
/// 2. Reductions over a view whose offset lives in <c>Shape.offset</c> — sliced
///    (a[1:3,1:3]) and negative-stride (a[::-1], a[:,::-1]) views. The NpyIter
///    REDUCE path (op_axes branch) did not add Shape.offset to the operand base
///    pointer, so it read from the buffer base — wrong cells, and after
///    FlipNegativeStrides moved the pointer, out-of-bounds (garbage / NaN). This
///    silently corrupted sum/mean/prod/min/max for every NpyIter-routed dtype
///    (double/single sum+mean, complex & decimal all ops, half sum+mean) on any
///    offset!=0 view. Contiguous / transpose / F-order / positive-strided views
///    (offset==0) were unaffected, which is why it hid so long.
/// </summary>
[TestClass]
public class ReduceOffsetStrideParityTests
{
    private static bool[] BoolArr(NDArray a)
    {
        var r = new bool[a.size];
        for (long i = 0; i < a.size; i++) r[i] = (bool)a.GetAtIndex(i);
        return r;
    }

    // Half is not IConvertible, so Convert.ToDouble(boxedHalf) throws — bridge it explicitly.
    private static double D(object o) => o is Half h ? (double)h : Convert.ToDouble(o);

    [TestMethod]
    public void Bool_AmaxAmin_AlongAxis_MatchesNumPy()
    {
        // [[T,F,T],[F,F,T]] — col1 is all-False (the bug: amax returned True there).
        var b = np.array(new bool[] { true, false, true, false, false, true }).reshape(2, 3);
        CollectionAssert.AreEqual(new[] { true, false, true }, BoolArr(np.amax(b, 0)), "amax axis0");
        CollectionAssert.AreEqual(new[] { false, false, true }, BoolArr(np.amin(b, 0)), "amin axis0");
        CollectionAssert.AreEqual(new[] { true, true }, BoolArr(np.amax(b, 1)), "amax axis1");
        CollectionAssert.AreEqual(new[] { false, false }, BoolArr(np.amin(b, 1)), "amin axis1");

        // [[F,F,F],[T,F,T]] — row0 all-False along axis1.
        var c = np.array(new bool[] { false, false, false, true, false, true }).reshape(2, 3);
        CollectionAssert.AreEqual(new[] { false, true }, BoolArr(np.amax(c, 1)), "amax axis1 (all-False row)");
        CollectionAssert.AreEqual(new[] { true, false, true }, BoolArr(np.amax(c, 0)), "amax axis0");

        // flat bool min/max were always correct — pin them so the fix can't regress them.
        Assert.AreEqual(true, (bool)np.amax(b).GetAtIndex(0), "amax flat");
        Assert.AreEqual(false, (bool)np.amin(b).GetAtIndex(0), "amin flat");
    }

    private static void AssertD(NDArray got, double[] expected, string ctx)
    {
        Assert.AreEqual(expected.Length, (int)got.size, $"{ctx}: size");
        for (long i = 0; i < expected.Length; i++)
            Assert.AreEqual(expected[i], D(got.GetAtIndex(i)), 1e-9, $"{ctx}[{i}]");
    }

    [TestMethod]
    public void Double_NegativeStride_And_OffsetSlice_Reduce_MatchesNumPy()
    {
        // a = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
        var a = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);

        var rev = a["::-1, :"]; // [[8,9,10,11],[4,5,6,7],[0,1,2,3]] (offset 8, stride -4)
        AssertD(np.sum(rev, 0), new double[] { 12, 15, 18, 21 }, "sum(rev,0)");
        AssertD(np.sum(rev, 1), new double[] { 38, 22, 6 }, "sum(rev,1)");
        AssertD(np.amax(rev, 0), new double[] { 8, 9, 10, 11 }, "amax(rev,0)");
        AssertD(np.amin(rev, 0), new double[] { 0, 1, 2, 3 }, "amin(rev,0)");
        AssertD(np.mean(rev, 0), new double[] { 4, 5, 6, 7 }, "mean(rev,0)");

        var revc = a[":, ::-1"]; // each row reversed (offset 3, stride -1 on axis1)
        AssertD(np.sum(revc, 0), new double[] { 21, 18, 15, 12 }, "sum(revcol,0)");
        AssertD(np.sum(revc, 1), new double[] { 6, 22, 38 }, "sum(revcol,1)");

        var sl = a["1:3, 1:3"]; // [[5,6],[9,10]] (offset 5, positive strides)
        AssertD(np.sum(sl, 0), new double[] { 14, 16 }, "sum(slice,0)");
        AssertD(np.sum(sl, 1), new double[] { 11, 19 }, "sum(slice,1)");
        AssertD(np.amax(sl, 0), new double[] { 9, 10 }, "amax(slice,0)");
        AssertD(np.mean(sl, 1), new double[] { 5.5, 9.5 }, "mean(slice,1)");

        // flat (axis=None) over the offset views
        Assert.AreEqual(66.0, D(np.sum(rev).GetAtIndex(0)), 1e-9, "sum(rev) flat");
        Assert.AreEqual(30.0, D(np.sum(sl).GetAtIndex(0)), 1e-9, "sum(slice) flat");
    }

    [TestMethod]
    public void Complex_NegativeStride_Reduce_MatchesContiguousCopy()
    {
        var data = new Complex[12];
        for (int i = 0; i < 12; i++) data[i] = new Complex(((i * 5) % 13) - 6, ((i * 3) % 11) - 5);
        var a = np.array(data).reshape(3, 4);
        foreach (var view in new[] { a["::-1, :"], a[":, ::-1"], a["1:3, 1:3"] })
        {
            for (int axis = 0; axis <= 1; axis++)
            {
                var g = np.sum(view, axis);
                var e = np.sum(view.copy(), axis);
                for (long i = 0; i < g.size; i++)
                {
                    var cg = (Complex)g.GetAtIndex(i);
                    var ce = (Complex)e.GetAtIndex(i);
                    Assert.AreEqual(ce.Real, cg.Real, 1e-9, $"complex sum axis{axis} [{i}] re");
                    Assert.AreEqual(ce.Imaginary, cg.Imaginary, 1e-9, $"complex sum axis{axis} [{i}] im");
                }
            }
        }
    }

    // Flat (axis=None) reduction of a BROADCAST view (stride-0 axis). The coordinate-walk
    // kernel was ~50× NumPy here (the bcast_reduce pathology canary); the fix materializes
    // the broadcast once and takes the fast contiguous kernel. Result must equal reducing the
    // materialized contiguous copy (and, for the integer/min/max cases, be exact).
    [TestMethod]
    public void Flat_Reduce_OverBroadcastView_MatchesMaterializedCopy()
    {
        // a (1,8) broadcast up to (64,8): every row identical → axis0 is stride-0.
        var row = (np.arange(8).astype(NPTypeCode.Double) % 5) + 1;        // [1,2,3,4,5,1,2,3]
        var bc = np.broadcast_to(row.reshape(1, 8), new Shape(64, 8));
        Assert.IsTrue(bc.Shape.IsBroadcasted, "precondition: view must be broadcasted");

        double sExp = 64.0 * (double)np.sum(row);
        Assert.AreEqual(sExp, D(np.sum(bc).GetAtIndex(0)), 1e-9, "broadcast flat sum");
        Assert.AreEqual((double)np.amin(row), D(np.amin(bc).GetAtIndex(0)), 0, "broadcast flat min (exact)");
        Assert.AreEqual((double)np.amax(row), D(np.amax(bc).GetAtIndex(0)), 0, "broadcast flat max (exact)");
        Assert.AreEqual((double)np.mean(row), D(np.mean(bc).GetAtIndex(0)), 1e-9, "broadcast flat mean");

        // every op must equal reducing the materialized contiguous copy, across dtypes
        var copy = bc.copy();
        foreach (var dt in new[] { NPTypeCode.Double, NPTypeCode.Single, NPTypeCode.Int32, NPTypeCode.Int64 })
        {
            var bcd = np.broadcast_to(((np.arange(8) % 5) + 1).astype(dt).reshape(1, 8), new Shape(64, 8));
            var cpy = bcd.copy();
            Assert.AreEqual(D(np.sum(cpy).GetAtIndex(0)), D(np.sum(bcd).GetAtIndex(0)), 1e-6, $"{dt} bcast sum");
            Assert.AreEqual(D(np.amin(cpy).GetAtIndex(0)), D(np.amin(bcd).GetAtIndex(0)), 0, $"{dt} bcast min");
            Assert.AreEqual(D(np.amax(cpy).GetAtIndex(0)), D(np.amax(bcd).GetAtIndex(0)), 0, $"{dt} bcast max");
        }
    }

    [DataTestMethod]
    [DataRow(NPTypeCode.Double)]
    [DataRow(NPTypeCode.Single)]
    [DataRow(NPTypeCode.Int32)]
    [DataRow(NPTypeCode.Int64)]
    [DataRow(NPTypeCode.Decimal)]
    [DataRow(NPTypeCode.Half)]
    public void Reduce_OverOffsetAndNegativeStrideViews_EqualsContiguousCopy(NPTypeCode dt)
    {
        // distinct positive values (1..24) so prod has no zero short-circuit and
        // every group's min/max/sum/prod is unambiguous.
        var a = (np.arange(24) + 1).astype(dt).reshape(4, 6);
        var views = new (string name, NDArray v)[]
        {
            ("rev", a["::-1, :"]),
            ("revcol", a[":, ::-1"]),
            ("slice", a["1:4, 1:5"]),
            ("transpose", a.T),
            ("forder", a.copy(order: 'F')),
        };
        string[] ops = { "sum", "min", "max", "mean", "prod" };
        double tol = dt == NPTypeCode.Half ? 5e-2 : dt == NPTypeCode.Single ? 1e-3 : 1e-9;

        foreach (var (name, v) in views)
        {
            var baseline = v.copy(); // C-contiguous, offset 0 — independently-correct reference
            for (int axis = 0; axis <= 1; axis++)
            {
                foreach (var op in ops)
                {
                    NDArray g = Reduce(op, v, axis);
                    NDArray e = Reduce(op, baseline, axis);
                    Assert.AreEqual(e.size, g.size, $"{dt} {op} {name} axis{axis}: size");
                    for (long i = 0; i < g.size; i++)
                    {
                        double dg = D(g.GetAtIndex(i));
                        double de = D(e.GetAtIndex(i));
                        // Half prod(1..24) overflows to inf — view and copy must agree
                        // on the SAME inf/NaN; finite values compare within tolerance.
                        bool eq = (double.IsNaN(de) && double.IsNaN(dg))
                               || (double.IsInfinity(de) && double.IsInfinity(dg) && Math.Sign(de) == Math.Sign(dg))
                               || Math.Abs(dg - de) <= tol * (1 + Math.Abs(de));
                        Assert.IsTrue(eq, $"{dt} {op} {name} axis{axis} [{i}]: view {dg} != copy {de}");
                    }
                }
            }
        }
    }

    private static NDArray Reduce(string op, NDArray a, int axis) => op switch
    {
        "sum" => np.sum(a, axis),
        "min" => np.amin(a, axis),
        "max" => np.amax(a, axis),
        "mean" => np.mean(a, axis),
        "prod" => np.prod(a, axis),
        _ => throw new ArgumentException(op),
    };

    /// <summary>
    /// Pins the per-dtype amin/amax-along-axis specialization (catastrophic-slowness fix):
    /// bool/char have no fractional/NaN domain and a total order identical to their unsigned
    /// integer bit pattern, so they reinterpret to the byte / uint16 SIMD reducer instead of
    /// the scalar double-bridge (~76× bool, ~219× char). Half routes to a boxing-free direct
    /// loop (no double round-trip). All three previously went through CombineScalarsPromoted's
    /// per-element ConvertToDouble → Math.Min → ConvertFromDouble bridge.
    ///
    /// This guards the two correctness invariants the speed path must preserve:
    ///   • char compares as UNSIGNED 16-bit (0x8000 &gt; 0x41, 0xFFFF is the max) — a signed
    ///     reinterpret would invert the top half of the range.
    ///   • half PROPAGATES NaN (a group containing NaN reduces to NaN), matching NumPy.
    /// Expected values are from NumPy 2.4.2.
    /// </summary>
    [TestMethod]
    public void CharAndHalf_AxisMinMax_PerDtypeSpecialization_MatchesNumPy()
    {
        // char spanning the uint16 range - 0x8000/0xFFFF must out-rank 'A' (0x41).
        var c = np.array(new[] { (char)0x0041, (char)0xFFFF, (char)0x0001, (char)0x8000, (char)0x0002, (char)0x7FFF }).reshape(2, 3);
        CollectionAssert.AreEqual(new[] { (char)0x8000, (char)0xFFFF, (char)0x7FFF }, CharArr(np.amax(c, 0)), "char amax axis0");
        CollectionAssert.AreEqual(new[] { (char)0x0041, (char)0x0002, (char)0x0001 }, CharArr(np.amin(c, 0)), "char amin axis0");
        CollectionAssert.AreEqual(new[] { (char)0xFFFF, (char)0x8000 }, CharArr(np.amax(c, 1)), "char amax axis1");
        CollectionAssert.AreEqual(new[] { (char)0x0001, (char)0x0002 }, CharArr(np.amin(c, 1)), "char amin axis1");

        // half NaN propagation: any group containing NaN reduces to NaN (NumPy parity).
        double n = double.NaN;
        var h = np.array(new[] { 1.0, n, 3.0, n, 2.0, -1.0 }).astype(NPTypeCode.Half).reshape(2, 3);
        AssertHalf(np.amax(h, 0), new[] { n, n, 3.0 }, "half amax axis0");
        AssertHalf(np.amin(h, 0), new[] { n, n, -1.0 }, "half amin axis0");
        AssertHalf(np.amax(h, 1), new[] { n, n }, "half amax axis1");
        AssertHalf(np.amin(h, 1), new[] { n, n }, "half amin axis1");

        // bool reinterpret-to-byte path: all-False group → False (no double-bridge True bug).
        var b = np.array(new bool[] { true, false, true, false, false, false }).reshape(2, 3);
        CollectionAssert.AreEqual(new[] { true, false, true }, BoolArr(np.amax(b, 0)), "bool amax axis0");
        CollectionAssert.AreEqual(new[] { false, false, false }, BoolArr(np.amin(b, 0)), "bool amin axis0");
    }

    /// <summary>
    /// Same per-dtype specialization, but for the FLAT (axis=None) reduction path — which
    /// dispatches separately (Max/MinElementwiseCharFallback / ...HalfFallback) from the axis
    /// path. char routes through the uint16 SIMD reducer via a zero-copy view (~110×); half uses
    /// a boxing-free direct-Half scan (~4×, and ~2.4× faster than NumPy, which widens f16→float).
    /// Invariants: char unsigned-16-bit ordering, half NaN propagation. Values from NumPy 2.4.2.
    /// </summary>
    [TestMethod]
    public void CharAndHalf_FlatMinMax_MatchesNumPy()
    {
        // char flat — 0x8000/0xFFFF must out-rank 'A'=0x41 (unsigned order).
        var c = np.array(new[] { (char)0x0041, (char)0xFFFF, (char)0x0001, (char)0x8000, (char)0x0002, (char)0x7FFF });
        Assert.AreEqual((char)0x0001, (char)np.amin(c).GetAtIndex(0), "char flat amin");
        Assert.AreEqual((char)0xFFFF, (char)np.amax(c).GetAtIndex(0), "char flat amax");
        // a strided (negative-stride) char view must reduce identically (view preserves strides).
        var cr = c["::-1"];
        Assert.AreEqual((char)0x0001, (char)np.amin(cr).GetAtIndex(0), "char flat amin (reversed view)");
        Assert.AreEqual((char)0xFFFF, (char)np.amax(cr).GetAtIndex(0), "char flat amax (reversed view)");

        // half flat — NaN propagates (a NaN anywhere → NaN result), then a clean no-NaN case.
        double n = double.NaN;
        var h = np.array(new[] { 1.0, n, 3.0, n, 2.0, -1.0 }).astype(NPTypeCode.Half);
        Assert.IsTrue(double.IsNaN(D(np.amin(h).GetAtIndex(0))), "half flat amin NaN-propagate");
        Assert.IsTrue(double.IsNaN(D(np.amax(h).GetAtIndex(0))), "half flat amax NaN-propagate");
        var hf = np.array(new[] { 1.0, 3.0, 2.0, -1.0 }).astype(NPTypeCode.Half);
        Assert.AreEqual(-1.0, D(np.amin(hf).GetAtIndex(0)), 1e-3, "half flat amin");
        Assert.AreEqual(3.0, D(np.amax(hf).GetAtIndex(0)), 1e-3, "half flat amax");
    }

    private static char[] CharArr(NDArray a)
    {
        var r = new char[a.size];
        for (long i = 0; i < a.size; i++) r[i] = (char)a.GetAtIndex(i);
        return r;
    }

    private static void AssertHalf(NDArray got, double[] expected, string ctx)
    {
        Assert.AreEqual(expected.Length, (int)got.size, $"{ctx}: size");
        for (long i = 0; i < expected.Length; i++)
        {
            double g = D(got.GetAtIndex(i));
            if (double.IsNaN(expected[i]))
                Assert.IsTrue(double.IsNaN(g), $"{ctx}[{i}]: expected NaN, got {g}");
            else
                Assert.AreEqual(expected[i], g, 1e-3, $"{ctx}[{i}]");
        }
    }
}
