using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the SIMD float|double → narrow-int cast (i8/u8/i16/u16/char) to NumPy 2.4.2.
///
/// These were the worst cells in the Phase-0 cast matrix — f32→i8 bottomed it at 0.09× NumPy,
/// because <c>ResolveStrategy</c> gave every srcFloat→dstInt pair except Single→Int32 a per-element
/// scalar loop. The fix (<c>DirectILKernelGenerator.Cast.FloatNarrow.cs</c>) is a
/// <c>cvtt(VCVTTPS2DQ/VCVTTPD2DQ) → truncating Vector.Narrow</c> chain, bit-exact with
/// <c>Converts.To{Narrow}(double) == unchecked((narrow)ToInt32(value))</c>.
///
/// The defining semantic — and what makes this NOT a saturating pack — is that NumPy float→narrow
/// **WRAPS** (takes the low bits of the int32 truncation), it does NOT clamp:
///   f32→i8: 128.5 → -128 (a saturating pack would give +127, wrong); 256 → 0; 300 → 44;
///           NaN / ±inf / overflow → 0 (int32 INT_MIN sentinel, low bits 0).
/// The oracle arrays below are produced by actual NumPy and pin exactly that.
/// </summary>
[TestClass]
public class FloatToNarrowCastParityTests
{
    // Inputs probed against NumPy (np.array(CASES, dtype).astype(narrow)).
    private static readonly double[] CASES =
    {
        128.5, -128.5, 255.9, 256.0, 300.0, -1.0, 65535.9, 65536.0, 70000.0,
        2147483648.0, 4294967296.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 1e20, -1e20
    };

    // NumPy 2.4.2 oracle outputs (identical for f32 and f64 sources — every CASES value is exactly
    // representable in float32, and the int32-intermediate wrap is width-driven, not precision-driven).
    private static readonly long[] ORACLE_I8  = { -128, -128, -1, 0, 44, -1, -1, 0, 112, 0, 0, 0, 0, 0, 0, 0 };
    private static readonly long[] ORACLE_U8  = {  128,  128, 255, 0, 44, 255, 255, 0, 112, 0, 0, 0, 0, 0, 0, 0 };
    private static readonly long[] ORACLE_I16 = {  128, -128, 255, 256, 300, -1, -1, 0, 4464, 0, 0, 0, 0, 0, 0, 0 };
    private static readonly long[] ORACLE_U16 = {  128, 65408, 255, 256, 300, 65535, 65535, 0, 4464, 0, 0, 0, 0, 0, 0, 0 };

    private static long L(NDArray a, int i) => Convert.ToInt64(a.GetAtIndex(i));

    [TestMethod]
    public void Oracle_Single_MatchesNumPy()
    {
        var f32 = np.array(CASES.Select(d => (float)d).ToArray());
        AssertOracle(f32.astype(NPTypeCode.SByte),  ORACLE_I8,  "f32->i8");
        AssertOracle(f32.astype(NPTypeCode.Byte),   ORACLE_U8,  "f32->u8");
        AssertOracle(f32.astype(NPTypeCode.Int16),  ORACLE_I16, "f32->i16");
        AssertOracle(f32.astype(NPTypeCode.UInt16), ORACLE_U16, "f32->u16");
    }

    [TestMethod]
    public void Oracle_Double_MatchesNumPy()
    {
        var f64 = np.array(CASES);
        AssertOracle(f64.astype(NPTypeCode.SByte),  ORACLE_I8,  "f64->i8");
        AssertOracle(f64.astype(NPTypeCode.Byte),   ORACLE_U8,  "f64->u8");
        AssertOracle(f64.astype(NPTypeCode.Int16),  ORACLE_I16, "f64->i16");
        AssertOracle(f64.astype(NPTypeCode.UInt16), ORACLE_U16, "f64->u16");
    }

    private static void AssertOracle(NDArray got, long[] want, string label)
    {
        Assert.AreEqual(want.Length, (int)got.size, $"{label}: size");
        for (int i = 0; i < want.Length; i++)
            Assert.AreEqual(want[i], L(got, i), $"{label}: CASES[{i}]={CASES[i]} expected {want[i]}, got {L(got, i)}");
    }

    // The SIMD bulk (multiple of 32/16) and the scalar tail must agree, across every memory layout.
    // The contiguous kernel is pinned to NumPy by the Oracle_* tests above; this guards that every
    // non-contiguous source (F, T, sliced, negrow, negcol, [:, ::2]) produces identical bytes —
    // i.e. that the strided driver's inner-contig Bulk, staged-strided Bulk, and scalar paths all
    // match the contiguous reference. Grid embeds the edge battery so views sample wrap/overflow.
    [TestMethod]
    public void AllLayouts_BitExact_VsContiguousMaterialization()
    {
        int R = 64, C = 64;
        var data = new float[R * C];
        var edge = CASES.Select(d => (float)d).ToArray();
        for (int i = 0; i < R * C; i++) data[i] = edge[(i * 13 + i / 5) % edge.Length];

        var gridF = np.array(data).reshape(R, C);
        var gridD = np.array(data.Select(x => (double)x).ToArray()).reshape(R, C);

        var dsts = new[] { NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Char };
        var layouts = new (string name, Func<NDArray, NDArray> f)[]
        {
            ("F",       g => g.copy(order: 'F')),
            ("T",       g => g.T),
            ("sliced",  g => g["1:" + (R - 1) + ", 1:" + (C - 1)]),
            ("negrow",  g => g["::-1, :"]),
            ("negcol",  g => g[":, ::-1"]),
            ("strided", g => g[":, ::2"]),
        };

        foreach (var grid in new[] { gridF, gridD })
        foreach (var (lname, lf) in layouts)
        foreach (var dst in dsts)
        {
            var view = lf(grid);
            var viaView = view.astype(dst);                 // strided kernel
            var viaContig = view.copy().astype(dst);        // materialize -> contiguous kernel (NumPy-pinned)
            Assert.AreEqual(viaContig.size, viaView.size, $"{grid.dtype.Name}|{lname}|{dst}: size");
            for (long i = 0; i < viaView.size; i++)
                Assert.AreEqual(L(viaContig, (int)i), L(viaView, (int)i),
                    $"{grid.dtype.Name}|{lname}|{dst}: element {i} (strided != contiguous)");
        }
    }
}
