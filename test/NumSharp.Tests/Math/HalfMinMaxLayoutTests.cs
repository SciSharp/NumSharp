using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.MathTests;

/// <summary>
/// Pins float16 <c>np.max</c> / <c>np.min</c> / <c>np.nanmax</c> / <c>np.nanmin</c> BITS on every layout, flat and per axis,
/// and <c>np.ptp</c> (their difference), against NumPy 2.4.2 (every expected pattern was produced by running the same call
/// on NumPy 2.4.2).
/// </summary>
/// <remarks>
/// <para>
/// NumPy's HALF_maximum / HALF_minimum / HALF_fmax / HALF_fmin are sequential SELECTIONS, so WHICH NaN comes back (payload,
/// sign, a signalling NaN unquieted) and which ±0 survives depend on the reduction's visiting order: each axis in increasing
/// logical index (NPY_ITER_DONT_NEGATE_STRIDES), a flat reduction in K order. Before 2026-10-03 the axis kernel let a LATER
/// NaN replace an earlier one, the flat non-contiguous path returned the canonical <see cref="Half.NaN"/> for any NaN, and
/// the flat walk visited reversed / permuted views in another order — values right, bits wrong; the oracle tokenizes NaN,
/// so nothing saw it.
/// </para>
/// <para>
/// The data uses distinct NaNs (0x7e01, 0xfe02, signalling 0x7c04, …) so every wrong "first" shows up as a different bit
/// pattern. The larger fixtures reach the walks' SIMD paths: rows of 100 (forward and reversed cores, NaN at every
/// unroll seam), a 3×4100 slab (the 4096-lane block seam), a stepped slab (the gather), rows of 3 (the branch-free fold).
/// </para>
/// </remarks>
[TestClass]
public class HalfMinMaxLayoutTests
{
    /// <summary>A 1-D float16 array holding exactly <paramref name="bits"/>.</summary>
    private static NDArray H(params ushort[] bits)
    {
        var nd = np.zeros(new Shape(bits.Length), NPTypeCode.Half);
        unsafe
        {
            ushort* p = (ushort*)nd.Address;
            for (int i = 0; i < bits.Length; i++)
                p[i] = bits[i];
        }

        return nd;
    }

    /// <summary>A float16 array of <paramref name="rows"/> × <paramref name="cols"/> whose element (r, c) is <c>f(r, c)</c>.</summary>
    private static NDArray H2(int rows, int cols, Func<int, int, ushort> f)
    {
        var bits = new ushort[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                bits[r * cols + c] = f(r, c);
        return H(bits).reshape(rows, cols);
    }

    /// <summary>The result's bits in C order, as "0x7e01 0xfe02 …" for a readable failure.</summary>
    private static string Hex(NDArray r)
    {
        var bytes = np.ascontiguousarray(r).tobytes();
        return string.Join(" ", Enumerable.Range(0, bytes.Length / 2).Select(i => $"0x{BitConverter.ToUInt16(bytes, 2 * i):x4}"));
    }

    private static string Hex(params ushort[] bits) => string.Join(" ", bits.Select(b => $"0x{b:x4}"));

    private static void AssertBits(NDArray r, string because, params ushort[] expected) => Hex(r).Should().Be(Hex(expected), because);

    private static NDArray Reduce(string op, NDArray v, int? axis, bool keepdims = false) => op switch
    {
        "max" => np.max(v, axis, keepdims),
        "min" => np.min(v, axis, keepdims),
        "nanmax" => np.nanmax(v, axis, keepdims),
        _ => np.nanmin(v, axis, keepdims),
    };

    /// <summary>The 3×4 fixture: a distinct NaN in most rows and columns, signalling ones included (0x7c04, 0x7c07).</summary>
    private static NDArray Small() => H(
        0x3c00, 0x7e01, 0x4000, 0xfe02,
        0xfc03, 0x3800, 0x7c04, 0x4200,
        0x7e05, 0xfd06, 0xbc00, 0x7c07).reshape(3, 4);

    [TestMethod]
    public void SmallMatrix_EveryLayoutOpAndAxis_MatchesNumPyBits()
    {
        var m = Small();
        var views = new (string Name, NDArray View)[]
        {
            ("C", m), ("T", m.T), ("rev0", m["::-1"]), ("rev1", m[":, ::-1"]), ("T_rev", m.T["::-1, ::-1"]), ("step", m[":, ::2"]),
        };
        // (view, op, axis) → NumPy 2.4.2's bits.
        var expected = new (string, string, int?, ushort[])[]
        {
            ("C", "max", null, new ushort[] { 0x7e01 }),
            ("C", "max", 0, new ushort[] { 0xfc03, 0x7e01, 0x7c04, 0xfe02 }),
            ("C", "max", 1, new ushort[] { 0x7e01, 0xfc03, 0x7e05 }),
            ("C", "min", 0, new ushort[] { 0xfc03, 0x7e01, 0x7c04, 0xfe02 }),
            ("C", "nanmax", null, new ushort[] { 0x4200 }),
            ("C", "nanmax", 0, new ushort[] { 0x3c00, 0x3800, 0x4000, 0x4200 }),
            ("C", "nanmin", 1, new ushort[] { 0x3c00, 0x3800, 0xbc00 }),
            ("T", "max", null, new ushort[] { 0x7e01 }),
            ("T", "max", 0, new ushort[] { 0x7e01, 0xfc03, 0x7e05 }),
            ("T", "max", 1, new ushort[] { 0xfc03, 0x7e01, 0x7c04, 0xfe02 }),
            ("T", "nanmin", 0, new ushort[] { 0x3c00, 0x3800, 0xbc00 }),
            ("rev0", "max", null, new ushort[] { 0x7e05 }),
            ("rev0", "max", 0, new ushort[] { 0x7e05, 0xfd06, 0x7c04, 0x7c07 }),
            ("rev0", "min", 1, new ushort[] { 0x7e05, 0xfc03, 0x7e01 }),
            ("rev0", "nanmax", 1, new ushort[] { 0xbc00, 0x4200, 0x4000 }),
            ("rev1", "max", null, new ushort[] { 0xfe02 }),
            ("rev1", "max", 0, new ushort[] { 0xfe02, 0x7c04, 0x7e01, 0xfc03 }),
            ("rev1", "min", 1, new ushort[] { 0xfe02, 0x7c04, 0x7c07 }),
            ("rev1", "nanmin", 0, new ushort[] { 0x4200, 0xbc00, 0x3800, 0x3c00 }),
            ("T_rev", "max", null, new ushort[] { 0x7c07 }),
            ("T_rev", "max", 0, new ushort[] { 0x7c07, 0x7c04, 0xfe02 }),
            ("T_rev", "min", 1, new ushort[] { 0x7c07, 0x7c04, 0xfd06, 0x7e05 }),
            ("T_rev", "nanmin", 1, new ushort[] { 0x4200, 0xbc00, 0x3800, 0x3c00 }),
            ("step", "max", null, new ushort[] { 0xfc03 }),
            ("step", "max", 0, new ushort[] { 0xfc03, 0x7c04 }),
            ("step", "min", 1, new ushort[] { 0x3c00, 0xfc03, 0x7e05 }),
            ("step", "nanmax", 1, new ushort[] { 0x4000, 0xfc03, 0xbc00 }),
        };
        foreach (var (name, op, axis, bits) in expected)
        {
            var view = views.Single(v => v.Name == name).View;
            using var r = Reduce(op, view, axis);
            AssertBits(r, $"np.{op}({name}, axis={axis?.ToString() ?? "None"})", bits);
        }
    }

    [TestMethod]
    public void SimdRows_ForwardAndReversed_KeepTheFirstVisitedNaN()
    {
        // Rows of 100 (the SIMD cores): the first NaN of row r sits at an unroll / vector seam, a second, different NaN 37
        // elements later. Reversed rows are visited from the HIGH address down, so their first NaN is the other one.
        int[] pos = { 0, 15, 16, 31, 32, 63, 64, 65, 99 };
        var b = H2(pos.Length, 100, (r, c) =>
        {
            if (c == pos[r]) return (ushort)(0x7e00 | (r + 1));
            if (c == (pos[r] + 37) % 100) return (ushort)(0xfe00 | (r + 0x20));
            return (ushort)(0x3c00 + ((r * 37 + c * 11) % 0x3ff));
        });
        foreach (var op in new[] { "max", "min" })
        {
            using var fwd = Reduce(op, b, 1);
            AssertBits(fwd, $"{op} forward rows", 0x7e01, 0x7e02, 0x7e03, 0x7e04, 0x7e05, 0xfe25, 0xfe26, 0xfe27, 0xfe28);
            using var rev = Reduce(op, b[":, ::-1"], 1);
            AssertBits(rev, $"{op} reversed rows", 0xfe20, 0xfe21, 0xfe22, 0xfe23, 0xfe24, 0x7e06, 0x7e07, 0x7e08, 0x7e09);
            using var flat = Reduce(op, b, null);
            AssertBits(flat, $"{op} flat C", 0x7e01);
            using var flatT = Reduce(op, b.T, null);
            AssertBits(flatT, $"{op} flat of the transpose (K order = memory order)", 0x7e01);
            using var flatRev = Reduce(op, b[":, ::-1"], null);
            AssertBits(flatRev, $"{op} flat of reversed rows (each run visited backwards)", 0xfe20);
        }
    }

    [TestMethod]
    public void StridedLongRows_OutOfLineFold_FirstNaNFirstZeroAndAllNaN()
    {
        // (3, 80)[:, ::2]: rows of 40 at stride 2 — longer than the branch-free short-row fold, not a ±1 run, so the
        // out-of-line priority-key fold (HalfStridedWinnerBits) decides every row. Row 0: two NaNs, the first wins. Row 1:
        // negatives with a -0 before a +0 (max keeps the first zero). Row 2: all NaN (fmax keeps the first element).
        var a = H2(3, 80, (r, c) =>
        {
            if (r == 0 && c == 6) return 0x7d01;
            if (r == 0 && c == 30) return 0xfe02;
            if (r == 1 && c == 10) return 0x8000;
            if (r == 1 && c == 40) return 0x0000;
            if (r == 1) return (ushort)(c % 2 == 0 ? 0xbc00 + (c % 7) : 0x3c00);
            if (r == 2) return (ushort)(c % 2 == 0 ? 0x7c00 | (1 + c) : 0x3c00);
            return (ushort)(0x3c00 + ((r * 13 + c * 5) % 0x200));
        });
        var v = a[":, ::2"];
        using (var r = np.max(v, 1)) AssertBits(r, "max of stepped rows", 0x7d01, 0x8000, 0x7c01);
        using (var r = np.min(v, 1)) AssertBits(r, "min of stepped rows", 0x7d01, 0xbc06, 0x7c01);
        using (var r = np.nanmax(v, 1)) AssertBits(r, "nanmax of stepped rows", 0x3d86, 0x8000, 0x7c01);
        using (var r = np.nanmin(v, 1)) AssertBits(r, "nanmin of stepped rows", 0x3c00, 0xbc06, 0x7c01);
    }

    [TestMethod]
    public void ReversedSimdRows_RescanFromTheTop_ForEveryRule()
    {
        // Rows of 64 visited backwards (the SIMD cores' reverse rescans). Row 0: two NaNs in ONE 16-lane vector (columns 50
        // and 53) — the reversed walk meets 53 first. Row 1: negatives and two zeros (+0 at 40 is met before -0 at 20).
        // Row 2: all NaN (fmax keeps the first VISITED element, column 63). Row 3: positives and two zeros (-0 at 60 first).
        var b = H2(4, 64, (r, c) => (r, c) switch
        {
            (0, 50) => 0x7e0a,
            (0, 53) => 0xfe0b,
            (1, 20) => 0x8000,
            (1, 40) => 0x0000,
            (1, _) => (ushort)(0xbc00 | (c & 0xf)),
            (2, _) => (ushort)(0x7c00 | (0x40 + c)),
            (3, 5) => 0x0000,
            (3, 60) => 0x8000,
            (3, _) => (ushort)(0x3c00 | (c & 0xf)),
            _ => (ushort)(0x4000 + ((r * 7 + c * 3) % 0x100)),
        });
        var v = b[":, ::-1"];
        using (var r = np.max(v, 1)) AssertBits(r, "max of reversed rows", 0xfe0b, 0x0000, 0x7c7f, 0x3c0f);
        using (var r = np.min(v, 1)) AssertBits(r, "min of reversed rows", 0xfe0b, 0xbc0f, 0x7c7f, 0x8000);
        using (var r = np.nanmax(v, 1)) AssertBits(r, "nanmax of reversed rows", 0x40bd, 0x0000, 0x7c7f, 0x3c0f);
        using (var r = np.nanmin(v, 1)) AssertBits(r, "nanmin of reversed rows", 0x4000, 0xbc0f, 0x7c7f, 0x8000);
    }

    /// <summary>The 3×4100 slab: NaNs of rows 1 and 2 straddle the 4096-lane block seam; column 0 holds a signalling NaN.</summary>
    private static NDArray Slab() => H2(3, 4100, (r, c) =>
    {
        if (r == 2 && c == 0) return 0x7c01;
        if (c is 4094 or 4095 or 4096 or 4097 or 4099)
        {
            if (r == 1) return (ushort)(0x7e00 | (c & 0xff));
            if (r == 2) return (ushort)(0xfe00 | ((c + 1) & 0xff));
        }

        return (ushort)((r == 0 ? 0x3c00 : r == 1 ? 0x3800 : 0x4000) + (c % 0x3ff));
    });

    private static NDArray Pick(NDArray r, params int[] at) => np.array(at.Select(i => r.GetAtIndex<Half>(i)).ToArray());

    [TestMethod]
    public void Slab_AcrossTheBlockSeam_AndReversed_MatchesNumPyBits()
    {
        var s = Slab();
        int[] at = { 0, 1, 4093, 4094, 4095, 4096, 4097, 4098, 4099 };
        (string Op, ushort[] Fwd, ushort[] Rev)[] cases =
        {
            ("max", new ushort[] { 0x7c01, 0x4001, 0x4001, 0x7efe, 0x7eff, 0x7e00, 0x7e01, 0x4006, 0x7e03 },
                new ushort[] { 0x7e03, 0x4006, 0x4006, 0x4005, 0x4004, 0x4003, 0x4002, 0x4001, 0x7c01 }),
            ("min", new ushort[] { 0x7c01, 0x3801, 0x3801, 0x7efe, 0x7eff, 0x7e00, 0x7e01, 0x3806, 0x7e03 },
                new ushort[] { 0x7e03, 0x3806, 0x3806, 0x3805, 0x3804, 0x3803, 0x3802, 0x3801, 0x7c01 }),
            ("nanmax", new ushort[] { 0x3c00, 0x4001, 0x4001, 0x3c02, 0x3c03, 0x3c04, 0x3c05, 0x4006, 0x3c07 },
                new ushort[] { 0x3c07, 0x4006, 0x4006, 0x4005, 0x4004, 0x4003, 0x4002, 0x4001, 0x3c00 }),
            ("nanmin", new ushort[] { 0x3800, 0x3801, 0x3801, 0x3c02, 0x3c03, 0x3c04, 0x3c05, 0x3806, 0x3c07 },
                new ushort[] { 0x3c07, 0x3806, 0x3806, 0x3805, 0x3804, 0x3803, 0x3802, 0x3801, 0x3800 }),
        };
        foreach (var (op, fwd, rev) in cases)
        {
            using var r = Reduce(op, s, 0);
            AssertBits(Pick(r, at), $"{op} along axis 0 (contiguous slab rows)", fwd);
            using var rr = Reduce(op, s[":, ::-1"], 0);
            AssertBits(Pick(rr, at), $"{op} along axis 0 of [:, ::-1] (a reversed inner run, gathered)", rev);
        }
    }

    [TestMethod]
    public void Slab_SteppedInnerRun_IsGatheredAndExact()
    {
        var g = H2(4, 300, (r, c) => (r, c) switch
        {
            (1, 10) => 0x7e11,
            (3, 10) => 0xfe33,
            (2, 20) => 0xfc22,
            (3, 20) => 0x7e44,
            (3, 21) => 0x7e55,      // an odd column: outside the stepped view
            _ => (ushort)(0x3c00 + ((r * 101 + c * 7) % 0x3ff)),
        });
        var v = g[":, ::2"];
        using var mx = np.max(v, 0);
        AssertBits(Pick(mx, 5, 10, 0), "max along axis 0 of [:, ::2]", 0x7e11, 0xfc22, 0x3d2f);
        using var mn = np.min(v, 0);
        AssertBits(Pick(mn, 5, 10, 0), "min along axis 0 of [:, ::2]", 0x7e11, 0xfc22, 0x3c00);
    }

    [TestMethod]
    public void ShortRows_BranchFreeFold_FirstNaNAndFirstZero()
    {
        var sr = H(0x8000, 0x0000, 0xbc00, 0x0000, 0x8000, 0x8000, 0x7e01, 0x7e02, 0x3c00,
                   0x3c00, 0xfe03, 0x7c04, 0x4000, 0x4000, 0x3c00, 0xbc00, 0x8000, 0x0000).reshape(6, 3);
        using (var r = np.max(sr, 1)) AssertBits(r, "max of rows of 3", 0x8000, 0x0000, 0x7e01, 0xfe03, 0x4000, 0x8000);
        using (var r = np.min(sr, 1)) AssertBits(r, "min of rows of 3", 0xbc00, 0x0000, 0x7e01, 0xfe03, 0x3c00, 0xbc00);
        using (var r = np.nanmax(sr, 1)) AssertBits(r, "nanmax of rows of 3", 0x8000, 0x0000, 0x3c00, 0x3c00, 0x4000, 0x8000);
        using (var r = np.nanmin(sr, 1)) AssertBits(r, "nanmin of rows of 3", 0xbc00, 0x0000, 0x3c00, 0x3c00, 0x3c00, 0xbc00);
    }

    [TestMethod]
    public void SignedZeroTies_KeepTheFirstZeroInVisitingOrder()
    {
        var z = H(0x8000, 0x0000, 0xbc00, 0x0000, 0x0000, 0x8000, 0x8000, 0xbc00).reshape(2, 4);
        using (var r = np.max(z)) AssertBits(r, "max flat C", 0x8000);
        using (var r = np.max(z, 0)) AssertBits(r, "max axis 0", 0x8000, 0x0000, 0x8000, 0x0000);
        using (var r = np.max(z, 1)) AssertBits(r, "max axis 1", 0x8000, 0x0000);
        using (var r = np.max(z.T)) AssertBits(r, "max flat of the transpose (memory order)", 0x8000);
        using (var r = np.max(z[":, ::-1"])) AssertBits(r, "max flat of reversed rows", 0x0000);
        using (var r = np.max(z[":, ::-1"], 0)) AssertBits(r, "max axis 0 of reversed rows", 0x0000, 0x8000, 0x0000, 0x8000);
        using (var r = np.max(z[":, ::-1"], 1)) AssertBits(r, "max axis 1 of reversed rows", 0x0000, 0x8000);

        var zp = H(0x0000, 0x8000, 0x3c00, 0x8000, 0x0000, 0x4000).reshape(2, 3);
        using (var r = np.min(zp, 0)) AssertBits(r, "min axis 0", 0x0000, 0x8000, 0x3c00);
        using (var r = np.min(zp.T, 1)) AssertBits(r, "min axis 1 of the transpose", 0x0000, 0x8000, 0x3c00);
        using (var r = np.min(zp.T)) AssertBits(r, "min flat of the transpose", 0x0000);
    }

    [TestMethod]
    public void NanMaxNanMin_AllNaNSlice_ReturnsItsFirstElementVerbatim()
    {
        var an = H(0xfe01, 0x7e02, 0x3c00, 0x7c03, 0xfc04, 0x7e05).reshape(2, 3);
        foreach (var op in new[] { "nanmax", "nanmin" })
        {
            using (var r = Reduce(op, an, 0)) AssertBits(r, $"{op} axis 0", 0xfe01, 0x7e02, 0x3c00);
            using (var r = Reduce(op, an, 1)) AssertBits(r, $"{op} axis 1", 0x3c00, 0x7c03);
            using (var r = Reduce(op, an.T, 0)) AssertBits(r, $"{op} axis 0 of the transpose", 0x3c00, 0x7c03);
            using (var r = Reduce(op, an.T, 1)) AssertBits(r, $"{op} axis 1 of the transpose", 0xfe01, 0x7e02, 0x3c00);
        }
    }

    [TestMethod]
    public void BroadcastInputs_FirstNaNPerOutput()
    {
        var bc = np.broadcast_to(H(0x3c00, 0x7e01, 0xfe02, 0x4000), new Shape(3, 4));
        foreach (var op in new[] { "max", "min" })
        {
            using (var r = Reduce(op, bc, null)) AssertBits(r, $"{op} flat", 0x7e01);
            using (var r = Reduce(op, bc, 0)) AssertBits(r, $"{op} along the broadcast axis", 0x3c00, 0x7e01, 0xfe02, 0x4000);
            using (var r = Reduce(op, bc, 1)) AssertBits(r, $"{op} along the real axis", 0x7e01, 0x7e01, 0x7e01);
        }

        var bc2 = np.broadcast_to(H(0x7e09, 0x3c00, 0xfe0a).reshape(3, 1), new Shape(3, 4));
        using (var r = np.max(bc2)) AssertBits(r, "max flat of a broadcast column", 0x7e09);
        using (var r = np.max(bc2.T)) AssertBits(r, "max flat of its transpose", 0x7e09);
    }

    [TestMethod]
    public void Keepdims_ShapesAndBits()
    {
        var m = Small();
        using var flat = np.max(m, null, keepdims: true);
        flat.shape.Should().Equal(1L, 1L);
        AssertBits(flat, "flat keepdims", 0x7e01);
        using var ax = np.max(m, 1, keepdims: true);
        ax.shape.Should().Equal(3L, 1L);
        AssertBits(ax, "axis-1 keepdims", 0x7e01, 0xfc03, 0x7e05);
    }

    [TestMethod]
    public void Permuted3D_EveryAxis()
    {
        var bits = Enumerable.Range(0, 24).Select(i => (ushort)(0x3c00 + i)).ToArray();
        bits[5] = 0x7e01;
        bits[9] = 0xfe02;
        bits[17] = 0x7c03;
        bits[22] = 0xfc04;
        var v = H(bits).reshape(2, 3, 4).transpose(new[] { 2, 0, 1 });
        using (var r = np.max(v)) AssertBits(r, "flat (K order = memory order)", 0x7e01);
        using (var r = np.max(v, 0)) AssertBits(r, "axis 0", 0x3c03, 0x7e01, 0xfe02, 0x3c0f, 0x7c03, 0xfc04);
        using (var r = np.max(v, 1))
            AssertBits(r, "axis 1", 0x3c0c, 0x3c10, 0x3c14, 0x3c0d, 0x7e01, 0xfe02, 0x3c0e, 0x3c12, 0xfc04, 0x3c0f, 0x3c13, 0x3c17);
        using (var r = np.max(v, 2)) AssertBits(r, "axis 2", 0x3c08, 0x3c14, 0x7e01, 0x7c03, 0x3c0a, 0xfc04, 0x3c0b, 0x3c17);
    }

    [TestMethod]
    public void DtypeFloat16_OverAFloat32Input_CastsThenReduces()
    {
        // NumSharp's dtype= extension computes in the requested dtype, as the axis path's casting kernel does. The flat path
        // used to hand the float32 buffer to the float16 kernel unconverted (5.0 here, 0.0 for the stepped view).
        var f = np.array(new float[] { 1f, 5f, -2f, 3f, 100f, 0.5f, 7f, 8f });
        using (var r = np.max(f, null, false, np.float16)) AssertBits(r, "max(float32, dtype=float16)", 0x5640);
        using (var r = np.max(f["::2"], null, false, np.float16)) AssertBits(r, "max(float32[::2], dtype=float16)", 0x5640);
        using (var r = np.min(f, null, false, np.float16)) AssertBits(r, "min(float32, dtype=float16)", 0xc000);
    }

    [TestMethod]
    public void Evaluate_FlatMaxMin_OverFloat16_FirstNaN()
    {
        var a = H(0x3c00, 0xfe01, 0x4000, 0x7e02, 0x7c03, 0x3800);
        NDExpr.ExactMinMaxRuns = 0;
        using (var r = np.evaluate(NDExpr.Max((NDExpr)a))) AssertBits(r, "evaluate Max(leaf)", 0xfe01);
        Assert.AreEqual(1, NDExpr.ExactMinMaxRuns, "the float16 leaf takes NumPy's sequential fold");
        using (var r = np.evaluate(NDExpr.Min((NDExpr)a))) AssertBits(r, "evaluate Min(leaf)", 0xfe01);
        using (var r = np.evaluate(NDExpr.Max((NDExpr)a.reshape(2, 3).T))) AssertBits(r, "evaluate Max(transposed leaf)", 0xfe01);
        // A computed child: NumPy materializes -a (negative flips the sign bit, payload kept) and reduces it in order.
        using (var r = np.evaluate(NDExpr.Max(-(NDExpr)a))) AssertBits(r, "evaluate Max(-a)", 0x7e01);
        using (var r = np.evaluate(NDExpr.Max(-(NDExpr)a.reshape(2, 3).T))) AssertBits(r, "evaluate Max(-a.T)", 0x7e01);
        // A stepped leaf is reduced through ITS strides: the skipped elements hold earlier NaNs that must not be read.
        var s = H(0x3c00, 0x7e01, 0x4000, 0xfe02, 0x3800, 0x4200, 0x7c03, 0x3c00);
        using (var r = np.evaluate(NDExpr.Max((NDExpr)s["::2"]))) AssertBits(r, "evaluate Max(stepped leaf)", 0x7c03);
        using (var r = np.evaluate(NDExpr.Min((NDExpr)s["::-2"]))) AssertBits(r, "evaluate Min(reversed stepped leaf)", 0xfe02);
    }

    [TestMethod]
    public void Ptp_IsTheDifferenceOfTheTwoWalks_FirstNaNQuieted()
    {
        // NumPy's ptp is subtract(maximum.reduce(a), minimum.reduce(a)) (numpy/_core/_methods.py _ptp). Both walks return the
        // SAME first NaN, and HALF_subtract computes in float32 and narrows once, which quiets it: the signalling 0x7c01
        // comes back as 0x7e01. Before 2026-10-03 ptp inherited the wrong walks (3,622 of a 6,034-case NumPy probe had
        // other NaN bits).
        var m = H(
            0x7c01, 0x3c00, 0x0000, 0xfc00,
            0xfe05, 0x8000, 0x8000, 0x4000,
            0x7e07, 0x0000, 0xbc00, 0x7c00).reshape(3, 4);
        using (var r = np.ptp(m, 0))
            AssertBits(r, "axis 0: column 0's first NaN quieted; 1 - (-0); +0 - (-1); +inf - (-inf)", 0x7e01, 0x3c00, 0x3c00, 0x7c00);
        using (var r = np.ptp(m, 1)) AssertBits(r, "axis 1: each row's first NaN, quieted", 0x7e01, 0xfe05, 0x7e07);
        using (var r = np.ptp(m)) AssertBits(r, "flat", 0x7e01);
        using (var r = np.ptp(m["::-1"], 0))
            AssertBits(r, "reversed rows, axis 0: the view's row 0 is visited first", 0x7e07, 0x3c00, 0x3c00, 0x7c00);
        using (var r = np.ptp(m["::-1"])) AssertBits(r, "reversed rows, flat", 0x7e07);
        using (var r = np.ptp(m.T, 0)) AssertBits(r, "transposed, axis 0", 0x7e01, 0xfe05, 0x7e07);
        using (var r = np.ptp(m.T)) AssertBits(r, "transposed, flat (K order = memory order)", 0x7e01);
        using (var kd = np.ptp(m, 1, null, keepdims: true))
        {
            kd.shape.Should().Equal(3L, 1L);
            AssertBits(kd, "axis-1 keepdims", 0x7e01, 0xfe05, 0x7e07);
        }

        using (var r = np.ptp(H(0x8000, 0x0000, 0x8000))) AssertBits(r, "(-0) - (-0): both walks keep the first zero", 0x0000);

        // A 70-element run reaches the SIMD cores, forward and reversed: the first NaN in the visiting direction wins.
        var run = Enumerable.Repeat((ushort)0x3c00, 70).ToArray();
        run[10] = 0xfc22;
        run[40] = 0x7c11;
        using (var r = np.ptp(H(run))) AssertBits(r, "run of 70", 0xfe22);
        using (var r = np.ptp(H(run)["::-1"])) AssertBits(r, "reversed run of 70", 0x7e11);
        using (var r = np.ptp(H(0x7bff, 0x0001, 0x3555))) AssertBits(r, "finite: rounded once from the float32 difference", 0x7bff);

        // inf - inf is the HARDWARE default NaN: negative on x86 (0xfe00, NumPy's win-amd64 answer), positive on arm64. This
        // is why the oracle compares float16 ptp by raw bytes on x86 hosts only.
        if (System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture
            is System.Runtime.InteropServices.Architecture.X64 or System.Runtime.InteropServices.Architecture.X86)
        {
            using var r = np.ptp(H(0x7c00));
            AssertBits(r, "inf - inf on x86", 0xfe00);
        }
    }

    [TestMethod]
    public unsafe void DirectAxisKernel_HalfMinMax_RunsTheExactWalk()
    {
        // The (Half, Half) Min / Max axis kernel key is no longer reached by np.max / np.min (they take the walk first), but
        // stays exact for any other caller: it adapts the kernel contract onto NumPyHalfMinMaxReduce.Axis.
        var kernel = DirectILKernelGenerator.TryGetAxisReductionKernel(
            new AxisReductionKernelKey(NPTypeCode.Half, NPTypeCode.Half, ReductionOp.Max, InnerAxisContiguous: false));
        kernel.Should().NotBeNull();
        var m = Small().T;      // (4, 3), strides (1, 4): reduce axis 1 → the first NaN of each column of the original
        using var o = np.zeros(new Shape(4), NPTypeCode.Half);
        var dims = m.Shape.dimensions;
        var strides = m.Shape.strides;
        var os = o.Shape.strides;
        fixed (long* pd = dims)
        fixed (long* ps = strides)
        fixed (long* po = os)
            kernel!((byte*)m.Address + m.Shape.offset * 2, o.Address, ps, pd, po, 1, dims[1], 2, 4);
        AssertBits(o, "the adapted Direct kernel", 0xfc03, 0x7e01, 0x7c04, 0xfe02);

        // 3-D over the LAST axis: the contract's output strides are per OUTPUT axis, so input axis d < axis reads
        // outputStrides[d] — a slip there writes outputs over each other. The engine's walk (NumPy-verified above) is the
        // expectation.
        var bits = Enumerable.Range(0, 24).Select(i => (ushort)(0x3c00 + (i * 7) % 24)).ToArray();
        bits[3] = 0x7e0c;
        bits[13] = 0xfe0d;
        var c3 = H(bits).reshape(2, 3, 4);
        using var o3 = np.zeros(new Shape(2, 3), NPTypeCode.Half);
        var d3 = c3.Shape.dimensions;
        var s3 = c3.Shape.strides;
        var os3 = o3.Shape.strides;
        fixed (long* pd = d3)
        fixed (long* ps = s3)
        fixed (long* po = os3)
            kernel!(c3.Address, o3.Address, ps, pd, po, 2, d3[2], 3, 6);
        using var expected = np.max(c3, 2);
        Hex(o3).Should().Be(Hex(expected), "the adapter over the last axis of a 3-D input equals the engine's walk");
    }
}
