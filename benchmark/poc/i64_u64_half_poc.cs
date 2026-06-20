#:property AllowUnsafeBlocks=true
// Standalone bit-exactness proof for the AVX2 i64/u64 -> f16 cast (bucket A).
// Reads NumPy reference vectors, runs SIMD-bulk + scalar, compares to NumPy.
//   Run: dotnet run -c Release - < benchmark/poc/i64_u64_half_poc.cs
using System;
using System.IO;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

const string DIR = @"K:\source\NumSharp\benchmark\poc\_xref";

unsafe {
    // ---------- i64 ----------
    long[] tv = ReadI64(Path.Combine(DIR, "i64_tv.bin"));
    ushort[] refb = ReadU16(Path.Combine(DIR, "i64_ref.bin"));
    ushort[] simd = new ushort[tv.Length];
    ushort[] scal = new ushort[tv.Length];
    fixed (long* s = tv) fixed (ushort* d = simd) { long i = H.BulkInt64ToHalf(s, d, tv.Length); for (; i < tv.Length; i++) d[i] = H.Int64ToHalfBits(s[i]); }
    for (int i = 0; i < tv.Length; i++) scal[i] = H.Int64ToHalfBits(tv[i]);
    ReportI64("i64->f16", tv, refb, simd, scal);

    // ---------- u64 ----------
    ulong[] utv = ReadU64(Path.Combine(DIR, "u64_tv.bin"));
    ushort[] urefb = ReadU16(Path.Combine(DIR, "u64_ref.bin"));
    ushort[] usimd = new ushort[utv.Length];
    ushort[] uscal = new ushort[utv.Length];
    fixed (ulong* s = utv) fixed (ushort* d = usimd) { long i = H.BulkUInt64ToHalf(s, d, utv.Length); for (; i < utv.Length; i++) d[i] = H.UInt64ToHalfBits(s[i]); }
    for (int i = 0; i < utv.Length; i++) uscal[i] = H.UInt64ToHalfBits(utv[i]);
    ReportU64("u64->f16", utv, urefb, usimd, uscal);
}

static void ReportI64(string name, long[] tv, ushort[] refb, ushort[] simd, ushort[] scal)
{
    int ds = 0, ds2 = 0, first = -1, n = tv.Length;
    for (int i = 0; i < n; i++) { if (simd[i] != refb[i]) { if (first < 0) first = i; ds++; } if (scal[i] != refb[i]) ds2++; }
    Console.WriteLine($"{name}: N={n}  SIMD-diff={ds}  scalar-diff={ds2}  {(ds == 0 && ds2 == 0 ? "BIT-EXACT OK" : "MISMATCH")}");
    if (first >= 0) for (int i = first, c = 0; i < n && c < 8; i++) if (simd[i] != refb[i]) { Console.WriteLine($"    [{i}] val={tv[i]} simd=0x{simd[i]:X4} ref=0x{refb[i]:X4}"); c++; }
}
static void ReportU64(string name, ulong[] tv, ushort[] refb, ushort[] simd, ushort[] scal)
{
    int ds = 0, ds2 = 0, first = -1, n = tv.Length;
    for (int i = 0; i < n; i++) { if (simd[i] != refb[i]) { if (first < 0) first = i; ds++; } if (scal[i] != refb[i]) ds2++; }
    Console.WriteLine($"{name}: N={n}  SIMD-diff={ds}  scalar-diff={ds2}  {(ds == 0 && ds2 == 0 ? "BIT-EXACT OK" : "MISMATCH")}");
    if (first >= 0) for (int i = first, c = 0; i < n && c < 8; i++) if (simd[i] != refb[i]) { Console.WriteLine($"    [{i}] val={tv[i]} simd=0x{simd[i]:X4} ref=0x{refb[i]:X4}"); c++; }
}
static long[] ReadI64(string p) { var b = File.ReadAllBytes(p); var a = new long[b.Length / 8]; Buffer.BlockCopy(b, 0, a, 0, b.Length); return a; }
static ulong[] ReadU64(string p) { var b = File.ReadAllBytes(p); var a = new ulong[b.Length / 8]; Buffer.BlockCopy(b, 0, a, 0, b.Length); return a; }
static ushort[] ReadU16(string p) { var b = File.ReadAllBytes(p); var a = new ushort[b.Length / 2]; Buffer.BlockCopy(b, 0, a, 0, b.Length); return a; }

static class H
{
    internal static readonly Vector256<int> RtoSel = Vector256.Create(0, 2, 4, 6, 0, 2, 4, 6);

    internal static Vector256<int> FloatToHalfBits(Vector256<float> fv)
    {
        var x0 = fv.AsInt32();
        var sign = Avx2.And(x0, Vector256.Create(unchecked((int)0x80000000)));
        var x = Avx2.Xor(x0, sign);
        var f32inf = Vector256.Create(255 << 23);
        var f16max = Vector256.Create((127 + 16) << 23);
        var denMagic = Vector256.Create(((127 - 15) + (23 - 10) + 1) << 23);
        var sub = Avx2.Subtract(Avx2.Add(x.AsSingle(), denMagic.AsSingle()).AsInt32(), denMagic);
        var mantOdd = Avx2.And(Avx2.ShiftRightLogical(x, 13), Vector256.Create(1));
        var xn = Avx2.Add(Avx2.Add(x, Vector256.Create(((15 - 127) << 23) + 0xfff)), mantOdd);
        var normal = Avx2.ShiftRightArithmetic(xn, 13);
        var payload = Avx2.ShiftRightLogical(Avx2.And(x, Vector256.Create(0x7fffff)), 13);
        var pz = Avx2.And(Avx2.CompareEqual(payload, Vector256<int>.Zero), Vector256.Create(1));
        var nanRes = Avx2.Or(Vector256.Create(0x7c00), Avx2.Or(payload, pz));
        var isNan = Avx2.CompareGreaterThan(x, f32inf);
        var infnan = Avx2.BlendVariable(Vector256.Create(0x7c00), nanRes, isNan);
        var isInfNan = Avx2.CompareGreaterThan(x, Avx2.Subtract(f16max, Vector256.Create(1)));
        var isSub = Avx2.CompareGreaterThan(Vector256.Create(113 << 23), x);
        var res = Avx2.BlendVariable(normal, sub, isSub);
        res = Avx2.BlendVariable(res, infnan, isInfNan);
        return Avx2.Or(res, Avx2.ShiftRightLogical(sign, 16));
    }
    internal static ushort SingleToHalfBits(float fval)
    {
        int x0 = BitConverter.SingleToInt32Bits(fval);
        int sign = x0 & unchecked((int)0x80000000);
        int x = x0 ^ sign;
        int denMagic = ((127 - 15) + (23 - 10) + 1) << 23;
        int sub = BitConverter.SingleToInt32Bits(BitConverter.Int32BitsToSingle(x) + BitConverter.Int32BitsToSingle(denMagic)) - denMagic;
        int mantOdd = (int)((uint)x >> 13) & 1;
        int xn = x + (((15 - 127) << 23) + 0xfff) + mantOdd;
        int normal = xn >> 13;
        int payload = (int)((uint)(x & 0x7fffff) >> 13);
        int pz = payload == 0 ? 1 : 0;
        int nanRes = 0x7c00 | payload | pz;
        int f32inf = 255 << 23, f16max = (127 + 16) << 23;
        int infnan = x > f32inf ? nanRes : 0x7c00;
        int res = ((113 << 23) > x) ? sub : normal;
        if (x > f16max - 1) res = infnan;
        res |= (int)((uint)sign >> 16);
        return (ushort)res;
    }

    // ===== i64 -> f16 (clamp |v|>=65520 -> +-70000 sentinel; low-32 narrow exact; cvtdq2ps + Giesen) =====
    internal static Vector256<int> Int64x8ToHalfBits(Vector256<long> a, Vector256<long> b)
    {
        var c = Vector256.Create(65519L); var cm = Vector256.Create(-65519L);
        var hi = Vector256.Create(70000L); var lo = Vector256.Create(-70000L);
        a = Avx2.BlendVariable(a.AsByte(), hi.AsByte(), Avx2.CompareGreaterThan(a, c).AsByte()).AsInt64();
        a = Avx2.BlendVariable(a.AsByte(), lo.AsByte(), Avx2.CompareGreaterThan(cm, a).AsByte()).AsInt64();
        b = Avx2.BlendVariable(b.AsByte(), hi.AsByte(), Avx2.CompareGreaterThan(b, c).AsByte()).AsInt64();
        b = Avx2.BlendVariable(b.AsByte(), lo.AsByte(), Avx2.CompareGreaterThan(cm, b).AsByte()).AsInt64();
        var ai = Avx2.PermuteVar8x32(a.AsInt32(), RtoSel).GetLower();
        var bi = Avx2.PermuteVar8x32(b.AsInt32(), RtoSel).GetLower();
        var packed = Vector256.Create(ai, bi);
        return FloatToHalfBits(Avx.ConvertToVector256Single(packed));
    }
    internal static ushort Int64ToHalfBits(long v)
    {
        if (v >= 65520) return 0x7C00;
        if (v <= -65520) return 0xFC00;
        return SingleToHalfBits((float)v);
    }
    internal static unsafe long BulkInt64ToHalf(void* s, void* d, long n)
    {
        long* src = (long*)s; ushort* dst = (ushort*)d; long i = 0;
        for (; i + 16 <= n; i += 16)
        {
            var lo = Int64x8ToHalfBits(Vector256.Load(src + i), Vector256.Load(src + i + 4));
            var hi = Int64x8ToHalfBits(Vector256.Load(src + i + 8), Vector256.Load(src + i + 12));
            Vector256.Store(Vector256.Narrow(lo, hi).AsUInt16(), dst + i);
        }
        return i;
    }

    // ===== u64 -> f16 (unsigned clamp via bias trick) =====
    internal static Vector256<int> UInt64x8ToHalfBits(Vector256<ulong> a, Vector256<ulong> b)
    {
        var bias = Vector256.Create(long.MinValue);
        var thr = Vector256.Create(65519L ^ long.MinValue);
        var hi = Vector256.Create(70000UL);
        var ga = Avx2.CompareGreaterThan(Avx2.Xor(a.AsInt64(), bias), thr);
        var gb = Avx2.CompareGreaterThan(Avx2.Xor(b.AsInt64(), bias), thr);
        a = Avx2.BlendVariable(a.AsByte(), hi.AsByte(), ga.AsByte()).AsUInt64();
        b = Avx2.BlendVariable(b.AsByte(), hi.AsByte(), gb.AsByte()).AsUInt64();
        var ai = Avx2.PermuteVar8x32(a.AsInt32(), RtoSel).GetLower();
        var bi = Avx2.PermuteVar8x32(b.AsInt32(), RtoSel).GetLower();
        var packed = Vector256.Create(ai, bi);
        return FloatToHalfBits(Avx.ConvertToVector256Single(packed));
    }
    internal static ushort UInt64ToHalfBits(ulong v)
    {
        if (v >= 65520) return 0x7C00;
        return SingleToHalfBits((float)v);
    }
    internal static unsafe long BulkUInt64ToHalf(void* s, void* d, long n)
    {
        ulong* src = (ulong*)s; ushort* dst = (ushort*)d; long i = 0;
        for (; i + 16 <= n; i += 16)
        {
            var lo = UInt64x8ToHalfBits(Vector256.Load(src + i), Vector256.Load(src + i + 4));
            var hi = UInt64x8ToHalfBits(Vector256.Load(src + i + 8), Vector256.Load(src + i + 12));
            Vector256.Store(Vector256.Narrow(lo, hi).AsUInt16(), dst + i);
        }
        return i;
    }
}
