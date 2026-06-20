// cast_narrow_shootout.cs — PROOF harness for CAST_BEAT_NUMPY_PLAN §0.2
// Benchmarks MULTIPLE implementations of the worst cast cliff (float->narrow-int, f32->i8 = 0.09x
// vs NumPy) for BOTH correctness (vs the NumPy-faithful Converts scalar) AND speed, plus per-dtype
// (i8/u8/i16/u16, f64 source) variants. Establishes that:
//   - the SIMD back-end must be a TRUNCATING narrow (mask + unsigned-pack + permute), NOT a
//     saturating pack (vpackss): the saturate path is the FASTEST and the WRONGEST (3.47M diffs).
//   - cvtt is the engine; per-width lane fixups differ (vpermd for 8-bit, vpermq for 16-bit).
//   - every (correct) variant beats NumPy 1.9-4.1x.
//
// Run:  dotnet run -c Release - < benchmark/poc/cast_narrow_shootout.cs
// NumPy baselines (4M, best-of-7) measured with the python block at the bottom of this file.
#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

const int N = 4_000_000;

// ============ f32->i8 : five implementations ============
unsafe static void V1(float* s, sbyte* d) { for (int i = 0; i < 4_000_000; i++) d[i] = Converts.ToSByte(s[i]); }
unsafe static void V2(float* s, sbyte* d) { // cvtt SIMD + scalar low-byte narrow
    int i = 0, n = 4_000_000; int* tmp = stackalloc int[8];
    for (; i + 8 <= n; i += 8) { Avx.Store(tmp, Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i))); for (int k = 0; k < 8; k++) d[i + k] = (sbyte)tmp[k]; }
    for (; i < n; i++) d[i] = Converts.ToSByte(s[i]);
}
unsafe static void V3(float* s, sbyte* d) { // cvtt + &0xFF + 2x vpackuswb + vpermd  (PRODUCTION)
    int i = 0, n = 4_000_000; var mask = Vector256.Create(0xFF); var perm = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
    for (; i + 32 <= n; i += 32) {
        var i0 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 0)), mask);
        var i1 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 8)), mask);
        var i2 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 16)), mask);
        var i3 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 24)), mask);
        var w0 = Avx2.PackUnsignedSaturate(i0, i1); var w1 = Avx2.PackUnsignedSaturate(i2, i3);
        var b = Avx2.PackUnsignedSaturate(w0.AsInt16(), w1.AsInt16());
        Avx.Store((byte*)(d + i), Avx2.PermuteVar8x32(b.AsInt32(), perm).AsByte());
    }
    for (; i < n; i++) d[i] = Converts.ToSByte(s[i]);
}
unsafe static void V5(float* s, sbyte* d) { // cvtt + pshufb low-byte gather + 64-bit moves
    int i = 0, n = 4_000_000;
    var sh = Vector256.Create((byte)0,4,8,12,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,
                                    0,4,8,12,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80);
    for (; i + 32 <= n; i += 32) {
        var s0 = Avx2.Shuffle(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 0)).AsByte(), sh).AsInt64();
        var s1 = Avx2.Shuffle(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 8)).AsByte(), sh).AsInt64();
        var s2 = Avx2.Shuffle(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 16)).AsByte(), sh).AsInt64();
        var s3 = Avx2.Shuffle(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 24)).AsByte(), sh).AsInt64();
        *(long*)(d + i + 0) = s0.GetElement(0) | (s0.GetElement(2) << 32);
        *(long*)(d + i + 8) = s1.GetElement(0) | (s1.GetElement(2) << 32);
        *(long*)(d + i + 16) = s2.GetElement(0) | (s2.GetElement(2) << 32);
        *(long*)(d + i + 24) = s3.GetElement(0) | (s3.GetElement(2) << 32);
    }
    for (; i < n; i++) d[i] = Converts.ToSByte(s[i]);
}
unsafe static void V4(float* s, sbyte* d) { // WRONG: cvtt + SATURATING pack (vpackss) -> saturates, != NumPy wrap
    int i = 0, n = 4_000_000; var perm = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
    for (; i + 32 <= n; i += 32) {
        var i0 = Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 0));
        var i1 = Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 8));
        var i2 = Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 16));
        var i3 = Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 24));
        var w0 = Avx2.PackSignedSaturate(i0, i1); var w1 = Avx2.PackSignedSaturate(i2, i3);
        var b = Avx2.PackSignedSaturate(w0, w1);
        Avx.Store(d + i, Avx2.PermuteVar8x32(b.AsInt32(), perm).AsSByte());
    }
    for (; i < n; i++) d[i] = Converts.ToSByte(s[i]);
}

// ============ per-dtype production kernels ============
unsafe static void Narrow8(float* s, byte* d) { // i8/u8 share this (bit-identical)
    int i = 0, n = 4_000_000; var mask = Vector256.Create(0xFF); var perm = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
    for (; i + 32 <= n; i += 32) {
        var i0 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 0)), mask);
        var i1 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 8)), mask);
        var i2 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 16)), mask);
        var i3 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 24)), mask);
        var w0 = Avx2.PackUnsignedSaturate(i0, i1); var w1 = Avx2.PackUnsignedSaturate(i2, i3);
        var b = Avx2.PackUnsignedSaturate(w0.AsInt16(), w1.AsInt16());
        Avx.Store(d + i, Avx2.PermuteVar8x32(b.AsInt32(), perm).AsByte());
    }
    for (; i < n; i++) d[i] = (byte)Converts.ToSByte(s[i]);
}
unsafe static void Narrow16(float* s, ushort* d) { // i16/u16 share this; cheaper vpermq fixup
    int i = 0, n = 4_000_000; var mask = Vector256.Create(0xFFFF);
    for (; i + 16 <= n; i += 16) {
        var i0 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 0)), mask);
        var i1 = Avx2.And(Avx.ConvertToVector256Int32WithTruncation(Avx.LoadVector256(s + i + 8)), mask);
        Avx.Store(d + i, Avx2.Permute4x64(Avx2.PackUnsignedSaturate(i0, i1).AsUInt64(), 0xD8).AsUInt16());
    }
    for (; i < n; i++) d[i] = (ushort)Converts.ToInt16(s[i]);
}
unsafe static void D2I32(double* s, int* d) { // f64->i32: cvttpd2dq + store
    int i = 0, n = 4_000_000;
    for (; i + 4 <= n; i += 4) Sse2.Store(d + i, Avx.ConvertToVector128Int32WithTruncation(Avx.LoadVector256(s + i)));
    for (; i < n; i++) d[i] = Converts.ToInt32(s[i]);
}
unsafe static void D2I16(double* s, short* d) { // f64->i16: cvttpd2dq + 128-bit packus (no lane cross)
    int i = 0, n = 4_000_000; var mask = Vector128.Create(0xFFFF);
    for (; i + 8 <= n; i += 8) {
        var a = Sse2.And(Avx.ConvertToVector128Int32WithTruncation(Avx.LoadVector256(s + i + 0)), mask);
        var b = Sse2.And(Avx.ConvertToVector128Int32WithTruncation(Avx.LoadVector256(s + i + 4)), mask);
        Sse2.Store((ushort*)(d + i), Sse41.PackUnsignedSaturate(a, b));
    }
    for (; i < n; i++) d[i] = Converts.ToInt16(s[i]);
}

// ============ drivers ============
var rndF = new Random(12345);
float[] fsp = { 300f, 128.5f, 255.5f, 256.7f, -129.5f, -300f, 32768.9f, 65535.9f, 1e9f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 127.4f, -128.6f };
var srcF = new float[N];
for (int i = 0; i < N; i++) { int r = rndF.Next(100); srcF[i] = r < 10 ? fsp[rndF.Next(fsp.Length)] : (float)(rndF.NextDouble() * 120000 - 60000); }

var rndD = new Random(3);
double[] dsp = { 3e9, 128.5, 255.5, -300, 2147483653.0, 1e18, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 100000.0, -100000.0 };
var srcD = new double[N];
for (int i = 0; i < N; i++) { int r = rndD.Next(100); srcD[i] = r < 10 ? dsp[rndD.Next(dsp.Length)] : (rndD.NextDouble() * 1e10 - 5e9); }

unsafe {
    // ---- f32->i8 shootout ----
    var rR = new sbyte[N]; var r2 = new sbyte[N]; var r3 = new sbyte[N]; var r5 = new sbyte[N]; var r4 = new sbyte[N];
    fixed (float* s = srcF)
    fixed (sbyte* pR = rR, p2 = r2, p3 = r3, p5 = r5, p4 = r4) {
        V1(s, pR); V2(s, p2); V3(s, p3); V5(s, p5); V4(s, p4);
        long e2 = 0, e3 = 0, e5 = 0, e4 = 0;
        for (int i = 0; i < N; i++) { if (p2[i] != pR[i]) e2++; if (p3[i] != pR[i]) e3++; if (p5[i] != pR[i]) e5++; if (p4[i] != pR[i]) e4++; }
        static double B(delegate*<float*, sbyte*, void> f, float* s, sbyte* d) { double b = 1e9; for (int r = 0; r < 7; r++) { var sw = Stopwatch.StartNew(); f(s, d); sw.Stop(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds); } return b; }
        double t1 = B(&V1, s, pR), t2 = B(&V2, s, p2), t3 = B(&V3, s, p3), t5 = B(&V5, s, p5), t4 = B(&V4, s, p4);
        Console.WriteLine("== f32->i8 shootout (4M, best-of-7) ==  NumPy=1.305ms");
        Console.WriteLine($"  V1 scalar (current)          {t1,7:F3}  x{t1 / t1,4:F1}  diffs=0          NPY/NS {1.305 / t1:F2}");
        Console.WriteLine($"  V2 cvtt+scalar-narrow        {t2,7:F3}  x{t1 / t2,4:F1}  diffs={e2,-8}  NPY/NS {1.305 / t2:F2}");
        Console.WriteLine($"  V3 cvtt+mask+pack+vpermd     {t3,7:F3}  x{t1 / t3,4:F1}  diffs={e3,-8}  NPY/NS {1.305 / t3:F2}  <- PRODUCTION");
        Console.WriteLine($"  V5 cvtt+pshufb+64move        {t5,7:F3}  x{t1 / t5,4:F1}  diffs={e5,-8}  NPY/NS {1.305 / t5:F2}");
        Console.WriteLine($"  V4 cvtt+SATURATE (WRONG)     {t4,7:F3}  x{t1 / t4,4:F1}  diffs={e4,-8}  <- fastest & WRONG");
    }

    // ---- per-dtype ----
    var i8R = new sbyte[N]; var u8R = new byte[N]; var i16R = new short[N]; var u16R = new ushort[N];
    var i8B = new sbyte[N]; var u8B = new byte[N]; var i16B = new short[N]; var u16B = new ushort[N];
    fixed (float* s = srcF)
    fixed (sbyte* p8r = i8R, p8b = i8B) fixed (byte* pu8r = u8R, pu8b = u8B)
    fixed (short* p16r = i16R, p16b = i16B) fixed (ushort* pu16r = u16R, pu16b = u16B) {
        for (int i = 0; i < N; i++) { p8r[i] = Converts.ToSByte(s[i]); pu8r[i] = Converts.ToByte(s[i]); p16r[i] = Converts.ToInt16(s[i]); pu16r[i] = Converts.ToUInt16(s[i]); }
        Narrow8(s, (byte*)p8b); Narrow8(s, pu8b); Narrow16(s, (ushort*)p16b); Narrow16(s, pu16b);
        long a = 0, b = 0, c = 0, e = 0;
        for (int i = 0; i < N; i++) { if (p8b[i] != p8r[i]) a++; if (pu8b[i] != pu8r[i]) b++; if (p16b[i] != p16r[i]) c++; if (pu16b[i] != pu16r[i]) e++; }
        static double B8(delegate*<float*, byte*, void> f, float* s, byte* d) { double x = 1e9; for (int r = 0; r < 7; r++) { var sw = Stopwatch.StartNew(); f(s, d); sw.Stop(); x = Math.Min(x, sw.Elapsed.TotalMilliseconds); } return x; }
        static double B16(delegate*<float*, ushort*, void> f, float* s, ushort* d) { double x = 1e9; for (int r = 0; r < 7; r++) { var sw = Stopwatch.StartNew(); f(s, d); sw.Stop(); x = Math.Min(x, sw.Elapsed.TotalMilliseconds); } return x; }
        double t8 = B8(&Narrow8, s, pu8b), t16 = B16(&Narrow16, s, pu16b);
        Console.WriteLine("== f32->narrow per-dtype (0-diff required) ==");
        Console.WriteLine($"  f32->i8   {t8,7:F3}  diffs={a}   NumPy 1.305  NPY/NS {1.305 / t8:F2}");
        Console.WriteLine($"  f32->u8   {t8,7:F3}  diffs={b}   NumPy 1.565  NPY/NS {1.565 / t8:F2}  (same kernel)");
        Console.WriteLine($"  f32->i16  {t16,7:F3}  diffs={c}   NumPy 2.110  NPY/NS {2.110 / t16:F2}");
        Console.WriteLine($"  f32->u16  {t16,7:F3}  diffs={e}   NumPy 2.008  NPY/NS {2.008 / t16:F2}  (same kernel)");
    }

    // ---- f64 source ----
    var di32R = new int[N]; var di32B = new int[N]; var di16R = new short[N]; var di16B = new short[N];
    fixed (double* s = srcD)
    fixed (int* qr = di32R, qb = di32B) fixed (short* hr = di16R, hb = di16B) {
        for (int i = 0; i < N; i++) { qr[i] = Converts.ToInt32(s[i]); hr[i] = Converts.ToInt16(s[i]); }
        D2I32(s, qb); D2I16(s, hb);
        long a = 0, b = 0; for (int i = 0; i < N; i++) { if (qb[i] != qr[i]) a++; if (hb[i] != hr[i]) b++; }
        static double B32(delegate*<double*, int*, void> f, double* s, int* d) { double x = 1e9; for (int r = 0; r < 7; r++) { var sw = Stopwatch.StartNew(); f(s, d); sw.Stop(); x = Math.Min(x, sw.Elapsed.TotalMilliseconds); } return x; }
        static double B16(delegate*<double*, short*, void> f, double* s, short* d) { double x = 1e9; for (int r = 0; r < 7; r++) { var sw = Stopwatch.StartNew(); f(s, d); sw.Stop(); x = Math.Min(x, sw.Elapsed.TotalMilliseconds); } return x; }
        double t32 = B32(&D2I32, s, qb), t16 = B16(&D2I16, s, hb);
        Console.WriteLine("== f64->int (cvttpd2dq) ==");
        Console.WriteLine($"  f64->i32  {t32,7:F3}  diffs={a}   NumPy 3.256  NPY/NS {3.256 / t32:F2}");
        Console.WriteLine($"  f64->i16  {t16,7:F3}  diffs={b}   NumPy 2.633  NPY/NS {2.633 / t16:F2}");
    }
}

/* NumPy baselines (4M, best-of-7) — run with:  python THIS_BLOCK
import numpy as np, time
N = 4_000_000; rng = np.random.default_rng(7)
f = (rng.random(N, dtype=np.float32) * 1000 - 500).astype(np.float32)
d = (rng.random(N) * 1e10 - 5e9)
def best(arr, dt):
    b = 1e9
    for _ in range(7):
        t = time.perf_counter(); arr.astype(dt); b = min(b, (time.perf_counter() - t) * 1000)
    return b
for dt in [np.int8, np.uint8, np.int16, np.uint16]:
    print(f"f32->{np.dtype(dt).name:6} {best(f, dt):7.3f} ms")
for dt in [np.int32, np.int16]:
    print(f"f64->{np.dtype(dt).name:6} {best(d, dt):7.3f} ms")
*/
