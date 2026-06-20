#:property AllowUnsafeBlocks=true
// Pin the 6x strided gap + verify fixes. Same deinterleave math, structures:
//   A inlined + const bounds (0.020ms baseline)
//   C odometer + separate InnerCopy method call per row (library structure, slow)
//   D odometer + AggressiveInlining InnerCopy
//   E ss-dispatched-once + odometer with INLINED deinterleave (no per-row method)
// Run: dotnet run -c Release - < benchmark/poc/subword_structure_poc.cs
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;
if (!Avx2.IsSupported) { Console.Error.WriteLine("no avx2"); return; }

const int R = 1000, C = 1000, CV = 500, it = 50, wm = 10, rd = 7;
double Best(Action f) { for (int i = 0; i < wm; i++) f(); double b = 1e9; for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); } return b; }

unsafe
{
    nint S = (nint)NativeMemory.Alloc((nuint)(R * C));
    nint D = (nint)NativeMemory.Alloc((nuint)(R * CV));
    byte* s0 = (byte*)S; for (int i = 0; i < R * C; i++) s0[i] = (byte)((i % 251) + 1);

    Console.WriteLine($"A inlined+const       : {Best(() => A(S, D)):F4} ms");
    Console.WriteLine($"C odometer+method     : {Best(() => Cc(S, D, R, C, CV)):F4} ms");
    Console.WriteLine($"D odometer+aggrinline : {Best(() => Dd(S, D, R, C, CV)):F4} ms");
    Console.WriteLine($"E dispatch+inlined    : {Best(() => Ee(S, D, R, C, CV)):F4} ms");
    Console.WriteLine($"F odometer+inline-if  : {Best(() => Ff(S, D, R, C, CV)):F4} ms");
    NativeMemory.Free((void*)S); NativeMemory.Free((void*)D);
}

static unsafe void A(nint sp, nint dp)
{
    byte* s1 = (byte*)sp, d1 = (byte*)dp; var mask = Vector256.Create((short)0x00FF);
    for (int r = 0; r < R; r++)
    {
        byte* s = s1 + r * C, d = d1 + r * CV; int k = 0;
        for (; k + 32 <= CV; k += 32)
        {
            var v0 = Vector256.Load((short*)(s + 2 * k)); var v1 = Vector256.Load((short*)(s + 2 * k + 32));
            var p = Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask));
            Vector256.Store(Avx2.Permute4x64(p.AsInt64(), 0xD8).AsByte(), d + k);
        }
        for (; k < CV; k++) d[k] = s[2 * k];
    }
}
static unsafe void Cc(nint sp, nint dp, int rr, int cc, int cv)
{
    byte* s1 = (byte*)sp, d1 = (byte*)dp; long srcOff = 0, dstOff = 0;
    for (int o = 0; o < rr; o++) { Inner(s1 + srcOff, d1 + dstOff, cv, 2, 1); srcOff += cc; dstOff += cv; }
}
static unsafe void Inner(byte* s, byte* d, long n, long ss, long ds)
{
    long i = 0; var mask = Vector256.Create((short)0x00FF);
    if (ds == 1 && ss == 2)
        for (; i + 32 <= n; i += 32)
        {
            var v0 = Vector256.Load((short*)(s + 2 * i)); var v1 = Vector256.Load((short*)(s + 2 * i + 32));
            var p = Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask));
            Vector256.Store(Avx2.Permute4x64(p.AsInt64(), 0xD8).AsByte(), d + i);
        }
    for (; i < n; i++) d[i * ds] = s[i * ss];
}
static unsafe void Dd(nint sp, nint dp, int rr, int cc, int cv)
{
    byte* s1 = (byte*)sp, d1 = (byte*)dp; long srcOff = 0, dstOff = 0;
    for (int o = 0; o < rr; o++) { InnerAg(s1 + srcOff, d1 + dstOff, cv, 2, 1); srcOff += cc; dstOff += cv; }
}
[MethodImpl(MethodImplOptions.AggressiveInlining)]
static unsafe void InnerAg(byte* s, byte* d, long n, long ss, long ds)
{
    long i = 0; var mask = Vector256.Create((short)0x00FF);
    if (ds == 1 && ss == 2)
        for (; i + 32 <= n; i += 32)
        {
            var v0 = Vector256.Load((short*)(s + 2 * i)); var v1 = Vector256.Load((short*)(s + 2 * i + 32));
            var p = Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask));
            Vector256.Store(Avx2.Permute4x64(p.AsInt64(), 0xD8).AsByte(), d + i);
        }
    for (; i < n; i++) d[i * ds] = s[i * ss];
}
// E: dispatch on ss ONCE, odometer with the deinterleave inlined directly in the loop body.
static unsafe void Ee(nint sp, nint dp, int rr, int cc, int cv)
{
    byte* s1 = (byte*)sp, d1 = (byte*)dp; var mask = Vector256.Create((short)0x00FF); long srcOff = 0, dstOff = 0;
    for (int o = 0; o < rr; o++)
    {
        byte* s = s1 + srcOff, d = d1 + dstOff; long i = 0;
        for (; i + 32 <= cv; i += 32)
        {
            var v0 = Vector256.Load((short*)(s + 2 * i)); var v1 = Vector256.Load((short*)(s + 2 * i + 32));
            var p = Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask));
            Vector256.Store(Avx2.Permute4x64(p.AsInt64(), 0xD8).AsByte(), d + i);
        }
        for (; i < cv; i++) d[i] = s[2 * i];
        srcOff += cc; dstOff += cv;
    }
}
// F: ONE odometer, inline if-else-if on ss in the body (no method call).
static unsafe void Ff(nint sp, nint dp, int rr, int cc, int cv)
{
    byte* s1 = (byte*)sp, d1 = (byte*)dp; var mask = Vector256.Create((short)0x00FF);
    long ss = 2, ds = 1; long srcOff = 0, dstOff = 0;
    for (int o = 0; o < rr; o++)
    {
        byte* s = s1 + srcOff, d = d1 + dstOff; long i = 0;
        if (ds == 1 && ss == 2)
        {
            for (; i + 32 <= cv; i += 32)
            {
                var v0 = Vector256.Load((short*)(s + 2 * i)); var v1 = Vector256.Load((short*)(s + 2 * i + 32));
                var p = Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask));
                Vector256.Store(Avx2.Permute4x64(p.AsInt64(), 0xD8).AsByte(), d + i);
            }
        }
        for (; i < cv; i++) d[i * ds] = s[i * ss];
        srcOff += cc; dstOff += cv;
    }
}
