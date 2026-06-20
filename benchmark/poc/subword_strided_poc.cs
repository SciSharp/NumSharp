#:property AllowUnsafeBlocks=true
// Standalone PoC — NO project ref (pure intrinsics) so it builds Release-clean fast.
// Proves & benchmarks SIMD deinterleave (stride-2) and reverse (stride-1) for
// same-type sub-word (1B/2B) copies vs the scalar inner loop the generic strided
// cast kernel currently emits, vs NumPy's measured baseline.
//
//   Scenario mirrors cast_matrix_bench: 1000x1000 source, view [:, ::2] -> 1000x500
//   (strided) and [:, ::-1] (negcol). Output pre-allocated warm (isolates KERNEL).
//
//   NumPy baselines to beat (best-of-3 sweep, cast_results.tsv):
//     i8|strided 0.0927   i16|strided 0.1071   i8|negcol 0.2212   u8|negcol 0.2379
//   Current NumSharp (scalar inner): i8|strided 0.175, i16|strided 0.182.
//
// Run: dotnet run -c Release - < benchmark/poc/subword_strided_poc.cs
using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;

if (!Avx2.IsSupported) { Console.Error.WriteLine("AVX2 required"); return; }

const int R = 1000, C = 1000, CV = C / 2;   // source RxC, view R x CV (stride-2)
const int it = 50, wm = 10, rd = 7;          // best-of-7

double Best(Action f)
{
    for (int i = 0; i < wm; i++) f();
    double b = 1e9;
    for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); }
    return b;
}

unsafe
{
    nint S1 = (nint)NativeMemory.Alloc((nuint)(R * C));
    nint D1 = (nint)NativeMemory.Alloc((nuint)(R * C));        // sized R*C (reverse reuses)
    nint REF1 = (nint)NativeMemory.Alloc((nuint)(R * C));
    nint S2 = (nint)NativeMemory.Alloc((nuint)(R * C * 2));
    nint D2 = (nint)NativeMemory.Alloc((nuint)(R * CV * 2));
    nint REF2 = (nint)NativeMemory.Alloc((nuint)(R * CV * 2));
    byte* s1 = (byte*)S1; short* s2 = (short*)S2;
    for (int i = 0; i < R * C; i++) { s1[i] = (byte)((i % 251) + 1); s2[i] = (short)((i % 65521) - 30000); }

    // verify
    Scalar2_1b(S1, D1); Buffer.MemoryCopy((void*)D1, (void*)REF1, R * CV, R * CV);
    Simd2_1b(S1, D1); long bad = 0; byte* d1 = (byte*)D1; byte* rr1 = (byte*)REF1; for (int i = 0; i < R * CV; i++) if (d1[i] != rr1[i]) bad++;
    Console.WriteLine($"verify stride2 1B: {(bad == 0 ? "OK" : $"FAIL {bad}")}");
    Scalar2_2b(S2, D2); Buffer.MemoryCopy((void*)D2, (void*)REF2, R * CV * 2, R * CV * 2);
    Simd2_2b(S2, D2); bad = 0; short* d2 = (short*)D2; short* rr2 = (short*)REF2; for (int i = 0; i < R * CV; i++) if (d2[i] != rr2[i]) bad++;
    Console.WriteLine($"verify stride2 2B: {(bad == 0 ? "OK" : $"FAIL {bad}")}");
    ScalarRev_1b(S1, REF1); SimdRev_1b(S1, D1); bad = 0; for (int i = 0; i < R * C; i++) if (d1[i] != rr1[i]) bad++;
    Console.WriteLine($"verify rev 1B:     {(bad == 0 ? "OK" : $"FAIL {bad}")}");

    // bench
    Console.WriteLine("\n--- stride-2 deinterleave (view [:, ::2], 1000x500 out) ---");
    Console.WriteLine($"  1B scalar : {Best(() => Scalar2_1b(S1, D1)):F4} ms   (NumPy i8 0.0927, NS now 0.175)");
    Console.WriteLine($"  1B simd   : {Best(() => Simd2_1b(S1, D1)):F4} ms");
    Console.WriteLine($"  1B memcpy : {Best(() => Memcpy1b(S1, D1)):F4} ms   (floor)");
    Console.WriteLine($"  2B scalar : {Best(() => Scalar2_2b(S2, D2)):F4} ms   (NumPy i16 0.1071, NS now 0.182)");
    Console.WriteLine($"  2B simd   : {Best(() => Simd2_2b(S2, D2)):F4} ms");
    Console.WriteLine($"  2B memcpy : {Best(() => Memcpy2b(S2, D2)):F4} ms   (floor)");
    Console.WriteLine("\n--- stride-(-1) reverse (view [:, ::-1], 1000x1000 out) ---");
    Console.WriteLine($"  1B scalar : {Best(() => ScalarRev_1b(S1, D1)):F4} ms   (NumPy i8 negcol 0.2212)");
    Console.WriteLine($"  1B simd   : {Best(() => SimdRev_1b(S1, D1)):F4} ms");

    NativeMemory.Free((void*)S1); NativeMemory.Free((void*)D1); NativeMemory.Free((void*)REF1);
    NativeMemory.Free((void*)S2); NativeMemory.Free((void*)D2); NativeMemory.Free((void*)REF2);
}

// ---------- kernels (static unsafe; nint args so the bench lambdas capture cleanly) ----------
static unsafe void Scalar2_1b(nint sp, nint dp) { byte* s1 = (byte*)sp, d1 = (byte*)dp; for (int r = 0; r < R; r++) { byte* s = s1 + r * C, d = d1 + r * CV; for (int k = 0; k < CV; k++) d[k] = s[2 * k]; } }
static unsafe void Scalar2_2b(nint sp, nint dp) { short* s2 = (short*)sp, d2 = (short*)dp; for (int r = 0; r < R; r++) { short* s = s2 + r * C, d = d2 + r * CV; for (int k = 0; k < CV; k++) d[k] = s[2 * k]; } }

static unsafe void Simd2_1b(nint sp, nint dp)
{
    byte* s1 = (byte*)sp, d1 = (byte*)dp; var mask = Vector256.Create((short)0x00FF);
    for (int r = 0; r < R; r++)
    {
        byte* s = s1 + r * C, d = d1 + r * CV; int k = 0;
        for (; k + 32 <= CV; k += 32)
        {
            var v0 = Vector256.Load((short*)(s + 2 * k));
            var v1 = Vector256.Load((short*)(s + 2 * k + 32));
            var p = Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask));
            Vector256.Store(Avx2.Permute4x64(p.AsInt64(), 0xD8).AsByte(), d + k);
        }
        for (; k < CV; k++) d[k] = s[2 * k];
    }
}
static unsafe void Simd2_2b(nint sp, nint dp)
{
    short* s2 = (short*)sp, d2 = (short*)dp; var mask = Vector256.Create(0x0000FFFF);
    for (int r = 0; r < R; r++)
    {
        short* s = s2 + r * C, d = d2 + r * CV; int k = 0;
        for (; k + 16 <= CV; k += 16)
        {
            var v0 = Vector256.Load((int*)(s + 2 * k));
            var v1 = Vector256.Load((int*)(s + 2 * k + 16));
            var p = Avx2.PackUnsignedSaturate(Avx2.And(v0, mask), Avx2.And(v1, mask));
            Vector256.Store(Avx2.Permute4x64(p.AsInt64(), 0xD8).AsInt16(), d + k);
        }
        for (; k < CV; k++) d[k] = s[2 * k];
    }
}

static unsafe void ScalarRev_1b(nint sp, nint dp) { byte* s1 = (byte*)sp, d1 = (byte*)dp; for (int r = 0; r < R; r++) { byte* s = s1 + r * C + (C - 1), d = d1 + r * C; for (int k = 0; k < C; k++) d[k] = s[-k]; } }
static unsafe void SimdRev_1b(nint sp, nint dp)
{
    byte* s1 = (byte*)sp, d1 = (byte*)dp;
    var rev = Vector256.Create((byte)15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0, 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    for (int r = 0; r < R; r++)
    {
        byte* s = s1 + r * C + (C - 1), d = d1 + r * C; int k = 0;
        for (; k + 32 <= C; k += 32)
        {
            var v = Vector256.Load(s - k - 31);
            var sh = Avx2.Shuffle(v, rev);
            Vector256.Store(Avx2.Permute4x64(sh.AsInt64(), 0x4E).AsByte(), d + k);
        }
        for (; k < C; k++) d[k] = s[-k];
    }
}

static unsafe void Memcpy1b(nint sp, nint dp) { Buffer.MemoryCopy((void*)sp, (void*)dp, R * CV, R * CV); }
static unsafe void Memcpy2b(nint sp, nint dp) { Buffer.MemoryCopy((void*)sp, (void*)dp, R * CV * 2, R * CV * 2); }
