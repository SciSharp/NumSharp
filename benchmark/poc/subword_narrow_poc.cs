#:property AllowUnsafeBlocks=true
// Standalone PoC: SIMD 2B->1B narrowing for strided/negcol views (the cross-WIDTH
// continuation of SubwordCopy). Two-stage: deinterleave/reverse the even/reversed 2B
// elements, then narrow to 1B (low byte for int, !=0 for bool). vs scalar + NumPy.
//
//   Scenario: 1000x1000 i16 source. strided=[:, ::2] -> 1000x500 i8; negcol=[:, ::-1]
//   -> 1000x1000 i8. NumPy baselines (cast_results.tsv best-of-3):
//     i16|strided|i8 0.757 (np 0.107)   char|strided|bool 0.676 (np 0.161)
//     i16|negcol|i8  0.893               i16|strided|bool 0.647 (np 0.143)
// Run: dotnet run -c Release - < benchmark/poc/subword_narrow_poc.cs
using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;
if (!Avx2.IsSupported) { Console.Error.WriteLine("no avx2"); return; }

const int R = 1000, C = 1000, CV = 500, it = 50, wm = 10, rd = 7;
double Best(Action f) { for (int i = 0; i < wm; i++) f(); double b = 1e9; for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); } return b; }

unsafe
{
    nint S = (nint)NativeMemory.Alloc((nuint)(R * C * 2));   // i16 source
    nint D = (nint)NativeMemory.Alloc((nuint)(R * C));       // i8 dst (sized R*C for negcol)
    nint REF = (nint)NativeMemory.Alloc((nuint)(R * C));
    short* s0 = (short*)S; for (int i = 0; i < R * C; i++) s0[i] = (short)((i % 65521) - 30000);

    // verify int narrow (stride-2)
    ScalarNarrow2to1(S, D, R, C, CV, 2); Buffer.MemoryCopy((void*)D, (void*)REF, R * CV, R * CV);
    SimdDeintNarrow(S, D, R, C, CV); long bad = 0; byte* d = (byte*)D; byte* rf = (byte*)REF; for (int i = 0; i < R * CV; i++) if (d[i] != rf[i]) bad++;
    Console.WriteLine($"verify deint-narrow 2B->1B: {(bad == 0 ? "OK" : $"FAIL {bad}")}");
    // verify reverse narrow (negcol)
    ScalarRevNarrow(S, REF, R, C); SimdRevNarrow(S, D, R, C); bad = 0; for (int i = 0; i < R * C; i++) if (d[i] != rf[i]) bad++;
    Console.WriteLine($"verify rev-narrow   2B->1B: {(bad == 0 ? "OK" : $"FAIL {bad}")}");
    // verify ->bool (stride-2)
    ScalarToBool2(S, REF, R, C, CV); SimdDeintBool(S, D, R, C, CV); bad = 0; for (int i = 0; i < R * CV; i++) if (d[i] != rf[i]) bad++;
    Console.WriteLine($"verify deint-bool   2B->1B: {(bad == 0 ? "OK" : $"FAIL {bad}")}");

    Console.WriteLine($"\n--- 2B->1B int narrow, stride-2 [:, ::2] (NumPy i16->i8 0.107) ---");
    Console.WriteLine($"  scalar : {Best(() => ScalarNarrow2to1(S, D, R, C, CV, 2)):F4} ms");
    Console.WriteLine($"  simd   : {Best(() => SimdDeintNarrow(S, D, R, C, CV)):F4} ms");
    Console.WriteLine($"--- 2B->1B int narrow, negcol [:, ::-1] (NumPy i16->i8 negcol ~0.24) ---");
    Console.WriteLine($"  scalar : {Best(() => ScalarRevNarrow(S, D, R, C)):F4} ms");
    Console.WriteLine($"  simd   : {Best(() => SimdRevNarrow(S, D, R, C)):F4} ms");
    Console.WriteLine($"--- 2B->bool, stride-2 (NumPy i16->bool 0.143) ---");
    Console.WriteLine($"  scalar : {Best(() => ScalarToBool2(S, D, R, C, CV)):F4} ms");
    Console.WriteLine($"  simd   : {Best(() => SimdDeintBool(S, D, R, C, CV)):F4} ms");

    NativeMemory.Free((void*)S); NativeMemory.Free((void*)D); NativeMemory.Free((void*)REF);
}

// ---- scalar baselines ----
static unsafe void ScalarNarrow2to1(nint sp, nint dp, int rr, int cc, int cv, int ss)
{ short* s1 = (short*)sp; byte* d1 = (byte*)dp; for (int r = 0; r < rr; r++) { short* s = s1 + r * cc; byte* d = d1 + r * cv; for (int k = 0; k < cv; k++) d[k] = (byte)s[ss * k]; } }
static unsafe void ScalarRevNarrow(nint sp, nint dp, int rr, int cc)
{ short* s1 = (short*)sp; byte* d1 = (byte*)dp; for (int r = 0; r < rr; r++) { short* s = s1 + r * cc + (cc - 1); byte* d = d1 + r * cc; for (int k = 0; k < cc; k++) d[k] = (byte)s[-k]; } }
static unsafe void ScalarToBool2(nint sp, nint dp, int rr, int cc, int cv)
{ short* s1 = (short*)sp; byte* d1 = (byte*)dp; for (int r = 0; r < rr; r++) { short* s = s1 + r * cc; byte* d = d1 + r * cv; for (int k = 0; k < cv; k++) d[k] = (byte)(s[2 * k] != 0 ? 1 : 0); } }

// ---- SIMD: deinterleave even 2B elems -> low byte (int narrow) ----
static unsafe void SimdDeintNarrow(nint sp, nint dp, int rr, int cc, int cv)
{
    short* s1 = (short*)sp; byte* d1 = (byte*)dp;
    var wmask = Vector256.Create(0x0000FFFF);          // keep even short (stage1)
    var bmask = Vector256.Create((short)0x00FF);       // keep low byte (stage2)
    for (int r = 0; r < rr; r++)
    {
        short* s = s1 + r * cc; byte* d = d1 + r * cv; int k = 0;
        for (; k + 16 <= cv; k += 16)
        {
            var v0 = Vector256.Load((int*)(s + 2 * k));
            var v1 = Vector256.Load((int*)(s + 2 * k + 16));
            var evens = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(Avx2.And(v0, wmask), Avx2.And(v1, wmask)).AsInt64(), 0xD8).AsInt16(); // 16 even shorts
            var lo = Avx2.And(evens, bmask);            // 16 shorts, low byte
            var packed = Avx2.PackUnsignedSaturate(lo, lo);                 // [lo.lo,lo.lo,lo.hi,lo.hi] bytes
            var bytes16 = Avx2.Permute4x64(packed.AsInt64(), 0xD8).GetLower(); // 16 bytes
            Vector128.Store(bytes16.AsByte(), d + k);
        }
        for (; k < cv; k++) d[k] = (byte)s[2 * k];
    }
}
// ---- SIMD: reverse 2B elems -> low byte (negcol int narrow) ----
static unsafe void SimdRevNarrow(nint sp, nint dp, int rr, int cc)
{
    short* s1 = (short*)sp; byte* d1 = (byte*)dp;
    var revw = Vector256.Create((byte)14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1, 14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1);
    var bmask = Vector256.Create((short)0x00FF);
    for (int r = 0; r < rr; r++)
    {
        short* s = s1 + r * cc + (cc - 1); byte* d = d1 + r * cc; int k = 0;
        for (; k + 16 <= cc; k += 16)
        {
            var v = Vector256.Load(s - k - 15);
            var rev = Avx2.Permute4x64(Avx2.Shuffle(v.AsByte(), revw).AsInt64(), 0x4E).AsInt16(); // 16 reversed shorts
            var lo = Avx2.And(rev, bmask);
            var packed = Avx2.PackUnsignedSaturate(lo, lo);
            Vector128.Store(Avx2.Permute4x64(packed.AsInt64(), 0xD8).GetLower().AsByte(), d + k);
        }
        for (; k < cc; k++) d[k] = (byte)s[-k];
    }
}
// ---- SIMD: deinterleave even 2B elems -> !=0 (bool) ----
static unsafe void SimdDeintBool(nint sp, nint dp, int rr, int cc, int cv)
{
    short* s1 = (short*)sp; byte* d1 = (byte*)dp;
    var wmask = Vector256.Create(0x0000FFFF);
    var one = Vector256.Create((short)1);
    var zero = Vector256<short>.Zero;
    for (int r = 0; r < rr; r++)
    {
        short* s = s1 + r * cc; byte* d = d1 + r * cv; int k = 0;
        for (; k + 16 <= cv; k += 16)
        {
            var v0 = Vector256.Load((int*)(s + 2 * k));
            var v1 = Vector256.Load((int*)(s + 2 * k + 16));
            var evens = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(Avx2.And(v0, wmask), Avx2.And(v1, wmask)).AsInt64(), 0xD8).AsInt16();
            var nz = Avx2.AndNot(Avx2.CompareEqual(evens, zero), one);      // 1 if !=0 else 0
            var packed = Avx2.PackUnsignedSaturate(nz, nz);
            Vector128.Store(Avx2.Permute4x64(packed.AsInt64(), 0xD8).GetLower().AsByte(), d + k);
        }
        for (; k < cv; k++) d[k] = (byte)(s[2 * k] != 0 ? 1 : 0);
    }
}
