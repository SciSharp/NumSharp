#:property AllowUnsafeBlocks=true
// Standalone PoC: SIMD 1B->2B widening for strided/negcol views (inverse of SubwordNarrow).
// deint/reverse the 1B elements (SubwordCopy shuffles), then Vector256.WidenLower/Upper
// (sign-extend for i8, zero-extend for u8/bool). vs scalar + NumPy.
//   src 1000x1000 i8/u8. strided=[:, ::2]->1000x500 i16; negcol=[:, ::-1]->1000x1000 i16.
//   NumPy baselines (best-of-7 from verify_present): i8->i16 strided ~0.115, bool->i16 ~0.114.
// Run: dotnet run -c Release - < benchmark/poc/subword_widen_poc.cs
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
    nint S = (nint)NativeMemory.Alloc((nuint)(R * C));         // 1B src
    nint D = (nint)NativeMemory.Alloc((nuint)(R * C * 2));     // 2B dst (sized R*C for negcol)
    nint REF = (nint)NativeMemory.Alloc((nuint)(R * C * 2));
    sbyte* s0 = (sbyte*)S; for (int i = 0; i < R * C; i++) s0[i] = (sbyte)((i % 256) - 128);  // full signed range incl negatives

    // verify signed deint widen (stride-2)
    ScalarWidenSigned(S, D, R, C, CV, 2); Buffer.MemoryCopy((void*)D, (void*)REF, R * CV * 2, R * CV * 2);
    SimdDeintWidenSigned(S, D, R, C, CV); long bad = 0; short* d = (short*)D; short* rf = (short*)REF; for (int i = 0; i < R * CV; i++) if (d[i] != rf[i]) bad++;
    Console.WriteLine($"verify deint-widen i8->i16 : {(bad == 0 ? "OK" : $"FAIL {bad}")}");
    // verify unsigned deint widen
    ScalarWidenUnsigned(S, D, R, C, CV, 2); Buffer.MemoryCopy((void*)D, (void*)REF, R * CV * 2, R * CV * 2);
    SimdDeintWidenUnsigned(S, D, R, C, CV); bad = 0; for (int i = 0; i < R * CV; i++) if (d[i] != rf[i]) bad++;
    Console.WriteLine($"verify deint-widen u8->u16 : {(bad == 0 ? "OK" : $"FAIL {bad}")}");
    // verify signed reverse widen (negcol)
    ScalarRevWidenSigned(S, REF, R, C); SimdRevWidenSigned(S, D, R, C); bad = 0; for (int i = 0; i < R * C; i++) if (d[i] != rf[i]) bad++;
    Console.WriteLine($"verify rev-widen   i8->i16 : {(bad == 0 ? "OK" : $"FAIL {bad}")}");

    Console.WriteLine($"\n--- 1B->2B widen, stride-2 [:, ::2] (NumPy ~0.115) ---");
    Console.WriteLine($"  i8->i16 scalar : {Best(() => ScalarWidenSigned(S, D, R, C, CV, 2)):F4} ms");
    Console.WriteLine($"  i8->i16 simd   : {Best(() => SimdDeintWidenSigned(S, D, R, C, CV)):F4} ms");
    Console.WriteLine($"  u8->u16 simd   : {Best(() => SimdDeintWidenUnsigned(S, D, R, C, CV)):F4} ms");
    Console.WriteLine($"--- 1B->2B widen, negcol [:, ::-1] (NumPy ~0.24) ---");
    Console.WriteLine($"  i8->i16 scalar : {Best(() => ScalarRevWidenSigned(S, D, R, C)):F4} ms");
    Console.WriteLine($"  i8->i16 simd   : {Best(() => SimdRevWidenSigned(S, D, R, C)):F4} ms");

    NativeMemory.Free((void*)S); NativeMemory.Free((void*)D); NativeMemory.Free((void*)REF);
}

static unsafe void ScalarWidenSigned(nint sp, nint dp, int rr, int cc, int cv, int ss)
{ sbyte* s1 = (sbyte*)sp; short* d1 = (short*)dp; for (int r = 0; r < rr; r++) { sbyte* s = s1 + r * cc; short* d = d1 + r * cv; for (int k = 0; k < cv; k++) d[k] = s[ss * k]; } }
static unsafe void ScalarWidenUnsigned(nint sp, nint dp, int rr, int cc, int cv, int ss)
{ byte* s1 = (byte*)sp; ushort* d1 = (ushort*)dp; for (int r = 0; r < rr; r++) { byte* s = s1 + r * cc; ushort* d = d1 + r * cv; for (int k = 0; k < cv; k++) d[k] = s[ss * k]; } }
static unsafe void ScalarRevWidenSigned(nint sp, nint dp, int rr, int cc)
{ sbyte* s1 = (sbyte*)sp; short* d1 = (short*)dp; for (int r = 0; r < rr; r++) { sbyte* s = s1 + r * cc + (cc - 1); short* d = d1 + r * cc; for (int k = 0; k < cc; k++) d[k] = s[-k]; } }

// deinterleave even bytes (SubwordCopy 1B deint) -> 32 even bytes in a Vector256<byte>
static unsafe Vector256<byte> DeintEven(short* p0)
{
    var v0 = Vector256.Load(p0); var v1 = Vector256.Load(p0 + 16);
    return Avx2.Permute4x64(Avx2.PackUnsignedSaturate(Avx2.And(v0, Vector256.Create((short)0x00FF)), Avx2.And(v1, Vector256.Create((short)0x00FF))).AsInt64(), 0xD8).AsByte();
}
static unsafe void SimdDeintWidenSigned(nint sp, nint dp, int rr, int cc, int cv)
{
    sbyte* s1 = (sbyte*)sp; short* d1 = (short*)dp;
    for (int r = 0; r < rr; r++)
    {
        sbyte* s = s1 + r * cc; short* d = d1 + r * cv; int k = 0;
        for (; k + 32 <= cv; k += 32)
        {
            var ev = DeintEven((short*)(s + 2 * k)).AsSByte();   // 32 even bytes (raw)
            Vector256.Store(Vector256.WidenLower(ev), d + k);     // sign-extend lo 16
            Vector256.Store(Vector256.WidenUpper(ev), d + k + 16);// sign-extend hi 16
        }
        for (; k < cv; k++) d[k] = s[2 * k];
    }
}
static unsafe void SimdDeintWidenUnsigned(nint sp, nint dp, int rr, int cc, int cv)
{
    byte* s1 = (byte*)sp; ushort* d1 = (ushort*)dp;
    for (int r = 0; r < rr; r++)
    {
        byte* s = s1 + r * cc; ushort* d = d1 + r * cv; int k = 0;
        for (; k + 32 <= cv; k += 32)
        {
            var ev = DeintEven((short*)(s + 2 * k));              // 32 even bytes
            Vector256.Store(Vector256.WidenLower(ev), d + k);     // zero-extend lo 16
            Vector256.Store(Vector256.WidenUpper(ev), d + k + 16);// zero-extend hi 16
        }
        for (; k < cv; k++) d[k] = s[2 * k];
    }
}
static unsafe void SimdRevWidenSigned(nint sp, nint dp, int rr, int cc)
{
    sbyte* s1 = (sbyte*)sp; short* d1 = (short*)dp;
    for (int r = 0; r < rr; r++)
    {
        sbyte* s = s1 + r * cc + (cc - 1); short* d = d1 + r * cc; int k = 0;
        for (; k + 32 <= cc; k += 32)
        {
            var v = Vector256.Load((byte*)(s - k - 31));
            var rev = Avx2.Permute4x64(Avx2.Shuffle(v, Vector256.Create((byte)15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0, 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)).AsInt64(), 0x4E).AsByte().AsSByte();
            Vector256.Store(Vector256.WidenLower(rev), d + k);
            Vector256.Store(Vector256.WidenUpper(rev), d + k + 16);
        }
        for (; k < cc; k++) d[k] = s[-k];
    }
}
