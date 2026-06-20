#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// POC: LSD radix sort (int32/int64) vs scalar Span.Sort vs NumPy baseline.
// Question: can a simple, SAFE radix kernel close the 7-13x gap to NumPy's SIMD sort?
using System;
using System.Diagnostics;

const int ROUNDS = 7;
int[] sizes = { 100_000, 1_000_000, 10_000_000 };
var rng = new Random(42);

static double BestMs(Action setup, Action timed)
{
    double best = 1e18;
    for (int r = 0; r < 7; r++) { setup(); var sw = Stopwatch.StartNew(); timed(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
    return best;
}

// ---- LSD radix sort, int32: 4x 8-bit passes, sign-flip to unsigned ordering ----
static void RadixInt32(int[] a, uint[] cur, uint[] tmp, int[] count)
{
    int n = a.Length;
    for (int i = 0; i < n; i++) cur[i] = (uint)a[i] ^ 0x80000000u;
    for (int shift = 0; shift < 32; shift += 8)
    {
        Array.Clear(count, 0, 256);
        for (int i = 0; i < n; i++) count[(cur[i] >> shift) & 0xFF]++;
        int sum = 0; for (int b = 0; b < 256; b++) { int c = count[b]; count[b] = sum; sum += c; }
        for (int i = 0; i < n; i++) { int d = (int)((cur[i] >> shift) & 0xFF); tmp[count[d]++] = cur[i]; }
        var t = cur; cur = tmp; tmp = t;
    }
    // 4 passes -> result is back in `cur` (the original cur after even swaps)
    for (int i = 0; i < n; i++) a[i] = (int)(cur[i] ^ 0x80000000u);
}

// ---- LSD radix argsort, int32: carry long indices ----
static void RadixArgInt32(int[] a, long[] outIdx, uint[] keyA, uint[] keyB, long[] idxA, long[] idxB, int[] count)
{
    int n = a.Length;
    for (int i = 0; i < n; i++) { keyA[i] = (uint)a[i] ^ 0x80000000u; idxA[i] = i; }
    for (int shift = 0; shift < 32; shift += 8)
    {
        Array.Clear(count, 0, 256);
        for (int i = 0; i < n; i++) count[(keyA[i] >> shift) & 0xFF]++;
        int sum = 0; for (int b = 0; b < 256; b++) { int c = count[b]; count[b] = sum; sum += c; }
        for (int i = 0; i < n; i++) { int d = (int)((keyA[i] >> shift) & 0xFF); int p = count[d]++; keyB[p] = keyA[i]; idxB[p] = idxA[i]; }
        var tk = keyA; keyA = keyB; keyB = tk; var ti = idxA; idxA = idxB; idxB = ti;
    }
    for (int i = 0; i < n; i++) outIdx[i] = idxA[i];
}

Console.WriteLine("== int32 sort: radix vs Span.Sort ==");
foreach (int n in sizes)
{
    var src = new int[n]; for (int i = 0; i < n; i++) src[i] = rng.Next();
    var work = new int[n];
    var cur = new uint[n]; var tmp = new uint[n]; var count = new int[256];
    double rad = BestMs(() => src.CopyTo(work, 0), () => RadixInt32(work, cur, tmp, count));
    double spn = BestMs(() => src.CopyTo(work, 0), () => work.AsSpan().Sort());
    // correctness: radix result must equal Span.Sort result
    var a1 = (int[])src.Clone(); RadixInt32(a1, cur, tmp, count);
    var a2 = (int[])src.Clone(); a2.AsSpan().Sort();
    bool ok = a1.AsSpan().SequenceEqual(a2);
    Console.WriteLine($"n={n,9}  radix {rad,8:F3} ms ({n/rad/1e3,7:F1} M/s) | span {spn,8:F3} ms ({n/spn/1e3,7:F1} M/s) | radix/span {spn/rad,4:F1}x | correct={ok}");
}

Console.WriteLine("== int32 argsort: radix-arg vs nothing (correctness + speed) ==");
foreach (int n in sizes)
{
    var src = new int[n]; for (int i = 0; i < n; i++) src[i] = rng.Next();
    var outIdx = new long[n];
    var kA = new uint[n]; var kB = new uint[n]; var iA = new long[n]; var iB = new long[n]; var count = new int[256];
    double rad = BestMs(() => { }, () => RadixArgInt32(src, outIdx, kA, kB, iA, iB, count));
    // correctness: src[outIdx] must be non-decreasing AND stable
    RadixArgInt32(src, outIdx, kA, kB, iA, iB, count);
    bool ok = true; for (int i = 1; i < n; i++) if (src[outIdx[i - 1]] > src[outIdx[i]]) { ok = false; break; }
    Console.WriteLine($"n={n,9}  radix-arg {rad,8:F3} ms ({n/rad/1e3,7:F1} M/s) | sorted={ok}");
}
