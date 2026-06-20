#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// POC2: 11-bit radix (fewer passes) for int32; 8-bit radix for int64 & float64 (+NaN-last partition).
using System;
using System.Diagnostics;

int[] sizes = { 1_000_000, 10_000_000 };
var rng = new Random(42);
static double BestMs(Action setup, Action timed)
{
    double best = 1e18;
    for (int r = 0; r < 7; r++) { setup(); var sw = Stopwatch.StartNew(); timed(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
    return best;
}

// ---- 11-bit radix int32: 3 passes (11+11+10 bits) ----
static void Radix11Int32(int[] a, uint[] cur, uint[] tmp, int[] count)
{
    const int BITS = 11, MASK = (1 << BITS) - 1;
    int n = a.Length;
    for (int i = 0; i < n; i++) cur[i] = (uint)a[i] ^ 0x80000000u;
    for (int shift = 0; shift < 32; shift += BITS)
    {
        Array.Clear(count, 0, MASK + 1);
        for (int i = 0; i < n; i++) count[(cur[i] >> shift) & MASK]++;
        int sum = 0; for (int b = 0; b <= MASK; b++) { int c = count[b]; count[b] = sum; sum += c; }
        for (int i = 0; i < n; i++) { int d = (int)((cur[i] >> shift) & MASK); tmp[count[d]++] = cur[i]; }
        var t = cur; cur = tmp; tmp = t;
    }
    // 3 passes => result in tmp-after-odd-swaps == cur variable now points to last-written
    for (int i = 0; i < n; i++) a[i] = (int)(cur[i] ^ 0x80000000u);
}

// ---- 8-bit radix int64: 8 passes ----
static void Radix8Int64(long[] a, ulong[] cur, ulong[] tmp, int[] count)
{
    int n = a.Length;
    for (int i = 0; i < n; i++) cur[i] = (ulong)a[i] ^ 0x8000000000000000UL;
    for (int shift = 0; shift < 64; shift += 8)
    {
        Array.Clear(count, 0, 256);
        for (int i = 0; i < n; i++) count[(int)((cur[i] >> shift) & 0xFF)]++;
        int sum = 0; for (int b = 0; b < 256; b++) { int c = count[b]; count[b] = sum; sum += c; }
        for (int i = 0; i < n; i++) { int d = (int)((cur[i] >> shift) & 0xFF); tmp[count[d]++] = cur[i]; }
        var t = cur; cur = tmp; tmp = t;
    }
    for (int i = 0; i < n; i++) a[i] = (long)(cur[i] ^ 0x8000000000000000UL);
}

// ---- radix float64: NaN-partition to end, then bit-transform radix on the non-NaN prefix ----
static int PartitionNaNToEnd(double[] a)
{
    int n = a.Length, w = 0;
    for (int i = 0; i < n; i++) if (!double.IsNaN(a[i])) a[w++] = a[i];
    for (int i = w; i < n; i++) a[i] = double.NaN;
    return w; // count of non-NaN
}
static void Radix8Double(double[] a, int m, ulong[] cur, ulong[] tmp, int[] count)
{
    // monotonic double->uint64 key: flip sign bit if positive, flip all if negative
    for (int i = 0; i < m; i++)
    {
        ulong b = BitConverter.DoubleToUInt64Bits(a[i]);
        b ^= (ulong)((long)b >> 63) | 0x8000000000000000UL;
        cur[i] = b;
    }
    for (int shift = 0; shift < 64; shift += 8)
    {
        Array.Clear(count, 0, 256);
        for (int i = 0; i < m; i++) count[(int)((cur[i] >> shift) & 0xFF)]++;
        int sum = 0; for (int b = 0; b < 256; b++) { int c = count[b]; count[b] = sum; sum += c; }
        for (int i = 0; i < m; i++) { int d = (int)((cur[i] >> shift) & 0xFF); tmp[count[d]++] = cur[i]; }
        var t = cur; cur = tmp; tmp = t;
    }
    for (int i = 0; i < m; i++)
    {
        ulong b = cur[i];
        b ^= ((b >> 63) - 1) | 0x8000000000000000UL;
        a[i] = BitConverter.UInt64BitsToDouble(b);
    }
}

Console.WriteLine("== int32 sort: 11-bit radix (vs NumPy 264/227 M/s @1M/10M) ==");
foreach (int n in sizes)
{
    var src = new int[n]; for (int i = 0; i < n; i++) src[i] = rng.Next();
    var work = new int[n]; var cur = new uint[n]; var tmp = new uint[n]; var count = new int[2048];
    double r = BestMs(() => src.CopyTo(work, 0), () => Radix11Int32(work, cur, tmp, count));
    var a1 = (int[])src.Clone(); Radix11Int32(a1, cur, tmp, count); var a2 = (int[])src.Clone(); a2.AsSpan().Sort();
    Console.WriteLine($"n={n,9}  radix11 {r,8:F3} ms ({n/r/1e3,7:F1} M/s)  correct={a1.AsSpan().SequenceEqual(a2)}");
}
Console.WriteLine("== int64 sort: 8-bit radix (vs NumPy 95/83 M/s) ==");
foreach (int n in sizes)
{
    var src = new long[n]; for (int i = 0; i < n; i++) src[i] = ((long)rng.Next() << 20) ^ rng.Next();
    var work = new long[n]; var cur = new ulong[n]; var tmp = new ulong[n]; var count = new int[256];
    double r = BestMs(() => src.CopyTo(work, 0), () => Radix8Int64(work, cur, tmp, count));
    var a1 = (long[])src.Clone(); Radix8Int64(a1, cur, tmp, count); var a2 = (long[])src.Clone(); a2.AsSpan().Sort();
    Console.WriteLine($"n={n,9}  radix64 {r,8:F3} ms ({n/r/1e3,7:F1} M/s)  correct={a1.AsSpan().SequenceEqual(a2)}");
}
Console.WriteLine("== float64 sort: NaN-partition + 8-bit radix (vs NumPy 137/114 M/s) ==");
foreach (int n in sizes)
{
    var src = new double[n]; for (int i = 0; i < n; i++) src[i] = rng.NextDouble() * 2 - 1;
    // sprinkle some NaNs
    for (int i = 0; i < n; i += 997) src[i] = double.NaN;
    var work = new double[n]; var cur = new ulong[n]; var tmp = new ulong[n]; var count = new int[256];
    double r = BestMs(() => src.CopyTo(work, 0), () => { int m = PartitionNaNToEnd(work); Radix8Double(work, m, cur, tmp, count); });
    // correctness vs NumPy-style comparer sort
    var a1 = (double[])src.Clone(); { int m = PartitionNaNToEnd(a1); Radix8Double(a1, m, cur, tmp, count); }
    var a2 = (double[])src.Clone(); Array.Sort(a2, (x, y) => { if (x < y) return -1; if (x > y) return 1; bool xn = double.IsNaN(x), yn = double.IsNaN(y); return xn == yn ? 0 : (xn ? 1 : -1); });
    bool ok = true; for (int i = 0; i < n; i++) { if (double.IsNaN(a1[i]) != double.IsNaN(a2[i]) || (!double.IsNaN(a1[i]) && a1[i] != a2[i])) { ok = false; break; } }
    Console.WriteLine($"n={n,9}  radixF64 {r,8:F3} ms ({n/r/1e3,7:F1} M/s)  correct={ok}");
}
