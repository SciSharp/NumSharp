#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// POC3: optimized LSD radix — ONE combined-histogram pass + skip-trivial-digit.
// 8-byte types: 1 read pass builds all 8 histograms; passes whose digit is constant are skipped.
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

// Generic-ish 8-bit radix over ulong keys with combined histogram + skip.
// keys already transformed; sorts cur in place (using tmp scratch); returns buffer holding result.
static ulong[] RadixUlong(ulong[] cur, ulong[] tmp, int n, int nbytes, int[][] hist)
{
    // 1) one pass: all histograms
    for (int p = 0; p < nbytes; p++) Array.Clear(hist[p], 0, 256);
    for (int i = 0; i < n; i++)
    {
        ulong k = cur[i];
        for (int p = 0; p < nbytes; p++) hist[p][(int)((k >> (8 * p)) & 0xFF)]++;
    }
    // 2) scatter only non-trivial passes
    var src = cur; var dst = tmp;
    for (int p = 0; p < nbytes; p++)
    {
        int[] h = hist[p];
        if (h[(int)((src[0] >> (8 * p)) & 0xFF)] == n) continue; // a bucket holds everything? quick check below is better
        // robust trivial check: any bucket == n
        bool trivial = false;
        for (int b = 0; b < 256; b++) { if (h[b] == n) { trivial = true; break; } if (h[b] != 0 && h[b] != n) break; }
        if (trivial) continue;
        int sum = 0; for (int b = 0; b < 256; b++) { int c = h[b]; h[b] = sum; sum += c; }
        for (int i = 0; i < n; i++) { int d = (int)((src[i] >> (8 * p)) & 0xFF); dst[h[d]++] = src[i]; }
        var t = src; src = dst; dst = t;
    }
    return src;
}

Console.WriteLine("== int32 (4-byte) optimized radix (NumPy 264/227 M/s) ==");
foreach (int n in sizes)
{
    var s = new int[n]; for (int i = 0; i < n; i++) s[i] = rng.Next();
    var cur = new ulong[n]; var tmp = new ulong[n]; var hist = new int[4][]; for (int p = 0; p < 4; p++) hist[p] = new int[256];
    var work = new int[n];
    double r = BestMs(() => s.CopyTo(work, 0), () =>
    {
        for (int i = 0; i < n; i++) cur[i] = (uint)work[i] ^ 0x80000000u;
        var res = RadixUlong(cur, tmp, n, 4, hist);
        for (int i = 0; i < n; i++) work[i] = (int)((uint)res[i] ^ 0x80000000u);
    });
    var a2 = (int[])s.Clone(); a2.AsSpan().Sort();
    Console.WriteLine($"n={n,9}  {r,8:F3} ms ({n/r/1e3,7:F1} M/s)  correct={work.AsSpan().SequenceEqual(a2)}");
}

Console.WriteLine("== int64 (8-byte) optimized radix (NumPy 95/83 M/s) ==");
foreach (int n in sizes)
{
    var s = new long[n]; for (int i = 0; i < n; i++) s[i] = ((long)rng.Next() << 20) ^ rng.Next();
    var cur = new ulong[n]; var tmp = new ulong[n]; var hist = new int[8][]; for (int p = 0; p < 8; p++) hist[p] = new int[256];
    var work = new long[n];
    double r = BestMs(() => s.CopyTo(work, 0), () =>
    {
        for (int i = 0; i < n; i++) cur[i] = (ulong)work[i] ^ 0x8000000000000000UL;
        var res = RadixUlong(cur, tmp, n, 8, hist);
        for (int i = 0; i < n; i++) work[i] = (long)(res[i] ^ 0x8000000000000000UL);
    });
    var a2 = (long[])s.Clone(); a2.AsSpan().Sort();
    Console.WriteLine($"n={n,9}  {r,8:F3} ms ({n/r/1e3,7:F1} M/s)  correct={work.AsSpan().SequenceEqual(a2)}");
}

Console.WriteLine("== float64 optimized radix, full random + limited-range (NumPy 137/114 M/s) ==");
foreach (var (label, gen) in new (string, Func<int, double>)[] { ("limited[-1,1]", i => rng.NextDouble() * 2 - 1), ("fullrange", i => BitConverter.Int64BitsToDouble(((long)rng.Next() << 32) ^ rng.Next())) })
{
    foreach (int n in sizes)
    {
        var s = new double[n]; for (int i = 0; i < n; i++) { double v = gen(i); s[i] = double.IsNaN(v) ? 0 : v; }
        for (int i = 0; i < n; i += 997) s[i] = double.NaN;
        var cur = new ulong[n]; var tmp = new ulong[n]; var hist = new int[8][]; for (int p = 0; p < 8; p++) hist[p] = new int[256];
        var work = new double[n];
        double r = BestMs(() => s.CopyTo(work, 0), () =>
        {
            int m = 0; for (int i = 0; i < n; i++) if (!double.IsNaN(work[i])) work[m++] = work[i];
            int nn = m;
            for (int i = nn; i < n; i++) work[i] = double.NaN;
            for (int i = 0; i < nn; i++) { ulong b = BitConverter.DoubleToUInt64Bits(work[i]); b ^= (ulong)((long)b >> 63) | 0x8000000000000000UL; cur[i] = b; }
            var res = RadixUlong(cur, tmp, nn, 8, hist);
            for (int i = 0; i < nn; i++) { ulong b = res[i]; b ^= ((b >> 63) - 1) | 0x8000000000000000UL; work[i] = BitConverter.UInt64BitsToDouble(b); }
        });
        var a2 = (double[])s.Clone(); Array.Sort(a2, (x, y) => { if (x < y) return -1; if (x > y) return 1; bool xn = double.IsNaN(x), yn = double.IsNaN(y); return xn == yn ? 0 : (xn ? 1 : -1); });
        bool ok = true; for (int i = 0; i < n; i++) { if (double.IsNaN(work[i]) != double.IsNaN(a2[i]) || (!double.IsNaN(work[i]) && work[i] != a2[i])) { ok = false; break; } }
        Console.WriteLine($"{label,-14} n={n,9}  {r,8:F3} ms ({n/r/1e3,7:F1} M/s)  correct={ok}");
    }
}
