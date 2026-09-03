#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// narrow_probe.cs — the 2-D block kernel challenged outside the regime the first report measured.
//
//   DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 \
//     dotnet run -c Release benchmark/nditer/probes/narrow_probe.cs [sections]
//   OPENBLAS_NUM_THREADS=1 python benchmark/nditer/probes/numpy_twins.py narrow        # NumPy side
//
// docs/NDITER_2D_BLOCK_KERNEL.md measured 2M f64 (DRAM-bound), widths >= one full vector, and ops
// with a vector body. This probe covers what it did not:
//   (A) f64 (rows, w) out= at 100K / 1M / 2M for w = 1..64 — sub-vector widths and cache-resident sizes
//   (B) other dtypes at 1M — f32 / u8 (RGB-style w=3) / i32
//   (C) ops WITHOUT a vector body (exp f64, mod) — those never reach the 2-D kernel
//   (D) the broadcast shapes the report lists as "left": add(A, col), view * scalar, add(A, row)
//   (E) 3-D: flattenable x[:, :, :w] vs non-flattenable x[::2, :, :w]
//   (F) hand-written floors for the narrow widths (what a perfect kernel could reach)
//
// Every row prints microseconds per call (best of rounds) and ns per outer row.
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp;
using NumSharp.Backends;

// NS_PROBE_AFFINITY=<hex mask> pins the process to given logical CPUs (e.g. 0x4 = one P-core on a
// hybrid part). On a P/E-core host an unpinned benchmark thread can land on an E-core and read
// 2-3x slower for EVERYTHING; pin the NumPy twin to the same mask (never run both at once).
if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } affinity)
    System.Diagnostics.Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(affinity, 16);

var sections = args.Length == 0 ? "ABCDEF" : string.Concat(args);

static double BestUs(Action body)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 60) body();
    var pilot = Stopwatch.StartNew(); int pc = 0; while (pilot.ElapsedMilliseconds < 20) { body(); pc++; }
    double perCallMs = pilot.Elapsed.TotalMilliseconds / pc;
    int it = Math.Clamp((int)Math.Round(2.0 / Math.Max(perCallMs, 1e-6)), 1, 100_000);
    int rds = Math.Max(5, (int)Math.Ceiling(150.0 / Math.Max(it * perCallMs, 1e-9)));
    rds = Math.Min(rds, 25);
    double best = double.MaxValue;
    for (int r = 0; r < rds; r++)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < it; i++) body();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1e3 / it);
    }
    return best;
}

unsafe
{
    if (sections.Contains('A'))
    {
        Console.WriteLine("=== (A) f64 (rows, w) strided view, out=: positive | add | sqrt ===");
        foreach (int total in new[] { 100_000, 1_000_000, 2_097_152 })
        {
            foreach (int w in new[] { 1, 2, 3, 4, 8, 16, 64 })
            {
                int rows = total / w;
                var back = np.arange(rows * 2 * w).astype(np.float64).reshape(rows, 2 * w);
                var back2 = (np.arange(rows * 2 * w).astype(np.float64) % 7.0).reshape(rows, 2 * w);
                var sv = back[$":, :{w}"]; var sv2 = back2[$":, :{w}"];
                var dst = np.empty(new Shape(rows, w), np.float64);
                double tPos = BestUs(() => np.positive(sv, dst));
                double tAdd = BestUs(() => np.add(sv, sv2, dst));
                double tSqrt = BestUs(() => np.sqrt(sv, dst));
                Console.WriteLine($"N={total,8} w={w,3} rows={rows,7}: positive {tPos,9:F1} [{tPos * 1e3 / rows,5:F2}]  add {tAdd,9:F1} [{tAdd * 1e3 / rows,5:F2}]  sqrt {tSqrt,9:F1} [{tSqrt * 1e3 / rows,5:F2}]   (us [ns/row])");
            }
        }
    }

    if (sections.Contains('B'))
    {
        Console.WriteLine("=== (B) 1M other dtypes (rows, w) out=: positive | add ===");
        void Dt(Type dt, string name, int[] widths)
        {
            const int total = 1_000_000;
            foreach (int w in widths)
            {
                int rows = total / w;
                var back = (np.arange(rows * 2 * w) % 100).astype(dt).reshape(rows, 2 * w);
                var back2 = (np.arange(rows * 2 * w) % 7).astype(dt).reshape(rows, 2 * w);
                var sv = back[$":, :{w}"]; var sv2 = back2[$":, :{w}"];
                var dst = np.empty(new Shape(rows, w), dt);
                double tPos = BestUs(() => np.positive(sv, dst));
                double tAdd = BestUs(() => np.add(sv, sv2, dst));
                Console.WriteLine($"{name,-4} w={w,3} rows={rows,7}: positive {tPos,9:F1} [{tPos * 1e3 / rows,5:F2}]  add {tAdd,9:F1} [{tAdd * 1e3 / rows,5:F2}]");
            }
        }
        Dt(np.float32, "f32", new[] { 3, 4, 8, 16 });
        Dt(np.uint8, "u8", new[] { 3, 16, 32 });
        Dt(np.int32, "i32", new[] { 3, 4, 8 });
    }

    if (sections.Contains('C'))
    {
        Console.WriteLine("=== (C) 1M f64 ops WITHOUT a vector body (per-chunk route): exp(out) | mod(sv, 3.0, out) | contiguous twins ===");
        foreach (int w in new[] { 4, 16, 64 })
        {
            const int total = 1_000_000; int rows = total / w;
            var back = ((np.arange(rows * 2 * w).astype(np.float64) % 13.0) + 0.5).reshape(rows, 2 * w);
            var sv = back[$":, :{w}"];
            var dst = np.empty(new Shape(rows, w), np.float64);
            var flat = np.ascontiguousarray(sv); var fdst = np.empty(new Shape(rows, w), np.float64);
            double tExp = BestUs(() => np.exp(sv, dst));
            double tExpC = BestUs(() => np.exp(flat, fdst));
            double tMod = BestUs(() => np.mod(sv, 3.0, dst));
            double tModC = BestUs(() => np.mod(flat, 3.0, fdst));
            Console.WriteLine($"w={w,3} rows={rows,7}: exp {tExp,9:F1} [{tExp * 1e3 / rows,5:F2}] (contig {tExpC,8:F1})   mod {tMod,9:F1} [{tMod * 1e3 / rows,5:F2}] (contig {tModC,8:F1})");
        }
    }

    if (sections.Contains('D'))
    {
        Console.WriteLine("=== (D) 1M f64 broadcast shapes, out=: add(A,col) | multiply(view,2.0) | view*2.0 | add(A,row) | add(view,col) ===");
        foreach (int c in new[] { 4, 16, 64 })
        {
            const int total = 1_000_000; int r = total / c;
            var A = (np.arange(total).astype(np.float64) % 97.0).reshape(r, c);
            var col = (np.arange(r).astype(np.float64) % 5.0).reshape(r, 1);
            var row = (np.arange(c).astype(np.float64) % 5.0).reshape(1, c);
            var back = np.arange(r * 2 * c).astype(np.float64).reshape(r, 2 * c);
            var sv = back[$":, :{c}"];
            var dst = np.empty(new Shape(r, c), np.float64);
            double tCol = BestUs(() => np.add(A, col, dst));
            double tMulS = BestUs(() => np.multiply(sv, 2.0, dst));
            double tOpS = BestUs(() => { var t = sv * 2.0; t.Dispose(); });
            double tRow = BestUs(() => np.add(A, row, dst));
            double tVCol = BestUs(() => np.add(sv, col, dst));
            Console.WriteLine($"c={c,3} rows={r,7}: add(A,col) {tCol,8:F1} [{tCol * 1e3 / r,5:F2}]  multiply(sv,2.0,out) {tMulS,8:F1} [{tMulS * 1e3 / r,5:F2}]  sv*2.0 {tOpS,8:F1}  add(A,row) {tRow,8:F1} [{tRow * 1e3 / r,5:F2}]  add(sv,col) {tVCol,8:F1} [{tVCol * 1e3 / r,5:F2}]");
        }
    }

    if (sections.Contains('E'))
    {
        Console.WriteLine("=== (E) 2M f64 3-D: flattenable x[:, :, :w] vs non-flattenable x[::2, :, :w], out=: positive | add ===");
        foreach (int w in new[] { 4, 16 })
        {
            const int total = 2_097_152; int d1 = 64; int d0 = total / (w * d1);
            var x = np.arange(d0 * d1 * 2 * w).astype(np.float64).reshape(d0, d1, 2 * w);
            var y = (np.arange(d0 * d1 * 2 * w).astype(np.float64) % 7.0).reshape(d0, d1, 2 * w);
            var flat = x[$":, :, :{w}"]; var flat2 = y[$":, :, :{w}"];
            var dstF = np.empty(new Shape(d0, d1, w), np.float64);
            var half = x[$"::2, :, :{w}"]; var half2 = y[$"::2, :, :{w}"];
            var dstH = np.empty(new Shape((d0 + 1) / 2, d1, w), np.float64);
            long rowsF = (long)d0 * d1, rowsH = (long)((d0 + 1) / 2) * d1;
            double tPF = BestUs(() => np.positive(flat, dstF)), tAF = BestUs(() => np.add(flat, flat2, dstF));
            double tPH = BestUs(() => np.positive(half, dstH)), tAH = BestUs(() => np.add(half, half2, dstH));
            Console.WriteLine($"w={w,3}: flat  positive {tPF,8:F1} [{tPF * 1e3 / rowsF,5:F2}]  add {tAF,8:F1} [{tAF * 1e3 / rowsF,5:F2}]   |  ::2  positive {tPH,8:F1} [{tPH * 1e3 / rowsH,5:F2}]  add {tAH,8:F1} [{tAH * 1e3 / rowsH,5:F2}]");
        }
    }

    if (sections.Contains('F'))
    {
        Console.WriteLine("=== (F) hand-written floors, f64 copy (positive) on (rows, w) — us [ns/row] ===");
        foreach (int total in new[] { 100_000, 1_000_000 })
        {
            foreach (int w in new[] { 1, 2, 3, 4, 8 })
            {
                int rows = total / w;
                var back = np.arange(rows * 2 * w).astype(np.float64).reshape(rows, 2 * w);
                var sv = back[$":, :{w}"];
                var dst = np.empty(new Shape(rows, w), np.float64);
                byte* ps = (byte*)sv.Address; byte* pd = (byte*)dst.Address;
                long ss = 2L * w * 8, sd = (long)w * 8;
                double tProd = BestUs(() => np.positive(sv, dst));
                double tGen = BestUs(() => Floors.GenericLoop(ps, pd, w, ss, sd, rows));
                double tSpec = w switch
                {
                    1 => BestUs(() => Floors.W1(ps, pd, ss, sd, rows)),
                    2 => BestUs(() => Floors.W2(ps, pd, ss, sd, rows)),
                    3 => BestUs(() => Floors.W3(ps, pd, ss, sd, rows)),
                    4 => BestUs(() => Floors.W4(ps, pd, ss, sd, rows)),
                    _ => BestUs(() => Floors.W8(ps, pd, ss, sd, rows)),
                };
                double tSpec4 = w switch
                {
                    1 => BestUs(() => Floors.W1x4(ps, pd, ss, sd, rows)),
                    2 => BestUs(() => Floors.W2pack(ps, pd, ss, sd, rows)),
                    4 => BestUs(() => Floors.W4x4(ps, pd, ss, sd, rows)),
                    _ => double.NaN,
                };
                Console.WriteLine($"N={total,8} w={w,2} rows={rows,7}: production {tProd,8:F1} [{tProd * 1e3 / rows,5:F2}]  generic-loop {tGen,8:F1} [{tGen * 1e3 / rows,5:F2}]  unrolled-w {tSpec,8:F1} [{tSpec * 1e3 / rows,5:F2}]  multi-row {tSpec4,8:F1} [{tSpec4 * 1e3 / rows,5:F2}]");
            }
        }
    }
}

static unsafe class Floors
{
    // The shape of the CURRENT emitted kernel: hoisted bounds, per-row unroll-check / 1-vector remainder / scalar tail.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GenericLoop(byte* ps, byte* pd, long w, long ss, long sd, long rows)
    {
        long unrollEnd = w - 16, vecEnd = w - 4;
        for (long r = 0; r < rows; r++)
        {
            long i = 0;
            for (; i <= unrollEnd; i += 16)
            {
                Vector256.Store(Vector256.Load((double*)ps + i), (double*)pd + i);
                Vector256.Store(Vector256.Load((double*)ps + i + 4), (double*)pd + i + 4);
                Vector256.Store(Vector256.Load((double*)ps + i + 8), (double*)pd + i + 8);
                Vector256.Store(Vector256.Load((double*)ps + i + 12), (double*)pd + i + 12);
            }
            for (; i <= vecEnd; i += 4)
                Vector256.Store(Vector256.Load((double*)ps + i), (double*)pd + i);
            for (; i < w; i++)
                ((double*)pd)[i] = ((double*)ps)[i];
            ps += ss; pd += sd;
        }
    }

    // Width-specialized straight-line row bodies (what a (key, w)-specialized kernel would emit).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void W1(byte* ps, byte* pd, long ss, long sd, long rows)
    {
        for (long r = 0; r < rows; r++) { *(double*)pd = *(double*)ps; ps += ss; pd += sd; }
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void W1x4(byte* ps, byte* pd, long ss, long sd, long rows)
    {
        long r = 0;
        for (; r + 4 <= rows; r += 4)
        {
            double a = *(double*)ps, b = *(double*)(ps + ss), c = *(double*)(ps + 2 * ss), d = *(double*)(ps + 3 * ss);
            *(double*)pd = a; *(double*)(pd + sd) = b; *(double*)(pd + 2 * sd) = c; *(double*)(pd + 3 * sd) = d;
            ps += 4 * ss; pd += 4 * sd;
        }
        for (; r < rows; r++) { *(double*)pd = *(double*)ps; ps += ss; pd += sd; }
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void W2(byte* ps, byte* pd, long ss, long sd, long rows)
    {
        for (long r = 0; r < rows; r++) { Vector128.Store(Vector128.Load((double*)ps), (double*)pd); ps += ss; pd += sd; }
    }
    // two rows packed into one 256-bit vector (Create(lo, hi) → op → GetLower/GetUpper)
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void W2pack(byte* ps, byte* pd, long ss, long sd, long rows)
    {
        long r = 0;
        for (; r + 2 <= rows; r += 2)
        {
            var v = Vector256.Create(Vector128.Load((double*)ps), Vector128.Load((double*)(ps + ss)));
            Vector128.Store(v.GetLower(), (double*)pd); Vector128.Store(v.GetUpper(), (double*)(pd + sd));
            ps += 2 * ss; pd += 2 * sd;
        }
        for (; r < rows; r++) { Vector128.Store(Vector128.Load((double*)ps), (double*)pd); ps += ss; pd += sd; }
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void W3(byte* ps, byte* pd, long ss, long sd, long rows)
    {
        for (long r = 0; r < rows; r++)
        {
            Vector128.Store(Vector128.Load((double*)ps), (double*)pd);
            ((double*)pd)[2] = ((double*)ps)[2];
            ps += ss; pd += sd;
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void W4(byte* ps, byte* pd, long ss, long sd, long rows)
    {
        for (long r = 0; r < rows; r++) { Vector256.Store(Vector256.Load((double*)ps), (double*)pd); ps += ss; pd += sd; }
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void W4x4(byte* ps, byte* pd, long ss, long sd, long rows)
    {
        long r = 0;
        for (; r + 4 <= rows; r += 4)
        {
            var a = Vector256.Load((double*)ps); var b = Vector256.Load((double*)(ps + ss));
            var c = Vector256.Load((double*)(ps + 2 * ss)); var d = Vector256.Load((double*)(ps + 3 * ss));
            Vector256.Store(a, (double*)pd); Vector256.Store(b, (double*)(pd + sd));
            Vector256.Store(c, (double*)(pd + 2 * sd)); Vector256.Store(d, (double*)(pd + 3 * sd));
            ps += 4 * ss; pd += 4 * sd;
        }
        for (; r < rows; r++) { Vector256.Store(Vector256.Load((double*)ps), (double*)pd); ps += ss; pd += sd; }
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void W8(byte* ps, byte* pd, long ss, long sd, long rows)
    {
        for (long r = 0; r < rows; r++)
        {
            Vector256.Store(Vector256.Load((double*)ps), (double*)pd);
            Vector256.Store(Vector256.Load((double*)ps + 4), (double*)pd + 4);
            ps += ss; pd += sd;
        }
    }
}
