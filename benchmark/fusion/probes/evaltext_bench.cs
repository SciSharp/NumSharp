#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaltext_bench.cs — np.evaluate over text (docs/plans/evaluate-text.md, phase E1), end to end.
//
// Times four trees over a text column and numeric columns, fused (np.evaluate) against the same statements unfused in
// NumSharp; the NumPy twin (evaltext_bench.py) times NumPy's statements over the identical data, and evaltext_sheet.py
// joins the two outputs into the plan's table (section 5, Gates).
//   filter   np.evaluate(NDExpr.Equal(names, "paris") & ((NDExpr)age > 30))
//   count    np.evaluate(NDExpr.CountNonzero(NDExpr.Equal(names, "paris")))
//   where    np.evaluate(NDExpr.Where(NDExpr.Less(names, "m"), (NDExpr)price * 2, price))
//   compare  np.evaluate(NDExpr.Equal(names, "paris"))
// Text kinds U (<U6), S (S6) and T (StringDType); layouts contiguous and strided ([::2] of a 2n array); n = 1K / 100K /
// 1M. The names are eight city names drawn by a 64-bit LCG that the twin reproduces bit for bit; age is int64, price
// float64. Every cell is checked fused == unfused before it is timed; best of 7 rounds after a paused double warm-up, two
// interleaved rounds per route so host drift hits both alike.
//
// Run (Release only; pinned to two P-cores — one CPU starves the background tier-1 compiler):
//   dotnet run -c Release benchmark/fusion/probes/evaltext_bench.cs > cs.out
//   python benchmark/fusion/probes/evaltext_bench.py > py.out
//   python benchmark/fusion/probes/evaltext_sheet.py cs.out py.out
// NS_PROBE_AFFINITY (hex) overrides the default mask 0x14; the twin reads the same variable.
// Output: one line per cell, kind,layout,n,tree,fused_us,unfused_us.
// =============================================================================
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends.Iteration;

// The pin is set before any array exists, so the tier-1 compiler threads inherit it with the rest of the process.
string affinity = Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } mask ? mask : "14";
Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(affinity, 16);

#if DEBUG
Console.WriteLine("FATAL: Debug build — rerun with -c Release");
return;
#pragma warning disable CS0162
#endif

// A Release script over a Debug NumSharp.Core still times unoptimized kernels (about 2x): refuse it.
var dbg = (DebuggableAttribute?)Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute));
if (dbg is { IsJITOptimizerDisabled: true })
{
    Console.WriteLine("FATAL: NumSharp.Core built Debug (JIT optimizer disabled) — rerun with -c Release");
    return;
}

var words = new[] { "paris", "london", "berlin", "madrid", "rome", "vienna", "oslo", "lisbon" };

foreach (string kind in new[] { "U", "S", "T" })
foreach (string layout in new[] { "contig", "strided" })
foreach (int n in new[] { 1_000, 100_000, 1_000_000 })
{
    // The data: a strided operand is every second element of a 2n array, so both layouts read n elements.
    int total = layout == "contig" ? n : 2 * n;
    var strs = new string[total];
    ulong state = 12345;
    for (int i = 0; i < total; i++)
    {
        // Knuth's MMIX LCG; bits 33..63 pick the word, the twin's (state >> 33) % 8 exactly.
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        strs[i] = words[(int)(state >> 33) % words.Length];
    }
    NDArray baseNames = kind switch
    {
        "U" => np.array(strs),
        "S" => np.array(strs).astype(np.dtype("S6")),
        _ => np.array(strs).astype(np.dtypes.StringDType()),
    };
    NDArray baseAge = np.arange(total).astype(NPTypeCode.Int64) % 97;
    NDArray basePrice = np.arange(total).astype(NPTypeCode.Double) * 0.5;
    NDArray names = layout == "contig" ? baseNames : baseNames["::2"];
    NDArray age = layout == "contig" ? baseAge : baseAge["::2"];
    NDArray price = layout == "contig" ? basePrice : basePrice["::2"];
    // NumPy compares S with bytes and U / T with str: an S column gets a 0-d S literal, the others a C# string (str).
    NDArray paris = kind == "S" ? np.array(new[] { "paris" }).astype(np.dtype("S5")).reshape(Shape.NewScalar()) : (NDArray)"paris";
    NDArray m = kind == "S" ? np.array(new[] { "m" }).astype(np.dtype("S1")).reshape(Shape.NewScalar()) : (NDArray)"m";

    var trees = new (string name, Func<NDArray> fused, Func<NDArray> unfused)[]
    {
        ("filter", () => np.evaluate(NDExpr.Equal(names, paris) & ((NDExpr)age > 30)),
                   () => { using var a = np.equal(names, paris); using var b = age > 30; return a & b; }),
        ("count", () => np.evaluate(NDExpr.CountNonzero(NDExpr.Equal(names, paris))),
                  () => { using var a = np.equal(names, paris); return np.count_nonzero(a); }),
        ("where", () => np.evaluate(NDExpr.Where(NDExpr.Less(names, m), (NDExpr)price * 2, price)),
                  () => { using var c = np.less(names, m); using var p2 = price * 2; return np.where(c, p2, price); }),
        ("compare", () => np.evaluate(NDExpr.Equal(names, paris)),
                    () => np.equal(names, paris)),
    };

    foreach (var (name, fused, unfused) in trees)
    {
        // A cell is timed only once both routes agree on every element.
        using (var r1 = fused())
        using (var r2 = unfused())
            if (!np.array_equal(r1, r2))
                throw new Exception($"{kind} {layout} {n} {name}: fused != unfused");

        // Best per-call time of f in microseconds. Two warm-up bursts with a pause between them: the first queues the
        // hot methods for tier-1, the pause lets the background compiler install them, the second runs the tier-1 code
        // once before timing starts. Results are disposed per call so the buffer pool, not the GC, recycles them.
        double Time(Func<NDArray> f)
        {
            for (int w = 0; w < 20; w++) f().Dispose();
            Thread.Sleep(100);
            for (int w = 0; w < 20; w++) f().Dispose();
            int reps = n <= 1000 ? 3000 : n <= 100_000 ? 200 : 25;
            double best = double.MaxValue;
            for (int r = 0; r < 7; r++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < reps; i++) f().Dispose();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000 / reps);
            }
            return best;
        }

        double tf = double.MaxValue, tu = double.MaxValue;
        for (int round = 0; round < 2; round++)
        {
            tf = Math.Min(tf, Time(fused));
            tu = Math.Min(tu, Time(unfused));
        }
        Console.WriteLine($"{kind},{layout},{n},{name},{tf:F2},{tu:F2}");
    }
}
