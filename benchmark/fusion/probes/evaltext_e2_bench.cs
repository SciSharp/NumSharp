#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaltext_e2_bench.cs — np.evaluate over text, phase E2 (docs/plans/evaluate-text.md), end to end.
//
// Times eight phase-E2 trees fused (np.evaluate) against the same statements unfused in NumSharp; the NumPy twin
// (evaltext_e2_bench.py) times NumPy's statements over the identical data, and evaltext_sheet.py joins the outputs.
//   truth_and   np.evaluate(NDExpr.LogicalAnd(names, (NDExpr)age > 30))                    the truth of text, fused
//   truth_where np.evaluate(NDExpr.Where(names, (NDExpr)price * 2, price))                 a text condition
//   truth_count np.evaluate(NDExpr.CountNonzero(names))                                    a truth reduction
//   strlen      np.evaluate((NDExpr.StrLen(names) > 5) & ((NDExpr)age > 30))              str_len, fused
//   starts      np.evaluate(NDExpr.CountNonzero(NDExpr.StartsWith(names, "pa") & ((NDExpr)age > 30)))
//   find        np.evaluate(NDExpr.Where(NDExpr.Find(names, "o") >= 0, price, 0))
//   isalpha     np.evaluate(NDExpr.IsAlpha(names))                                         a bare stage
//   cast        np.evaluate(NDExpr.Cast(digits, NPTypeCode.Int64) * 2)                     NumPy's parse, fused
// Text kinds U, S and T (StringDType); layouts contiguous and strided ([::2] of a 2n array); n = 1K / 100K / 1M. The
// names are eight city names drawn by a 64-bit LCG the twin reproduces bit for bit, the digit texts that LCG's values
// below 1000; age is int64, price float64. Every cell is checked fused == unfused before it is timed; best of 7 rounds
// after a paused double warm-up, two interleaved rounds per route.
//
// Run (Release only; pinned to two P-cores — one CPU starves the background tier-1 compiler):
//   dotnet run -c Release --artifacts-path <its own folder> benchmark/fusion/probes/evaltext_e2_bench.cs > cs.out
//   python benchmark/fusion/probes/evaltext_e2_bench.py > py.out
//   python benchmark/fusion/probes/evaltext_sheet.py cs.out py.out
// Give every probe script its own --artifacts-path: scripts sharing AssemblyName and one artifacts folder overwrite one
// another's build. NS_PROBE_AFFINITY (hex) overrides the default mask 0x14; the twin reads the same variable.
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

// The text operand of a kind: U as built, S6 bytes, or StringDType.
NDArray AsKind(string[] texts, string kind, string width) => kind switch
{
    "U" => np.array(texts),
    "S" => np.array(texts).astype(np.dtype(width)),
    _ => np.array(texts).astype(np.dtypes.StringDType()),
};

// A text literal of a kind: a C# string (NumPy's str) for U / T, a 0-d S array (NumPy's bytes) for S.
NDArray Literal(string text, string kind)
    => kind == "S" ? np.array(new[] { text }).astype(np.dtype($"S{text.Length}")).reshape(Shape.NewScalar()) : (NDArray)text;

foreach (string kind in new[] { "U", "S", "T" })
foreach (string layout in new[] { "contig", "strided" })
foreach (int n in new[] { 1_000, 100_000, 1_000_000 })
{
    int total = layout == "contig" ? n : 2 * n;
    var strs = new string[total];
    var digitTexts = new string[total];
    ulong state = 12345;
    for (int i = 0; i < total; i++)
    {
        // Knuth's MMIX LCG; bits 33..63 pick the word, the twin's (state >> 33) % 8 exactly.
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        strs[i] = words[(int)(state >> 33) % words.Length];
        digitTexts[i] = ((state >> 33) % 1000).ToString();
    }
    NDArray baseNames = AsKind(strs, kind, "S6");
    NDArray baseDigits = AsKind(digitTexts, kind, "S3");
    NDArray baseAge = np.arange(total).astype(NPTypeCode.Int64) % 97;
    NDArray basePrice = np.arange(total).astype(NPTypeCode.Double) * 0.5;
    NDArray names = layout == "contig" ? baseNames : baseNames["::2"];
    NDArray digits = layout == "contig" ? baseDigits : baseDigits["::2"];
    NDArray age = layout == "contig" ? baseAge : baseAge["::2"];
    NDArray price = layout == "contig" ? basePrice : basePrice["::2"];
    NDArray pa = Literal("pa", kind);
    NDArray o = Literal("o", kind);

    var trees = new (string name, Func<NDArray> fused, Func<NDArray> unfused)[]
    {
        ("truth_and", () => np.evaluate(NDExpr.LogicalAnd(names, (NDExpr)age > 30)),
                      () => { using var b = age > 30; return np.logical_and(names, b); }),
        ("truth_where", () => np.evaluate(NDExpr.Where(names, (NDExpr)price * 2, price)),
                        () => { using var p2 = price * 2; return np.where(names, p2, price); }),
        ("truth_count", () => np.evaluate(NDExpr.CountNonzero(names)),
                        () => NDArray.Scalar(np.count_nonzero(names))),
        ("strlen", () => np.evaluate((NDExpr.StrLen(names) > 5) & ((NDExpr)age > 30)),
                   () => { using var l = np.strings.str_len(names); using var a = l > 5; using var b = age > 30; return a & b; }),
        ("starts", () => np.evaluate(NDExpr.CountNonzero(NDExpr.StartsWith(names, pa) & ((NDExpr)age > 30))),
                   () => { using var s = np.strings.startswith(names, pa); using var b = age > 30; using var m = s & b; return NDArray.Scalar(np.count_nonzero(m)); }),
        ("find", () => np.evaluate(NDExpr.Where(NDExpr.Find(names, o) >= 0, price, 0)),
                 () => { using var f = np.strings.find(names, o); using var c = f >= 0; return np.where(c, price, 0); }),
        ("isalpha", () => np.evaluate(NDExpr.IsAlpha(names)),
                    () => np.strings.isalpha(names)),
        ("cast", () => np.evaluate(NDExpr.Cast(digits, NPTypeCode.Int64) * 2),
                 () => { using var v = digits.astype(NPTypeCode.Int64); return v * 2; }),
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
