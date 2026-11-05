#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaltext_e3_bench.cs — np.evaluate over text, phase E3 (docs/plans/evaluate-text.md), end to end.
//
// Times seven phase-E3 trees fused (np.evaluate) against the same statements unfused in NumSharp; the NumPy twin
// (evaltext_e3_bench.py) times NumPy's statements over the identical data, and evaltext_sheet.py joins the outputs.
//   concat       np.evaluate(NDExpr.Add(names, "!"))                                       a text result (bare stage)
//   concat_where np.evaluate(NDExpr.Where((NDExpr)age > 30, NDExpr.Add(names, "+"), names))  text branches, a computed
//                                                                                           condition
//   where_text   np.evaluate(NDExpr.Where((NDExpr)age > 30, names, "-"))                   NumPy's where over text
//   eq_concat    np.evaluate(NDExpr.Equal(NDExpr.Add(names, "x"), "parisx"))              a text result read by a
//                                                                                           comparison
//   strlen_cat   np.evaluate(NDExpr.CountNonzero(NDExpr.StrLen(NDExpr.Add(names, names)) > 10))  read by numpy.strings,
//                                                                                           reduced
//   argmax       np.evaluate(NDExpr.ArgMax(names))                                         a reduction of text
//   cast_text    np.evaluate(NDExpr.Cast(age, "U2"))                                       numbers formatted as text
// Text kinds U, S and T (StringDType); layouts contiguous and strided ([::2] of a 2n array); n = 1K / 100K / 1M. The
// names are eight city names drawn by a 64-bit LCG the twin reproduces bit for bit; age is int64. An S literal is a 0-d S
// array (NumPy's bytes); a U / T one a C# string (NumPy's str). Every cell is checked fused == unfused before it is
// timed; best of 7 rounds after a paused double warm-up, two interleaved rounds per route.
//
// Run (Release only; pinned to two P-cores — one CPU starves the background tier-1 compiler):
//   dotnet run -c Release --artifacts-path <its own folder> benchmark/fusion/probes/evaltext_e3_bench.cs > cs.out
//   python benchmark/fusion/probes/evaltext_e3_bench.py > py.out
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
NDArray AsKind(string[] texts, string kind) => kind switch
{
    "U" => np.array(texts),
    "S" => np.array(texts).astype(np.dtype("S6")),
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
    ulong state = 12345;
    for (int i = 0; i < total; i++)
    {
        // Knuth's MMIX LCG; bits 33..63 pick the word, the twin's (state >> 33) % 8 exactly.
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        strs[i] = words[(int)(state >> 33) % words.Length];
    }
    NDArray baseNames = AsKind(strs, kind);
    NDArray baseAge = np.arange(total).astype(NPTypeCode.Int64) % 97;
    NDArray names = layout == "contig" ? baseNames : baseNames["::2"];
    NDArray age = layout == "contig" ? baseAge : baseAge["::2"];
    NDArray bang = Literal("!", kind), plus = Literal("+", kind), dash = Literal("-", kind), x = Literal("x", kind);
    NDArray parisx = Literal("parisx", kind);

    var trees = new (string name, Func<NDArray> fused, Func<NDArray> unfused)[]
    {
        ("concat", () => np.evaluate(NDExpr.Add(names, bang)),
                   () => np.add(names, bang)),
        ("concat_where", () => np.evaluate(NDExpr.Where((NDExpr)age > 30, NDExpr.Add(names, plus), names)),
                         () => { using var c = age > 30; using var a = np.add(names, plus); return np.where(c, a, names); }),
        ("where_text", () => np.evaluate(NDExpr.Where((NDExpr)age > 30, names, dash)),
                       () => { using var c = age > 30; return np.where(c, names, dash); }),
        ("eq_concat", () => np.evaluate(NDExpr.Equal(NDExpr.Add(names, x), parisx)),
                      () => { using var a = np.add(names, x); return np.equal(a, parisx); }),
        ("strlen_cat", () => np.evaluate(NDExpr.CountNonzero(NDExpr.StrLen(NDExpr.Add(names, names)) > 10)),
                       () => { using var a = np.add(names, names); using var l = np.strings.str_len(a); using var m = l > 10; return NDArray.Scalar(np.count_nonzero(m)); }),
        ("argmax", () => np.evaluate(NDExpr.ArgMax(names)),
                   () => NDArray.Scalar(np.argmax(names))),
        ("cast_text", () => np.evaluate(NDExpr.Cast(age, "U2")),
                      () => age.astype(np.dtype("U2"))),
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
