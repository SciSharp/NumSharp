#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// strings_bench.cs — NumSharp side of the benchmark/strings subsystem (strings phase 2, P2.7).
// Times NumSharp's text computations over S, U and StringDType (T) on the SAME arrays the NumPy twin
// (strings_bench.py) wrote with np.save into NUMSHARP_STRINGS_BENCH_DATA — loaded with np.load_npy (a T array is
// NumPy's pickle, read with allow_pickle) — at 1K and 100K elements, 1M for a core set, and stepped views. Every call runs
// once untimed (40 times at 1K) before the first one is timed, so no cell measures tier-0 code.
//   Output key:  {op}|{kind}|{n}\t{ms}     (identical keys to the NumPy side)
//   A call NumSharp refuses for a kind emits nothing, as the twin does for NumPy's refusals.
//   Driven by:   strings_sheet.py  →  strings_results.md (the benchmark/strings subsystem of run_benchmark.py)
// Run the twin first (it writes the data), then:  dotnet run -c Release - < benchmark/strings/strings_bench.cs
// =============================================================================
using System.Diagnostics;
using NumSharp;

// Host pin: a driven run (the sheet / run_benchmark.py) already inherits the pinned core; this makes a
// standalone `dotnet run -c Release - < file` honor NUMSHARP_BENCHMARK_AFFINITY (or the probes' NS_PROBE_AFFINITY).
if ((Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY")
     ?? Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY")) is { Length: > 0 } hostAffinity)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(hostAffinity, 16);

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core — rerun with -c Release"); return; }

string dataDir = Environment.GetEnvironmentVariable("NUMSHARP_STRINGS_BENCH_DATA");
if (string.IsNullOrEmpty(dataDir) || !Directory.Exists(dataDir))
{
    Console.Error.WriteLine("NUMSHARP_STRINGS_BENCH_DATA must name the directory strings_bench.py wrote its data into");
    return;
}

// Best-of-rounds ms per call, the twin's policy: a warm-up whose timed tail is the pilot, ~200 ms of batched rounds,
// and ~1 s of single-call rounds (5 to 100) for a call over 20 ms.
double Best(Action f)
{
    string depth = (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_DEPTH") ?? "measure").ToLowerInvariant();
    if (depth == "pass") { var once = Stopwatch.StartNew(); f(); once.Stop(); return once.Elapsed.TotalMilliseconds; }
    int wm = depth == "light" ? 3 : 5;
    f();
    var pw = Stopwatch.StartNew();
    for (int i = 1; i < wm; i++) f();
    pw.Stop();
    double perCall = pw.Elapsed.TotalMilliseconds / (wm - 1);
    bool slow = perCall > 20.0;
    int it = slow ? 1 : Math.Clamp((int)Math.Round(1.0 / Math.Max(perCall, 1e-6)), 1, 1_000_000);
    double budget = 200.0 / (depth == "light" ? 6.0 : 1.0);
    int rd = slow ? (depth == "light" ? 3 : Math.Clamp((int)(1000.0 / perCall) + 1, 5, 100))
                  : Math.Max(1, (int)(budget / Math.Max(it * perCall, 1e-9)) + 1);
    double b = 1e9;
    for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); }
    return b;
}

// A result is disposed after each call (its buffer back to the pool, as NumPy frees it); a tuple result each part.
void Drop(object r)
{
    switch (r)
    {
        case NDArray a: a.Dispose(); break;
        case ValueTuple<NDArray, NDArray, NDArray> t: t.Item1.Dispose(); t.Item2.Dispose(); t.Item3.Dispose(); break;
    }
}

int[] sizes = { 1_000, 100_000 };
const int CoreSize = 1_000_000;
var coreOps = new HashSet<string> { "eq", "add", "sort", "str_len", "upper", "astype_f8" };
string[] kinds = { "u8", "u8n", "u40", "s8", "t8", "t40", "tna", "u8d", "s8d", "t8d" };
string[] coreKinds = { "u8", "t8", "u8d", "t8d" };
var sTable = new byte[256];
for (int i = 0; i < 256; i++) sTable[i] = (byte)i;
sTable['a'] = (byte)'x'; sTable['b'] = (byte)'y'; sTable['c'] = (byte)'z';
var uTable = new Dictionary<int, string> { [97] = "x", [98] = "y", [99] = "z" };

NDArray Load(string kind, int n) => np.load_npy(Path.Combine(dataDir, $"{kind.Replace('@', '_')}_{n}.npy"), allow_pickle: true);

// A search / fill argument of the operand's text class: a 0-d S array (Python bytes) for S, a str otherwise.
NDArray Needle(NDArray a, string text) => a.dtype.kind == 'S' ? np.array(text).astype(np.dtype("S" + text.Length)) : np.array(text);

List<(string op, Func<object> call)> OpsFor(string kind, NDArray a, NDArray b)
{
    string textKind = kind.Split('@')[0];
    if (textKind is "u8d" or "s8d" or "t8d")
        return new() { ("astype_f8", () => a.astype(np.float64)), ("isdigit", () => np.strings.isdigit(a)) };
    var head = a[":100"];
    var nA = Needle(a, "a");
    var nAb = Needle(a, "ab");
    var nXy = Needle(a, "xy");
    var ops = new List<(string, Func<object>)>
    {
        ("eq", () => a == b),
        ("lt", () => a < b),
        ("add", () => np.strings.add(a, b)),
        ("mul3", () => np.strings.multiply(a, np.array(3L))),
        ("max", () => np.max(a)),
        ("sort", () => np.sort(a)),
        ("argsort", () => np.argsort(a)),
        ("unique", () => (NDArray)np.unique(a)),
        ("isin", () => np.isin(a, head)),
        ("str_len", () => np.strings.str_len(a)),
        ("find", () => np.strings.find(a, nA)),
        ("count", () => np.strings.count(a, nA)),
        ("startswith", () => np.strings.startswith(a, nA)),
        ("strip", () => np.strings.strip(a, nAb)),
        ("replace", () => np.strings.replace(a, nA, nXy)),
        ("center", () => np.strings.center(a, np.array(12L))),
        ("zfill", () => np.strings.zfill(a, np.array(10L))),
        ("partition", () => np.strings.partition(a, nA)),
        ("slice", () => np.strings.slice(a, np.array(1L), np.array(4L))),
        ("upper", () => np.strings.upper(a)),
        ("title", () => np.strings.title(a)),
        ("isalpha", () => np.strings.isalpha(a)),
        ("translate", () => np.strings.translate(a, a.dtype.kind == 'S' ? sTable : uTable)),
    };
    if (a.dtype.kind == 'T')
        ops.Add(("sum", () => np.sum(a)));
    if (a.dtype.kind != 'S')
    {
        var fmt = np.full(a.Shape, "<%s>", a.dtype.kind == 'T' ? a.dtype : np.dtype("U4"));
        ops.Add(("mod", () => np.strings.mod(fmt, a)));
        ops.Add(("encode", () => np.strings.encode(a, "utf-8")));
    }
    else
        ops.Add(("decode", () => np.strings.decode(a, "utf-8")));
    return ops;
}

var sets = new List<(int n, string kind, NDArray a)>();
foreach (int n in sizes)
{
    foreach (var kind in kinds)
        sets.Add((n, kind, Load(kind, n)));
    // The stepped layout: the [::2] view of the 2n-element base the twin saved.
    foreach (var kind in new[] { "u8@step2", "t8@step2" })
        sets.Add((n, kind, Load(kind, n)["::2"]));
}
foreach (var kind in coreKinds)
    sets.Add((CoreSize, kind, Load(kind, CoreSize)));

// The calls each set times: a refused call (NumPy refuses it too) is dropped here, so it has no comparand.
var plan = new List<(int n, string kind, string op, Func<object> call)>();
foreach (var (n, kind, a) in sets)
{
    var b = np.roll(a, 1);
    foreach (var (op, call) in OpsFor(kind, a, b))
    {
        if ((n == CoreSize && !coreOps.Contains(op)) || (op == "sum" && n > sizes[0]))
            continue;
        try
        {
            Drop(call());
        }
        catch (Exception)
        {
            continue;   // NumSharp refuses this op for this kind (as NumPy does): no comparand
        }
        plan.Add((n, kind, op, call));
    }
}

// An untimed warm-up pass over EVERY call before any is timed: the first set to run a code path otherwise times it while
// its methods are still tier-0 (the background tier-1 compiles of a whole call tree outlast one cell's ~200 ms), which read
// the first U set 2-5x slower than its siblings. 40 calls of each 1K cell pass the 30-call tier-up threshold; the larger
// sizes reach their own size-dependent paths once or twice. The pause lets the tier-1 queue drain.
foreach (var (n, _, _, call) in plan)
    for (int i = 0, reps = n == sizes[0] ? 40 : n == CoreSize ? 1 : 2; i < reps; i++)
        Drop(call());
Thread.Sleep(1500);

foreach (var (n, kind, op, call) in plan)
{
    double ms = Best(() => Drop(call()));
    Console.WriteLine($"{op}|{kind}|{n}\t{ms:G17}");
}
