using BenchmarkDotNet.Running;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using NumSharp.Benchmark.CSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

// Host stability: pin this process to the orchestrator's performance core BEFORE any path runs —
// interactive menu, --filter, the audits — so every BenchmarkDotNet config in this process
// (Official, NumSharpBenchmarkConfig, Quick) measures on it. Idempotent with the call inside
// OfficialBenchmarkConfig; an absent NUMSHARP_BENCHMARK_AFFINITY leaves the process unpinned.
BenchmarkHost.PinToBenchmarkCore();

if (BenchmarkCheckpoint.TryWritePlan(typeof(Program).Assembly, args))
    return;

// Machine-readable, no-timing discovery used by benchmark/scripts/generate_scenarios_html.py.
// It reflects BenchmarkDotNet's own [Benchmark] / [ParamsSource] metadata, so the dtype audit
// cannot drift from the cases BDN would schedule.
if (args.Any(arg => string.Equals(arg, "--audit-scenarios-json", StringComparison.OrdinalIgnoreCase)))
{
    ScenarioAudit.WriteJson(typeof(Program).Assembly);
    return;
}

// One direct invocation of every official function × dtype cell at the smallest declared size.
// This is an execution/correctness gate, not a timing run; it makes the green audit matrix
// independently reproducible without paying BenchmarkDotNet process/statistics overhead.
if (args.Any(arg => string.Equals(arg, "--verify-scenarios", StringComparison.OrdinalIgnoreCase)))
{
    Environment.ExitCode = ScenarioAudit.VerifyExecution(typeof(Program).Assembly) ? 0 : 1;
    return;
}

// Run all benchmarks or specific ones based on command line args
if (args.Length == 0)
{
    // Interactive menu
    Console.WriteLine("NumSharp Performance Benchmarks");
    Console.WriteLine("================================");
    Console.WriteLine();
    Console.WriteLine("=== Original Benchmarks ===");
    Console.WriteLine("1. Dispatch Mechanism Comparison (DynamicMethod vs Static vs Struct)");
    Console.WriteLine("2. Fusion Pattern Benchmarks (fused vs multi-pass)");
    Console.WriteLine("3. NumSharp Current Performance");
    Console.WriteLine("4. DynamicMethod Emission (#544) - NumSharp vs DynMethod per-op");
    Console.WriteLine();
    Console.WriteLine("=== Comprehensive Benchmarks ===");
    Console.WriteLine("5. Arithmetic Operations (add, sub, mul, div, mod)");
    Console.WriteLine("6. Unary Operations (math, exp/log, trig, power)");
    Console.WriteLine("7. Reduction Operations (sum, mean, var/std, min/max)");
    Console.WriteLine("8. Broadcasting Operations");
    Console.WriteLine("9. Array Creation Operations");
    Console.WriteLine("10. Shape Manipulation (reshape, transpose, stack)");
    Console.WriteLine("11. Slicing Operations");
    Console.WriteLine("12. Multi-dimensional Arrays");
    Console.WriteLine();
    Console.WriteLine("=== Performance Analysis ===");
    Console.WriteLine("13. SIMD vs Scalar Comparison (binary ops, reductions)");
    Console.WriteLine();
    Console.WriteLine("=== Meta Options ===");
    Console.WriteLine("A. All Benchmarks");
    Console.WriteLine("Q. Quick smoke test (dry run)");
    Console.WriteLine();
    Console.Write("Select benchmark suite: ");

    var choice = Console.ReadLine()?.Trim().ToUpperInvariant();
    args = choice switch
    {
        "1" => ["--filter", "*Dispatch*"],
        "2" => ["--filter", "*Fusion*"],
        "3" => ["--filter", "*NumSharpBenchmarks*"],
        "4" => ["--filter", "*DynamicEmission*"],
        "5" => ["--filter", "*Arithmetic*"],
        "6" => ["--filter", "*Unary*,*Math*,*ExpLog*,*Trig*,*Power*"],
        "7" => ["--filter", "*Reduction*,*Sum*,*Mean*,*VarStd*,*MinMax*,*Prod*"],
        "8" => ["--filter", "*Broadcast*"],
        "9" => ["--filter", "*Creation*"],
        "10" => ["--filter", "*Manipulation*,*Reshape*,*Stack*,*Dims*"],
        "11" => ["--filter", "*Slice*"],
        "12" => ["--filter", "*MultiDim*"],
        "13" => ["--filter", "*SimdVsScalar*,*SimdReductionType*"],
        "A" => [],
        "Q" => ["--job", "Dry"],
        _ => []
    };
}

// Apply the official config (InProcessEmitToolchain + Full rigor) as the base so the
// out-of-process CsProj toolchain — which fails here due to duplicate project names in
// sibling .claude/worktrees/ checkouts — is never used. CLI args (e.g. --filter) extend it.
try
{
    BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, new OfficialBenchmarkConfig());
}
catch (Exception ex)
{
    // BenchmarkDotNet aborts a run by THROWING out of BenchmarkSwitcher.Run — e.g. the in-process
    // executor's "takes too long to run" timeout (see OfficialToolchain). Left unhandled, that
    // exception takes the runtime's crash path: the unhandled-exception printer writes to stderr
    // in cooperative-GC mode, so with a stalled console on stderr the process can neither exit
    // nor be suspended for a GC (2026-09-07: the finalizer thread spun one core at 100 % for hours
    // with the process alive and undebuggable). Report it here and exit non-zero instead. The
    // classes already exported to BenchmarkDotNet.Artifacts/results are intact; the orchestrator
    // (check=False on measure runs) collects them and moves on to the next suite.
    Console.Error.WriteLine($"NumSharp.Benchmark.CSharp: BenchmarkDotNet aborted the run: {ex}");
    Environment.ExitCode = 70;
}
