using BenchmarkDotNet.Running;
using NumSharp;
using NumSharp.Benchmark.CSharp.Benchmarks.LinearAlgebra;
using NumSharp.Benchmark.CSharp.Infrastructure;
using NumSharp.Interop.OpenBLAS;

if (BenchmarkCheckpoint.TryWritePlan(typeof(LinAlgBenchmarks).Assembly, args))
    return;

// A separate executable is deliberate: the ordinary BenchmarkDotNet project references Core only,
// so its Managed profile can never accidentally load or install a native backend. This runner reuses
// the EXACT SAME benchmark classes and OfficialBenchmarkConfig, but installs single-thread OpenBLAS
// before BenchmarkDotNet discovers/executes them.
// Host stability: pin to the orchestrator's performance core first (same seam as the Core runner).
BenchmarkHost.PinToBenchmarkCore();
np.multithreading(false);
OpenBlasEngine.Enable(threads: 1);
if (!OpenBlasEngine.Enabled)
    throw new InvalidOperationException("The OpenBLAS benchmark profile could not enable its backend.");

Console.WriteLine($"PROFILE openblas; {OpenBlasEngine.Info}");
try
{
    BenchmarkSwitcher.FromAssembly(typeof(LinAlgBenchmarks).Assembly)
        .Run(args, new OfficialBenchmarkConfig());
}
catch (Exception ex)
{
    // Same contract as the Core runner's Program.cs: a BenchmarkDotNet abort (e.g. the in-process
    // executor's timeout, see OfficialToolchain) is reported here and exits non-zero rather than
    // taking the runtime's unhandled-exception path, whose stderr write in cooperative-GC mode
    // wedges the process for good when the console is stalled. Exported classes stay on disk.
    Console.Error.WriteLine($"NumSharp.Benchmark.CSharp.OpenBLAS: BenchmarkDotNet aborted the run: {ex}");
    Environment.ExitCode = 70;
}
