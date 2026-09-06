using BenchmarkDotNet.Running;
using NumSharp;
using NumSharp.Benchmark.CSharp.Benchmarks.LinearAlgebra;
using NumSharp.Benchmark.CSharp.Infrastructure;
using NumSharp.Interop.OpenBLAS;

// A separate executable is deliberate: the ordinary BenchmarkDotNet project references Core only,
// so its Managed profile can never accidentally load or install a native backend. This runner reuses
// the EXACT SAME benchmark classes and OfficialBenchmarkConfig, but installs single-thread OpenBLAS
// before BenchmarkDotNet discovers/executes them.
np.multithreading(false);
OpenBlasEngine.Enable(threads: 1);
if (!OpenBlasEngine.Enabled)
    throw new InvalidOperationException("The OpenBLAS benchmark profile could not enable its backend.");

Console.WriteLine($"PROFILE openblas; {OpenBlasEngine.Info}");
BenchmarkSwitcher.FromAssembly(typeof(LinAlgBenchmarks).Assembly)
    .Run(args, new OfficialBenchmarkConfig());
