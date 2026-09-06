using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Toolchains.InProcess.Emit;

namespace NumSharp.Benchmark.Unmanaged
{
    /// <summary>
    ///     Shared BenchmarkDotNet configuration for the <c>Unmanaged/*</c> micro-benchmarks.
    ///     <para/>
    ///     These use the <see cref="InProcessEmitToolchain"/> instead of BenchmarkDotNet's default
    ///     out-of-process CsProj toolchain. In-process emission is MANDATORY for these benchmarks to
    ///     run at all in this repository, for two independent reasons:
    ///     <list type="number">
    ///         <item>Under .NET 10 SDKs the out-of-process toolchain fails to build the generated
    ///         benchmark host and reports <c>executed benchmarks: 0</c> (every result <c>NA</c>).</item>
    ///         <item>BenchmarkDotNet's project search finds same-named copies of this project under the
    ///         sibling git worktrees in <c>.claude/worktrees/</c> and aborts with
    ///         "benchmark project names need to be unique".</item>
    ///     </list>
    ///     In-process emission sidesteps both. It also measures in a warm, long-lived process and
    ///     removes per-benchmark build/launch overhead. A <b>Release</b> build is required for valid
    ///     numbers; <see cref="NumSharp.Benchmark.Program"/> disables the optimizations validator so a
    ///     Debug run merely warns instead of hard-failing.
    ///     <para/>
    ///     <b>Accuracy caveat:</b> InProcessEmit is BenchmarkDotNet's less-isolated toolchain (all
    ///     benchmarks share one process, so JIT/GC state can bleed between arms). Every number carries
    ///     that caveat until the out-of-process toolchain can be used again. Also note the
    ///     <see cref="MemoryDiagnoser"/> counts <i>managed</i> GC allocations only — an
    ///     <c>NDArray</c>'s unmanaged storage is invisible to it, so allocation columns understate the
    ///     true cost of NDArray-producing arms relative to <c>int[]</c> arms.
    /// </summary>
    public abstract class UnmanagedConfigBase : ManualConfig
    {
        protected UnmanagedConfigBase()
        {
            AddDiagnoser(MemoryDiagnoser.Default);
            AddLogger(ConsoleLogger.Default);
            AddExporter(MarkdownExporter.GitHub);
            AddExporter(HtmlExporter.Default);
            AddColumn(StatisticColumn.Mean, StatisticColumn.StdDev, StatisticColumn.Median,
                      StatisticColumn.Min, StatisticColumn.Max);
            AddColumn(BaselineRatioColumn.RatioMean);
            AddColumn(RankColumn.Arabic);
            WithSummaryStyle(SummaryStyle.Default.WithMaxParameterColumnWidth(40));
        }

        /// <summary>A default in-process job pinned to the given run strategy.</summary>
        private protected static Job InProcess(RunStrategy strategy) =>
            Job.Default.WithToolchain(InProcessEmitToolchain.Instance).WithStrategy(strategy);
    }

    /// <summary>
    ///     A single steady-state Throughput job (warmup + outlier removal). This is the ONLY job the
    ///     Unmanaged benchmarks use.
    ///     <para/>
    ///     <b>ColdStart was intentionally dropped.</b> Every Unmanaged benchmark wraps an internal
    ///     repetition loop (and reports per-op via <c>OperationsPerInvoke</c>). A ColdStart job forces
    ///     <c>invocationCount = unrollFactor = 1</c> to capture single-shot cost — but the one measured
    ///     invocation still runs the whole internal loop, which warms itself, so a "ColdStart" row here
    ///     measured a fully-warmed loop executed once, not a genuine cold start. It only added a
    ///     mislabeled, noisier row and is therefore removed.
    /// </summary>
    public sealed class UnmanagedThroughputConfig : UnmanagedConfigBase
    {
        public UnmanagedThroughputConfig()
        {
            AddJob(InProcess(RunStrategy.Throughput).WithWarmupCount(5).WithIterationCount(20));
        }
    }
}
