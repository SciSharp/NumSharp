using System;
using System.Runtime.CompilerServices;
using System.Threading;

// ---------------------------------------------------------------------------------------------------------------------
// Test METHODS run in PARALLEL, any test on any worker; [DoNotParallelize] classes and methods run afterwards, alone.
// See TestParallelization below for the rules a test must follow and the measurements behind the settings.
// ---------------------------------------------------------------------------------------------------------------------
[assembly: Parallelize(Workers = 0, Scope = ExecutionScope.MethodLevel)]

namespace NumSharp.Tests
{
    /// <summary>
    ///     How this assembly runs its tests in parallel, which tests must opt out, and the thread-pool floor the
    ///     parallel run needs. The <c>[assembly: Parallelize]</c> attribute above this type is the switch; the
    ///     result-batching half of the setup is <c>NumSharp.Tests.runsettings</c> next to the project file.
    /// </summary>
    /// <remarks>
    ///     <para><b>What runs in parallel.</b> MSTest (3.8) queues every test — every data row of a data-driven test
    ///     too — and lets <c>Workers = 0</c> (= <see cref="Environment.ProcessorCount"/>) workers drain the queue in the
    ///     order the tests were discovered. Every test carrying <c>[DoNotParallelize]</c>, on its class or on the
    ///     method, is held back and runs after the parallel phase has finished, sequentially and alone, so it sees the
    ///     process exactly as the serial runner showed it. Measured 2026-10-16 on an i9-13900K (32 logical CPUs, a
    ///     quarter to half of them busy with other work), Release, CI filter: the serial runner took 108 s (net10.0)
    ///     and 117 s (net8.0); one test per worker takes 22-27 s and 22-28 s (about 15 s of parallel phase, 9-10 s of
    ///     [DoNotParallelize] tests); pinned to four CPUs, as a CI runner has, 48 s. <c>-- MSTest.Parallelize.Workers=1</c>
    ///     on the <c>dotnet test</c> command line runs the suite serially again.</para>
    ///     <para><b>Why method level.</b> A class's tests used to run one after another on one worker, and a few classes
    ///     hold several seconds of work each (polynomial kernel sweeps, the comparison-engine sweeps, the storm tests);
    ///     they set the end of the parallel phase (30-33 s per run with one class per worker). With one test per worker,
    ///     the heavy tests of one class run side by side. The suite was written under TUnit, which ran every test in
    ///     parallel, until April 2026, so its tests do not share mutable <c>static</c> state inside a class. A test that
    ///     flips process-wide state that ANY other test reads — a sibling in its own class included — must therefore
    ///     carry <c>[DoNotParallelize]</c>.</para>
    ///     <para><b>Where the parallel phase's time goes.</b> It is CPU-bound: the parallel run spends ~230 CPU seconds,
    ///     the serial one ~170, of which the JIT takes ~40 (some 80,000 methods: the test methods, the library's code
    ///     paths and the IL kernels the tests generate) and the collection storms ~53. The tests are queued in the order
    ///     the compiler emitted their classes, namespaces in reverse alphabetical order, so the heavy kernel sweeps of
    ///     <c>Backends</c> come last and decide the end of the phase.</para>
    ///     <para><b>The rule — a class (or method) must carry <c>[DoNotParallelize]</c> when it</b>:
    ///     <list type="bullet">
    ///       <item>reads or resets a process-wide counter: <c>SizeBucketedBufferPool</c> hits/misses/returns,
    ///         <c>GC.CollectionCount</c>, <c>GC.GetTotalAllocatedBytes</c>, the size of a process-wide kernel or
    ///         program cache (<c>GeneratedDelegates.InnerLoopCount</c>, <c>NDExprProgramCache</c>) — another test's
    ///         work moves them. Per-thread counters (<c>GC.GetAllocatedBytesForCurrentThread</c>, the
    ///         <c>[ThreadStatic]</c> <c>…Runs</c> hooks of the engine) need no opt-out;</item>
    ///       <item>needs a GC window nobody disturbs (<c>GcQuiescence.OpenWindow</c> / <c>Undisturbed</c>), counts
    ///         finalized buffers, or forces collections (<c>GC.Collect</c>, <c>GcQuiescence</c>): a forced collection
    ///         suspends every thread of the process, so in the parallel phase it stalls every worker;</item>
    ///       <item>flips process-wide state: <c>OpenBlasEngine.Enable</c>/<c>Disable</c> (the <c>TensorEngine.Blas</c>
    ///         seam — every other test asserts NumSharp's own kernels), <c>Environment.SetEnvironmentVariable</c>,
    ///         <c>Console.SetOut</c>, <c>np.multithreading</c>, <c>np.ma.masked_print_option</c>, a
    ///         non-<c>[ThreadStatic]</c> engine hook such as <c>ComparisonRun.Disabled</c>;</item>
    ///       <item>seeds or reads the state of the GLOBAL <c>np.random</c> singleton and then expects a particular
    ///         stream (an instance — <c>np.random.RandomState(seed)</c>, <c>default_rng(seed)</c>, a bit generator —
    ///         is private to the test and needs nothing);</item>
    ///       <item>asserts wall-clock time.</item>
    ///     </list>
    ///     Thread-local state is safe as it is: <c>np.setbufsize</c> (thread-local, like NumPy's), the print options
    ///     (<c>AsyncLocal</c>), the culture, the engine's <c>[ThreadStatic]</c> test hooks and counters. A hook a test
    ///     sets is seen only by the thread that set it, so a test that hands its work to other threads
    ///     (<c>Parallel.For</c>, its own threads) must set the hook on each of them.</para>
    ///     <para><b>Concurrency storms run in the parallel phase.</b> The collection storm tests spawn their own threads;
    ///     their per-push budgets were calibrated against planted defects with the test host pinned to four CPUs
    ///     (see <c>StressBudget</c> and the budget comments at each storm), the same kind of oversubscription the
    ///     parallel phase gives them. A storm whose window is wall-clock opens it only once its threads are all
    ///     running, so a late-scheduled thread cannot find the window already closed.</para>
    /// </remarks>
    internal static class TestParallelization
    {
        /// <summary>
        ///     Thread-pool threads the pool may hand out immediately per processor, before it starts injecting new
        ///     threads at its slow starvation rate.
        /// </summary>
        /// <remarks>
        ///     MSTest's workers are <c>Task.Run</c> work items, so the parallel phase occupies one pool thread per worker
        ///     — as many as the pool's default minimum. A test that then uses the pool itself (<c>Parallel.For</c>,
        ///     <c>Task.Run</c>, a continuation that completes an awaited task) would wait for the pool's starvation
        ///     injection, about one new thread per half second. Two per processor leaves as many threads again for the
        ///     tests' own fan-out; the pool still creates a thread only when work is queued for it.
        /// </remarks>
        private const int MinPoolThreadsPerProcessor = 2;

        /// <summary>
        ///     Runs when the test assembly loads, before the first test: raises the thread pool's minimum so the parallel
        ///     workers never starve a test's own pool work.
        /// </summary>
        /// <remarks>
        ///     A module initializer, like <see cref="TestHostQualityOfService"/>, so it needs no MSTest hook and runs for
        ///     a filtered run too. The floor is only raised, never lowered: a host that configured more keeps it.
        /// </remarks>
        [ModuleInitializer]
        internal static void RaiseThreadPoolFloor()
        {
            ThreadPool.GetMinThreads(out int worker, out int completion);
            int wanted = Environment.ProcessorCount * MinPoolThreadsPerProcessor;
            if (worker < wanted)
                ThreadPool.SetMinThreads(wanted, completion);
        }
    }
}
