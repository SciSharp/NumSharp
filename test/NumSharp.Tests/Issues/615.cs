using System;
using System.Threading;
using System.Threading.Tasks;
using NumSharp.Backends;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #615, "Intermittent AccessViolation (unmanaged-storage lifetime race)"
    ///     (https://github.com/SciSharp/NumSharp/issues/615): the reported crash was <c>np.left_shift</c> of a 100,000-element
    ///     bool array, whose widened <c>astype</c> temporary could be collected and its buffer reused while the SIMD shift
    ///     kernel still read it. Since 0.80.0 that temporary is disposed only after the kernel returns.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     This class replays the reported call on the same route (a bool value under
    ///     <c>DefaultEngine.BoolSimdScalarShiftMaxElements</c> takes the <c>astype</c> + SIMD path) with temporary operands,
    ///     while a second thread forces collections and fills freshly pooled buffers of the same size with a poison pattern.
    ///     A kernel that read a reclaimed buffer would see poison and return wrong values. The race exists only in optimized
    ///     code (a Debug build keeps every local alive), so the test guards the Release build CI runs.
    ///     </para>
    ///     <para>
    ///     The answer's library-wide claims are held by existing gates: the lifetime oracle
    ///     (<c>NumSharp.Tests.Oracle</c>, <c>LifetimeOracleTests.Lifetime</c> and <c>LifetimeScaled_EveryOpKey</c>, both
    ///     <c>FuzzMatrix</c>, so CI runs them) replays NumPy calls with temporary operands under a collecting, poisoning
    ///     stress thread in an optimized child process, and <c>OperandRootsTests</c> reads the woven IL to check that every
    ///     scope-opening or <c>[NDAnchored]</c> entry point roots its operands.
    ///     </para>
    ///     <para><c>[DoNotParallelize]</c>: forced collections suspend every thread of the process (TestParallelization).</para>
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class Issue615
    {
        /// <summary>The issue's array size; it is under the bool route's cutoff, so the astype + SIMD route runs.</summary>
        private const int N = 100_000;

        /// <summary>
        ///     500 shifts of a fresh temporary bool array by 2 each give NumPy 2.4.2's int64 result (all 4s) while another
        ///     thread collects, drains finalizers and poisons pooled buffers of the widened operand's size; the process
        ///     survives and no result carries poison.
        /// </summary>
        [TestMethod]
        public void BoolLeftShift_OfTemporaries_UnderForcedCollections_StaysCorrect()
        {
            ((long)N).Should().BeLessThanOrEqualTo(DefaultEngine.BoolSimdScalarShiftMaxElements,
                "the test must drive the astype + SIMD route the crash was reported on");

            using var stop = new CancellationTokenSource();
            var thief = Task.Run(() =>
            {
                while (!stop.IsCancellationRequested)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    // Take a buffer of the widened operand's size (N int64s) from the pool, poison it and give it back: a
                    // reclaimed operand buffer would be handed out here and overwritten under a kernel that still reads it.
                    using var poison = np.full(new Shape(N), unchecked((long)0x5A5A5A5A5A5A5A5AUL));
                }
            });

            try
            {
                for (int i = 0; i < 500; i++)
                {
                    // The operand is a temporary nobody holds: the shape of the reported crash.
                    using var r = np.left_shift(np.ones(new Shape(N), np.@bool), 2);

                    r.typecode.Should().Be(NPTypeCode.Int64, "bool << a Python int is int64 in NumPy 2.4.2");
                    r.GetInt64(0).Should().Be(4);
                    r.GetInt64(N - 1).Should().Be(4);
                    np.sum(r).GetInt64().Should().Be(4L * N, $"iteration {i} read a reclaimed buffer");
                }
            }
            finally
            {
                stop.Cancel();
                thief.Wait();
            }
        }
    }
}
