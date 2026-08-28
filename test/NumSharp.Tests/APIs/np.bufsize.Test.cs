using System;
using System.IO;
using System.Threading;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     Tests for <see cref="np.getbufsize"/> / <see cref="np.setbufsize(long)"/>. Every expected
    ///     value and error message comes from running NumPy 2.4.2 in the implementing session
    ///     (see <c>numpy._core._ufunc_config.setbufsize</c> + <c>umath/extobj.c</c>).
    /// </summary>
    [TestClass]
    public class np_bufsize_Test
    {
        // bufsize is thread-local and persists until changed; reset after every test so a value
        // set on the runner thread never leaks into the next test.
        [TestCleanup]
        public void Reset() => np.setbufsize(8192);

        // ------------------------------------------------------------------- happy path

        [TestMethod]
        public void Default_Is_8192()
        {
            // numpy: np.getbufsize() == 8192 (NPY_BUFSIZE)
            np.getbufsize().Should().Be(8192L);
        }

        [TestMethod]
        public void SetBufsize_ReturnsOld_AndUpdatesCurrent()
        {
            // numpy: setbufsize returns the PREVIOUS size; getbufsize then reports the new one.
            long old = np.setbufsize(4096);
            old.Should().Be(8192L);
            np.getbufsize().Should().Be(4096L);
        }

        [TestMethod]
        public void SetBufsize_Persists_WithinThread()
        {
            // No errstate in NumSharp, so a bare setbufsize just persists (like numpy without errstate).
            np.setbufsize(160);
            np.getbufsize().Should().Be(160L);
            np.getbufsize().Should().Be(160L);
        }

        [DataTestMethod]
        [DataRow(16L)]           // smallest valid (>=5 AND multiple of 16)
        [DataRow(32L)]
        [DataRow(4096L)]
        [DataRow(10_000_000L)]   // largest valid (10e6, inclusive)
        public void SetBufsize_ValidValues_Accepted(long v)
        {
            np.setbufsize(v).Should().Be(8192L);  // returns the previous (default) size
            np.getbufsize().Should().Be(v);
        }

        // ------------------------------------------------------------------- validation (verbatim)

        [TestMethod]
        public void Negative_Raises_NonNegative()
        {
            Action a = () => np.setbufsize(-1);
            a.Should().Throw<ValueError>()
             .Which.Message.Should().Be("buffer size must be non-negative");
        }

        [DataTestMethod]
        [DataRow(0L)]
        [DataRow(4L)]
        public void TooSmall_Raises(long v)
        {
            Action a = () => np.setbufsize(v);
            a.Should().Throw<ValueError>()
             .Which.Message.Should().Be($"Buffer size, {v}, is too small");
        }

        [DataTestMethod]
        [DataRow(5L)]    // >=5 but not a multiple of 16 -> mult-16 message wins
        [DataRow(8L)]
        [DataRow(15L)]
        [DataRow(17L)]
        public void NotMultipleOf16_Raises(long v)
        {
            Action a = () => np.setbufsize(v);
            a.Should().Throw<ValueError>()
             .Which.Message.Should().Be($"Buffer size, {v}, is not a multiple of 16");
        }

        [DataTestMethod]
        [DataRow(10_000_001L)]
        [DataRow(10_000_016L)]
        [DataRow(1_000_000_000_000_000_000L)]  // numpy: 10**18 -> "too big" (fits int64)
        public void TooBig_Raises(long v)
        {
            Action a = () => np.setbufsize(v);
            a.Should().Throw<ValueError>()
             .Which.Message.Should().Be($"Buffer size, {v}, is too big");
        }

        [TestMethod]
        public void ValueError_Is_ArgumentException()
        {
            // NumPy raises ValueError; NumSharp's ValueError : ArgumentException, so both catch.
            Action a = () => np.setbufsize(-1);
            a.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void FailedSet_LeavesCurrentUnchanged()
        {
            np.setbufsize(4096);
            Action a = () => np.setbufsize(7);   // invalid, must not mutate current
            a.Should().Throw<ValueError>();
            np.getbufsize().Should().Be(4096L);
        }

        // ------------------------------------------------------------------- semantics

        [TestMethod]
        public void Bufsize_DoesNotChangeResults()
        {
            // Buffering is a chunking knob only: a buffered cast is bit-identical at 16 vs 8192.
            var a = np.arange(100000).astype(np.float64) / 3.0;
            np.setbufsize(16);   var r1 = a.astype(np.int32);
            np.setbufsize(8192); var r2 = a.astype(np.int32);
            np.array_equal(r1, r2).Should().BeTrue();
        }

        [TestMethod]
        public void Bufsize_IsThreadLocal()
        {
            np.setbufsize(4096);
            long seen = -1, after = -1;
            var t = new Thread(() =>
            {
                seen = np.getbufsize();   // a fresh thread sees the 8192 default, not 4096
                np.setbufsize(160);
                after = np.getbufsize();
            });
            t.Start();
            t.Join();
            seen.Should().Be(8192L);              // isolated from the main thread's 4096
            after.Should().Be(160L);              // worker changed only its own thread
            np.getbufsize().Should().Be(4096L);   // main thread unaffected by the worker
        }

        [TestMethod]
        public void Current_Equals_GetBufsize()
        {
            // The value NDIter reads at construction (NDIter.cs default resolution) is exactly getbufsize().
            np.setbufsize(320);
            NDIterBufferManager.CurrentBufferSize.Should().Be(np.getbufsize());
            NDIterBufferManager.CurrentBufferSize.Should().Be(320L);
        }

        [TestMethod]
        public void SetBufsize_ChangesNdIterBuffer()
        {
            // Functional wiring, not a no-op: a buffered nditer's BufferSize follows setbufsize.
            ObserveNdIterBufferSize(16).Should().Be(16L);
            ObserveNdIterBufferSize(8192).Should().Be(8192L);
        }

        private static long ObserveNdIterBufferSize(long bufsize)
        {
            np.setbufsize(bufsize);
            using var it = np.nditer(np.arange(64).reshape(8, 8),
                                     flags: new[] { "buffered", "external_loop" });
            var sw = new StringWriter();
            TextWriter prev = Console.Out;
            try
            {
                Console.SetOut(sw);
                it.debug_print();
            }
            finally
            {
                Console.SetOut(prev);
            }
            foreach (var line in sw.ToString().Split('\n'))
                if (line.Contains("BufferSize:"))
                    return long.Parse(line.Split(':')[1].Trim());
            return -1;
        }
    }
}
