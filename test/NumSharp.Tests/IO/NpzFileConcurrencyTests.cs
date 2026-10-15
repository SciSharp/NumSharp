using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using NumSharp.IO;

namespace NumSharp.Tests.IO
{
    /// <summary>
    ///     <see cref="NpzFile"/> shared by several threads: a cached member is returned without a lock, and everything that
    ///     reads the archive (a member's first load, <see cref="NpzFile.GetRawBytes"/>, <see cref="NpzFile.IsArray"/>'s
    ///     magic check, <see cref="NpzFile.Dispose"/>) runs under one gate, because a <c>ZipArchive</c> reads one underlying
    ///     stream. These tests pin that concurrent first loads decode the right bytes, that every access to a member returns
    ///     one instance, and that disposing during reads ends them with <see cref="ObjectDisposedException"/>.
    /// </summary>
    [TestClass]
    public class NpzFileConcurrencyTests
    {
        /// <summary>Members per archive.</summary>
        private const int Members = 12;

        /// <summary>
        ///     Round after round, a fresh archive is opened and several threads read every member at once, two threads
        ///     starting on each member so first loads of the same member collide. Every array has its member's values,
        ///     every raw read has its member's bytes, <see cref="NpzFile.IsArray"/> answers true, and all threads get the
        ///     same instance of a member within a round.
        /// </summary>
        [TestMethod]
        public void Members_LoadedByManyThreadsAtOnce_HaveTheirValues_AndOneInstanceEach()
        {
            const int rounds = 24;
            int threads = System.Math.Clamp(Environment.ProcessorCount, 4, 8);
            var (bytes, keys, expected, expectedRaw) = BuildArchive(elements: 8_192);

            var problems = new List<string>();
            var got = new NDArray[threads, Members];
            NpzFile current = np.load_npz(bytes);

            // Two phases per round: everyone starts the round together, then (after everyone finished it) the last thread
            // in checks that the round handed out one instance per member, releases them, and opens the next archive.
            // Disposed only after every worker left: a worker still waiting in SignalAndWait would get an
            // ObjectDisposedException on its own thread, where nothing catches it, and that would end the test host.
            var barrier = new Barrier(threads, b =>
            {
                if (b.CurrentPhaseNumber % 2 == 0)
                    return;
                for (int k = 0; k < Members; k++)
                {
                    for (int t = 1; t < threads; t++)
                        if (!ReferenceEquals(got[t, k], got[0, k]))
                            Report($"round {b.CurrentPhaseNumber / 2}: member {keys[k]} was handed out as two instances");
                    got[0, k]?.Dispose();
                    for (int t = 0; t < threads; t++)
                        got[t, k] = null;
                }
                current.Dispose();
                current = np.load_npz(bytes);
            });

            var workers = new Thread[threads];
            for (int w = 0; w < threads; w++)
            {
                int worker = w;
                workers[w] = new Thread(() =>
                {
                    try
                    {
                        for (int round = 0; round < rounds; round++)
                        {
                            barrier.SignalAndWait();
                            var npz = Volatile.Read(ref current);
                            try
                            {
                                // Two threads start on each member, so their first loads of it collide.
                                for (int step = 0; step < Members; step++)
                                {
                                    int k = (worker / 2 + step) % Members;
                                    NDArray a = npz[keys[k]];
                                    got[worker, k] = a;
                                    if (!np.array_equal(a, expected[k]))
                                        Report($"round {round}: {keys[k]} has wrong values");
                                }

                                int rawKey = (worker + round) % Members;
                                if (!npz.GetRawBytes(keys[rawKey]).AsSpan().SequenceEqual(expectedRaw[rawKey]))
                                    Report($"round {round}: raw bytes of {keys[rawKey]} differ");
                                if (!npz.IsArray(keys[rawKey]))
                                    Report($"round {round}: IsArray({keys[rawKey]}) is false");
                            }
                            catch (Exception e)
                            {
                                Report($"round {round}, thread {worker}: {e}");
                            }
                            barrier.SignalAndWait();
                        }
                    }
                    catch (Exception e)
                    {
                        // A failing post-phase action reaches every worker as a BarrierPostPhaseException; catching it
                        // here ends all of them together instead of leaving the others waiting.
                        Report($"thread {worker}: {e}");
                    }
                }) { IsBackground = true, Name = $"npz reader {w}" };
            }

            foreach (var t in workers)
                t.Start();
            foreach (var t in workers)
                t.Join(TimeSpan.FromMinutes(1)).Should().BeTrue("a reader must finish instead of hanging the run");
            barrier.Dispose();
            current.Dispose();
            DisposeAll(expected);

            problems.Should().BeEmpty();

            void Report(string problem)
            {
                lock (problems)
                    if (problems.Count < 20)
                        problems.Add(problem);
            }
        }

        /// <summary>
        ///     Threads keep reading raw member bytes (every call reads the archive) and members while the archive is
        ///     disposed under them. Each read either returns its member's exact bytes or raises
        ///     <see cref="ObjectDisposedException"/>; nothing else is thrown, every thread ends, an access after
        ///     <see cref="NpzFile.Dispose"/> raises <see cref="ObjectDisposedException"/>, and arrays handed out before
        ///     keep their values.
        /// </summary>
        [TestMethod]
        public void Dispose_WhileThreadsRead_EndsEveryReadWithObjectDisposed()
        {
            int threads = System.Math.Clamp(Environment.ProcessorCount, 4, 8);
            // 10,000 float64 is 80,128 bytes per member: every GetRawBytes copy stays under the 85,000-byte large-object
            // threshold, so the loop below leaves gen0 garbage rather than large-object-heap garbage behind.
            var (bytes, keys, expected, expectedRaw) = BuildArchive(elements: 10_000);
            var npz = np.load_npz(bytes);
            NDArray before = npz[keys[0]];
            // The arrays the archive hands out belong to their readers, so the test releases them at the end.
            var handedOut = new NDArray[Members];
            handedOut[0] = before;

            var problems = new List<string>();
            int reads = 0;
            var workers = new Thread[threads];
            for (int w = 0; w < threads; w++)
            {
                int worker = w;
                workers[w] = new Thread(() =>
                {
                    for (int i = worker; ; i++)
                    {
                        int k = i % Members;
                        try
                        {
                            if (!npz.GetRawBytes(keys[k]).AsSpan().SequenceEqual(expectedRaw[k]))
                                Report($"thread {worker}: raw bytes of {keys[k]} differ");
                            NDArray member = npz[keys[k]];
                            Interlocked.CompareExchange(ref handedOut[k], member, null);
                            if (!np.array_equal(member, expected[k]))
                                Report($"thread {worker}: {keys[k]} has wrong values");
                            Interlocked.Increment(ref reads);
                        }
                        catch (ObjectDisposedException)
                        {
                            return;
                        }
                        catch (Exception e)
                        {
                            Report($"thread {worker}: {e}");
                            return;
                        }
                    }
                }) { IsBackground = true, Name = $"npz reader {w}" };
            }

            foreach (var t in workers)
                t.Start();
            // Dispose only once every thread is in its loop, so the close lands among reads in progress.
            var deadline = DateTime.UtcNow.AddSeconds(30);
            while (Volatile.Read(ref reads) < threads * 4 && DateTime.UtcNow < deadline)
                Thread.Yield();
            npz.Dispose();
            foreach (var t in workers)
                t.Join(TimeSpan.FromMinutes(1)).Should().BeTrue("a reader must end with ObjectDisposedException instead of hanging");

            problems.Should().BeEmpty();
            Action after = () => _ = npz[keys[1]];
            after.Should().Throw<ObjectDisposedException>();
            np.array_equal(before, expected[0]).Should().BeTrue("an array handed out before Dispose belongs to its reader and stays valid");
            npz.Dispose();   // a second Dispose does nothing
            DisposeAll(handedOut);
            DisposeAll(expected);

            void Report(string problem)
            {
                lock (problems)
                    if (problems.Count < 20)
                        problems.Add(problem);
            }
        }

        /// <summary>
        ///     Builds a compressed archive of <see cref="Members"/> float64 members with distinct values (compressed, so a
        ///     read that decodes another member's bytes fails loudly), plus each member's expected array and raw bytes, read
        ///     single-threaded.
        /// </summary>
        /// <param name="elements">Elements per member.</param>
        /// <returns>The archive bytes, the member keys, each member's array and each member's raw bytes.</returns>
        private static (byte[] bytes, string[] keys, NDArray[] expected, byte[][] expectedRaw) BuildArchive(int elements)
        {
            var keys = Enumerable.Range(0, Members).Select(k => $"m{k}").ToArray();
            var expected = new NDArray[Members];
            var members = new Dictionary<string, NDArray>();
            for (int k = 0; k < Members; k++)
            {
                using var indices = np.arange(elements);
                using var asDouble = indices.astype(np.float64);
                expected[k] = asDouble + (k + 1) * 0.25;
                members[keys[k]] = expected[k];
            }

            byte[] bytes = np.savez_compressed(members);
            var expectedRaw = new byte[Members][];
            using (var npz = np.load_npz(bytes))
                for (int k = 0; k < Members; k++)
                    expectedRaw[k] = npz.GetRawBytes(keys[k]);
            return (bytes, keys, expected, expectedRaw);
        }

        /// <summary>
        ///     Disposes every distinct array in <paramref name="arrays"/>. The tests release what they allocate: an
        ///     undisposed array returns its pooled buffer only through its finalizer, and a first version of these tests
        ///     that left about 20 MB that way made a wall-clock test later in the suite
        ///     (<c>T2_NonZero_Is_Much_Slower_Than_NumPy</c>) fail on both frameworks.
        /// </summary>
        /// <param name="arrays">The arrays; null entries and repeated instances are skipped.</param>
        private static void DisposeAll(IEnumerable<NDArray> arrays)
        {
            var done = new HashSet<NDArray>(ReferenceEqualityComparer.Instance);
            foreach (var a in arrays)
                if (a is not null && done.Add(a))
                    a.Dispose();
        }
    }
}
