using System;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The ARC every OrtValue the bridge touches gets (<c>OrtValueBooking</c>; the owner's decision of 2026-11-07: "ARC
    ///     over every value it touches", for viewing ORT's memory through NumSharp and ORT's view of NumSharp's): any number
    ///     of threads may dispose a value at once, while a verb reads it, or while a view of it is alive, and ONNX Runtime
    ///     releases the native value exactly once, after the last NumSharp user. Measured before the takeover (ORT 1.31.1):
    ///     two Disposes of one value corrupted the heap (4 of 4 processes), a Dispose during ToNDArray / AsNDArray was an
    ///     access violation, and a view read after its value's Dispose returned another value's data (400 of 400).
    /// </summary>
    /// <remarks>
    ///     <para>The race tests run their trials back to back, every side on its own thread released by one barrier, the
    ///     disposing side after a random delay so the Dispose lands at every point of the other side's work. A defect here
    ///     is usually a crash of the test host ("Test host process crashed"), which fails the run as surely as an
    ///     assertion. The release count comes from <c>OrtValueBooking.TakenOverReleases</c>, process-wide: the suite runs
    ///     without parallelism (<c>[assembly: DoNotParallelize]</c>) and the base class settles the finalizer queue first.</para>
    /// </remarks>
    [TestClass]
    public class OrtValueArcTests : OnnxTestBase
    {
        /// <summary>Races per test: enough that a Dispose lands everywhere in a verb's work (a trial is ~0.1 ms).</summary>
        private const int Trials = 200;

        /// <summary>The elements of the 64 x 64 float32 arrays the tests cross (16,384 bytes).</summary>
        private const int Elements = 64 * 64;

        /// <summary>
        ///     The takeover hooks into the loaded ONNX Runtime: every member it needs exists at the floor and the current
        ///     release. A rename in a future ORT is caught here instead of silently falling back to ORT's own rules.
        /// </summary>
        [TestMethod]
        public void Arc_HooksIntoTheLoadedOnnxRuntime()
        {
            OrtValueBooking.CanTakeOver.Should().BeTrue(
                "OrtValue._memHandle, _handle, _disposed and the internal (IntPtr, OnnxValueType) constructor exist at 1.16.0 and 1.31.1");
        }

        /// <summary>
        ///     Eight threads dispose one ToOrtValue result at the same instant, 200 times: ONNX Runtime releases each value
        ///     exactly once and every booking comes back. ORT's own Dispose released it once per thread that passed its
        ///     unguarded flag (heap corruption).
        /// </summary>
        [TestMethod]
        public void ToOrtValue_EightThreadsDisposeOneValue_ReleasesItOnce()
        {
            using NDArray src = np.full(new Shape(64, 64), 2f);
            long releases = OrtValueBooking.TakenOverReleases;
            int live = OrtValueBooking.LiveBookings;
            long booked = OrtValueBooking.BookedBytes;
            for (int trial = 0; trial < Trials; trial++)
            {
                OrtValue value = src.ToOrtValue();
                RunTogether(8, _ => value.Dispose()).Should().OnlyContain(e => e == null, "a Dispose never throws");
            }

            (OrtValueBooking.TakenOverReleases - releases).Should().Be(Trials, "one native release per value, however many threads disposed it");
            OrtValueBooking.LiveBookings.Should().Be(live, "every booking ended with its value's one release");
            OrtValueBooking.BookedBytes.Should().Be(booked);
        }

        /// <summary>
        ///     One thread copies a value with ToNDArray in a loop while another disposes it: every copy holds the value's
        ///     data or is refused with ObjectDisposedException, and the value is released once. The copy's lease makes the
        ///     Dispose wait for the copy in flight; it used to release the value under GetTensorTypeAndShape.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_DisposedWhileAnotherThreadCopiesIt_EveryCopyIsRightOrRefused()
        {
            using NDArray src = np.full(new Shape(64, 64), 2f);
            long releases = OrtValueBooking.TakenOverReleases;
            var random = new Random(11);
            int wrong = 0, refused = 0;
            for (int trial = 0; trial < Trials; trial++)
            {
                OrtValue value = src.ToOrtValue();
                int delay = random.Next(0, 20_000);
                Exception[] errors = RunTogether(2, side =>
                {
                    if (side == 1)
                    {
                        Delay(delay);
                        value.Dispose();
                        return;
                    }
                    for (int i = 0; i < 100; i++)
                    {
                        using NDArray copy = value.ToNDArray();
                        if (copy.GetSingle(0, 0) != 2f || copy.GetSingle(63, 63) != 2f)
                            Interlocked.Increment(ref wrong);
                    }
                });
                refused += CountRefusals(errors);
            }

            wrong.Should().Be(0, "a copy is made under the value's lease, so it never reads released memory");
            refused.Should().BeGreaterThan(0, "some Disposes land before the reader is done (otherwise the test raced nothing)");
            (OrtValueBooking.TakenOverReleases - releases).Should().Be(Trials, "each value is released once, after the copy in flight");
        }

        /// <summary>
        ///     One thread views a value with AsNDArray in a loop and reads each view while another disposes the value; the
        ///     reader keeps its last view past the Dispose and reads it again. Every view reads the value's data, a view
        ///     started after the Dispose is refused, and each value is released once - when its last view is disposed.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_DisposedWhileAnotherThreadViewsIt_EveryViewReadsTheValue()
        {
            using NDArray src = np.full(new Shape(64, 64), 2f);
            long releases = OrtValueBooking.TakenOverReleases;
            var random = new Random(12);
            int wrong = 0;
            for (int trial = 0; trial < Trials; trial++)
            {
                OrtValue value = src.ToOrtValue();
                int delay = random.Next(0, 20_000);
                Exception[] errors = RunTogether(2, side =>
                {
                    if (side == 1)
                    {
                        Delay(delay);
                        value.Dispose();
                        return;
                    }
                    NDArray last = null;
                    try
                    {
                        for (int i = 0; i < 100; i++)
                        {
                            NDArray view = value.AsNDArray();
                            last?.Dispose();
                            last = view;
                            if (view.GetSingle(0, 0) != 2f || view.GetSingle(63, 63) != 2f)
                                Interlocked.Increment(ref wrong);
                        }
                    }
                    finally
                    {
                        // The last view outlives the Dispose (when the Dispose came first, it was refused before this one).
                        Thread.Sleep(0);
                        if (last is not null && (last.GetSingle(0, 0) != 2f || last.GetSingle(63, 63) != 2f))
                            Interlocked.Increment(ref wrong);
                        last?.Dispose();
                    }
                });
                CountRefusals(errors);
            }

            wrong.Should().Be(0, "a view keeps ONNX Runtime's memory until it is disposed, Dispose of the value or not");
            (OrtValueBooking.TakenOverReleases - releases).Should().Be(Trials, "each value is released once, after its last view");
        }

        /// <summary>
        ///     One thread: a view and a NumSharp slice of it outlive the value's Dispose while ONNX Runtime's allocator hands
        ///     out eight fresh blocks of the same size. The slice still reads the value's own data, the bytes stay booked,
        ///     and the native value is released exactly when the slice - the last view - is disposed. Before the takeover
        ///     400 of 400 such reads returned another value's data.
        /// </summary>
        [TestMethod]
        public void View_OutlivesItsValuesDispose_ReadsTheValuesMemoryUntilTheLastViewIsGone()
        {
            using NDArray src = np.full(new Shape(64, 64), 2f);
            using NDArray sevens = np.full(new Shape(64, 64), 7f);
            long booked = OrtValueBooking.BookedBytes;
            OrtValue value = src.ToOrtValue();
            NDArray view = value.AsNDArray();
            NDArray slice = view["1:, ::2"];   // a NumSharp view of the view: it holds the memory too
            view.Dispose();

            long releases = OrtValueBooking.TakenOverReleases;
            value.Dispose();
            OrtValueBooking.TakenOverReleases.Should().Be(releases, "the slice still reads the value's memory, so the release waits for it");
            OrtValueBooking.BookedBytes.Should().Be(booked + Elements * 4, "the memory is alive, so its booking is too");

            var reuse = new OrtValue[8];
            try
            {
                for (int i = 0; i < reuse.Length; i++)
                    reuse[i] = sevens.ToOrtValue();   // what used to take the freed block
                ((float)np.sum(slice)).Should().Be(2f * slice.size, "every element is still the value's 2, not another value's 7");
                slice.GetSingle(0, 0).Should().Be(2f);
                slice.GetSingle(62, 31).Should().Be(2f);
            }
            finally
            {
                foreach (OrtValue other in reuse)
                    other?.Dispose();
            }

            releases = OrtValueBooking.TakenOverReleases;
            slice.Dispose();
            OrtValueBooking.TakenOverReleases.Should().Be(releases + 1, "the last view's disposal made the one release");
            OrtValueBooking.BookedBytes.Should().Be(booked, "and ended the booking");
        }

        /// <summary>
        ///     An output of ONNX Runtime's OWN session.Run is adopted by its first bridge verb: once the caller disposes the
        ///     results, the view keeps the output's memory, which is booked from then on (only NumSharp keeps it alive), and
        ///     the release comes with the view's disposal.
        /// </summary>
        [TestMethod]
        public void RunOutput_AdoptedByItsFirstVerb_ItsViewOutlivesTheResultsDispose()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray src = np.full(new Shape(64, 64), 2f);
            using NDArray sevens = np.full(new Shape(64, 64), 7f);
            using OrtValue input = src.ToOrtValue();
            using var options = new RunOptions();
            IDisposableReadOnlyCollection<OrtValue> results = session.Run(options, new[] { session.InputNames[0] }, new[] { input }, new[] { session.OutputNames[0] });
            OrtValue output = results[0];
            OrtValueBooking.HasToken(output).Should().BeFalse("ORT made the output; nothing of NumSharp touched it yet");
            NDArray view = output.AsNDArray();
            OrtValueBooking.HasToken(output).Should().BeTrue("the first verb adopted the output");

            long foreign = SizeBucketedBufferPool.OutstandingForeignBytes;
            long releases = OrtValueBooking.TakenOverReleases;
            results.Dispose();
            OrtValueBooking.TakenOverReleases.Should().Be(releases, "the view still reads the output's memory");
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(foreign + Elements * 4,
                "the bytes only a NumSharp view keeps alive now are booked until the view is gone");

            var reuse = new OrtValue[8];
            try
            {
                for (int i = 0; i < reuse.Length; i++)
                    reuse[i] = sevens.ToOrtValue();
                ((float)np.sum(view)).Should().Be(2f * Elements, "the view reads the output's own data after the results' Dispose");
            }
            finally
            {
                foreach (OrtValue other in reuse)
                    other?.Dispose();
            }

            releases = OrtValueBooking.TakenOverReleases;   // after the reuse values' own releases
            view.Dispose();
            OrtValueBooking.TakenOverReleases.Should().Be(releases + 1, "the view's disposal released the output, once");
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(foreign, "and gave the deferred booking back");
        }

        /// <summary>
        ///     The value of an AsOrtValue handle (ONNX Runtime's view of NumSharp memory) disposed by the handle and twice
        ///     directly, all at once, 200 times: one native release each, and the NumSharp buffer is released by the handle
        ///     alone.
        /// </summary>
        [TestMethod]
        public void AsOrtValueValue_DisposedByItsHandleAndDirectlyAtOnce_ReleasesItOnce()
        {
            using NDArray src = np.full(new Shape(64, 64), 2f);
            long releases = OrtValueBooking.TakenOverReleases;
            for (int trial = 0; trial < Trials; trial++)
            {
                OrtTensor handle = src.AsOrtValue();
                OrtValue value = handle.Value;
                OrtValueBooking.HasToken(value).Should().BeTrue("a value over NumSharp memory carries its ARC from birth");
                RunTogether(3, side =>
                {
                    if (side == 0)
                        handle.Dispose();
                    else
                        value.Dispose();
                }).Should().OnlyContain(e => e == null, "a Dispose never throws");
            }

            (OrtValueBooking.TakenOverReleases - releases).Should().Be(Trials, "one native release per value");
            ((float)np.sum(src)).Should().Be(2f * Elements, "the source array's buffer was never released under it");
        }

        /// <summary>
        ///     An owning view and the caller's own (wrong) Disposes of the value, all at once: the value is released once,
        ///     when the view is gone.
        /// </summary>
        [TestMethod]
        public void OwningView_OfAValueTheCallerAlsoDisposes_ReleasesItOnce()
        {
            using NDArray src = np.full(new Shape(64, 64), 2f);
            long releases = OrtValueBooking.TakenOverReleases;
            int wrong = 0;
            for (int trial = 0; trial < Trials; trial++)
            {
                OrtValue value = src.ToOrtValue();
                NDArray view = value.AsNDArray(ownsValue: true);
                RunTogether(3, side =>
                {
                    if (side == 0)
                    {
                        if (view.GetSingle(63, 63) != 2f)
                            Interlocked.Increment(ref wrong);
                        view.Dispose();
                    }
                    else
                    {
                        value.Dispose();
                    }
                }).Should().OnlyContain(e => e == null, "a Dispose never throws");
            }

            wrong.Should().Be(0);
            (OrtValueBooking.TakenOverReleases - releases).Should().Be(Trials, "one release per value, whichever Dispose came first");
        }

        /// <summary>
        ///     The owning verbs whose result needs nothing from the value (a Copy-mode view, a text copy, an empty array)
        ///     dispose the value after their own lease ended: disposing it under the lease would wait for that very lease
        ///     forever. Each call must return (within a generous deadline) and release its value once.
        /// </summary>
        [TestMethod]
        public void OwningVerbs_ThatDisposeTheirValue_NeverWaitForTheirOwnLease()
        {
            using NDArray src = np.full(new Shape(64, 64), 2f);
            using NDArray empty = np.zeros(new Shape(0, 3), NPTypeCode.Single);
            using NDArray text = np.array(new[] { "p", "qq" });
            long releases = OrtValueBooking.TakenOverReleases;

            NDArray copy = null, emptyView = null, textCopy = null;
            Task work = Task.Run(() =>
            {
                copy = src.ToOrtValue().AsNDArray(ownsValue: true, mode: OnnxCodecMode.Copy);
                emptyView = empty.ToOrtValue().AsNDArray(ownsValue: true);
                textCopy = text.ToOrtValue().AsNDArray(ownsValue: true);
            });
            try
            {
                work.Wait(TimeSpan.FromSeconds(60)).Should().BeTrue("a verb that disposed its value under its own lease would never return");
                ((float)np.sum(copy)).Should().Be(2f * Elements);
                emptyView.size.Should().Be(0);
                textCopy.size.Should().Be(2);
                (OrtValueBooking.TakenOverReleases - releases).Should().Be(3, "each owned value was released once, by its verb");
            }
            finally
            {
                copy?.Dispose();
                emptyView?.Dispose();
                textCopy?.Dispose();
            }
        }

        /// <summary>
        ///     A sequence output of ONNX Runtime's own Run, adopted on the creating thread, read with ToNDArrays by one thread
        ///     while another disposes the results: every read is complete or refused, and the sequence is released once.
        /// </summary>
        [TestMethod]
        public void Sequence_DisposedWhileAnotherThreadReadsIt_EveryReadIsRightOrRefused()
        {
            using InferenceSession session = OpenSession("sequence_out");
            using NDArray x = np.array(new[] { 1f, 2f, 3f });
            using OrtTensor input = x.AsOrtValue();
            long releases = OrtValueBooking.TakenOverReleases;
            var random = new Random(13);
            int wrong = 0;
            for (int trial = 0; trial < Trials; trial++)
            {
                using var options = new RunOptions();
                IDisposableReadOnlyCollection<OrtValue> results = session.Run(options, new[] { "X" }, new[] { input.Value }, new[] { "seq" });
                OrtValue sequence = results[0];
                NDArray[] expected = sequence.ToNDArrays();   // the first verb, on the creating thread: adopts the sequence
                int delay = random.Next(0, 20_000);
                Exception[] errors = RunTogether(2, side =>
                {
                    if (side == 1)
                    {
                        Delay(delay);
                        results.Dispose();
                        return;
                    }
                    for (int i = 0; i < 50; i++)
                    {
                        NDArray[] read = sequence.ToNDArrays();
                        try
                        {
                            if (read.Length != expected.Length || !SameValues(read, expected))
                                Interlocked.Increment(ref wrong);
                        }
                        finally
                        {
                            foreach (NDArray element in read)
                                element.Dispose();
                        }
                    }
                });
                CountRefusals(errors);
                foreach (NDArray element in expected)
                    element.Dispose();
            }

            wrong.Should().Be(0, "a read holds the sequence's lease, so it never reads a released sequence");
            (OrtValueBooking.TakenOverReleases - releases).Should().Be(Trials, "each sequence is released once, after the read in flight");
        }

        /// <summary>
        ///     The values the lease cannot cover keep ONNX Runtime's own rules and get no token: a value over memory ORT does
        ///     not own (its slot holds that memory's pin) and a sequence built from member values (ORT disposes the members
        ///     after the hook). A value the bridge hands out (GetMapKeys) carries its token from birth.
        /// </summary>
        [TestMethod]
        public void Adoption_SkipsWhatCannotBeLeased_AndBridgeResultsCarryTheirTokenFromBirth()
        {
            float[] managed = { 1f, 2f, 3f };
            using (OrtValue overManaged = OrtValue.CreateTensorValueFromMemory(managed, new long[] { 3 }))
            using (NDArray view = overManaged.AsNDArray())
            {
                OrtValueBooking.HasToken(overManaged).Should().BeFalse("its slot holds the managed array's pin");
                view.GetSingle(2).Should().Be(3f);
            }

            using (NDArray member = np.array(new[] { 4f, 5f }))
            {
                OrtValue element = member.ToOrtValue();
                // CreateSequence takes the element over by emptying the (mutable) collection it is given.
                using OrtValue sequence = OrtValue.CreateSequence(new System.Collections.Generic.List<OrtValue> { element });
                NDArray[] read = sequence.ToNDArrays();
                try
                {
                    OrtValueBooking.HasToken(sequence).Should().BeFalse("a value built from members is not adopted");
                    read[0].GetSingle(1).Should().Be(5f);
                }
                finally
                {
                    foreach (NDArray nd in read)
                        nd.Dispose();
                }
            }

            using InferenceSession session = OpenSession("zipmap_int64");
            using NDArray probabilities = np.array(new[,] { { 0.1f, 0.7f, 0.2f } });
            using OrtTensor input = probabilities.AsOrtValue();
            using var options = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> results = session.Run(options, new[] { "X" }, new[] { input.Value }, new[] { "Z" });
            using OrtValue map = results[0].GetValue(0, OrtAllocator.DefaultInstance);
            using OrtValue keys = map.GetMapKeys();
            OrtValueBooking.HasToken(keys).Should().BeTrue("GetMapKeys hands out a value with its ARC from birth");
            OrtValueBooking.HasToken(map).Should().BeTrue("and its map argument was adopted by the verb");
        }

        // ---- helpers ---------------------------------------------------------------------------------

        /// <summary>
        ///     Runs <paramref name="body"/> on <paramref name="threads"/> dedicated threads released together by one barrier
        ///     and waits for all of them.
        /// </summary>
        /// <param name="threads">How many sides race.</param>
        /// <param name="body">One side's work, given its index.</param>
        /// <returns>Each side's exception, or null for a side that returned.</returns>
        internal static Exception[] RunTogether(int threads, Action<int> body)
        {
            var errors = new Exception[threads];
            using var start = new Barrier(threads);
            var workers = new Thread[threads];
            for (int i = 0; i < threads; i++)
            {
                int side = i;
                workers[side] = new Thread(() =>
                {
                    start.SignalAndWait();
                    try
                    {
                        body(side);
                    }
                    catch (Exception e)
                    {
                        errors[side] = e;
                    }
                }) { IsBackground = true };
                workers[side].Start();
            }
            foreach (Thread worker in workers)
                worker.Join();
            return errors;
        }

        /// <summary>
        ///     Counts the sides refused with <see cref="ObjectDisposedException"/>, and fails on any other exception: a race
        ///     may refuse a verb that came after the Dispose, never fail it otherwise.
        /// </summary>
        /// <param name="errors">What <see cref="RunTogether"/> returned.</param>
        /// <returns>The number of refusals.</returns>
        internal static int CountRefusals(Exception[] errors)
        {
            int refused = 0;
            foreach (Exception e in errors)
            {
                if (e is null)
                    continue;
                e.Should().BeOfType<ObjectDisposedException>("a verb racing a Dispose is refused, nothing else: {0}", e);
                refused++;
            }
            return refused;
        }

        /// <summary>
        ///     Busy-waits about <paramref name="iterations"/> loop turns, so a racing side starts at a random point of the
        ///     other's work. A plain managed loop (not <see cref="Thread.SpinWait"/>, which on .NET 8 is a call the GC cannot
        ///     suspend).
        /// </summary>
        /// <param name="iterations">The loop count.</param>
        internal static void Delay(int iterations)
        {
            int sink = 0;
            for (int i = 0; i < iterations; i++)
                Volatile.Write(ref sink, i);
        }

        /// <summary>Do two lists of arrays hold the same values, element by element (same shapes)?</summary>
        /// <param name="read">One list.</param>
        /// <param name="expected">The other.</param>
        /// <returns><c>true</c> when every pair is equal.</returns>
        private static bool SameValues(NDArray[] read, NDArray[] expected)
        {
            for (int i = 0; i < read.Length; i++)
            {
                if (read[i].size != expected[i].size)
                    return false;
                using NDArray same = np.equal(read[i], expected[i]);
                if (!(bool)np.all(same))
                    return false;
            }
            return true;
        }
    }
}
