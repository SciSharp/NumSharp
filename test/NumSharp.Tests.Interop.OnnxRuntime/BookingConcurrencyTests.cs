using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     P8's bookings under concurrency (the owner's request of 2026-11-06 to prove multithreading, memory safety and GC
    ///     load): <see cref="NDArrayOnnxInterop.ToOrtValue"/> values and import views made on many threads, disposed on their
    ///     own thread, disposed on ANOTHER thread or dropped for the finalizer, while another thread collects - every value
    ///     reads its own data, every bound value stays correct after its managed value is gone, and every booking comes back.
    /// </summary>
    /// <remarks>
    ///     The suite runs without parallelism (<c>[assembly: DoNotParallelize]</c>), so the process-wide booking counters
    ///     move only with each test's own threads. The base class fails a test that leaves a live import or export behind.
    /// </remarks>
    [TestClass]
    public class BookingConcurrencyTests : OnnxTestBase
    {
        /// <summary>Worker threads per test.</summary>
        private const int Threads = 8;

        /// <summary>
        ///     Eight threads each make 300 <see cref="NDArrayOnnxInterop.ToOrtValue"/> copies of a 64 KB array holding the
        ///     thread's own value, read the copy back, then dispose it on the same thread, hand it to another thread to
        ///     dispose, or drop it - while another thread runs young collections. Every copy reads its own thread's value, and
        ///     once the dropped values are collected the live bookings, the booked bytes and the pool's foreign balance are
        ///     back where they started.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_FromManyThreads_DisposedAnywhereOrDropped_BooksBackToBaseline()
        {
            int liveBefore = OrtValueBooking.LiveBookings;
            long bookedBefore = OrtValueBooking.BookedBytes;
            long foreignBefore = SizeBucketedBufferPool.OutstandingForeignBytes;
            var sources = new NDArray[Threads];
            for (int t = 0; t < Threads; t++)
                sources[t] = np.full(new Shape(64, 256), (float)(t + 1));   // 64 KB, one value per thread
            int wrong = 0;
            var crossThread = new ConcurrentQueue<OrtValue>();
            try
            {
                using (StartCollector(out CancellationTokenSource stop))
                {
                    Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, t =>
                    {
                        for (int i = 0; i < 300; i++)
                        {
                            OrtValue value = sources[t].ToOrtValue();
                            ReadOnlySpan<float> data = value.GetTensorDataAsSpan<float>();
                            if (data[0] != t + 1 || data[data.Length - 1] != t + 1)
                                Interlocked.Increment(ref wrong);
                            switch (i % 3)
                            {
                                case 0:
                                    value.Dispose();
                                    break;
                                case 1:
                                    crossThread.Enqueue(value);   // disposed by whichever thread dequeues it
                                    if (crossThread.TryDequeue(out OrtValue other))
                                        other.Dispose();
                                    break;
                                default:
                                    break;                        // dropped: the GC collects it, the token's finalizer ends the booking
                            }
                        }
                    });
                    stop.Cancel();
                }
                while (crossThread.TryDequeue(out OrtValue rest))
                    rest.Dispose();
            }
            finally
            {
                foreach (NDArray source in sources)
                    source.Dispose();
            }

            wrong.Should().Be(0, "every copy holds its own thread's values");
            WaitFor(() => OrtValueBooking.LiveBookings == liveBefore && OrtValueBooking.BookedBytes == bookedBefore &&
                          SizeBucketedBufferPool.OutstandingForeignBytes == foreignBefore, 10_000)
                .Should().BeTrue($"every booking ends once its value is disposed or collected (live {OrtValueBooking.LiveBookings} vs {liveBefore}, " +
                                 $"booked {OrtValueBooking.BookedBytes} vs {bookedBefore}, foreign {SizeBucketedBufferPool.OutstandingForeignBytes} vs {foreignBefore})");
        }

        /// <summary>
        ///     Four threads, each with its own session and <see cref="OrtIoBinding"/>, bind a fresh
        ///     <see cref="NDArrayOnnxInterop.ToOrtValue"/> copy as input A of <c>add_f32</c> (C = A + B, B = zeros), dispose or
        ///     drop the managed value and the source array, churn the pool with same-size arrays of 7777, and run the binding:
        ///     every output is the bound array's own value. ORT owns each copy, so letting go of the managed value never frees
        ///     memory a binding still reads, on any thread.
        /// </summary>
        [TestMethod]
        public void BoundValues_FromManyThreads_StayCorrect_AfterTheirManagedValuesAreDisposedOrDropped()
        {
            const int Workers = 4, Rounds = 12;
            long[] dims = { 64, 64 };
            int wrong = 0;
            var failures = new ConcurrentQueue<Exception>();
            using (StartCollector(out CancellationTokenSource stop))
            {
                Parallel.For(0, Workers, new ParallelOptions { MaxDegreeOfParallelism = Workers }, w =>
                {
                    try
                    {
                        using InferenceSession session = OpenSession("add_f32");
                        using OrtIoBinding binding = session.CreateIoBinding();
                        using var runOptions = new RunOptions();
                        float[] output = new float[64 * 64];
                        using OrtValue b = OrtValue.CreateTensorValueFromMemory(new float[64 * 64], dims);
                        using OrtValue c = OrtValue.CreateTensorValueFromMemory(output, dims);
                        for (int r = 0; r < Rounds; r++)
                        {
                            float expected = w * 100 + r + 1;
                            BindAndLetGo(binding, expected, drop: r % 2 == 1);
                            binding.BindInput("B", b);
                            binding.BindOutput("C", c);
                            if (r % 3 == 0)
                                GC.Collect(0);
                            var thieves = new List<NDArray>();
                            for (int k = 0; k < 4; k++)
                                thieves.Add(np.full(new Shape(64, 64), 7777f));   // same-size pool buffers taken in between
                            session.RunWithBinding(runOptions, binding);
                            foreach (float x in output)
                            {
                                if (x != expected)
                                {
                                    Interlocked.Increment(ref wrong);
                                    break;
                                }
                            }
                            binding.ClearBoundInputs();
                            binding.ClearBoundOutputs();
                            foreach (NDArray thief in thieves)
                                thief.Dispose();
                        }
                    }
                    catch (Exception e)
                    {
                        failures.Enqueue(e);
                    }
                });
                stop.Cancel();
            }
            failures.Should().BeEmpty(string.Join(" | ", failures));
            wrong.Should().Be(0, "a bound copy reads its own values after its managed value is disposed or dropped");
        }

        /// <summary>
        ///     Eight threads make zero-copy views - of a <see cref="DenseTensor{T}"/> over a fresh managed array (booked: the pin
        ///     keeps the array alive) and of an ORT-allocated value the view owns (booked: the view disposes it) - each filled
        ///     with its own value. Each thread keeps its last eight views alive and checks a view's values again when it lets
        ///     go of it, after the other threads' collections; then the view is disposed on the same thread, on another
        ///     thread, or dropped. Every view reads its own values to the end, and the bookings come back to the byte.
        /// </summary>
        [TestMethod]
        public void ImportViews_FromManyThreads_ReadTheirOwnMemory_AndBookBackToBaseline()
        {
            long pressureBefore = NDArrayOnnxInterop.ImportPressureBytes;
            long foreignBefore = SizeBucketedBufferPool.OutstandingForeignBytes;
            int wrong = 0;
            var crossThread = new ConcurrentQueue<NDArray>();
            using (StartCollector(out CancellationTokenSource stop))
            {
                Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, t =>
                {
                    var window = new Queue<(NDArray View, float Value)>();
                    for (int i = 0; i < 200; i++)
                    {
                        float value = t * 1000 + i;
                        NDArray view = i % 2 == 0 ? DenseTensorView(value) : OwningView(value);
                        if (!Holds(view, value))
                            Interlocked.Increment(ref wrong);
                        window.Enqueue((view, value));
                        if (window.Count > 8)
                            LetGo(window.Dequeue(), i, crossThread, ref wrong);
                    }
                    while (window.Count > 0)
                        LetGo(window.Dequeue(), 0, crossThread, ref wrong);
                });
                stop.Cancel();
            }
            while (crossThread.TryDequeue(out NDArray rest))
                rest.Dispose();

            wrong.Should().Be(0, "every view reads its own memory for as long as it lives");
            WaitFor(() => NDArrayOnnxInterop.ImportPressureBytes == pressureBefore &&
                          SizeBucketedBufferPool.OutstandingForeignBytes == foreignBefore, 10_000)
                .Should().BeTrue($"every view's booking ends with its lease (import {NDArrayOnnxInterop.ImportPressureBytes} vs {pressureBefore}, " +
                                 $"foreign {SizeBucketedBufferPool.OutstandingForeignBytes} vs {foreignBefore})");
        }

        /// <summary>A view of a <see cref="DenseTensor{T}"/> over a fresh managed array of 4,096 copies of <paramref name="value"/>.</summary>
        /// <param name="value">The fill value.</param>
        /// <returns>The view; it pins (and books) the array.</returns>
        private static NDArray DenseTensorView(float value)
        {
            var data = new float[4096];
            Array.Fill(data, value);
            return new DenseTensor<float>(data, new[] { 4096 }).AsNDArray();
        }

        /// <summary>An owning view of an ORT-allocated value of 4,096 copies of <paramref name="value"/>.</summary>
        /// <param name="value">The fill value.</param>
        /// <returns>The view; it owns (and books) the value.</returns>
        private static NDArray OwningView(float value)
        {
            OrtValue owned = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float, new long[] { 4096 });
            owned.GetTensorMutableDataAsSpan<float>().Fill(value);
            return owned.AsNDArray(ownsValue: true);
        }

        /// <summary>Does <paramref name="view"/> still hold <paramref name="value"/> at its first, middle and last element?</summary>
        /// <param name="view">A 4,096-element float32 view.</param>
        /// <param name="value">The value it was filled with.</param>
        /// <returns><c>true</c> when all three elements hold the value.</returns>
        private static bool Holds(NDArray view, float value) =>
            view.GetSingle(0) == value && view.GetSingle(2048) == value && view.GetSingle(4095) == value;

        /// <summary>
        ///     Checks a view's values one last time, then disposes it here, hands it to another thread to dispose, or drops it
        ///     (the third way leaves it to the finalizer).
        /// </summary>
        /// <param name="entry">The view and the value it was filled with.</param>
        /// <param name="i">The iteration, which picks the way.</param>
        /// <param name="crossThread">The queue other threads dispose from.</param>
        /// <param name="wrong">The count of views that read wrong values.</param>
        private static void LetGo((NDArray View, float Value) entry, int i, ConcurrentQueue<NDArray> crossThread, ref int wrong)
        {
            if (!Holds(entry.View, entry.Value))
                Interlocked.Increment(ref wrong);
            switch (i % 3)
            {
                case 0:
                    entry.View.Dispose();
                    break;
                case 1:
                    crossThread.Enqueue(entry.View);
                    if (crossThread.TryDequeue(out NDArray other))
                        other.Dispose();
                    break;
                default:
                    break;   // dropped: the finalizer releases the lease and gives the booking back
            }
        }

        /// <summary>
        ///     Binds a fresh <see cref="NDArrayOnnxInterop.ToOrtValue"/> copy of an array of <paramref name="value"/> as input A,
        ///     then disposes the source array and disposes or drops the managed value, in a frame of its own.
        /// </summary>
        /// <param name="binding">The binding.</param>
        /// <param name="value">The fill value.</param>
        /// <param name="drop"><c>true</c> to drop the managed value; <c>false</c> to dispose it.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void BindAndLetGo(OrtIoBinding binding, float value, bool drop)
        {
            using NDArray source = np.full(new Shape(64, 64), value);
            OrtValue a = source.ToOrtValue();
            binding.BindInput("A", a);
            if (!drop)
                a.Dispose();
        }

        /// <summary>
        ///     Starts a thread that runs young collections until cancelled, so finalizers of dropped values and views run while
        ///     the workers are busy.
        /// </summary>
        /// <param name="stop">Cancel it to stop the collector; the returned scope waits for the thread.</param>
        /// <returns>A scope that, when disposed, cancels the collector (if still running) and waits for it.</returns>
        private static IDisposable StartCollector(out CancellationTokenSource stop)
        {
            var cts = new CancellationTokenSource();
            Task collector = Task.Run(() =>
            {
                while (!cts.IsCancellationRequested)
                {
                    GC.Collect(0);
                    Thread.Sleep(1);
                }
            });
            stop = cts;
            return new CollectorScope(cts, collector);
        }

        /// <summary>The collector's lifetime: cancelled and awaited on dispose, so no collection outlives the test.</summary>
        private sealed class CollectorScope : IDisposable
        {
            /// <summary>The collector's cancellation.</summary>
            private readonly CancellationTokenSource _stop;

            /// <summary>The collector's task.</summary>
            private readonly Task _collector;

            /// <summary>Wraps a running collector.</summary>
            /// <param name="stop">Its cancellation.</param>
            /// <param name="collector">Its task.</param>
            public CollectorScope(CancellationTokenSource stop, Task collector)
            {
                _stop = stop;
                _collector = collector;
            }

            /// <summary>Cancels the collector and waits for its thread to finish.</summary>
            public void Dispose()
            {
                _stop.Cancel();
                _collector.Wait();
                _stop.Dispose();
            }
        }
    }
}
