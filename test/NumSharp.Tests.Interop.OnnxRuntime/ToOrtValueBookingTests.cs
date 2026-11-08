using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
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
    ///     <see cref="NDArrayOnnxInterop.ToOrtValue"/>'s booking (P8 / F20 of the bridge QA, the owner's decision of
    ///     2026-11-06): ONNX Runtime keeps owning and freeing the copy, and NumSharp books its bytes until the caller disposes
    ///     the value or the GC collects it, so a forgotten value is no longer invisible to the GC.
    /// </summary>
    /// <remarks>
    ///     <para>Each test pins one half of the contract: the booking ends at the caller's Dispose (seen through ORT's own
    ///     <c>MemoryHandle</c> hook) or at collection, exactly once; a forgotten value paces collections and a disposed one
    ///     never does; and - the reason ORT keeps the memory - a value bound to an <see cref="OrtIoBinding"/> stays correct
    ///     after the managed value is disposed or collected (a NumSharp-owned copy tied to the managed value returned another
    ///     array's values there).</para>
    ///     <para>The suite runs without parallelism (<c>[assembly: DoNotParallelize]</c>), so the process-wide booking counters
    ///     move only with each test's own values.</para>
    /// </remarks>
    [TestClass]
    public class ToOrtValueBookingTests : OnnxTestBase
    {
        /// <summary>
        ///     The ONNX Runtime this suite runs against has the private <c>OrtValue._memHandle</c> slot the booking hooks into.
        ///     It does at the floor and the current release (1.16.0, 1.31.1); a future version that renames it would make
        ///     <c>ToOrtValue</c> quietly unbooked, so this test is what notices.
        /// </summary>
        [TestMethod]
        public void Booking_HooksIntoTheLoadedOnnxRuntime()
        {
            OrtValueBooking.IsAvailable.Should().BeTrue(
                "OrtValue._memHandle (a MemoryHandle?) exists from ONNX Runtime 1.16.0 to 1.31.1; without it ToOrtValue falls back to an unbooked value");
        }

        /// <summary>A value is booked for its byte size the moment it exists, and the caller's Dispose ends the booking at once.</summary>
        [TestMethod]
        public void ToOrtValue_BooksItsBytes_AndDisposeGivesThemBackAtOnce()
        {
            using NDArray src = Arange(NPTypeCode.Single, 64, 64);
            int live = OrtValueBooking.LiveBookings;
            long booked = OrtValueBooking.BookedBytes, foreign = SizeBucketedBufferPool.OutstandingForeignBytes;

            OrtValue value = src.ToOrtValue();
            OrtValueBooking.LiveBookings.Should().Be(live + 1);
            OrtValueBooking.BookedBytes.Should().Be(booked + 64 * 64 * 4, "the copy's 16,384 bytes are booked");
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(foreign + 64 * 64 * 4, "the pool counts them as foreign memory");

            value.Dispose();
            OrtValueBooking.LiveBookings.Should().Be(live, "Dispose ends the booking at once - no collection needed");
            OrtValueBooking.BookedBytes.Should().Be(booked);
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(foreign);
        }

        /// <summary>Disposing a value twice gives its booking back once (ORT's Dispose is idempotent, and so is the token).</summary>
        [TestMethod]
        public void ToOrtValue_DisposedTwice_GivesTheBookingBackOnce()
        {
            using NDArray src = Arange(NPTypeCode.Double, 100);
            long booked = OrtValueBooking.BookedBytes;
            OrtValue value = src.ToOrtValue();
            value.Dispose();
            value.Dispose();
            OrtValueBooking.BookedBytes.Should().Be(booked, "the second Dispose gives nothing back a second time");
        }

        /// <summary>A value the caller drops undisposed gives its booking back when the GC collects it.</summary>
        [TestMethod]
        public void ToOrtValue_Forgotten_GivesTheBookingBackWhenCollected()
        {
            using NDArray src = Arange(NPTypeCode.Single, 32, 32);
            int live = OrtValueBooking.LiveBookings;
            long booked = OrtValueBooking.BookedBytes;

            MakeAndDrop(src, 10);
            OrtValueBooking.LiveBookings.Should().Be(live + 10, "dropped values stay booked until a collection finds them");

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            OrtValueBooking.LiveBookings.Should().Be(live, "the collection ran each value's finalizers, which ended the bookings");
            OrtValueBooking.BookedBytes.Should().Be(booked);
        }

        /// <summary>
        ///     The GC hears about forgotten values: 64 dropped values of 1 MiB grow the bookings by four pacing windows and
        ///     pace collections; 64 disposed values never grow them and pace none.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_ForgottenValuesPaceCollections_DisposedValuesNever()
        {
            if (!SizeBucketedBufferPool.GcPacingEnabled)
                Assert.Inconclusive("GC pacing is switched off (NUMSHARP_POOL_GC_PACING=0)");
            using NDArray src = Arange(NPTypeCode.Single, 256, 1024);   // 1 MiB
            GC.Collect();
            GC.WaitForPendingFinalizers();

            long paced = SizeBucketedBufferPool.PacedCollections;
            for (int i = 0; i < 64; i++)
            {
                using OrtValue value = src.ToOrtValue();
            }
            SizeBucketedBufferPool.PacedCollections.Should().Be(paced, "disposed values give their bookings back at once: nothing grows, nothing is paced");

            MakeAndDrop(src, 64);
            SizeBucketedBufferPool.PacedCollections.Should().BeGreaterThanOrEqualTo(paced + 3,
                "64 MiB of dropped values grow the bookings by four 16 MiB windows; each window paces one collection");

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        /// <summary>
        ///     ORT keeps owning the copy, so a value bound to an <see cref="OrtIoBinding"/> stays correct after the managed
        ///     value is disposed (the binding holds its own native reference). Same-size NumSharp arrays are taken from the
        ///     pool in between: a copy that NumSharp freed under the binding would be handed to them and read back as 7777.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_BoundValue_StaysCorrectAfterItsDispose() => BoundValueStaysCorrect(dropInsteadOfDispose: false);

        /// <summary>As <see cref="ToOrtValue_BoundValue_StaysCorrectAfterItsDispose"/>, with the managed value dropped and collected instead.</summary>
        [TestMethod]
        public void ToOrtValue_BoundValue_StaysCorrectAfterItsCollection() => BoundValueStaysCorrect(dropInsteadOfDispose: true);

        /// <summary>A text array's string tensor is not booked: ORT keeps its strings in allocations of unknown size.</summary>
        [TestMethod]
        public void ToOrtValue_Text_IsNotBooked()
        {
            using NDArray text = np.array(new[] { "ab", "c" });
            long booked = OrtValueBooking.BookedBytes;
            using (OrtValue value = text.ToOrtValue())
                OrtValueBooking.BookedBytes.Should().Be(booked, "a string tensor is not booked");
        }

        /// <summary>An empty array books nothing (no bytes) and still crosses.</summary>
        [TestMethod]
        public void ToOrtValue_Empty_IsNotBooked()
        {
            using NDArray empty = np.zeros(new Shape(0, 3), np.float32);
            int live = OrtValueBooking.LiveBookings;
            using (OrtValue value = empty.ToOrtValue())
            {
                OrtValueBooking.LiveBookings.Should().Be(live, "zero bytes are not booked");
                value.GetTensorTypeAndShape().Shape.Should().Equal(new long[] { 0, 3 });
            }
        }

        /// <summary>
        ///     A booked value handed to <see cref="OrtValue.CreateSequence(ICollection{OrtValue})"/> belongs to the sequence;
        ///     disposing the sequence disposes the member, which ends its booking.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_BookingEndsWhenASequenceOwningTheValueIsDisposed()
        {
            using NDArray src = Arange(NPTypeCode.Single, 16);
            long booked = OrtValueBooking.BookedBytes;
            var members = new List<OrtValue> { src.ToOrtValue() };
            using (OrtValue sequence = OrtValue.CreateSequence(members))
                OrtValueBooking.BookedBytes.Should().Be(booked + 64, "the member is still booked inside the sequence");
            OrtValueBooking.BookedBytes.Should().Be(booked, "disposing the sequence disposed its member");
        }

        /// <summary>Makes <paramref name="count"/> booked values and drops them, in a frame of its own so nothing keeps them reachable.</summary>
        /// <param name="src">The array to copy.</param>
        /// <param name="count">How many values to make and drop.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void MakeAndDrop(NDArray src, int count)
        {
            for (int i = 0; i < count; i++)
                src.ToOrtValue();
        }

        /// <summary>
        ///     Binds a <see cref="NDArrayOnnxInterop.ToOrtValue"/> result as A of <c>add_f32</c> (C = A + B, B = zeros), lets go
        ///     of the managed value, takes same-size buffers from the pool, runs the binding and compares C with the array.
        /// </summary>
        /// <param name="dropInsteadOfDispose"><c>true</c> to drop and collect the managed value; <c>false</c> to dispose it.</param>
        private void BoundValueStaysCorrect(bool dropInsteadOfDispose)
        {
            using NDArray src = Arange(NPTypeCode.Single, 64, 64);
            long[] dims = { 64, 64 };
            using InferenceSession session = OpenSession("add_f32");
            using OrtIoBinding binding = session.CreateIoBinding();
            using var runOptions = new RunOptions();
            float[] output = new float[64 * 64];
            using OrtValue b = OrtValue.CreateTensorValueFromMemory(new float[64 * 64], dims);
            using OrtValue c = OrtValue.CreateTensorValueFromMemory(output, dims);

            BindAndLetGo(binding, src, dropInsteadOfDispose);
            binding.BindInput("B", b);
            binding.BindOutput("C", c);
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            var thieves = new List<NDArray>();
            for (int i = 0; i < 16; i++)
                thieves.Add(np.full(new Shape(64, 64), 7777f));

            session.RunWithBinding(runOptions, binding);
            output.Should().Equal(src.ToArray<float>(), "ORT owns the bound copy: letting go of the managed value frees nothing under the binding");

            binding.ClearBoundInputs();
            foreach (NDArray thief in thieves)
                thief.Dispose();
        }

        /// <summary>Binds a fresh value as input A, then disposes it or drops it, in a frame of its own.</summary>
        /// <param name="binding">The binding.</param>
        /// <param name="src">The array to copy.</param>
        /// <param name="drop"><c>true</c> to drop the value; <c>false</c> to dispose it.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void BindAndLetGo(OrtIoBinding binding, NDArray src, bool drop)
        {
            OrtValue a = src.ToOrtValue();
            binding.BindInput("A", a);
            if (!drop)
                a.Dispose();
        }
    }
}
