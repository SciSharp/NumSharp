using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     The handle / lease lifetime model, gated by the process-global <see cref="NDArrayTensorsInterop.LiveExports"/>
    ///     / <see cref="NDArrayTensorsInterop.LiveImports"/> counters (the base class already asserts they return to
    ///     baseline after every test).
    /// </summary>
    [TestClass]
    public class LifetimeTests : TensorsTestBase
    {
        [TestMethod]
        public void ExportHandle_TracksLiveExports_AndDisposeIsIdempotent()
        {
            int before = NDArrayTensorsInterop.LiveExports;
            using var a = Arange(NPTypeCode.Single, 4);
            var h = a.AsTensorSpan<float>();
            NDArrayTensorsInterop.LiveExports.Should().Be(before + 1);

            h.IsDisposed.Should().BeFalse();
            h.Dispose();
            h.IsDisposed.Should().BeTrue();
            NDArrayTensorsInterop.LiveExports.Should().Be(before);

            h.Dispose();  // idempotent
            NDArrayTensorsInterop.LiveExports.Should().Be(before);
        }

        [TestMethod]
        public void DisposedHandle_Span_Throws()
        {
            using var a = Arange(NPTypeCode.Single, 4);
            var h = a.AsTensorSpan<float>();
            h.Dispose();
            new Action(() => { _ = h.Span; }).Should().Throw<ObjectDisposedException>();
            new Action(() => { _ = h.ReadOnlySpan; }).Should().Throw<ObjectDisposedException>();
        }

        [TestMethod]
        public void SourceDisposedWhileHandleLive_MemoryStaysValid()
        {
            using var h = ExportThenDisposeSource();      // the source NDArray is gone; the ARC pin keeps the buffer
            var s = h.ReadOnlySpan;
            s[new nint[] { 0 }].Should().Be(0f);
            s[new nint[] { 3 }].Should().Be(3f);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static TensorSpanHandle<float> ExportThenDisposeSource()
        {
            NDArray a = Arange(NPTypeCode.Single, 4);
            TensorSpanHandle<float> h = a.AsTensorSpan<float>();
            a.Dispose();                                  // handle owns its own ARC reference
            return h;
        }

        [TestMethod]
        public void UndisposedHandle_FinalizerReleasesThePin()
        {
            int before = NDArrayTensorsInterop.LiveExports;
            LeakOneHandle();
            WaitFor(() => NDArrayTensorsInterop.LiveExports == before, 10_000)
                .Should().BeTrue("the finalizer must drop the ARC pin of a handle that was never disposed");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void LeakOneHandle()
        {
            using NDArray a = Arange(NPTypeCode.Single, 4);
            var h = a.AsTensorSpan<float>();              // deliberately NOT disposed
            NDArrayTensorsInterop.LiveExports.Should().BeGreaterThan(0);
            GC.KeepAlive(h);
        }

        [TestMethod]
        public void ImportView_TracksLiveImports_ReleasedOnDispose()
        {
            int before = NDArrayTensorsInterop.LiveImports;
            var t = Tensor.Create(new double[] { 1, 2, 3, 4 }, new nint[] { 2, 2 });
            NDArray v = t.AsNDArray();
            NDArrayTensorsInterop.LiveImports.Should().Be(before + 1);

            v.Dispose();
            NDArrayTensorsInterop.LiveImports.Should().Be(before, "disposing the sole view releases the lease synchronously");
        }

        [TestMethod]
        public void ImportLease_ReleasesOnlyWhenLastViewDies()
        {
            int before = NDArrayTensorsInterop.LiveImports;
            var t = Tensor.Create(new double[] { 1, 2, 3, 4 }, new nint[] { 4 });
            NDArray v1 = t.AsNDArray();
            NDArray v2 = v1["0:2"];                        // a derived slice shares the ARC block + lease
            NDArrayTensorsInterop.LiveImports.Should().Be(before + 1);

            v1.Dispose();
            NDArrayTensorsInterop.LiveImports.Should().Be(before + 1, "v2 still holds the memory");

            v2.Dispose();
            NDArrayTensorsInterop.LiveImports.Should().Be(before, "the last view releases the lease");
        }

        [TestMethod]
        public void ImportView_KeepsTensorAlive_AfterLocalRefDropped()
        {
            NDArray v = ImportAndDropTensor();
            using (v)
            {
                Settle();                                 // the tensor object is unreferenced; the lease roots it
                v.GetDouble(1).Should().Be(2.0, "the pinned backing store is still valid");
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray ImportAndDropTensor()
        {
            var t = Tensor.Create(new double[] { 1, 2, 3, 4 }, new nint[] { 4 });
            return t.AsNDArray();                          // t goes out of scope; the view must keep it alive
        }

        /// <summary>
        ///     Both export verbs refuse an array whose buffer has been released, whatever its layout. Regression:
        ///     <c>ToTensor</c> densified a strided source (<c>copy()</c>) BEFORE its disposal check, so a released
        ///     buffer was read silently — once the pool reused it, <c>ToTensor</c> returned another array's data.
        ///     Only the contiguous path threw, and an empty array was never checked at all.
        /// </summary>
        [TestMethod]
        public void ExportVerbs_ReleasedBuffer_Throw_EveryLayout()
        {
            foreach (string layout in new[] { "contiguous slice", "transposed", "stepped", "broadcast", "empty" })
            {
                NDArray released = Released(layout);
                new Action(() => released.ToTensor<double>()).Should().Throw<ObjectDisposedException>($"ToTensor of a released {layout} array");
                new Action(() => released.AsTensorSpan<double>()).Should().Throw<ObjectDisposedException>($"AsTensorSpan of a released {layout} array");
            }
        }

        /// <summary>
        ///     Builds a float64 view of the requested layout, then disposes it together with every array it was derived
        ///     from, so no reference is left and the buffer goes back to the pool.
        /// </summary>
        /// <param name="layout">One of "contiguous slice", "transposed", "stepped", "broadcast" or "empty".</param>
        /// <returns>The disposed view, whose buffer has been released.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="layout"/> is not one of the names above.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray Released(string layout)
        {
            var made = new List<NDArray>();
            NDArray Keep(NDArray a)
            {
                made.Add(a);
                return a;
            }

            NDArray owner = Keep(layout == "empty" ? np.zeros(new Shape(3, 0), np.float64) : np.arange(6.0));
            NDArray view = layout switch
            {
                "contiguous slice" => Keep(owner["1:5"]),
                "transposed" => Keep(Keep(owner.reshape(2, 3)).T),
                "stepped" => Keep(owner["::2"]),
                "broadcast" => Keep(np.broadcast_to(Keep(owner["0:3"]), new Shape(2, 3))),
                "empty" => owner,
                _ => throw new ArgumentOutOfRangeException(nameof(layout), layout, "unknown layout"),
            };

            // Dispose the view and every array it was derived from: only then is the buffer's last reference gone.
            foreach (NDArray a in made)
                a.Dispose();
            return view;
        }
    }
}
